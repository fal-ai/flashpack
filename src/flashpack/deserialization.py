import json
import math
import os
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .constants import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_NUM_STREAMS,
    FILE_FORMAT_V3,
    MAGIC,
    U64LE,
)
from .utils import (
    get_module_and_attribute,
    human_num_elements,
    is_ignored_tensor_name,
    maybe_init_distributed,
    string_to_dtype,
    timer,
    torch_dtype_to_numpy_dtype,
)


def get_flashpack_file_metadata(path: str) -> dict[str, Any]:
    """
    Get the metadata from a flashpack file.
    """
    st = os.stat(path)
    with open(path, "rb") as f:
        if st.st_size < len(MAGIC) + U64LE.size:
            raise ValueError("File too small to contain footer")

        f.seek(st.st_size - len(MAGIC))
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic} != {MAGIC}")

        f.seek(st.st_size - len(MAGIC) - U64LE.size)
        (json_len,) = U64LE.unpack(f.read(U64LE.size))
        start = st.st_size - len(MAGIC) - U64LE.size - json_len
        if start < 0:
            raise ValueError("Corrupt footer length")

        f.seek(start)
        meta = json.loads(f.read(json_len).decode("utf-8"))
        if meta.get("format") != FILE_FORMAT_V3:
            raise ValueError(f"Unexpected format: {meta.get('format')}")

        return meta


def is_flashpack_file(path: str) -> bool:
    """
    Check if a file is a flashpack file.
    """
    try:
        get_flashpack_file_metadata(path)
        return True
    except Exception:
        return False


def read_flashpack_file(
    path: str,
    device: str | torch.device = "cpu",
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    num_streams: int = DEFAULT_NUM_STREAMS,
    silent: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Read the flashpack file and return the tensor and metadata.
    """
    with timer("read_metadata", silent):
        meta = get_flashpack_file_metadata(path)

    device = torch.device(device) if isinstance(device, str) else device
    target_dtype = string_to_dtype(meta["target_dtype"])
    total_elems = int(meta["total_elems"])
    elem_sz = torch.tensor([], dtype=target_dtype).element_size()

    with timer("mmap_payload", silent):
        np_dtype = torch_dtype_to_numpy_dtype(target_dtype)
        mm = np.memmap(path, dtype=np_dtype, mode="r", shape=(total_elems,))

    # Fast CPU path
    if device.type == "cpu":
        with timer("cpu_from_memmap", silent):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                flash_cpu = (
                    torch.from_numpy(mm)
                    if target_dtype != torch.bfloat16
                    else torch.from_numpy(mm.view(np.uint16)).view(torch.bfloat16)
                )
        return flash_cpu, meta

    if device.type != "cuda":
        raise ValueError(f"Unsupported device: {device}")

    with timer("alloc_device", silent):
        if target_dtype == torch.bfloat16:
            flash_dev = torch.empty(total_elems, dtype=torch.bfloat16, device=device)
            flash_dev_u16 = flash_dev.view(torch.uint16)
        else:
            flash_dev = torch.empty(total_elems, dtype=target_dtype, device=device)
            flash_dev_u16 = None

    # Advise kernel to read ahead (Linux only)
    try:
        import mmap as mmap_module

        # MADV_WILLNEED: tell kernel we'll need this data
        # MADV_SEQUENTIAL: we'll read sequentially
        mm._mmap.madvise(mmap_module.MADV_WILLNEED)
        mm._mmap.madvise(mmap_module.MADV_SEQUENTIAL)
    except:
        pass

    # Tune chunk size for the specific file
    total_bytes = total_elems * elem_sz

    # aim for 100-200 chunks total for good pipelining
    target_num_chunks = 150
    optimal_chunk_bytes = max(chunk_bytes, total_bytes // target_num_chunks)
    # But cap at 64MB to avoid too much staging memory
    optimal_chunk_bytes = min(optimal_chunk_bytes, 64 * 1024 * 1024)

    elems_per_chunk = max(1, (optimal_chunk_bytes // elem_sz))
    n_chunks = (total_elems + elems_per_chunk - 1) // elems_per_chunk

    with timer("read_and_copy", silent):
        # Pre-allocate a small number of pinned staging buffers for pipelining
        num_pipeline_buffers = min(num_streams, 8)  # Don't over-allocate
        dt = torch.uint16 if target_dtype == torch.bfloat16 else target_dtype
        staging_bufs = [
            torch.empty(elems_per_chunk, dtype=dt, pin_memory=True)
            for _ in range(num_pipeline_buffers)
        ]

        # Pre-allocate streams
        streams = [torch.cuda.Stream(device=device) for _ in range(min(num_streams, 8))]

        # Pipeline: fill first buffer, then alternate fill/copy
        for chunk_idx in range(n_chunks):
            start = chunk_idx * elems_per_chunk
            end = min(total_elems, start + elems_per_chunk)
            sz = end - start

            # Select staging buffer (round-robin)
            buf_idx = chunk_idx % num_pipeline_buffers
            buf = staging_bufs[buf_idx].narrow(0, 0, sz)

            # Select stream
            stream = streams[chunk_idx % len(streams)]

            # Wait for this buffer's previous use to complete
            if chunk_idx >= num_pipeline_buffers:
                stream.synchronize()

            # Copy from mmap to staging (CPU)
            np_view = mm[start:end]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                src_t = (
                    torch.from_numpy(np_view)
                    if target_dtype != torch.bfloat16
                    else torch.from_numpy(np_view.view(np.uint16))
                )
            buf.copy_(src_t, non_blocking=False)

            # Copy to device (GPU) on the selected stream
            with torch.cuda.stream(stream):
                if target_dtype == torch.bfloat16:
                    flash_dev_u16.narrow(0, start, sz).copy_(buf, non_blocking=True)
                else:
                    flash_dev.narrow(0, start, sz).copy_(buf, non_blocking=True)

        # Final sync
        torch.cuda.synchronize(device)

    del mm
    return flash_dev, meta


def iterate_from_flash_tensor(
    flash_tensor: torch.Tensor,
    metadata: dict[str, Any],
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Iterate over the tensors stored in the flash tensor.
    """
    index = metadata["index"]

    align_bytes = int(metadata.get("align_bytes", 0))
    if align_bytes:
        esz = flash_tensor.element_size()
        g = math.gcd(align_bytes, esz)
        align_elems = align_bytes // g
        bad = [rec for rec in index if (int(rec["offset"]) % align_elems) != 0]
        if bad:
            names = ", ".join(r["name"] for r in bad[:3])
            raise ValueError(
                f"{len(bad)} index entries are misaligned (e.g., {names})."
            )

    for rec in index:
        name = rec["name"]
        if is_ignored_tensor_name(name, ignore_names, ignore_prefixes, ignore_suffixes):
            continue

        shape = tuple(rec["shape"]) or (1,)
        off = int(rec["offset"])
        n = int(rec["length"])

        try:
            view = flash_tensor.narrow(0, off, n).view(
                *shape
            )  # contiguous 1D slice -> reshaped
            yield name, view
        except Exception as e:
            raise ValueError(f"Could not get tensor for record {rec}") from e


def assign_from_file(
    model: torch.nn.Module,
    path: str,
    device: str | torch.device | None = None,
    strict: bool | None = None,
    strict_params: bool = True,
    strict_buffers: bool = False,
    keep_flash_ref_on_model: bool = True,
    silent: bool = True,
    num_streams: int = DEFAULT_NUM_STREAMS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    use_distributed_loading: bool = False,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
    coerce_dtype: bool = False,
) -> None:
    """
    Assign the weights from a flashpack file to a model.
    """
    if device is None:
        try:
            device = model.device
        except AttributeError:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if use_distributed_loading:
        maybe_init_distributed(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )
        rank = dist.get_rank()
        if rank == 0:
            flash_tensor, meta = read_flashpack_file(
                path=path,
                device=device,
                silent=silent,
                num_streams=num_streams,
                chunk_bytes=chunk_bytes,
            )
        else:
            meta = get_flashpack_file_metadata(path)
            flash_tensor = torch.empty(
                meta["total_elems"],
                dtype=string_to_dtype(meta["target_dtype"]),
                device=device,
            )
        dist.broadcast(flash_tensor, src=0)
    else:
        flash_tensor, meta = read_flashpack_file(
            path=path,
            device=device,
            silent=silent,
            num_streams=num_streams,
            chunk_bytes=chunk_bytes,
        )

    if keep_flash_ref_on_model:
        setattr(model, "_flash_shared_storage", flash_tensor)
        setattr(model, "_flash_shared_storage_meta", meta)

    target_dtype = string_to_dtype(meta["target_dtype"])

    with timer("build_lookups", silent):
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

    assigned_param_names = []
    assigned_buffer_names = []
    all_discarded_names = []
    total_elements = 0

    with timer("assign", silent):
        try:
            for name, view in iterate_from_flash_tensor(
                flash_tensor, meta, ignore_names, ignore_prefixes, ignore_suffixes
            ):
                total_elements += view.numel()

                if name in params:
                    module, attr = get_module_and_attribute(model, name)
                    old_param = getattr(module, attr)
                    if not isinstance(old_param, torch.nn.Parameter):
                        raise TypeError(
                            f"Expected parameter at '{name}', got {type(old_param)}"
                        )
                    new_param = torch.nn.Parameter(
                        view, requires_grad=old_param.requires_grad
                    )
                    setattr(module, attr, new_param)
                    assigned_param_names.append(name)
                elif name in buffers:
                    module, attr = get_module_and_attribute(model, name)
                    old_buf = getattr(module, attr)
                    if not torch.is_tensor(old_buf):
                        raise TypeError(
                            f"Expected Tensor buffer at '{name}', got {type(old_buf)}"
                        )
                    if old_buf.dtype != target_dtype:
                        if coerce_dtype:
                            view = view.to(old_buf.dtype)
                        else:
                            raise TypeError(
                                f"dtype mismatch for buffer '{name}': model={old_buf.dtype} vs flash={target_dtype}."
                            )
                    module._buffers[attr] = view
                    assigned_buffer_names.append(name)
                else:
                    all_discarded_names.append(name)
        except Exception as e:
            raise ValueError(
                f"Error while assigning to {type(model).__name__} from {path}"
            ) from e

    if strict or strict_params or strict_buffers:
        if all_discarded_names:
            raise ValueError(
                f"Could not assign {len(all_discarded_names)} names: {all_discarded_names}"
            )

        missing_params = set(params.keys()) - set(assigned_param_names)
        missing_buffers = set(buffers.keys()) - set(assigned_buffer_names)

        missing_params = [
            name
            for name in missing_params
            if not is_ignored_tensor_name(
                name, ignore_names, ignore_prefixes, ignore_suffixes
            )
        ]
        missing_buffers = [
            name
            for name in missing_buffers
            if not is_ignored_tensor_name(
                name, ignore_names, ignore_prefixes, ignore_suffixes
            )
        ]

        is_strict_params = strict_params if strict is None else strict
        is_strict_buffers = strict_buffers if strict is None else strict

        if (
            missing_params
            and missing_buffers
            and is_strict_params
            and is_strict_buffers
        ):
            raise ValueError(
                f"Missing {len(missing_params)} parameters and {len(missing_buffers)} buffers: {missing_params} {missing_buffers}"
            )
        elif missing_params and is_strict_params:
            raise ValueError(
                f"Missing {len(missing_params)} parameters: {missing_params}"
            )
        elif missing_buffers and is_strict_buffers:
            raise ValueError(
                f"Missing {len(missing_buffers)} buffers: {missing_buffers}"
            )

        if missing_buffers and not silent:
            print(f"Ignoring {len(missing_buffers)} buffers: {missing_buffers}")
        if missing_params and not silent:
            print(f"Ignoring {len(missing_params)} parameters: {missing_params}")

    if all_discarded_names and not silent:
        print(f"Discarded {len(all_discarded_names)} names: {all_discarded_names}")

    if not silent:
        print(
            f"Assigned {human_num_elements(total_elements)} total parameters to {len(assigned_param_names)} parameters and {len(assigned_buffer_names)} buffers"
        )
