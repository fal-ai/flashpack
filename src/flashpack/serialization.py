import json
import math
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
import tqdm

from .constants import (
    DEFAULT_ALIGN_BYTES,
    DEFAULT_NUM_WRITE_WORKERS,
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    MAGIC,
    U64LE,
)
from .utils import dtype_to_string, get_packing_dtype, timer, torch_dtype_to_numpy_dtype


@dataclass
class TensorIndexRecord:
    name: str
    shape: list[int]
    offset: int  # element offset (not bytes)
    length: int  # number of elements
    macroblock: int = 0


@dataclass
class MacroblockPlan:
    dtype: torch.dtype
    offset_bytes: int
    length_bytes: int
    total_elems: int
    align_elems: int
    tensors: list[TensorIndexRecord]


def pack_to_file(
    state_dict_or_model: dict[str, torch.Tensor] | torch.nn.Module,
    destination_path: str,
    target_dtype: torch.dtype | None,
    name_order: list[str] | None = None,
    align_bytes: int = DEFAULT_ALIGN_BYTES,
    silent: bool = True,
    num_workers: int = DEFAULT_NUM_WRITE_WORKERS,
) -> None:
    """
    Pack the state dictionary or model to a flashpack file.
    """
    if isinstance(state_dict_or_model, torch.nn.Module):
        state_dict = state_dict_or_model.state_dict()
    else:
        state_dict = state_dict_or_model

    keys = list(state_dict.keys())
    if name_order is None:
        # Sort by size (largest first) for better UX
        names = sorted(keys, key=lambda k: state_dict[k].numel(), reverse=True)
    else:
        name_set = set(keys)
        names = [n for n in name_order if n in name_set]

    if not names:
        raise ValueError("No tensors to pack.")

    if align_bytes < 0:
        raise ValueError("align_bytes must be >= 0")

    def _validate_dtype(dtype: torch.dtype) -> torch.dtype:
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported dtype in state dict: {dtype}")
        torch_dtype_to_numpy_dtype(dtype)
        return dtype

    def _lcm(a: int, b: int) -> int:
        if a == 0 and b == 0:
            return 0
        if a == 0:
            return abs(b)
        if b == 0:
            return abs(a)
        return abs(a * b) // math.gcd(a, b)

    resolved_target_dtype = (
        _validate_dtype(target_dtype) if target_dtype is not None else None
    )

    dtype_to_names: dict[torch.dtype, list[str]] = {}
    dtype_order: list[torch.dtype] = []
    for name in names:
        tensor = state_dict[name]
        write_dtype = resolved_target_dtype or _validate_dtype(tensor.dtype)
        if write_dtype not in dtype_to_names:
            dtype_to_names[write_dtype] = []
            dtype_order.append(write_dtype)
        dtype_to_names[write_dtype].append(name)

    with timer("build_index", silent):
        macroblocks: list[MacroblockPlan] = []
        index: list[TensorIndexRecord] = []
        file_cursor = 0  # bytes

        for block_id, dtype in enumerate(dtype_order):
            names_for_dtype = dtype_to_names[dtype]
            elem_size = torch.tensor([], dtype=dtype).element_size()
            block_alignment = _lcm(align_bytes, elem_size) if align_bytes else elem_size
            if block_alignment:
                pad_bytes = (-file_cursor) % block_alignment
                file_cursor += pad_bytes
            block_offset = file_cursor

            g = math.gcd(align_bytes, elem_size) if align_bytes else 1
            align_elems = (align_bytes // g) if align_bytes else 0

            block_cursor = 0
            block_records: list[TensorIndexRecord] = []
            for name in names_for_dtype:
                tensor = state_dict[name]
                n = tensor.numel()
                if align_elems:
                    pad_elems = (-block_cursor) % align_elems
                    block_cursor += pad_elems

                rec = TensorIndexRecord(
                    name=name,
                    shape=list(tensor.shape),
                    offset=block_cursor,
                    length=n,
                    macroblock=block_id,
                )
                block_records.append(rec)
                index.append(rec)
                block_cursor += n

            block_size_bytes = block_cursor * elem_size
            macroblocks.append(
                MacroblockPlan(
                    dtype=dtype,
                    offset_bytes=block_offset,
                    length_bytes=block_size_bytes,
                    total_elems=block_cursor,
                    align_elems=align_elems,
                    tensors=block_records,
                )
            )
            file_cursor = block_offset + block_size_bytes

    total_payload_bytes = file_cursor
    if total_payload_bytes == 0:
        raise ValueError("Nothing to pack after alignment.")

    dest_dir = os.path.dirname(os.path.abspath(destination_path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd_tmp = None
    tmp_path = None

    try:
        # Create tempfile alongside destination
        fd_tmp, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".packtmp_")
        os.close(fd_tmp)

        with timer("create_memmap", silent):
            mm = np.memmap(
                tmp_path, dtype=np.uint8, mode="w+", shape=(total_payload_bytes,)
            )

            block_numpy_views: list[np.ndarray] = []
            block_views: list[torch.Tensor] = []
            for block in macroblocks:
                block_slice = mm[
                    block.offset_bytes : block.offset_bytes + block.length_bytes
                ]
                np_dtype = torch_dtype_to_numpy_dtype(block.dtype)
                typed_view = block_slice.view(np_dtype)
                block_numpy_views.append(typed_view)
                block_views.append(torch.from_numpy(typed_view))

        # Optimized copy: sequential with batched progress updates
        with timer("copy_to_memmap", silent):
            # Only show progress if not silent
            if not silent:
                progress = tqdm.tqdm(desc="Copying to memmap", total=len(index))

            # Determine if we should use any parallelism
            # Only use threads if we have GPU tensors that need transfer
            has_gpu_tensors = any(state_dict[rec.name].is_cuda for rec in index)
            use_parallel = has_gpu_tensors and num_workers > 1

            if use_parallel:
                # Use minimal parallelism (4 workers max) for GPU->CPU transfer overlap
                from concurrent.futures import ThreadPoolExecutor, as_completed

                actual_workers = min(4, num_workers)

                def copy_one(rec: TensorIndexRecord) -> None:
                    block = macroblocks[rec.macroblock]
                    dst_block = block_views[rec.macroblock]
                    src = state_dict[rec.name]
                    target_dtype = block.dtype
                    packing_dtype = get_packing_dtype(target_dtype)

                    if target_dtype != packing_dtype:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        src_bits = src_cpu.view(packing_dtype)
                        dst = dst_block.narrow(0, rec.offset, rec.length).view(
                            packing_dtype
                        )
                        dst.copy_(src_bits, non_blocking=False)
                    else:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        dst = dst_block.narrow(0, rec.offset, rec.length)
                        dst.copy_(src_cpu, non_blocking=False)

                with ThreadPoolExecutor(max_workers=actual_workers) as ex:
                    futures = [ex.submit(copy_one, rec) for rec in index]

                    # Update progress in batches
                    batch_size = max(1, len(futures) // 100)
                    for i, future in enumerate(as_completed(futures)):
                        future.result()
                        if not silent and (
                            i % batch_size == 0 or i == len(futures) - 1
                        ):
                            progress.update(
                                batch_size
                                if i + batch_size < len(futures)
                                else len(futures) - progress.n
                            )
            else:
                # Sequential processing for CPU tensors (fastest!)
                progress_update_interval = max(1, len(index) // 100)

                for i, rec in enumerate(index):
                    block = macroblocks[rec.macroblock]
                    dst_block = block_views[rec.macroblock]
                    src = state_dict[rec.name]
                    target_dtype = block.dtype
                    packing_dtype = get_packing_dtype(target_dtype)

                    if target_dtype != packing_dtype:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        src_bits = src_cpu.view(packing_dtype)
                        dst = dst_block.narrow(0, rec.offset, rec.length).view(
                            packing_dtype
                        )
                        dst.copy_(src_bits, non_blocking=False)
                    else:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        dst = dst_block.narrow(0, rec.offset, rec.length)
                        dst.copy_(src_cpu, non_blocking=False)

                    # Batch progress updates to reduce overhead
                    if not silent and (
                        i % progress_update_interval == 0 or i == len(index) - 1
                    ):
                        progress.update(
                            min(progress_update_interval, len(index) - progress.n)
                        )

            if not silent:
                progress.close()

        # Single sync operation (no double flush+fsync)
        with timer("flush_payload", silent):
            # Flush memory map
            mm.flush()

        # Append footer
        if len(macroblocks) == 1:
            block = macroblocks[0]
            meta_payload = {
                "format": FILE_FORMAT_V3,
                "target_dtype": dtype_to_string(block.dtype),
                "align_bytes": int(align_bytes),
                "total_elems": int(block.total_elems),
                "index": [
                    {
                        "name": r.name,
                        "shape": r.shape,
                        "offset": int(r.offset),
                        "length": int(r.length),
                    }
                    for r in index
                ],
            }
        else:
            meta_payload = {
                "format": FILE_FORMAT_V4,
                "align_bytes": int(align_bytes),
                "total_payload_bytes": int(total_payload_bytes),
                "total_elems": sum(block.total_elems for block in macroblocks),
                "macroblocks": [
                    {
                        "dtype": dtype_to_string(block.dtype),
                        "offset_bytes": int(block.offset_bytes),
                        "length_bytes": int(block.length_bytes),
                        "length_elems": int(block.total_elems),
                    }
                    for block in macroblocks
                ],
                "index": [
                    {
                        "name": r.name,
                        "shape": r.shape,
                        "offset": int(r.offset),
                        "length": int(r.length),
                        "macroblock": int(r.macroblock),
                    }
                    for r in index
                ],
            }
        footer_json = json.dumps(
            meta_payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

        with timer("append_footer", silent):
            with open(tmp_path, "ab") as f:
                f.write(footer_json)
                f.write(U64LE.pack(len(footer_json)))
                f.write(MAGIC)
                # Single fsync here is enough
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

        # Explicitly close memory map
        # `_mmap` in numpy <= 1.25, `base` in numpy >= 1.26
        mm_base = getattr(mm, "_mmap", None) or getattr(mm, "base", None)
        if mm_base is not None:
            mm_base.close()

        # Atomic replace
        with timer("atomic_rename", silent):
            os.replace(tmp_path, destination_path)
            tmp_path = None

    finally:
        # Cleanup on error
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
