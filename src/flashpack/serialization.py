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
    MAGIC,
    U64LE,
)
from .utils import dtype_to_string, timer, torch_dtype_to_numpy_dtype


@dataclass
class TensorIndexRecord:
    name: str
    shape: list[int]
    offset: int  # element offset (not bytes)
    length: int  # number of elements


def pack_to_file(
    state_dict_or_model: dict[str, torch.Tensor] | torch.nn.Module,
    destination_path: str,
    target_dtype: torch.dtype,
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

    element_size = torch.tensor([], dtype=target_dtype).element_size()
    g = math.gcd(align_bytes, element_size) if align_bytes else 1
    align_elems = (align_bytes // g) if align_bytes else 0

    with timer("build_index", silent):
        index: list[TensorIndexRecord] = []
        elem_cursor = 0
        for name in names:
            t = state_dict[name]
            n = t.numel()

            if align_elems:
                pad_elems = (-elem_cursor) % align_elems
                elem_cursor += pad_elems

            index.append(
                TensorIndexRecord(
                    name=name, shape=list(t.shape), offset=elem_cursor, length=n
                )
            )
            elem_cursor += n

    total_elems = elem_cursor
    if total_elems == 0:
        raise ValueError("Nothing to pack after alignment.")

    np_dtype = torch_dtype_to_numpy_dtype(target_dtype)
    dest_dir = os.path.dirname(os.path.abspath(destination_path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd_tmp = None
    tmp_path = None

    try:
        # Create tempfile alongside destination
        fd_tmp, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".packtmp_")
        os.close(fd_tmp)

        with timer("create_memmap", silent):
            mm = np.memmap(tmp_path, dtype=np_dtype, mode="w+", shape=(total_elems,))
            flash_view = (
                torch.from_numpy(mm)
                if target_dtype != torch.bfloat16
                else torch.from_numpy(mm.view(np.uint16))
            )

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
                    src = state_dict[rec.name]
                    if target_dtype == torch.bfloat16:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        src_bits = src_cpu.view(torch.uint16)
                        dst = flash_view.narrow(0, rec.offset, rec.length).view(
                            torch.uint16
                        )
                        dst.copy_(src_bits, non_blocking=False)
                    else:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        dst = flash_view.narrow(0, rec.offset, rec.length)
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
                    src = state_dict[rec.name]

                    if target_dtype == torch.bfloat16:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        src_bits = src_cpu.view(torch.uint16)
                        dst = flash_view.narrow(0, rec.offset, rec.length).view(
                            torch.uint16
                        )
                        dst.copy_(src_bits, non_blocking=False)
                    else:
                        src_cpu = src.view(-1).to(dtype=target_dtype, device="cpu")
                        dst = flash_view.narrow(0, rec.offset, rec.length)
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
        meta_payload = {
            "format": FILE_FORMAT_V3,
            "target_dtype": dtype_to_string(target_dtype),
            "align_bytes": int(align_bytes),
            "total_elems": int(total_elems),
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
