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
    DEFAULT_ZSTD_LEVEL,
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    FPZ_CODEC_SPLITPLANE_V1,
    FPZ_CODEC_SPLITPLANE_V2,
    FPZ_COMPRESS_BF16,
    FPZ_FRAME_ALIGN_BYTES,
    FPZ_FRAME_UNCOMPRESSED_BYTES,
    FPZ_HI_CHUNK_ALIGN_BYTES,
    FPZ_HI_CHUNK_UNCOMPRESSED_BYTES,
    MAGIC,
    U64LE,
)
from .utils import (
    dtype_to_string,
    get_packing_dtype,
    require_zstandard,
    timer,
    torch_dtype_to_numpy_dtype,
)


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


def _resolve_hi_chunk_bytes(hi_chunk_bytes: int | None) -> int:
    """Resolve and validate the v2 high-plane chunk size.

    Precedence: explicit ``hi_chunk_bytes`` arg > ``FLASHPACK_FPZ_CHUNK_BYTES``
    env (for the converter) > ``FPZ_HI_CHUNK_UNCOMPRESSED_BYTES`` default. An
    explicitly requested value must be a positive multiple of
    ``FPZ_FRAME_ALIGN_BYTES`` and no larger than a frame's high plane
    (``FPZ_FRAME_UNCOMPRESSED_BYTES // 2``). The default is clamped to the frame
    high plane rather than rejected (a chunk >= the high plane just yields one
    chunk per frame -- the case tests hit by shrinking the frame size).
    """
    half_frame = FPZ_FRAME_UNCOMPRESSED_BYTES // 2
    env = os.environ.get("FLASHPACK_FPZ_CHUNK_BYTES")
    if hi_chunk_bytes is None and not env:
        return max(1, min(FPZ_HI_CHUNK_UNCOMPRESSED_BYTES, half_frame))
    requested = hi_chunk_bytes if hi_chunk_bytes is not None else int(env)
    if requested < FPZ_FRAME_ALIGN_BYTES or requested % FPZ_FRAME_ALIGN_BYTES:
        raise ValueError(
            f"hi_chunk_bytes must be a positive multiple of "
            f"{FPZ_FRAME_ALIGN_BYTES} (got {requested})"
        )
    if requested > half_frame:
        raise ValueError(
            f"hi_chunk_bytes ({requested}) exceeds the frame high-plane "
            f"size ({half_frame})"
        )
    return requested


def pack_to_file(
    state_dict_or_model: dict[str, torch.Tensor] | torch.nn.Module,
    destination_path: str,
    target_dtype: torch.dtype | None,
    name_order: list[str] | None = None,
    align_bytes: int = DEFAULT_ALIGN_BYTES,
    silent: bool = True,
    num_workers: int = DEFAULT_NUM_WRITE_WORKERS,
    compress: str | None = None,
    hi_chunk_bytes: int | None = None,
) -> None:
    """
    Pack the state dictionary or model to a flashpack file.

    ``compress="fpz-bf16"`` enables split-plane zstd compression for bf16
    macroblocks only (see ``constants.py``); every other dtype is stored
    uncompressed, and the file falls back to the plain uncompressed format
    when no bf16 macroblock is present. Requires the optional ``zstandard``
    package. ``hi_chunk_bytes`` overrides the v2 high-plane chunk size (default
    ``FPZ_HI_CHUNK_UNCOMPRESSED_BYTES``, or ``FLASHPACK_FPZ_CHUNK_BYTES``);
    larger chunks mean fewer per-chunk wrapper objects for the GPU decoder.
    """
    if compress is not None and compress != FPZ_COMPRESS_BF16:
        raise ValueError(
            f"Unsupported compress option: {compress!r} "
            f"(expected None or {FPZ_COMPRESS_BF16!r})"
        )
    resolved_hi_chunk_bytes = _resolve_hi_chunk_bytes(hi_chunk_bytes)

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

    # fpz path: bf16 macroblocks are split-plane zstd-compressed, every other
    # dtype is stored verbatim. When no block is eligible the file is identical
    # to the uncompressed pack, so fall through to the plain path below.
    compress_flags = [
        compress == FPZ_COMPRESS_BF16 and block.dtype is torch.bfloat16
        for block in macroblocks
    ]
    if any(compress_flags):
        # Single pass, no uncompressed scratch: convert each tensor to CPU
        # bytes and stream them through a rolling frame buffer directly into
        # the final compressed file.
        with timer("fpz_stream_write", silent):
            _write_fpz_pack_streaming(
                state_dict=state_dict,
                macroblocks=macroblocks,
                index=index,
                align_bytes=align_bytes,
                compress_flags=compress_flags,
                destination_path=destination_path,
                dest_dir=dest_dir,
                silent=silent,
                hi_chunk_bytes=resolved_hi_chunk_bytes,
            )
        return

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


def _write_zeros(f, n: int) -> None:
    """Write ``n`` zero bytes to ``f`` in bounded chunks."""
    while n > 0:
        take = min(n, 1 << 20)
        f.write(b"\x00" * take)
        n -= take


def _iter_block_uncompressed_chunks(
    block: MacroblockPlan,
    state_dict: dict[str, torch.Tensor],
    progress: "tqdm.tqdm | None" = None,
):
    """Yield a macroblock's uncompressed payload in order, reproducing the
    memmap layout exactly.

    Emits ``("zeros", nbytes)`` for the inter-tensor element-alignment gaps
    (zero-filled, as a fresh memmap is) and ``("bytes", uint8_ndarray)`` for
    each tensor -- the target-dtype CPU reinterpretation (packing view) that
    the uncompressed copy loop writes. ``.to(device="cpu")`` handles the D2H
    transfer for GPU-source tensors.
    """
    elem_size = torch.tensor([], dtype=block.dtype).element_size()
    packing_dtype = get_packing_dtype(block.dtype)
    cursor_elems = 0
    for rec in block.tensors:
        if rec.offset > cursor_elems:
            yield ("zeros", (rec.offset - cursor_elems) * elem_size)
            cursor_elems = rec.offset
        src = state_dict[rec.name]
        src_cpu = src.view(-1).to(dtype=block.dtype, device="cpu")
        if block.dtype != packing_dtype:
            src_cpu = src_cpu.view(packing_dtype)
        raw = src_cpu.contiguous().view(torch.uint8).numpy()
        yield ("bytes", raw)
        cursor_elems += rec.length
        if progress is not None:
            progress.update(1)
    if block.total_elems > cursor_elems:
        yield ("zeros", (block.total_elems - cursor_elems) * elem_size)


def _fpz_encode_frame(f, block_start: int, frame_u8: np.ndarray, compressor) -> dict:
    """Encode one split-plane zstd frame from ``frame_u8`` (the uncompressed
    bytes of a single frame) and write it to ``f``.

    bf16 elements are little-endian, so even bytes are the low (mantissa-LSB)
    plane -- kept raw -- and odd bytes are the high (sign+exponent) plane --
    zstd-compressed. The frame payload start is padded to a 4096-byte boundary
    relative to the macroblock start.
    """
    n_out = int(frame_u8.shape[0])
    lo = np.ascontiguousarray(frame_u8[0::2])
    hi = np.ascontiguousarray(frame_u8[1::2])
    lo_bytes = lo.tobytes()
    hi_z = compressor.compress(hi.tobytes())

    payload_off = f.tell() - block_start
    pad = (-payload_off) % FPZ_FRAME_ALIGN_BYTES
    if pad:
        f.write(b"\x00" * pad)
        payload_off += pad

    f.write(lo_bytes)
    f.write(hi_z)
    return {
        "payload_off": int(payload_off),
        "lo_len": int(len(lo_bytes)),
        "hi_len": int(len(hi_z)),
        "n_out": int(n_out),
    }


# Default fpz codec version written by the streaming encoder. v2 (chunked high
# plane) is the GPU-decodable format; tests set this to 1 to exercise the
# v1-still-reads backward-compatibility path.
_DEFAULT_FPZ_VERSION = 2


def _fpz_encode_frame_v2(
    f, block_start: int, frame_u8: np.ndarray, chunk_compressor, chunk: int
) -> dict:
    """Encode one split-plane frame with a CHUNKED high plane (codec v2).

    Same low/high split as v1, but the high plane is compressed as a sequence of
    independent zstd frames of ``chunk`` uncompressed bytes each (the frame's
    last chunk holds the remainder). Many small chunks are what a GPU decoder
    needs to decompress in parallel; the frame record lists each chunk's
    compressed length so the reader locates them by prefix sum. Larger chunks
    mean fewer per-chunk wrapper objects for the GPU decoder to build (the read
    bottleneck once decode is parallel) at the cost of slightly less parallelism
    and a hair less ratio. Ratio drops slightly versus v1 because each chunk
    compresses without the neighbouring chunks' context.
    """
    n_out = int(frame_u8.shape[0])
    lo = np.ascontiguousarray(frame_u8[0::2])
    hi = np.ascontiguousarray(frame_u8[1::2])
    lo_bytes = lo.tobytes()
    half = int(hi.shape[0])
    hi_z_chunks = [
        chunk_compressor.compress(hi[off : off + chunk].tobytes())
        for off in range(0, half, chunk)
    ]

    payload_off = f.tell() - block_start
    pad = (-payload_off) % FPZ_FRAME_ALIGN_BYTES
    if pad:
        f.write(b"\x00" * pad)
        payload_off += pad

    f.write(lo_bytes)
    # Pad after the lo plane and after every chunk so each chunk STARTS
    # hi_align-aligned within the (FPZ_FRAME_ALIGN_BYTES-aligned) payload:
    # batched GPU decode requires aligned device chunk pointers. A full
    # frame's lo plane (32 MiB) is already aligned, but the tail frame's
    # arbitrary half-length is not. "hi_chunks" records TRUE zstd lengths;
    # the reader recomputes padded offsets from the footer's "hi_align".
    pad = (-len(lo_bytes)) % FPZ_HI_CHUNK_ALIGN_BYTES
    if pad:
        f.write(b"\x00" * pad)
    hi_chunks: list[int] = []
    for z in hi_z_chunks:
        f.write(z)
        hi_chunks.append(int(len(z)))
        pad = (-len(z)) % FPZ_HI_CHUNK_ALIGN_BYTES
        if pad:
            f.write(b"\x00" * pad)
    return {
        "payload_off": int(payload_off),
        "lo_len": int(len(lo_bytes)),
        "n_out": int(n_out),
        "hi_chunks": hi_chunks,
    }


def _fpz_stream_compress_block(
    f,
    block_start: int,
    block: MacroblockPlan,
    state_dict: dict[str, torch.Tensor],
    encode_frame,
    progress: "tqdm.tqdm | None",
) -> list[dict]:
    """Stream one bf16 macroblock through a rolling ``FPZ_FRAME_UNCOMPRESSED_BYTES``
    buffer, emitting a split-plane frame (via ``encode_frame``) each time it
    fills (and once more for the tail). Peak extra memory is one frame buffer
    plus one source tensor."""
    frame_bytes = FPZ_FRAME_UNCOMPRESSED_BYTES
    buf = np.empty(frame_bytes, dtype=np.uint8)
    fill = 0
    frames: list[dict] = []

    for kind, data in _iter_block_uncompressed_chunks(block, state_dict, progress):
        if kind == "zeros":
            remaining = data
            while remaining > 0:
                take = min(remaining, frame_bytes - fill)
                buf[fill : fill + take] = 0
                fill += take
                remaining -= take
                if fill == frame_bytes:
                    frames.append(encode_frame(f, block_start, buf))
                    fill = 0
        else:
            arr = data
            pos = 0
            n = int(arr.shape[0])
            while pos < n:
                take = min(n - pos, frame_bytes - fill)
                buf[fill : fill + take] = arr[pos : pos + take]
                fill += take
                pos += take
                if fill == frame_bytes:
                    frames.append(encode_frame(f, block_start, buf))
                    fill = 0

    if fill > 0:
        frames.append(encode_frame(f, block_start, buf[:fill]))
    return frames


def _write_fpz_pack_streaming(
    state_dict: dict[str, torch.Tensor],
    macroblocks: list[MacroblockPlan],
    index: list[TensorIndexRecord],
    align_bytes: int,
    compress_flags: list[bool],
    destination_path: str,
    dest_dir: str,
    silent: bool,
    hi_chunk_bytes: int,
) -> None:
    """Write a compressed (fpz) pack to ``destination_path`` atomically in a
    single pass -- no uncompressed scratch file.

    Each macroblock's payload is produced on the fly from ``state_dict`` (same
    dtype conversion and inter-tensor alignment as the uncompressed planner)
    and either streamed through split-plane zstd frames (bf16 blocks, per
    ``compress_flags``) or written verbatim. The footer/frame format is
    byte-compatible with the read path. ``hi_chunk_bytes`` is the v2 high-plane
    chunk size, recorded per fpz block so the reader reproduces the chunking.
    """
    zstandard = require_zstandard()
    version = _DEFAULT_FPZ_VERSION
    if version == 2:
        # v2 compresses each high-plane chunk (hi_chunk_bytes uncompressed) as
        # its own zstd frame. threads=-1 (one worker per core) does nothing for
        # a small input and only adds per-call overhead, so use a single-threaded
        # compressor; parallelism at repack time now comes from the many chunks,
        # not from one big multithreaded compress. (Chunks are compressed
        # serially here; a chunk-level thread pool is a repack-speed follow-up.)
        chunk_compressor = zstandard.ZstdCompressor(level=DEFAULT_ZSTD_LEVEL)

        def encode_frame(f_, block_start_, frame_u8_):
            return _fpz_encode_frame_v2(
                f_, block_start_, frame_u8_, chunk_compressor, hi_chunk_bytes
            )

        codec_name = FPZ_CODEC_SPLITPLANE_V2
    else:
        # threads=-1 = one worker per core: a ~19GB high plane at single-threaded
        # zstd-3 (~0.4 GB/s) would take ~45 min per repack; multithreaded frames
        # keep converter jobs in minutes. Frame outputs are byte-compatible.
        compressor = zstandard.ZstdCompressor(level=DEFAULT_ZSTD_LEVEL, threads=-1)

        def encode_frame(f_, block_start_, frame_u8_):
            return _fpz_encode_frame(f_, block_start_, frame_u8_, compressor)

        codec_name = FPZ_CODEC_SPLITPLANE_V1

    fd_tmp, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".packtmp_")
    os.close(fd_tmp)
    progress = None
    if not silent:
        progress = tqdm.tqdm(desc="Packing (fpz)", total=len(index))
    try:
        macroblock_records: list[dict] = []
        with open(tmp_path, "wb") as f:
            for block_id, block in enumerate(macroblocks):
                elem_size = torch.tensor([], dtype=block.dtype).element_size()
                block_alignment = (
                    math.lcm(align_bytes, elem_size) if align_bytes else elem_size
                )
                if block_alignment:
                    pad = (-f.tell()) % block_alignment
                    if pad:
                        f.write(b"\x00" * pad)
                block_offset = f.tell()

                record = {
                    "dtype": dtype_to_string(block.dtype),
                    "offset_bytes": int(block_offset),
                    "length_elems": int(block.total_elems),
                }
                if compress_flags[block_id]:
                    frames = _fpz_stream_compress_block(
                        f, block_offset, block, state_dict, encode_frame, progress
                    )
                    record["length_bytes"] = int(f.tell() - block_offset)
                    fpz_record: dict = {"codec": codec_name, "frames": frames}
                    if version == 2:
                        # Record the chunk size so the reader reproduces the
                        # chunking regardless of the current default, and the
                        # chunk-start alignment so it can recompute the padded
                        # offsets (absent = 1: pre-alignment packed layout).
                        fpz_record["hi_chunk_usize"] = int(hi_chunk_bytes)
                        fpz_record["hi_align"] = FPZ_HI_CHUNK_ALIGN_BYTES
                    record["fpz"] = fpz_record
                else:
                    for kind, data in _iter_block_uncompressed_chunks(
                        block, state_dict, progress
                    ):
                        if kind == "zeros":
                            _write_zeros(f, data)
                        else:
                            f.write(data)
                    record["length_bytes"] = int(block.length_bytes)
                macroblock_records.append(record)

            total_payload_bytes = f.tell()
            meta_payload = {
                "format": FILE_FORMAT_V4,
                "align_bytes": int(align_bytes),
                "total_payload_bytes": int(total_payload_bytes),
                "total_elems": sum(block.total_elems for block in macroblocks),
                "macroblocks": macroblock_records,
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
            f.write(footer_json)
            f.write(U64LE.pack(len(footer_json)))
            f.write(MAGIC)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        os.replace(tmp_path, destination_path)
        tmp_path = None
    finally:
        if progress is not None:
            progress.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
