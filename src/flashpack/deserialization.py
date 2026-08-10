import json
import logging
import math
import os
import queue
import threading
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tqdm

from .constants import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_NUM_STREAMS,
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    FILE_FORMAT_V5,
    FPZ_CODEC_SPLITPLANE_V1,
    FPZ_CODEC_SPLITPLANE_V2,
    FPZ_FRAME_UNCOMPRESSED_BYTES,
    FPZ_HI_CHUNK_UNCOMPRESSED_BYTES,
    MAGIC,
    U64LE,
)
from .parallel_read import (
    parallel_read_into_storage,
    parallel_read_supported,
)
from .utils import (
    effective_read_threads,
    get_module_and_attribute,
    get_packing_dtype,
    human_num_elements,
    is_ignored_tensor_name,
    maybe_init_distributed,
    require_zstandard,
    string_to_dtype,
    timer,
    torch_dtype_to_numpy_dtype,
)

logger = logging.getLogger(__name__)


@dataclass
class MacroblockSpec:
    dtype: torch.dtype
    offset_bytes: int
    length_bytes: int
    length_elems: int
    # For fpz-compressed blocks: the {"codec", "frames": [...]} record from the
    # footer. None for plain (uncompressed) blocks. offset_bytes/length_bytes
    # describe the compressed payload as stored; length_elems is always the
    # logical (uncompressed) element count.
    fpz: dict[str, Any] | None = None

    @property
    def uncompressed_bytes(self) -> int:
        return self.length_elems * torch.tensor([], dtype=self.dtype).element_size()


@dataclass
class FlashTensorStorage:
    blocks: list[torch.Tensor]
    backing_arrays: list[np.memmap] | None = None

    def block(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def device(self) -> torch.device:
        if not self.blocks:
            return torch.device("cpu")
        return self.blocks[0].device


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
        fmt = meta.get("format")
        if fmt not in (FILE_FORMAT_V3, FILE_FORMAT_V4, FILE_FORMAT_V5):
            raise ValueError(f"Unexpected format: {fmt}")

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


def _ensure_index_macroblocks(meta: dict[str, Any], num_blocks: int) -> None:
    index = meta.get("index", [])
    for rec in index:
        block_id = rec.get("macroblock")
        if block_id is None:
            block_id = 0
            rec["macroblock"] = block_id
        block_id = int(block_id)
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(
                f"Index entry references macroblock {block_id}, but only {num_blocks} blocks exist."
            )


def _build_macroblock_specs(meta: dict[str, Any]) -> list[MacroblockSpec]:
    fmt = meta.get("format")
    specs: list[MacroblockSpec] = []
    if fmt == FILE_FORMAT_V3:
        dtype = string_to_dtype(meta["target_dtype"])
        total_elems = int(meta["total_elems"])
        elem_sz = torch.tensor([], dtype=dtype).element_size()
        specs.append(
            MacroblockSpec(
                dtype=dtype,
                offset_bytes=0,
                length_bytes=total_elems * elem_sz,
                length_elems=total_elems,
            )
        )
    elif fmt in (FILE_FORMAT_V4, FILE_FORMAT_V5):
        macroblocks = meta.get("macroblocks")
        if not macroblocks:
            raise ValueError("Missing macroblock metadata for flashpack v4 file.")
        for block in macroblocks:
            dtype = string_to_dtype(block["dtype"])
            fpz = block.get("fpz")
            if fpz is not None:
                codec = fpz.get("codec")
                if codec not in (FPZ_CODEC_SPLITPLANE_V1, FPZ_CODEC_SPLITPLANE_V2):
                    raise ValueError(f"Unsupported fpz codec: {codec!r}")
            specs.append(
                MacroblockSpec(
                    dtype=dtype,
                    offset_bytes=int(block["offset_bytes"]),
                    length_bytes=int(block["length_bytes"]),
                    length_elems=int(block["length_elems"]),
                    fpz=fpz,
                )
            )
    else:
        raise ValueError(f"Unsupported flashpack format: {fmt}")

    _ensure_index_macroblocks(meta, len(specs))
    return specs


def _madvise_memmap(mm: np.memmap) -> None:
    try:
        import mmap as mmap_module

        mm._mmap.madvise(mmap_module.MADV_WILLNEED)
        mm._mmap.madvise(mmap_module.MADV_SEQUENTIAL)
    except Exception:
        pass


def _open_memmaps(path: str, specs: list[MacroblockSpec]) -> list[np.memmap]:
    memmaps = []
    for spec in specs:
        np_dtype = torch_dtype_to_numpy_dtype(spec.dtype)
        mm = np.memmap(
            path,
            dtype=np_dtype,
            mode="r",
            offset=spec.offset_bytes,
            shape=(spec.length_elems,),
        )
        _madvise_memmap(mm)
        memmaps.append(mm)
    return memmaps


def _cpu_storage_from_memmaps(
    memmaps: list[np.memmap], specs: list[MacroblockSpec]
) -> FlashTensorStorage:
    blocks: list[torch.Tensor] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for mm, spec in zip(memmaps, specs):
            tensor = torch.from_numpy(mm)
            packing_dtype = get_packing_dtype(spec.dtype)
            if spec.dtype != packing_dtype:
                tensor = tensor.view(spec.dtype)
            blocks.append(tensor)
    return FlashTensorStorage(blocks=blocks, backing_arrays=memmaps)


def _copy_memmaps_into_storage(
    memmaps: list[np.memmap],
    specs: list[MacroblockSpec],
    storage: FlashTensorStorage,
    device: torch.device,
    chunk_bytes: int,
    num_streams: int,
) -> None:
    # Drain the copy streams even on error: an exception escaping while
    # non-blocking H2D copies are still in flight lets the caller free the
    # destination blocks, which the caching allocator may hand to a retry
    # while the copy engine is still writing — silent weight corruption.
    try:
        _copy_memmaps_into_storage_inner(
            memmaps, specs, storage, device, chunk_bytes, num_streams
        )
    finally:
        torch.cuda.synchronize(device)


def _copy_memmaps_into_storage_inner(
    memmaps: list[np.memmap],
    specs: list[MacroblockSpec],
    storage: FlashTensorStorage,
    device: torch.device,
    chunk_bytes: int,
    num_streams: int,
) -> None:
    for idx, (mm, spec) in enumerate(zip(memmaps, specs)):
        total_elems = spec.length_elems
        elem_sz = torch.tensor([], dtype=spec.dtype).element_size()
        total_bytes = total_elems * elem_sz

        target_num_chunks = 150
        optimal_chunk_bytes = max(chunk_bytes, total_bytes // max(target_num_chunks, 1))
        optimal_chunk_bytes = min(optimal_chunk_bytes, 64 * 1024 * 1024)
        elems_per_chunk = max(1, (optimal_chunk_bytes // max(elem_sz, 1)))
        n_chunks = (total_elems + elems_per_chunk - 1) // elems_per_chunk

        block_tensor = storage.block(idx)
        num_pipeline_buffers = max(1, min(num_streams, 8))

        # For dtypes that require bit-reinterpretation (e.g. bfloat16 stored as uint16),
        # allocate staging buffers in the packing dtype
        packing_dtype = get_packing_dtype(spec.dtype)
        staging_bufs = [
            torch.empty(elems_per_chunk, dtype=packing_dtype, pin_memory=True)
            for _ in range(num_pipeline_buffers)
        ]
        num_cuda_streams = max(1, min(num_streams, 8))
        streams = [torch.cuda.Stream(device=device) for _ in range(num_cuda_streams)]

        for chunk_idx in range(n_chunks):
            start = chunk_idx * elems_per_chunk
            end = min(total_elems, start + elems_per_chunk)
            sz = end - start

            buf_idx = chunk_idx % num_pipeline_buffers
            buf_raw = staging_bufs[buf_idx].narrow(0, 0, sz)
            stream = streams[chunk_idx % num_cuda_streams]

            if chunk_idx >= num_pipeline_buffers:
                stream.synchronize()

            np_view = mm[start:end]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                src_t = torch.from_numpy(np_view)
            buf_raw.copy_(src_t, non_blocking=False)

            # Reinterpret bits if needed (e.g. uint16 -> bfloat16)
            if spec.dtype != packing_dtype:
                buf = buf_raw.view(spec.dtype)
            else:
                buf = buf_raw

            with torch.cuda.stream(stream):
                block_tensor.narrow(0, start, sz).copy_(buf, non_blocking=True)

        torch.cuda.synchronize(device)
    return None


def _allocate_empty_storage(
    specs: list[MacroblockSpec], device: torch.device
) -> FlashTensorStorage:
    blocks = [
        torch.empty(spec.length_elems, dtype=spec.dtype, device=device)
        for spec in specs
    ]
    return FlashTensorStorage(blocks=blocks)


def _allocate_aligned_cpu_storage(specs: list[MacroblockSpec]) -> FlashTensorStorage:
    """CPU blocks whose base address is 4096-byte aligned, so the parallel
    reader's O_DIRECT fast path applies to whole macroblocks. torch's CPU
    allocator only guarantees 64-byte alignment, so over-allocate raw bytes
    and slice at the aligned offset (the view keeps the raw storage alive).
    """
    align = 4096
    blocks: list[torch.Tensor] = []
    for spec in specs:
        # Size by the logical (uncompressed) byte count. This equals
        # spec.length_bytes for plain blocks, but for fpz blocks length_bytes
        # is the smaller compressed on-disk size, so the destination must be
        # sized from the element count instead.
        nbytes = spec.uncompressed_bytes
        raw = torch.empty(nbytes + align, dtype=torch.uint8)
        off = (-raw.data_ptr()) % align
        packing_dtype = get_packing_dtype(spec.dtype)
        block = raw.narrow(0, off, nbytes).view(packing_dtype)
        if spec.dtype != packing_dtype:
            block = block.view(spec.dtype)
        blocks.append(block)
    return FlashTensorStorage(blocks=blocks)


def _broadcast_storage(storage: FlashTensorStorage, src: int) -> None:
    """Broadcast every macroblock from ``src`` to all ranks, as raw bytes.

    Blocks are broadcast through a ``uint8`` view rather than their native
    dtype: the collective only moves bits, and torch's NCCL dtype map does
    not cover every dtype flashpack stores (``float8_e8m0fnu`` -- the mxfp8
    scale dtype -- is absent, so a native-dtype broadcast of an mxfp8 pack
    crashes; gloo similarly lacks the float8 family). The byte view is
    dtype-agnostic and free (no copy).
    """
    for block in storage.blocks:
        dist.broadcast(block.view(torch.uint8), src=src)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


if hasattr(os, "preadv"):

    def _pread_into(fd: int, offset: int, mv: memoryview) -> None:
        """Fill ``mv`` from ``fd`` at ``offset`` with ``preadv`` (reused reader
        machinery). Raises ``IOError`` on a short read (e.g. a truncated
        file)."""
        n = len(mv)
        got = 0
        while got < n:
            r = os.preadv(fd, [mv[got:]], offset + got)
            if r <= 0:
                raise IOError(f"short read: wanted {n} bytes at {offset}, got {got}")
            got += r

else:

    def _pread_into(fd: int, offset: int, mv: memoryview) -> None:
        """Portable fallback (Windows: no ``preadv``): seek + read. Safe
        because every fpz reader thread owns its file descriptor, so the
        fd offset is never shared."""
        n = len(mv)
        got = 0
        while got < n:
            os.lseek(fd, offset + got, os.SEEK_SET)
            b = os.read(fd, n - got)
            if not b:
                raise IOError(f"short read: wanted {n} bytes at {offset}, got {got}")
            mv[got : got + len(b)] = b
            got += len(b)


def _fpz_frame_tasks(specs: list[MacroblockSpec]) -> list[tuple]:
    """Build the per-block decode work list and validate frame coverage.

    Each item is ``("frame", block_idx, frame, out_pos)`` for an fpz frame or
    ``("raw", block_idx, None, 0)`` for a plain block. Raises ``ValueError`` if
    a block's frames do not exactly cover its uncompressed byte length -- the
    same error surface the single-threaded decoder used to raise.
    """
    tasks: list[tuple] = []
    for idx, spec in enumerate(specs):
        if spec.fpz is None:
            tasks.append(("raw", idx, None, 0))
            continue
        total = spec.uncompressed_bytes
        out_pos = 0
        for frame in spec.fpz["frames"]:
            n_out = int(frame["n_out"])
            if out_pos + n_out > total:
                raise ValueError("fpz frames exceed the macroblock size")
            tasks.append(("frame", idx, frame, out_pos))
            out_pos += n_out
        if out_pos != total:
            raise ValueError(f"fpz frames cover {out_pos} bytes, expected {total}")
    return tasks


def _align_up(n: int, align: int) -> int:
    """Round ``n`` up to a multiple of ``align`` (``align`` >= 1)."""
    return n + (-n % align)


def _fpz_read_frame_planes(
    fd: int,
    block_file_offset: int,
    frame: dict[str, Any],
    decompressor,
    chunk_u: int = FPZ_HI_CHUNK_UNCOMPRESSED_BYTES,
    hi_align: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one fpz frame and return its ``(lo, hi)`` byte planes as uint8
    numpy arrays, each ``n_out // 2`` bytes.

    Shared CPU decode step for both read paths and both codec versions. v1
    frames store the high plane as a single zstd frame (``hi_len``); v2 frames
    store it as many ``chunk_u``-uncompressed-byte chunks whose compressed
    lengths are in ``hi_chunks`` (``chunk_u`` is the block's ``hi_chunk_usize``,
    defaulting to the pre-parameterization 64 KiB when absent). zstd
    ``decompress`` releases the GIL and takes the compressed input as a buffer,
    so read targets are ``memoryview``s (no intermediate ``bytes`` copy) and N
    threads scale near linearly.
    """
    payload_off = int(frame["payload_off"])
    lo_len = int(frame["lo_len"])
    n_out = int(frame["n_out"])
    half = n_out - lo_len

    lo_raw = bytearray(lo_len)
    _pread_into(fd, block_file_offset + payload_off, memoryview(lo_raw))
    lo = np.frombuffer(lo_raw, dtype=np.uint8)
    # hi_align-packs pad after the lo plane and after each chunk so every
    # chunk STARTS aligned (GPU batched decode needs aligned device chunk
    # pointers); hi_chunks records true zstd lengths, offsets are padded.
    hi_base = block_file_offset + payload_off + _align_up(lo_len, hi_align)

    if "hi_chunks" in frame:
        # v2: decode each chunk (a standalone zstd frame) into its slice.
        hi = np.empty(half, dtype=np.uint8)
        uoff = 0
        src_off = hi_base
        for clen in frame["hi_chunks"]:
            clen = int(clen)
            usize = min(chunk_u, half - uoff)
            cbuf = bytearray(clen)
            _pread_into(fd, src_off, memoryview(cbuf))
            dec = decompressor.decompress(memoryview(cbuf), max_output_size=usize)
            if len(dec) != usize:
                raise ValueError("fpz v2 chunk size mismatch")
            hi[uoff : uoff + usize] = np.frombuffer(dec, dtype=np.uint8)
            uoff += usize
            src_off += _align_up(clen, hi_align)
        if lo_len * 2 != n_out or uoff != half or lo.shape[0] != lo_len:
            raise ValueError("fpz frame plane size mismatch")
        return lo, hi

    # v1: the high plane is a single zstd frame.
    hi_len = int(frame["hi_len"])
    hi_raw = bytearray(hi_len)
    _pread_into(fd, hi_base, memoryview(hi_raw))
    # memoryview input avoids a GIL-held full copy of the compressed plane;
    # decompress itself releases the GIL.
    hi_bytes = decompressor.decompress(memoryview(hi_raw), max_output_size=half)
    hi = np.frombuffer(hi_bytes, dtype=np.uint8)
    if lo_len * 2 != n_out or lo.shape[0] != lo_len or hi.shape[0] != half:
        raise ValueError("fpz frame plane size mismatch")
    return lo, hi


def _fpz_read_into_cpu_storage(
    path: str, specs: list[MacroblockSpec], blocks: list[torch.Tensor]
) -> None:
    """Fill pre-allocated (uncompressed-sized) CPU ``blocks`` from an fpz file
    with a pool of decode threads.

    Work is one item per fpz frame (or per plain block); frames write disjoint
    destination byte ranges, so threads never collide. Each thread owns a file
    descriptor and a zstd decompressor. The heavy step -- the zstd decode --
    releases the GIL, so throughput scales with ``FLASHPACK_READ_THREADS``
    (default 16) instead of running serially as it did before.
    """
    zstandard = require_zstandard()
    n_threads = effective_read_threads(_env_int("FLASHPACK_READ_THREADS", 16))

    dst_u8 = [b.view(torch.uint8).numpy() for b in blocks]
    tasks = _fpz_frame_tasks(specs)
    n_threads = min(n_threads, max(1, len(tasks)))

    work: queue.SimpleQueue = queue.SimpleQueue()
    for task in tasks:
        work.put(task)
    for _ in range(n_threads):
        work.put(None)

    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            decompressor = zstandard.ZstdDecompressor()
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    kind, blk, frame, out_pos = item
                    if kind == "raw":
                        spec = specs[blk]
                        _pread_into(fd, spec.offset_bytes, memoryview(dst_u8[blk]))
                        continue
                    spec = specs[blk]
                    chunk_u = int(
                        (spec.fpz or {}).get(
                            "hi_chunk_usize", FPZ_HI_CHUNK_UNCOMPRESSED_BYTES
                        )
                    )
                    hi_align = int((spec.fpz or {}).get("hi_align", 1))
                    lo, hi = _fpz_read_frame_planes(
                        fd, spec.offset_bytes, frame, decompressor, chunk_u, hi_align
                    )
                    n_out = int(frame["n_out"])
                    seg = dst_u8[blk][out_pos : out_pos + n_out]
                    # Disjoint destination ranges across threads; numpy releases
                    # the GIL for the strided byte copy.
                    seg[0::2] = lo
                    seg[1::2] = hi
            finally:
                os.close(fd)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_reader, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


_FPZ_CUDA_BUFFERS_PER_THREAD = 2


def _fpz_read_into_cuda_storage(
    path: str,
    specs: list[MacroblockSpec],
    blocks: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Fill pre-allocated device ``blocks`` from an fpz file with a pool of
    reader threads (mirrors ``parallel_read_into_storage``).

    Each reader owns a file descriptor, a CUDA stream, and a small ring of
    double-buffered pinned/device staging slots. A work item is one fpz frame
    (or a whole plain block); frames write disjoint destination segments so the
    readers never collide. Per frame: ``preadv`` the low plane straight into a
    pinned buffer, zstd-decode the high plane (GIL released) and copy it into a
    pinned buffer, then enqueue on the stream the two H2Ds and the two strided
    GPU copies (``dst_u8[0::2] = lo``, ``dst_u8[1::2] = hi``). A GPU decoder
    (nvcomp) would slot in by replacing the decompress step.

    There is NO per-frame ``stream.synchronize()``: a CUDA event per staging
    slot gates only buffer reuse, so a thread reads/decodes the next frame while
    the GPU is still consuming the previous one. Removing that per-frame sync
    (and the single-buffered staging) is the fix for the observed ~2.6 GB/s
    stall -- the CPU decode now overlaps the H2D/copy instead of blocking on it.

    GPU-untested locally (no CUDA device); the frame read + decode is exercised
    by the CPU tests via the shared ``_fpz_read_frame_planes`` helper.
    """
    zstandard = require_zstandard()
    n_threads = effective_read_threads(_env_int("FLASHPACK_READ_THREADS", 16))
    half_cap = FPZ_FRAME_UNCOMPRESSED_BYTES // 2
    n_slots = _FPZ_CUDA_BUFFERS_PER_THREAD

    byte_blocks = [b.view(torch.uint8) for b in blocks]

    # Order every reader stream after the destination allocation (same
    # wait_event pattern as parallel_read_into_storage).
    alloc_ready = torch.cuda.Event()
    alloc_ready.record(torch.cuda.current_stream(device))

    tasks = _fpz_frame_tasks(specs)
    work: queue.SimpleQueue = queue.SimpleQueue()
    for task in tasks:
        work.put(task)
    n_threads = min(n_threads, max(1, len(tasks)))
    for _ in range(n_threads):
        work.put(None)

    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            decompressor = zstandard.ZstdDecompressor()
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(alloc_ready)
            lo_pin = [
                torch.empty(half_cap, dtype=torch.uint8, pin_memory=True)
                for _ in range(n_slots)
            ]
            hi_pin = [
                torch.empty(half_cap, dtype=torch.uint8, pin_memory=True)
                for _ in range(n_slots)
            ]
            lo_dev = [
                torch.empty(half_cap, dtype=torch.uint8, device=device)
                for _ in range(n_slots)
            ]
            hi_dev = [
                torch.empty(half_cap, dtype=torch.uint8, device=device)
                for _ in range(n_slots)
            ]
            events = [torch.cuda.Event() for _ in range(n_slots)]
            for ev in events:
                ev.record(stream)
            i = 0
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    kind, blk, frame, out_pos = item
                    dst = byte_blocks[blk]
                    if kind == "raw":
                        spec = specs[blk]
                        buf = bytearray(spec.length_bytes)
                        _pread_into(fd, spec.offset_bytes, memoryview(buf))
                        host = torch.frombuffer(buf, dtype=torch.uint8)
                        stream.synchronize()
                        with torch.cuda.stream(stream):
                            dst.copy_(host, non_blocking=False)
                        continue

                    slot = i % n_slots
                    i += 1
                    # The slot's previous H2D must be done before we overwrite
                    # its pinned buffers.
                    events[slot].synchronize()

                    n_out = int(frame["n_out"])
                    spec = specs[blk]
                    chunk_u = int(
                        (spec.fpz or {}).get(
                            "hi_chunk_usize", FPZ_HI_CHUNK_UNCOMPRESSED_BYTES
                        )
                    )
                    hi_align = int((spec.fpz or {}).get("hi_align", 1))
                    # Shared CPU decode (handles both v1 single-frame and v2
                    # chunked high planes), then copy both planes into pinned.
                    lo_np, hi_np = _fpz_read_frame_planes(
                        fd, spec.offset_bytes, frame, decompressor, chunk_u, hi_align
                    )
                    half = int(hi_np.shape[0])
                    lo_pin[slot].numpy()[:half] = lo_np
                    hi_pin[slot].numpy()[:half] = hi_np

                    seg = dst.narrow(0, out_pos, n_out)
                    with torch.cuda.stream(stream):
                        lo_dev[slot][:half].copy_(
                            lo_pin[slot][:half], non_blocking=True
                        )
                        hi_dev[slot][:half].copy_(
                            hi_pin[slot][:half], non_blocking=True
                        )
                        seg[0::2].copy_(lo_dev[slot][:half], non_blocking=True)
                        seg[1::2].copy_(hi_dev[slot][:half], non_blocking=True)
                        events[slot].record(stream)
            finally:
                os.close(fd)
            stream.synchronize()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_reader, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize(device)
    if errors:
        raise errors[0]


# GPU decode for fpz v2 packs (opt-in via FLASHPACK_FPZ_GPU_DECODE=1).
#
# v2 stores each frame's high plane as many small independent zstd chunks --
# the shape a batched GPU decompressor needs. The decode runs through ctypes
# bindings to libnvcomp's batched Zstd API (see ``_nvcomp_ll``); when
# libnvcomp is unavailable, or a pack predates the v2 chunk alignment, reads
# fall back to the threaded CPU decode path. The chunks are standard zstd
# frames (python-zstandard output: single frame, embedded content size,
# 2 MiB window, no dictionary), which the batched interface decodes directly.
# ---------------------------------------------------------------------------

_FPZ_GPU_DECODE_WARNED = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _fpz_gpu_decode_enabled() -> bool:
    """Whether the opt-in GPU decode path is requested (env-gated)."""
    return _env_flag("FLASHPACK_FPZ_GPU_DECODE")


def fpz_gpu_warmup(device: "str | torch.device" = "cuda") -> bool:
    """Pay the batched GPU decoder's one-time init cost off the hot path.

    The first ``nvcompBatchedZstdDecompressAsync`` launch in a process pays
    CUDA module loading for nvcomp's decompress kernels (measured 4-25s on
    H200 under default lazy loading; ~1.6s residual with
    ``CUDA_MODULE_LOADING=EAGER`` set before the first CUDA call, which is
    the recommended companion setting). Apps can call this from ``setup()``
    -- e.g. while weights download -- so the first real load doesn't pay it.

    Decodes one tiny zstd chunk through the batched path end to end. Safe
    no-op returning ``False`` when CUDA, libnvcomp, or zstandard is
    unavailable; returns ``True`` only when the warmup decode round-tripped.
    """
    try:
        if not torch.cuda.is_available():
            return False
        from . import _nvcomp_ll

        ll = _nvcomp_ll.load()
        if ll is None:
            return False
        zstandard = require_zstandard()
        dev = torch.device(device)
        usize = 4096
        payload = zstandard.ZstdCompressor(level=1).compress(b"\x00" * usize)
        src = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(dev)
        dst = torch.empty(usize, dtype=torch.uint8, device=dev)
        # Single-chunk device tables: [src ptr, src len, dst capacity, dst ptr]
        tab = torch.tensor(
            [src.data_ptr(), len(payload), usize, dst.data_ptr()],
            dtype=torch.int64,
            device=dev,
        )
        actual = torch.zeros(1, dtype=torch.int64, device=dev)
        statuses = torch.full((1,), -1, dtype=torch.int32, device=dev)
        temp_bytes = ll.temp_size(1, usize, usize)
        temp = torch.empty(max(1, temp_bytes), dtype=torch.uint8, device=dev)
        stream = torch.cuda.current_stream(dev)
        ll.decompress_async(
            tab[0:1].data_ptr(),
            tab[1:2].data_ptr(),
            tab[2:3].data_ptr(),
            actual.data_ptr(),
            1,
            temp.data_ptr(),
            temp_bytes,
            tab[3:4].data_ptr(),
            statuses.data_ptr(),
            stream.cuda_stream,
        )
        stream.synchronize()
        return int(statuses.item()) == 0 and int(actual.item()) == usize
    except Exception:
        return False


# GPU-decode tuning knobs (env-overridable).
#
# Pipeline math (why these defaults). With v2 the GPU decode is fast and
# parallel, so the bottleneck moves to the file read: at the measured ~0.8 GB/s
# per-thread FUSE rate, and reading ~0.6x the logical bytes (the raw low plane
# plus the compressed high plane), hitting ~18 GB/s logical needs
# 0.6*18/0.8 ~= 14 read threads. So we restore the CPU path's read parallelism
# (~16 threads) instead of the 2 that the sync-free round used. Each thread
# reads AND decodes; preads (GIL released) overlap across threads and decodes
# overlap reads via the per-thread double-buffered slots.
#
# Memory: each slot holds the low / compressed-high / decompressed-high device
# planes (~3 x FPZ_FRAME_UNCOMPRESSED_BYTES/2 per frame) plus pinned host
# buffers (~2 x). Per-thread device ~= n_slots * batch_frames * 3 * 32 MiB and
# pinned ~= n_slots * batch_frames * 2 * 32 MiB; total scales by thread count.
# batch_frames=1 keeps per-thread memory small so we can afford ~16 threads
# (16*2*1*160 MiB ~= 5 GB device, ~2 GB pinned) -- and one 64 MiB frame already
# holds ~512 hi chunks, which is plenty of work for one nvcomp batched decode.
_FPZ_GPU_DEFAULT_THREADS = 16
_FPZ_GPU_DEFAULT_BATCH_FRAMES = 1
_FPZ_GPU_DEFAULT_BATCH_BYTES = 1024 * 1024 * 1024  # summed uncompressed per batch
_FPZ_GPU_DEFAULT_SLOTS = 2  # double-buffer depth per thread


def _fpz_hi_chunk_usizes(half: int, chunk_u: int) -> list[int]:
    """Uncompressed sizes of a v2 frame's high-plane chunks (pure function).

    ``half`` bytes split into ``chunk_u``-sized pieces, the last holding the
    remainder. Matches the encoder's chunking, and is the per-element shape the
    GPU decode's DecompressConfig is keyed on.
    """
    if half < 0 or chunk_u < 1:
        raise ValueError("half must be >= 0 and chunk_u >= 1")
    full, rem = divmod(half, chunk_u)
    sizes = [chunk_u] * full
    if rem:
        sizes.append(rem)
    return sizes


def plan_fpz_gpu_batches(
    frame_tasks: list[tuple],
    max_batch_frames: int,
    max_batch_bytes: int,
) -> list[list[tuple]]:
    """Group fpz frame tasks into nvcomp batched-decode groups (pure function).

    ``frame_tasks`` are the ``("frame", block_idx, frame, out_pos)`` items from
    :func:`_fpz_frame_tasks` (raw-block tasks are handled separately). Frames are
    grouped in file order into batches bounded by BOTH a frame count
    (``max_batch_frames``) and a summed uncompressed-output-byte budget
    (``max_batch_bytes``). The byte budget matters because each fpz frame is up
    to ``FPZ_FRAME_UNCOMPRESSED_BYTES`` (64 MiB) and every batched frame needs
    device staging proportional to that size, so an unbounded batch would blow
    the GPU memory budget.

    A single frame at or above the byte budget still forms its own size-1 batch
    (the budget never drops a frame). An empty input yields an empty list.
    """
    if max_batch_frames < 1:
        raise ValueError("max_batch_frames must be >= 1")
    if max_batch_bytes < 1:
        raise ValueError("max_batch_bytes must be >= 1")

    batches: list[list[tuple]] = []
    current: list[tuple] = []
    current_bytes = 0
    for task in frame_tasks:
        n_out = int(task[2]["n_out"])
        if current and (
            len(current) >= max_batch_frames or current_bytes + n_out > max_batch_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(task)
        current_bytes += n_out
    if current:
        batches.append(current)
    return batches


def _fpz_pack_chunk_layout(specs: list[MacroblockSpec]) -> tuple[int, int]:
    """(chunk_usize, chunk_alignment) recorded by the encoder.

    A pack is written with one chunk size and one chunk-start alignment;
    both are read from the first fpz block (absent fields mean the 64 KiB
    default / the unpadded pre-alignment layout). A frame whose block
    disagrees is caught by the chunk-count check in the read loop.
    """
    chunk_u = FPZ_HI_CHUNK_UNCOMPRESSED_BYTES
    hi_align = 1
    for spec in specs:
        if spec.fpz is not None:
            chunk_u = int(spec.fpz.get("hi_chunk_usize", chunk_u))
            hi_align = int(spec.fpz.get("hi_align", 1))
            break
    return chunk_u, hi_align


def _fpz_read_into_cuda_storage_gpu(
    path: str,
    specs: list[MacroblockSpec],
    blocks: list[torch.Tensor],
    device: torch.device,
    ll,
) -> None:
    """GPU-decode variant of :func:`_fpz_read_into_cuda_storage` for v2 packs.

    v2 stores each frame's high plane as many small independent zstd chunks,
    which the batched decompressor decodes in a single launch. Per reader
    thread: pread a frame's low plane and compressed high plane into pinned
    staging, H2D both (moving the high plane compressed cuts PCIe traffic by
    the compression ratio), decode every chunk of the batch with ONE batched
    call, then interleave the planes into the destination block (even bytes
    low, odd bytes high -- the same invariant as the CPU path).

    Design notes:

    * Per batch, Python fills one pinned int64 chunk table with vectorized
      numpy, issues one small H2D plus two on-device base-address adds, and
      makes one foreign call -- cost independent of the chunk count.
    * Decode never synchronizes: a slot's staging buffers are only reused
      after its CUDA event (recorded after the interleave) has fired, and
      per-chunk statuses/actual sizes fold into two on-stream scalars that
      are checked once at the end.
    * Reads dominate, so many threads overlap preads while decodes overlap
      reads via the slots.

    ``ll`` is the loaded :mod:`flashpack._nvcomp_ll` binding; the caller
    guarantees it is usable and that the pack's chunk alignment satisfies
    the decompressor's requirements.
    """
    half_cap = FPZ_FRAME_UNCOMPRESSED_BYTES // 2
    chunk_u, hi_align = _fpz_pack_chunk_layout(specs)
    max_chunks = (half_cap + chunk_u - 1) // chunk_u
    # Upper bound on a frame's compressed-high blob: the zstd bound plus
    # per-chunk frame-header overhead and chunk-start padding; aligned so
    # per-frame staging bases (k * comp_cap) preserve the chunk alignment
    # inside device staging.
    comp_cap = half_cap + (half_cap // 255) + max_chunks * 80 + 4096
    comp_cap = _align_up(comp_cap, max(16, hi_align))

    n_threads = effective_read_threads(
        _env_int("FLASHPACK_FPZ_GPU_DECODE_THREADS", _FPZ_GPU_DEFAULT_THREADS)
    )
    batch_frames = max(
        1, _env_int("FLASHPACK_FPZ_GPU_BATCH_FRAMES", _FPZ_GPU_DEFAULT_BATCH_FRAMES)
    )
    batch_bytes = max(
        1, _env_int("FLASHPACK_FPZ_GPU_BATCH_BYTES", _FPZ_GPU_DEFAULT_BATCH_BYTES)
    )
    n_slots = max(1, _env_int("FLASHPACK_FPZ_GPU_SLOTS", _FPZ_GPU_DEFAULT_SLOTS))

    byte_blocks = [b.view(torch.uint8) for b in blocks]

    # Order every reader stream after the destination allocation (same
    # wait_event pattern as parallel_read_into_storage / the CPU-decode path).
    alloc_ready = torch.cuda.Event()
    alloc_ready.record(torch.cuda.current_stream(device))

    tasks = _fpz_frame_tasks(specs)
    raw_tasks = [task for task in tasks if task[0] == "raw"]
    frame_tasks = [task for task in tasks if task[0] == "frame"]
    frame_batches = plan_fpz_gpu_batches(frame_tasks, batch_frames, batch_bytes)

    # Work items: raw blocks (whole-block H2D) and frame batches (GPU decode).
    work: queue.SimpleQueue = queue.SimpleQueue()
    for task in raw_tasks:
        work.put(("raw", task))
    for batch in frame_batches:
        work.put(("batch", batch))
    n_threads = min(n_threads, max(1, work.qsize()))
    for _ in range(n_threads):
        work.put(None)

    errors: list[BaseException] = []
    trace_on = _env_flag("FLASHPACK_FPZ_GPU_TRACE")
    traces: list[str] = []
    traces_lock = threading.Lock()

    def _reader(thread_idx: int) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(alloc_ready)
            # One contiguous staging set per slot; frame k lives at
            # k * half_cap (low / decompressed-high) or k * comp_cap
            # (compressed-high).
            lo_pin = [
                torch.empty(batch_frames * half_cap, dtype=torch.uint8, pin_memory=True)
                for _ in range(n_slots)
            ]
            hiz_pin = [
                torch.empty(batch_frames * comp_cap, dtype=torch.uint8, pin_memory=True)
                for _ in range(n_slots)
            ]
            lo_pin_view = [memoryview(b.numpy()) for b in lo_pin]
            hiz_pin_view = [memoryview(b.numpy()) for b in hiz_pin]
            lo_dev = [
                torch.empty(batch_frames * half_cap, dtype=torch.uint8, device=device)
                for _ in range(n_slots)
            ]
            hiz_dev = [
                torch.empty(batch_frames * comp_cap, dtype=torch.uint8, device=device)
                for _ in range(n_slots)
            ]
            hi_dev = [
                torch.empty(batch_frames * half_cap, dtype=torch.uint8, device=device)
                for _ in range(n_slots)
            ]
            # Per-batch chunk table, rows: (0) src offset within hiz staging,
            # (1) true compressed length, (2) dst offset within hi staging,
            # (3) expected uncompressed size. The pinned copy is per SLOT
            # (host reuse is gated by the slot event, like the other pinned
            # staging); the device table, scratch and result arrays are per
            # thread (their reuse is stream-ordered).
            cap_chunks = batch_frames * max_chunks
            ll_temp_bytes = ll.temp_size(cap_chunks, chunk_u, batch_frames * half_cap)
            ll_temp = torch.empty(
                max(1, ll_temp_bytes), dtype=torch.uint8, device=device
            )
            ll_tab_pin = [
                torch.empty((4, cap_chunks), dtype=torch.int64, pin_memory=True)
                for _ in range(n_slots)
            ]
            ll_tab_np = [t.numpy() for t in ll_tab_pin]
            ll_tab_dev = torch.empty((4, cap_chunks), dtype=torch.int64, device=device)
            ll_actual = torch.empty(cap_chunks, dtype=torch.int64, device=device)
            ll_statuses = torch.empty(cap_chunks, dtype=torch.int32, device=device)
            # Stream-side correctness accumulators: per-chunk statuses and
            # actual-size mismatches fold into two scalars ON the decode
            # stream (no syncs); read once after the final synchronize.
            ll_status_max = torch.zeros((), dtype=torch.int32, device=device)
            ll_size_bad = torch.zeros((), dtype=torch.bool, device=device)
            ll_usizes_cache: dict[int, np.ndarray] = {}
            ll_dst_rel_cache: dict[tuple[int, int], np.ndarray] = {}
            # Recorded now so the first synchronize on any slot is a no-op.
            events = [torch.cuda.Event() for _ in range(n_slots)]
            for ev in events:
                ev.record(stream)
            batch_idx = 0
            t_pread = t_h2d = t_table = t_decode = t_interleave = 0.0
            t_evsync = t_final = 0.0
            n_batches = n_frames_done = 0
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    kind, payload = item
                    if kind == "raw":
                        _, blk, _frame, _out_pos = payload
                        spec = specs[blk]
                        buf = bytearray(spec.length_bytes)
                        _pread_into(fd, spec.offset_bytes, memoryview(buf))
                        host = torch.frombuffer(buf, dtype=torch.uint8)
                        stream.synchronize()
                        with torch.cuda.stream(stream):
                            byte_blocks[blk].copy_(host, non_blocking=False)
                        continue

                    batch = payload
                    slot = batch_idx % n_slots
                    batch_idx += 1
                    # Wait for this slot's previous batch (its interleave)
                    # before overwriting its pinned/device buffers -- decode
                    # does not synchronize, so this event keeps reuse safe.
                    _t = time.perf_counter() if trace_on else 0.0
                    events[slot].synchronize()
                    if trace_on:
                        t_evsync += time.perf_counter() - _t

                    halves: list[int] = []
                    n_chunks = 0
                    for k, (_, blk, frame, _out_pos) in enumerate(batch):
                        payload_off = int(frame["payload_off"])
                        lo_len = int(frame["lo_len"])
                        n_out = int(frame["n_out"])
                        half = n_out - lo_len
                        hi_chunks = frame.get("hi_chunks")
                        if hi_chunks is None:
                            raise ValueError(
                                "GPU fpz decode requires a v2 (chunked) pack"
                            )
                        if lo_len != half or lo_len * 2 != n_out:
                            raise ValueError("fpz frame plane size mismatch")
                        clens = np.asarray(hi_chunks, dtype=np.int64)
                        m = int(clens.shape[0])
                        # Chunk starts are hi_align-padded in the payload (and
                        # therefore in staging); hi_chunks holds true lengths.
                        aligned = clens + (-clens) % hi_align
                        hi_len_total = int(aligned.sum())
                        if hi_len_total > comp_cap:
                            raise ValueError(
                                f"fpz compressed frame ({hi_len_total} bytes) "
                                f"exceeds staging capacity ({comp_cap} bytes)"
                            )
                        usizes = _fpz_hi_chunk_usizes(half, chunk_u)
                        if len(usizes) != m:
                            raise ValueError("fpz v2 chunk count mismatch")
                        base = specs[blk].offset_bytes + payload_off
                        lo_off = k * half_cap
                        hiz_off = k * comp_cap
                        _t = time.perf_counter() if trace_on else 0.0
                        _pread_into(
                            fd, base, lo_pin_view[slot][lo_off : lo_off + lo_len]
                        )
                        _pread_into(
                            fd,
                            base + _align_up(lo_len, hi_align),
                            hiz_pin_view[slot][hiz_off : hiz_off + hi_len_total],
                        )
                        if trace_on:
                            t_pread += time.perf_counter() - _t
                        halves.append(half)
                        _t = time.perf_counter() if trace_on else 0.0
                        with torch.cuda.stream(stream):
                            lo_dev[slot].narrow(0, lo_off, half).copy_(
                                lo_pin[slot].narrow(0, lo_off, half),
                                non_blocking=True,
                            )
                            hiz_dev[slot].narrow(0, hiz_off, hi_len_total).copy_(
                                hiz_pin[slot].narrow(0, hiz_off, hi_len_total),
                                non_blocking=True,
                            )
                        if trace_on:
                            t_h2d += time.perf_counter() - _t
                        # Vectorized table rows for this frame's chunks.
                        _t = time.perf_counter() if trace_on else 0.0
                        tab = ll_tab_np[slot]
                        starts = np.empty(m, dtype=np.int64)
                        starts[0] = 0
                        np.cumsum(aligned[: m - 1], out=starts[1:])
                        tab[0, n_chunks : n_chunks + m] = hiz_off + starts
                        tab[1, n_chunks : n_chunks + m] = clens
                        dst_rel = ll_dst_rel_cache.get((k, m))
                        if dst_rel is None:
                            dst_rel = k * half_cap + (
                                np.arange(m, dtype=np.int64) * chunk_u
                            )
                            ll_dst_rel_cache[(k, m)] = dst_rel
                        tab[2, n_chunks : n_chunks + m] = dst_rel
                        caps = ll_usizes_cache.get(half)
                        if caps is None:
                            caps = np.asarray(usizes, dtype=np.int64)
                            ll_usizes_cache[half] = caps
                        tab[3, n_chunks : n_chunks + m] = caps
                        n_chunks += m
                        if trace_on:
                            t_table += time.perf_counter() - _t

                    # One H2D of the table, two on-device base-address adds,
                    # ONE foreign call for the whole batch. The add outputs
                    # are fresh stream-local tensors; the decompressor reads
                    # them during the (stream-ordered) decode, so dropping
                    # the Python refs afterwards is safe.
                    _t = time.perf_counter() if trace_on else 0.0
                    with torch.cuda.stream(stream):
                        ll_tab_dev[:, :n_chunks].copy_(
                            ll_tab_pin[slot][:, :n_chunks], non_blocking=True
                        )
                        src_ptrs = ll_tab_dev[0, :n_chunks] + hiz_dev[slot].data_ptr()
                        dst_ptrs = ll_tab_dev[2, :n_chunks] + hi_dev[slot].data_ptr()
                    ll.decompress_async(
                        src_ptrs.data_ptr(),
                        ll_tab_dev[1].data_ptr(),
                        ll_tab_dev[3].data_ptr(),
                        ll_actual.data_ptr(),
                        n_chunks,
                        ll_temp.data_ptr(),
                        ll_temp_bytes,
                        dst_ptrs.data_ptr(),
                        ll_statuses.data_ptr(),
                        stream.cuda_stream,
                    )
                    with torch.cuda.stream(stream):
                        torch.maximum(
                            ll_status_max,
                            ll_statuses[:n_chunks].max(),
                            out=ll_status_max,
                        )
                        torch.logical_or(
                            ll_size_bad,
                            (ll_actual[:n_chunks] != ll_tab_dev[3, :n_chunks]).any(),
                            out=ll_size_bad,
                        )
                    if trace_on:
                        t_decode += time.perf_counter() - _t

                    # Interleave per frame (same invariant as the CPU path):
                    # even bytes low plane, odd bytes high plane.
                    _t = time.perf_counter() if trace_on else 0.0
                    for k, (_, blk, frame, out_pos) in enumerate(batch):
                        half = halves[k]
                        n_out = int(frame["n_out"])
                        seg = byte_blocks[blk].narrow(0, out_pos, n_out)
                        with torch.cuda.stream(stream):
                            seg[0::2].copy_(
                                lo_dev[slot].narrow(0, k * half_cap, half),
                                non_blocking=True,
                            )
                            seg[1::2].copy_(
                                hi_dev[slot].narrow(0, k * half_cap, half),
                                non_blocking=True,
                            )
                    events[slot].record(stream)
                    if trace_on:
                        t_interleave += time.perf_counter() - _t
                        n_batches += 1
                        n_frames_done += len(batch)
            finally:
                os.close(fd)
            _t = time.perf_counter() if trace_on else 0.0
            stream.synchronize()
            # Deferred per-chunk verification: both scalars were folded on
            # the decode stream per batch, so this is the only D2H.
            status_val = int(ll_status_max.item())
            if status_val != 0:
                raise RuntimeError(
                    "fpz batched GPU decode reported a per-chunk error: "
                    + ll.status_string(status_val)
                )
            if bool(ll_size_bad.item()):
                raise ValueError(
                    "fpz batched GPU decode produced a chunk size mismatch"
                )
            if trace_on:
                t_final += time.perf_counter() - _t
                # Enqueue phases (h2d, interleave) are async so their wall is
                # small; a large `decode` means the foreign call itself
                # blocks, while large `evsync`/`final` means the pipeline is
                # GPU-bound waiting on decode + interleave.
                line = (
                    f"fpz-gpu thread={thread_idx} frames={n_frames_done} "
                    f"batches={n_batches} pread={t_pread:.3f}s "
                    f"h2d_enq={t_h2d:.3f}s table={t_table:.3f}s "
                    f"decode={t_decode:.3f}s "
                    f"interleave_enq={t_interleave:.3f}s "
                    f"evsync={t_evsync:.3f}s final_sync={t_final:.3f}s"
                )
                with traces_lock:
                    traces.append(line)
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_reader, args=(i,), daemon=True)
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize(device)
    if trace_on:
        for line in traces:
            logger.debug(line)
    if errors:
        raise errors[0]


_FPZ_V1_GPU_WARNED = False
_FPZ_ALIGN_WARNED = False


def _fpz_specs_all_v2(specs: list[MacroblockSpec]) -> bool:
    """True if every fpz block uses the v2 (chunked) codec -- the only format
    the GPU decoder can decompress in parallel."""
    for spec in specs:
        if spec.fpz is not None and spec.fpz.get("codec") != FPZ_CODEC_SPLITPLANE_V2:
            return False
    return True


def _read_fpz_into_storage(
    path: str, specs: list[MacroblockSpec], device: torch.device
) -> FlashTensorStorage:
    """Materialize an fpz (partially compressed) pack into ``device`` storage."""
    global _FPZ_V1_GPU_WARNED
    if device.type == "cpu":
        storage = _allocate_aligned_cpu_storage(specs)
        _fpz_read_into_cpu_storage(path, specs, storage.blocks)
        return storage
    if device.type == "cuda":
        storage = _allocate_empty_storage(specs, device)
        ll = _fpz_gpu_decoder(specs) if _fpz_gpu_decode_enabled() else None
        if ll is not None:
            _fpz_read_into_cuda_storage_gpu(path, specs, storage.blocks, device, ll)
        else:
            _fpz_read_into_cuda_storage(path, specs, storage.blocks, device)
        return storage
    raise ValueError(f"Unsupported device: {device}")


def _fpz_gpu_decoder(specs: list[MacroblockSpec]):
    """Resolve the batched GPU decoder for this pack, or ``None`` to use the
    CPU decode path. Warns once per reason: libnvcomp missing, a v1 pack (one
    chunk per frame decodes serially -- the CPU path is faster), or a pack
    whose chunk layout predates the alignment the decompressor requires."""
    global _FPZ_GPU_DECODE_WARNED, _FPZ_V1_GPU_WARNED, _FPZ_ALIGN_WARNED
    from . import _nvcomp_ll

    ll = _nvcomp_ll.load()
    if ll is None:
        if not _FPZ_GPU_DECODE_WARNED:
            _FPZ_GPU_DECODE_WARNED = True
            warnings.warn(
                "FLASHPACK_FPZ_GPU_DECODE=1 but libnvcomp is not available; "
                "falling back to CPU zstd decode. Install the GPU extra "
                "with: pip install 'flashpack[fpz-gpu]'.",
                RuntimeWarning,
                stacklevel=3,
            )
        return None
    if not _fpz_specs_all_v2(specs):
        if not _FPZ_V1_GPU_WARNED:
            _FPZ_V1_GPU_WARNED = True
            warnings.warn(
                "FLASHPACK_FPZ_GPU_DECODE=1 but this pack uses the v1 fpz "
                "codec, which cannot be GPU-decoded in parallel; using the "
                "CPU decode path. Repack with the v2 encoder for GPU decode.",
                RuntimeWarning,
                stacklevel=3,
            )
        return None
    chunk_u, hi_align = _fpz_pack_chunk_layout(specs)
    req_in, req_out, _req_temp = ll.alignments()
    if hi_align % req_in != 0 or chunk_u % req_out != 0:
        if not _FPZ_ALIGN_WARNED:
            _FPZ_ALIGN_WARNED = True
            warnings.warn(
                f"FLASHPACK_FPZ_GPU_DECODE=1 but this pack's chunk layout "
                f"(hi_align={hi_align}, chunk_usize={chunk_u}) does not "
                f"satisfy the decompressor's alignment requirements "
                f"(input={req_in}, output={req_out}); using the CPU decode "
                "path. Repack with the current encoder for GPU decode.",
                RuntimeWarning,
                stacklevel=3,
            )
        return None
    return ll


def read_flashpack_file(
    path: str,
    device: str | torch.device = "cpu",
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    num_streams: int = DEFAULT_NUM_STREAMS,
    silent: bool = True,
    metadata: dict[str, Any] | None = None,
) -> tuple[FlashTensorStorage, dict[str, Any]]:
    """
    Read the flashpack file and return the macroblock storage and metadata.
    """
    with timer("read_metadata", silent):
        meta = metadata or get_flashpack_file_metadata(path)

    specs = _build_macroblock_specs(meta)
    device = torch.device(device) if isinstance(device, str) else device

    if any(spec.fpz is not None for spec in specs):
        with timer("read_fpz", silent):
            storage = _read_fpz_into_storage(path, specs, device)
        return storage, meta

    if device.type == "cpu":
        if parallel_read_supported(device):
            # Opt-in eager path (FLASHPACK_CPU_PARALLEL_READ=1): materialize
            # the payload into RAM with parallel reads instead of returning
            # lazy mmap views. See parallel_read.py for the measurements.
            with timer("alloc_cpu_aligned", silent):
                storage = _allocate_aligned_cpu_storage(specs)
            with timer("read_and_copy", silent):
                parallel_read_into_storage(path, specs, storage.blocks, device)
            return storage, meta
        with timer("mmap_payload", silent):
            memmaps = _open_memmaps(path, specs)
        with timer("cpu_from_memmap", silent):
            storage = _cpu_storage_from_memmaps(memmaps, specs)
        return storage, meta

    if device.type != "cuda":
        raise ValueError(f"Unsupported device: {device}")

    with timer("alloc_device", silent):
        storage = _allocate_empty_storage(specs, device)

    if parallel_read_supported(device):
        with timer("read_and_copy", silent):
            parallel_read_into_storage(path, specs, storage.blocks, device)
        return storage, meta

    with timer("mmap_payload", silent):
        memmaps = _open_memmaps(path, specs)

    with timer("read_and_copy", silent):
        _copy_memmaps_into_storage(
            memmaps,
            specs,
            storage,
            device=device,
            chunk_bytes=chunk_bytes,
            num_streams=num_streams,
        )

    del memmaps
    return storage, meta


def read_flashpack_file_distributed(
    path: str,
    device: str | torch.device = "cuda",
    src: int = 0,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    num_streams: int = DEFAULT_NUM_STREAMS,
    silent: bool = True,
    metadata: dict[str, Any] | None = None,
) -> tuple[FlashTensorStorage, dict[str, Any]]:
    """Rank-``src`` reads the pack from disk; every rank returns the full
    storage, received via broadcast.

    This removes the N-times read amplification of world-size-N loads: the
    reader deliberately bypasses the page cache (O_DIRECT), so without this
    every rank pays a full duplicate pack read, while an NVLink broadcast of
    the same bytes is 1-2 orders of magnitude faster than the read itself.

    Requires an initialized process group (see ``maybe_init_distributed``).
    Every rank must be able to read the pack FOOTER from ``path`` (payload
    is only read on ``src``); pass ``metadata`` to skip that requirement.
    Blocks are broadcast as raw bytes, so any pack dtype works regardless of
    the collective backend's dtype support.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "read_flashpack_file_distributed requires an initialized "
            "torch.distributed process group."
        )
    device = torch.device(device) if isinstance(device, str) else device
    if dist.get_backend() == "nccl" and device.type != "cuda":
        raise ValueError(
            "distributed flashpack loading with the NCCL backend requires "
            f"a cuda device, got {device}."
        )
    meta = metadata or get_flashpack_file_metadata(path)
    if dist.get_rank() == src:
        storage, meta = read_flashpack_file(
            path=path,
            device=device,
            chunk_bytes=chunk_bytes,
            num_streams=num_streams,
            silent=silent,
            metadata=meta,
        )
    else:
        specs = _build_macroblock_specs(meta)
        storage = _allocate_empty_storage(specs, device)
    _broadcast_storage(storage, src=src)
    return storage, meta


def iterate_from_flash_tensor(
    flash_tensor: FlashTensorStorage | torch.Tensor,
    metadata: dict[str, Any],
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Iterate over the tensors stored in the flash tensor.
    """
    storage = (
        flash_tensor
        if isinstance(flash_tensor, FlashTensorStorage)
        else FlashTensorStorage(blocks=[flash_tensor])
    )
    index = metadata["index"]

    align_bytes = int(metadata.get("align_bytes", 0))
    align_cache: dict[int, int] = {}

    def _get_align(block_idx: int) -> int:
        if not align_bytes:
            return 0
        if block_idx not in align_cache:
            esz = storage.block(block_idx).element_size()
            g = math.gcd(align_bytes, esz)
            align_cache[block_idx] = align_bytes // g if g else 0
        return align_cache[block_idx]

    if align_bytes:
        bad: list[dict[str, Any]] = []
        for rec in index:
            block_idx = int(rec.get("macroblock", 0))
            if block_idx < 0 or block_idx >= len(storage):
                raise ValueError(
                    f"Index entry references invalid macroblock {block_idx}."
                )
            align_elems = _get_align(block_idx)
            if align_elems and (int(rec["offset"]) % align_elems) != 0:
                bad.append(rec)
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
        block_idx = int(rec.get("macroblock", 0))
        if block_idx < 0 or block_idx >= len(storage):
            raise ValueError(f"Index entry references invalid macroblock {block_idx}.")
        block_tensor = storage.block(block_idx)

        try:
            view = block_tensor.narrow(0, off, n).view(
                *shape
            )  # contiguous 1D slice -> reshaped
            yield name, view
        except Exception as e:
            raise ValueError(f"Could not get tensor for record {rec}") from e


def revert_from_file(
    path: str,
    silent: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Revert a flashpack file to a state dictionary.
    """
    storage, meta = read_flashpack_file(path, silent=silent)
    state_dict = {}
    progress: tqdm.tqdm | None = None

    if not silent:
        progress = tqdm.tqdm(desc="Reverting from flashpack", total=len(storage))

    for name, view in iterate_from_flash_tensor(storage, meta):
        state_dict[name] = view.detach().cpu()
        if progress:
            progress.update(1)

    return state_dict


def assign_from_file(
    model: torch.nn.Module,
    path: str,
    device: str | torch.device | None = None,
    strict: bool | None = None,
    strict_params: bool = True,
    strict_buffers: bool = False,
    keep_flash_ref_on_model: bool = False,
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
        flash_storage, meta = read_flashpack_file_distributed(
            path=path,
            device=device,
            silent=silent,
            num_streams=num_streams,
            chunk_bytes=chunk_bytes,
        )
    else:
        flash_storage, meta = read_flashpack_file(
            path=path,
            device=device,
            silent=silent,
            num_streams=num_streams,
            chunk_bytes=chunk_bytes,
        )

    if keep_flash_ref_on_model:
        setattr(model, "_flash_shared_storage", flash_storage)
        setattr(model, "_flash_shared_storage_meta", meta)

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
                flash_storage, meta, ignore_names, ignore_prefixes, ignore_suffixes
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
                    if old_buf.dtype != view.dtype:
                        if coerce_dtype:
                            view = view.to(old_buf.dtype)
                        else:
                            raise TypeError(
                                f"dtype mismatch for buffer '{name}': model={old_buf.dtype} vs flash={view.dtype}."
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
