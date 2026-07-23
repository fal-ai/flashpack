import json
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
        if fmt not in (FILE_FORMAT_V3, FILE_FORMAT_V4):
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
    elif fmt == FILE_FORMAT_V4:
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
    for block in storage.blocks:
        dist.broadcast(block, src=src)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _pread_into(fd: int, offset: int, mv: memoryview) -> None:
    """Fill ``mv`` from ``fd`` at ``offset`` with ``preadv`` (reused reader
    machinery). Raises ``IOError`` on a short read (e.g. a truncated file)."""
    n = len(mv)
    got = 0
    while got < n:
        r = os.preadv(fd, [mv[got:]], offset + got)
        if r <= 0:
            raise IOError(f"short read: wanted {n} bytes at {offset}, got {got}")
        got += r


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


def _fpz_read_frame_planes(
    fd: int,
    block_file_offset: int,
    frame: dict[str, Any],
    decompressor,
    chunk_u: int = FPZ_HI_CHUNK_UNCOMPRESSED_BYTES,
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
    hi_base = block_file_offset + payload_off + lo_len

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
            src_off += clen
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
    n_threads = max(1, _env_int("FLASHPACK_READ_THREADS", 16))

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
            fd = os.open(path, os.O_RDONLY)
            decompressor = zstandard.ZstdDecompressor()
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    kind, blk, frame, out_pos = item
                    if kind == "raw":
                        spec = specs[blk]
                        _pread_into(
                            fd, spec.offset_bytes, memoryview(dst_u8[blk])
                        )
                        continue
                    spec = specs[blk]
                    chunk_u = int(
                        (spec.fpz or {}).get(
                            "hi_chunk_usize", FPZ_HI_CHUNK_UNCOMPRESSED_BYTES
                        )
                    )
                    lo, hi = _fpz_read_frame_planes(
                        fd, spec.offset_bytes, frame, decompressor, chunk_u
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
    n_threads = max(1, _env_int("FLASHPACK_READ_THREADS", 16))
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
            fd = os.open(path, os.O_RDONLY)
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
                    # Shared CPU decode (handles both v1 single-frame and v2
                    # chunked high planes), then copy both planes into pinned.
                    lo_np, hi_np = _fpz_read_frame_planes(
                        fd, spec.offset_bytes, frame, decompressor, chunk_u
                    )
                    half = int(hi_np.shape[0])
                    lo_pin[slot].numpy()[:half] = lo_np
                    hi_pin[slot].numpy()[:half] = hi_np

                    seg = dst.narrow(0, out_pos, n_out)
                    with torch.cuda.stream(stream):
                        lo_dev[slot][:half].copy_(lo_pin[slot][:half], non_blocking=True)
                        hi_dev[slot][:half].copy_(hi_pin[slot][:half], non_blocking=True)
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


# ---------------------------------------------------------------------------
# nvcomp GPU Zstd decode (prototype; opt-in via FLASHPACK_FPZ_GPU_DECODE=1).
#
# The CPU zstd decode caps fpz at ~9 GB/s logical (H200) versus ~21.6 GB/s for
# a page-hot raw pack, so decode -- not I/O -- is the fpz bottleneck. nvcomp's
# batched GPU Zstd decoder moves that work onto the device.
#
# API discovery -- nvidia-nvcomp-cu12 5.3.0 (pybind11 module ``nvidia.nvcomp``;
# signatures/docstrings read from the compiled nvcomp_impl .so, quoted below):
#
#   Codec(algorithm="Zstd", device_id=<int>, cuda_stream=<int>,
#         uncomp_chunk_size=65536, bitstream_kind=BitstreamKind.NVCOMP_NATIVE,
#         checksum_policy=NO_COMPUTE_NO_VERIFY, decompress_backend=...)
#     "Initialize codec."
#       algorithm      : name of the compression algorithm ("Zstd", "LZ4", ...).
#       device_id      : device to run on (default: current device).
#       cuda_stream    : cudaStream_t as a Python int (default: an internal
#                        stream). We pass each reader thread's own torch stream
#                        (``stream.cuda_stream``) so the decode is ordered on the
#                        SAME stream as our H2D copies and the interleave -- no
#                        cross-stream sync needed.
#       bitstream_kind : BitstreamKind.{NVCOMP_NATIVE, RAW, WITH_UNCOMPRESSED_SIZE}.
#                        We use RAW: "Compresses input data as is, just using the
#                        underlying compression algorithm. Does not add a header
#                        with nvCOMP metadata." The fpz high plane is a standard
#                        single zstd frame written by python-zstandard, so it must
#                        be decoded as RAW (NVCOMP_NATIVE expects nvcomp's own
#                        chunked container and would reject a bare zstd frame).
#
#   codec.decode(src, data_type="|u1", out=None, decompression_config=None)
#         -> nvcomp.Array                         "Decode a single Array."
#   codec.decode(srcs: list[Array], data_type=..., out=<list|None>,
#                decompression_config=None) -> list[Array]
#                                                 "Decode a batch of Arrays."
#     out : "An optional writable buffer to store decoded data. ... If it is an
#            externally-allocated buffer (e.g. cupy/numba array), its size is
#            fixed and a ValueError is raised when it is too small." We pass a
#            view over our pre-sized device high-plane tensor, so decode writes
#            straight into it -- no extra device copy and no host round trip.
#     data_type : output element type string; default "|u1" (uint8), which is
#            exactly the byte plane we want, so we never pass it.
#     decompression_config : when omitted, "decode internally calls
#            configure_decompression on src, forcing a stream synchronization"
#            on EVERY call -- that per-call sync serialized the whole pipeline
#            and measured ~0.6 GB/s on the H200. We instead build a reusable
#            DecompressConfig once per distinct batch shape via
#            codec.decompression_config(srcs) ("reusable across multiple decode
#            calls ... of the same uncompressed per-element shape") and pass it
#            to decode, which is then sync-free. fpz packs have only a few
#            shapes (full 64 MiB frames + one tail per block), so the one-time
#            build sync is paid a handful of times per thread, not per decode.
#            (A CompressConfig-derived config -- codec.decompression_config(
#            codec.compression_config(size)) -- would skip even the build sync,
#            but the docstring scopes that to same-process compress+decompress;
#            our frames are compressed offline by python-zstandard, so we use
#            the header-parsing overload that is proven against real frames.)
#
#   nvcomp.as_array(src_object, cuda_stream=None) -> Array
#     "Creates array from object with some standard interface." Zero-copy over
#     any object exposing __cuda_array_interface__ / __dlpack__. A contiguous
#     torch CUDA tensor qualifies, so we wrap the device staging tensors
#     directly (no copy). nvcomp.from_dlpack(...) is the explicit-DLPack
#     equivalent; as_array is sufficient here.
#
#   nvcomp.set_device_allocator(allocator) -- "Sets a new allocator ... for
#     future device allocations." allocator is
#     ``allocator(nbytes: int, stream: nvcomp.Stream) -> obj`` where obj has an
#     integer ``.ptr`` and frees on garbage collection. nvcomp grabs scratch
#     from this for every decode; its default (cudaMalloc/cudaFree) syncs the
#     device per call, so we install a torch-caching-allocator adapter (see
#     _install_torch_nvcomp_allocator) to serve scratch pool-side with no sync.
#
# Compatibility, verified locally against the pack side (python-zstandard
# level-3, threads=-1, one-shot ``compress``): every high plane is a SINGLE
# standard zstd frame (magic 0xFD2FB528) with the content size embedded, a
# 2 MiB window (windowLog 21), no dictionary and no checksum -- all within
# nvcomp GPU Zstd's limits. nvcomp itself cannot be exercised without a device,
# so confirming decode correctness on a real frame is step 0 of the H200 run.
# The correctness-critical interleave (even byte = low plane, odd byte = high
# plane) is identical to the CPU path and is covered by the CPU tests.
# ---------------------------------------------------------------------------

_FPZ_GPU_DECODE_WARNED = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _fpz_gpu_decode_enabled() -> bool:
    """Whether the opt-in nvcomp GPU decode path is requested (env-gated)."""
    return _env_flag("FLASHPACK_FPZ_GPU_DECODE")


def _load_nvcomp():
    """Import the optional nvcomp module for GPU Zstd decode.

    Returns the module, or ``None`` if it is not importable -- in which case it
    warns once (per process) so the caller can fall back to the CPU decode path
    without spamming. nvcomp is NVIDIA-proprietary and never a hard dependency;
    install it with ``pip install 'flashpack[fpz-gpu]'`` (see pyproject).
    """
    global _FPZ_GPU_DECODE_WARNED
    try:
        from nvidia import nvcomp

        return nvcomp
    except ImportError:
        if not _FPZ_GPU_DECODE_WARNED:
            _FPZ_GPU_DECODE_WARNED = True
            warnings.warn(
                "FLASHPACK_FPZ_GPU_DECODE=1 but the nvcomp package is not "
                "importable; falling back to CPU zstd decode. Install the GPU "
                "extra with: pip install 'flashpack[fpz-gpu]'.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None


def _env_flag_default(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# nvcomp calls its device allocator once per decode for scratch. Its default
# allocator is cudaMalloc/cudaFree, and each of those synchronizes the device --
# on the H200 that per-call sync (not the config sync) was the dominant fpz cost
# (~60ms/frame, tier-flat). Routing nvcomp's scratch through torch's stream-aware
# caching allocator serves it from an existing pool with no cudaMalloc/sync.
_fpz_nvcomp_alloc_tls = threading.local()
_FPZ_NVCOMP_ALLOC_INSTALLED = False
_FPZ_NVCOMP_ALLOC_WARNED = False


class _TorchNvcompDeviceBuffer:
    """Adapter exposing a torch caching-allocator block to nvcomp's allocator
    protocol: an object with an integer ``ptr`` that frees on ``__del__``.

    The allocation is tied to the calling reader thread's CUDA stream (stashed
    in a thread-local by the reader) so torch's caching allocator won't hand the
    block to another stream while nvcomp's decode -- which runs on that same
    stream -- is still using it.
    """

    __slots__ = ("_ptr",)

    def __init__(self, nbytes: int, device_index: int, stream) -> None:
        self._ptr = torch.cuda.caching_allocator_alloc(nbytes, device_index, stream)

    @property
    def ptr(self) -> int:
        return self._ptr

    def __del__(self) -> None:
        try:
            torch.cuda.caching_allocator_delete(self._ptr)
        except Exception:
            pass


def _install_torch_nvcomp_allocator(nvcomp, device: torch.device) -> bool:
    """Route nvcomp's per-decode device scratch through torch's caching allocator.

    Global and idempotent. Guarded: any API mismatch or failure leaves nvcomp on
    its default allocator (decode still works, just slower) and warns once.

    nvcomp API (from the wheel's ``set_device_allocator`` docstring): the
    allocator is ``allocator(nbytes: int, stream: nvcomp.Stream) -> obj`` where
    ``obj`` has an integer ``.ptr`` and releases its memory when garbage
    collected. We ignore nvcomp's ``stream`` arg and instead read the reader
    thread's torch stream from ``_fpz_nvcomp_alloc_tls`` (set per thread), which
    is the stream nvcomp actually decodes on.
    """
    global _FPZ_NVCOMP_ALLOC_INSTALLED, _FPZ_NVCOMP_ALLOC_WARNED
    if _FPZ_NVCOMP_ALLOC_INSTALLED:
        return True
    dev_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )

    def _alloc(nbytes, stream=None):
        return _TorchNvcompDeviceBuffer(
            int(nbytes), dev_index, getattr(_fpz_nvcomp_alloc_tls, "stream", None)
        )

    try:
        nvcomp.set_device_allocator(_alloc)
    except Exception:
        if not _FPZ_NVCOMP_ALLOC_WARNED:
            _FPZ_NVCOMP_ALLOC_WARNED = True
            warnings.warn(
                "Could not install the torch caching allocator into nvcomp "
                "(set_device_allocator failed); nvcomp keeps its default "
                "cudaMalloc allocator. Set FLASHPACK_FPZ_GPU_TORCH_ALLOC=0 to "
                "silence.",
                RuntimeWarning,
                stacklevel=2,
            )
        return False
    _FPZ_NVCOMP_ALLOC_INSTALLED = True
    return True


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


def _fpz_batch_signature(batch: list[tuple]) -> tuple[int, ...]:
    """Config-cache key for a frame batch: the per-frame decompressed high-plane
    sizes (``n_out // 2``), in order (pure function).

    An nvcomp ``DecompressConfig`` built from one batch is reusable for any other
    batch with the same per-element uncompressed shape, so batches that share
    this signature share a single config -- and the one-time
    ``decompression_config`` stream sync is paid once per distinct signature
    instead of once per decode call.
    """
    return tuple(int(frame["n_out"]) // 2 for _, _blk, frame, _out_pos in batch)


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


def _fpz_read_into_cuda_storage_gpu(
    path: str,
    specs: list[MacroblockSpec],
    blocks: list[torch.Tensor],
    device: torch.device,
    nvcomp,
) -> None:
    """GPU-decode variant of :func:`_fpz_read_into_cuda_storage` for v2 packs.

    v2 stores each frame's high plane as many small independent zstd chunks
    (``FPZ_HI_CHUNK_UNCOMPRESSED_BYTES`` each). That is nvcomp's native shape:
    the whole point of the GPU decoder is decoding MANY chunks in parallel. A v1
    single-frame high plane is one nvcomp chunk and decodes serially (~0.6 GB/s
    measured, tier-flat), which is why the caller routes v1 to the CPU path.

    Per reader thread: read a frame's low plane and its whole compressed-high
    blob into pinned staging, H2D both (moving the compressed high plane cuts
    PCIe traffic ~2.4x), then submit ALL of the frame's high chunks as one
    ``codec.decode`` batch (hundreds of Arrays), decoding straight into the
    device high-plane staging; finally the same strided interleave
    (``dst[0::2] = lo``, ``dst[1::2] = hi``).

    Two throughput levers, both load-bearing:

    * No per-decode sync. The naive path makes ``decode`` call
      ``configure_decompression`` (a stream sync) every call. We build a
      reusable ``DecompressConfig`` per distinct chunk-shape signature (one sync
      each) and pass it to ``decode``; since v2 chunks are almost all a uniform
      64 KiB, that is ~1-2 configs total per thread and every steady-state decode
      is sync-free. Reuse safety without the sync comes from an event-gated
      double buffer (``n_slots`` staging sets; ``synchronize`` a slot's event
      before reusing it, ``record`` it after decode+interleave).
    * Read parallelism. Read (not decode) is now the bottleneck, so we use many
      threads (see the tuning-knob pipeline math); preads overlap across threads
      and decodes overlap reads via the slots.

    Each thread owns its fd, stream, Codec and config cache. Raw (uncompressed)
    blocks take the same whole-block H2D as the CPU-decode path.
    """
    half_cap = FPZ_FRAME_UNCOMPRESSED_BYTES // 2
    # A pack is written with one chunk size; read it from the first fpz block
    # (absent for pre-parameterization v2 packs -> the 64 KiB default). A frame
    # whose block disagrees is caught by the chunk-count check in the loop.
    chunk_u = FPZ_HI_CHUNK_UNCOMPRESSED_BYTES
    for spec in specs:
        if spec.fpz is not None:
            chunk_u = int(spec.fpz.get("hi_chunk_usize", chunk_u))
            break
    max_chunks = (half_cap + chunk_u - 1) // chunk_u
    # Upper bound on a frame's whole compressed-high blob: the zstd bound for
    # half_cap uncompressed, plus per-chunk zstd frame-header overhead.
    comp_cap = half_cap + (half_cap // 255) + max_chunks * 64 + 4096

    n_threads = max(
        1, _env_int("FLASHPACK_FPZ_GPU_DECODE_THREADS", _FPZ_GPU_DEFAULT_THREADS)
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

    # Route nvcomp's per-decode scratch through torch's caching allocator to kill
    # the per-call cudaMalloc/cudaFree device sync (the round-2 bottleneck).
    # Global + idempotent + guarded; disable with FLASHPACK_FPZ_GPU_TORCH_ALLOC=0.
    if _env_flag_default("FLASHPACK_FPZ_GPU_TORCH_ALLOC", True):
        _install_torch_nvcomp_allocator(nvcomp, device)

    def _reader(thread_idx: int) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(alloc_ready)
            # nvcomp's device allocator (if installed) reads this thread's stream
            # from the thread-local, so decode scratch is tied to the decode
            # stream and torch won't reuse it out from under an in-flight decode.
            _fpz_nvcomp_alloc_tls.stream = stream
            # Resolve the concrete ordinal in this thread: an indexless "cuda"
            # device places both the stream and the staging tensors on this
            # thread's current device, and the Codec must match or nvcomp raises
            # "Input array and Codec device id mismatched".
            device_id = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            codec = nvcomp.Codec(
                algorithm="Zstd",
                bitstream_kind=nvcomp.BitstreamKind.RAW,
                device_id=device_id,
                cuda_stream=stream.cuda_stream,
            )
            # One contiguous staging set per slot; frame k lives at k*half_cap
            # (low / decompressed-high) or k*comp_cap (compressed-high). Slices
            # are contiguous, so nvcomp.as_array wraps them zero-copy.
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
            # Hoist the decode out= wrappers: one nvcomp.Array per
            # (slot, frame, chunk) over a fixed chunk_u slice at frame k's
            # chunk j offset, built ONCE (layout is data-independent). decode
            # writes the true (config-driven) size <= chunk_u into each, so the
            # same wrappers serve every batch. Indexed [slot][k*max_chunks + j].
            # (The compressed-in src wrappers stay per-batch, sized to each
            # chunk's exact compressed length, so nvcomp sees exactly one zstd
            # frame per Array.)
            out_wrap = [
                [
                    nvcomp.as_array(
                        hi_dev[s].narrow(0, k * half_cap + j * chunk_u, chunk_u)
                    )
                    for k in range(batch_frames)
                    for j in range(max_chunks)
                ]
                for s in range(n_slots)
            ]
            # Recorded now so the first synchronize on any slot is a no-op.
            events = [torch.cuda.Event() for _ in range(n_slots)]
            for ev in events:
                ev.record(stream)
            # Per-thread cache: batch shape signature -> reusable DecompressConfig.
            configs: dict[tuple[int, ...], object] = {}
            batch_idx = 0
            t_pread = t_h2d = t_decode = t_interleave = t_evsync = t_final = 0.0
            t_wrap = 0.0
            n_batches = n_frames_done = n_cfg = 0
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
                    # Wait for this slot's previous batch (its interleave) before
                    # overwriting its pinned/device buffers -- decode no longer
                    # synchronizes, so this event is what keeps reuse safe.
                    _t = time.perf_counter() if trace_on else 0.0
                    events[slot].synchronize()
                    if trace_on:
                        t_evsync += time.perf_counter() - _t

                    halves: list[int] = []
                    srcs: list = []
                    outs: list = []
                    sig_parts: list[int] = []
                    # Read every frame's planes into this slot's pinned staging,
                    # H2D them, and build the per-chunk src/out Array batch.
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
                        hi_len_total = sum(int(c) for c in hi_chunks)
                        if hi_len_total > comp_cap:
                            raise ValueError(
                                f"fpz compressed frame ({hi_len_total} bytes) exceeds "
                                f"staging capacity ({comp_cap} bytes)"
                            )
                        usizes = _fpz_hi_chunk_usizes(half, chunk_u)
                        if len(usizes) != len(hi_chunks):
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
                            base + lo_len,
                            hiz_pin_view[slot][hiz_off : hiz_off + hi_len_total],
                        )
                        if trace_on:
                            t_pread += time.perf_counter() - _t
                        halves.append(half)
                        _t = time.perf_counter() if trace_on else 0.0
                        with torch.cuda.stream(stream):
                            lo_dev[slot].narrow(0, lo_off, half).copy_(
                                lo_pin[slot].narrow(0, lo_off, half), non_blocking=True
                            )
                            hiz_dev[slot].narrow(0, hiz_off, hi_len_total).copy_(
                                hiz_pin[slot].narrow(0, hiz_off, hi_len_total),
                                non_blocking=True,
                            )
                        if trace_on:
                            t_h2d += time.perf_counter() - _t
                        # One src Array per compressed chunk (exact length) and
                        # its hoisted out wrapper; chunk usizes drive the config.
                        # This per-chunk wrapper building is GIL-bound Python and
                        # is the dominant residual cost at small chunk sizes --
                        # its own trace bucket so its share is visible.
                        _t = time.perf_counter() if trace_on else 0.0
                        coff = hiz_off
                        for j, clen in enumerate(hi_chunks):
                            clen = int(clen)
                            srcs.append(
                                nvcomp.as_array(hiz_dev[slot].narrow(0, coff, clen))
                            )
                            outs.append(out_wrap[slot][k * max_chunks + j])
                            coff += clen
                        sig_parts.extend(usizes)
                        if trace_on:
                            t_wrap += time.perf_counter() - _t

                    # Reusable config per chunk-shape signature: build once (one
                    # sync, waits on the H2D above), then decode sync-free here
                    # and on every later batch that shares the shape.
                    sig = tuple(sig_parts)
                    _t = time.perf_counter() if trace_on else 0.0
                    cfg = configs.get(sig)
                    if cfg is None:
                        cfg = codec.decompression_config(srcs)
                        configs[sig] = cfg
                        n_cfg += 1
                    codec.decode(srcs, out=outs, decompression_config=cfg)
                    if trace_on:
                        t_decode += time.perf_counter() - _t

                    # Strided interleave per frame (same invariant as the CPU
                    # path): even bytes low plane, odd bytes high plane.
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
            _fpz_nvcomp_alloc_tls.stream = None
            if trace_on:
                t_final += time.perf_counter() - _t
                # Enqueue phases (h2d, interleave) are async so their wall is
                # small; a large `decode` wall means the decode CALL itself
                # blocks (internal sync / scratch alloc), while a large
                # `evsync`/`final` means the pipeline is GPU-bound waiting on
                # decode+interleave to finish.
                line = (
                    f"[fpz-gpu-trace] thread={thread_idx} frames={n_frames_done} "
                    f"batches={n_batches} cfg_builds={n_cfg} "
                    f"pread={t_pread:.3f}s h2d_enq={t_h2d:.3f}s "
                    f"wrap={t_wrap:.3f}s decode={t_decode:.3f}s "
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
            print(line)
    if errors:
        raise errors[0]


_FPZ_V1_GPU_WARNED = False


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
        nvcomp = _load_nvcomp() if _fpz_gpu_decode_enabled() else None
        # The GPU decoder only helps v2 (chunked) packs; a v1 pack is one nvcomp
        # chunk per frame and decodes serially, so fall back to the threaded CPU
        # decode for it (still correct, and faster than serial GPU decode).
        if nvcomp is not None and not _fpz_specs_all_v2(specs):
            nvcomp = None
            if not _FPZ_V1_GPU_WARNED:
                _FPZ_V1_GPU_WARNED = True
                warnings.warn(
                    "FLASHPACK_FPZ_GPU_DECODE=1 but this pack uses the v1 fpz "
                    "codec, which cannot be GPU-decoded in parallel; using the "
                    "CPU decode path. Repack with the v2 encoder for GPU decode.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if nvcomp is not None:
            _fpz_read_into_cuda_storage_gpu(path, specs, storage.blocks, device, nvcomp)
        else:
            _fpz_read_into_cuda_storage(path, specs, storage.blocks, device)
        return storage
    raise ValueError(f"Unsupported device: {device}")


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
        rank = dist.get_rank()
        meta = get_flashpack_file_metadata(path)
        specs = _build_macroblock_specs(meta)
        if rank == 0:
            flash_storage, meta = read_flashpack_file(
                path=path,
                device=device,
                silent=silent,
                num_streams=num_streams,
                chunk_bytes=chunk_bytes,
                metadata=meta,
            )
        else:
            flash_storage = _allocate_empty_storage(specs, device)
        _broadcast_storage(flash_storage, src=0)
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
