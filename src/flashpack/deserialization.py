import json
import math
import os
import queue
import threading
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
    FPZ_FRAME_UNCOMPRESSED_BYTES,
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
                if codec != FPZ_CODEC_SPLITPLANE_V1:
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
    fd: int, block_file_offset: int, frame: dict[str, Any], decompressor
) -> tuple[np.ndarray, np.ndarray]:
    """Read one fpz frame and return its ``(lo, hi)`` byte planes as uint8
    numpy arrays, each ``n_out // 2`` bytes.

    Shared decode step for the CPU and CUDA read paths: the ``preadv`` read and
    the zstd decode of the high plane. zstd ``decompress`` releases the GIL and
    accepts the compressed input as a buffer, so the read target is passed as a
    ``memoryview`` (no intermediate ``bytes`` copy) and N threads scale near
    linearly. A future GPU decoder (nvcomp) can replace only the decompress
    call behind this same frame interface.
    """
    payload_off = int(frame["payload_off"])
    lo_len = int(frame["lo_len"])
    hi_len = int(frame["hi_len"])
    n_out = int(frame["n_out"])
    half = n_out - lo_len

    lo_raw = bytearray(lo_len)
    _pread_into(fd, block_file_offset + payload_off, memoryview(lo_raw))
    hi_raw = bytearray(hi_len)
    _pread_into(fd, block_file_offset + payload_off + lo_len, memoryview(hi_raw))

    # memoryview input avoids a GIL-held full copy of the compressed plane;
    # decompress itself releases the GIL.
    hi_bytes = decompressor.decompress(memoryview(hi_raw), max_output_size=half)
    lo = np.frombuffer(lo_raw, dtype=np.uint8)
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
                    lo, hi = _fpz_read_frame_planes(
                        fd, spec.offset_bytes, frame, decompressor
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
            lo_view = [memoryview(b.numpy()) for b in lo_pin]
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

                    payload_off = int(frame["payload_off"])
                    lo_len = int(frame["lo_len"])
                    hi_len = int(frame["hi_len"])
                    n_out = int(frame["n_out"])
                    half = n_out - lo_len
                    base = specs[blk].offset_bytes + payload_off

                    # Low plane: preadv straight into the pinned buffer (no
                    # intermediate numpy/bytearray copy).
                    _pread_into(fd, base, lo_view[slot][:lo_len])
                    # High plane: decode (GIL released), then one copy into pin.
                    hi_raw = bytearray(hi_len)
                    _pread_into(fd, base + lo_len, memoryview(hi_raw))
                    hi_bytes = decompressor.decompress(
                        memoryview(hi_raw), max_output_size=half
                    )
                    if lo_len != half or lo_len * 2 != n_out or len(hi_bytes) != half:
                        raise ValueError("fpz frame plane size mismatch")
                    hi_pin[slot][:half].copy_(
                        torch.frombuffer(hi_bytes, dtype=torch.uint8)
                    )

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
#            (once per decode/batch). The fully sync-free path needs a
#            DecompressConfig built from a CompressConfig in the SAME process
#            that compressed the data -- not our case (packs are compressed
#            offline) -- so the prototype accepts one sync per batch. That sync
#            is also what makes reusing the pinned staging buffers across
#            batches safe (see the reader loop). Eliminating it is the main
#            throughput lever left for the H200 tuning pass.
#
#   nvcomp.as_array(src_object, cuda_stream=None) -> Array
#     "Creates array from object with some standard interface." Zero-copy over
#     any object exposing __cuda_array_interface__ / __dlpack__. A contiguous
#     torch CUDA tensor qualifies, so we wrap the device staging tensors
#     directly (no copy). nvcomp.from_dlpack(...) is the explicit-DLPack
#     equivalent; as_array is sufficient here.
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


# GPU-decode tuning knobs (env-overridable). Each batched frame needs device
# staging for its low plane, its compressed high plane, and its decompressed
# high plane (~3 x FPZ_FRAME_UNCOMPRESSED_BYTES/2), plus pinned host staging for
# the low and compressed-high reads. Rough per-thread device footprint is
# ``batch_frames * 3 * (FPZ_FRAME_UNCOMPRESSED_BYTES/2)`` and pinned footprint
# ``batch_frames * 2 * (FPZ_FRAME_UNCOMPRESSED_BYTES/2)``; total scales by the
# thread count. Defaults target an H200-class GPU -- turn them down on smaller
# cards.
_FPZ_GPU_DEFAULT_THREADS = 4
_FPZ_GPU_DEFAULT_BATCH_FRAMES = 4
_FPZ_GPU_DEFAULT_BATCH_BYTES = 256 * 1024 * 1024  # summed uncompressed per batch


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
    """GPU-decode variant of :func:`_fpz_read_into_cuda_storage`.

    Same threaded reader structure and destination contract, but the high plane
    is decompressed on the device with nvcomp's batched Zstd decoder instead of
    on the CPU. Per reader thread: read the low and compressed-high planes into
    pinned staging, H2D both (moving the *compressed* high plane cuts PCIe
    traffic ~2.4x), decode a batch of high planes on the GPU straight into the
    device high-plane staging, then run the same strided interleave
    (``dst[0::2] = lo``, ``dst[1::2] = hi``).

    Each thread owns its file descriptor, CUDA stream, nvcomp Codec (bound to
    that stream) and a fixed set of reused staging buffers. All device work for
    a batch is enqueued on the one stream, so H2D -> decode -> interleave order
    is guaranteed without explicit per-op syncs; the decode's internal
    per-batch synchronization is what makes reusing the pinned buffers in the
    next batch safe. Raw (uncompressed) blocks take the same whole-block H2D as
    the CPU-decode path.
    """
    half_cap = FPZ_FRAME_UNCOMPRESSED_BYTES // 2
    # zstd worst-case output for a half_cap input; each frame's stored hi_len
    # never exceeds this (its input is at most half_cap bytes).
    comp_cap = half_cap + (half_cap // 255) + 4096

    n_threads = max(
        1, _env_int("FLASHPACK_FPZ_GPU_DECODE_THREADS", _FPZ_GPU_DEFAULT_THREADS)
    )
    batch_frames = max(
        1, _env_int("FLASHPACK_FPZ_GPU_BATCH_FRAMES", _FPZ_GPU_DEFAULT_BATCH_FRAMES)
    )
    batch_bytes = max(
        1, _env_int("FLASHPACK_FPZ_GPU_BATCH_BYTES", _FPZ_GPU_DEFAULT_BATCH_BYTES)
    )

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

    def _reader() -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(alloc_ready)
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
            # Fixed, reused staging (see the per-batch-sync note above).
            lo_pin = [
                torch.empty(half_cap, dtype=torch.uint8, pin_memory=True)
                for _ in range(batch_frames)
            ]
            hiz_pin = [
                torch.empty(comp_cap, dtype=torch.uint8, pin_memory=True)
                for _ in range(batch_frames)
            ]
            lo_view = [memoryview(b.numpy()) for b in lo_pin]
            hiz_view = [memoryview(b.numpy()) for b in hiz_pin]
            lo_dev = [
                torch.empty(half_cap, dtype=torch.uint8, device=device)
                for _ in range(batch_frames)
            ]
            hiz_dev = [
                torch.empty(comp_cap, dtype=torch.uint8, device=device)
                for _ in range(batch_frames)
            ]
            hi_dev = [
                torch.empty(half_cap, dtype=torch.uint8, device=device)
                for _ in range(batch_frames)
            ]
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
                    halves: list[int] = []
                    hi_lens: list[int] = []
                    # Read every frame's planes into pinned staging and enqueue
                    # the H2D of the low plane and the (small) compressed high
                    # plane onto the stream.
                    for k, (_, blk, frame, _out_pos) in enumerate(batch):
                        payload_off = int(frame["payload_off"])
                        lo_len = int(frame["lo_len"])
                        hi_len = int(frame["hi_len"])
                        n_out = int(frame["n_out"])
                        half = n_out - lo_len
                        if lo_len != half or lo_len * 2 != n_out:
                            raise ValueError("fpz frame plane size mismatch")
                        if hi_len > comp_cap:
                            raise ValueError(
                                f"fpz compressed frame ({hi_len} bytes) exceeds "
                                f"staging capacity ({comp_cap} bytes)"
                            )
                        base = specs[blk].offset_bytes + payload_off
                        _pread_into(fd, base, lo_view[k][:lo_len])
                        _pread_into(fd, base + lo_len, hiz_view[k][:hi_len])
                        halves.append(half)
                        hi_lens.append(hi_len)
                        with torch.cuda.stream(stream):
                            lo_dev[k][:half].copy_(lo_pin[k][:half], non_blocking=True)
                            hiz_dev[k][:hi_len].copy_(
                                hiz_pin[k][:hi_len], non_blocking=True
                            )

                    # Batched GPU Zstd decode straight into the device high-plane
                    # staging (out= views are externally-allocated, so decode
                    # writes in place). Runs on the codec's stream == our stream.
                    srcs = [
                        nvcomp.as_array(hiz_dev[k][: hi_lens[k]])
                        for k in range(len(batch))
                    ]
                    outs = [
                        nvcomp.as_array(hi_dev[k][: halves[k]])
                        for k in range(len(batch))
                    ]
                    codec.decode(srcs, out=outs)

                    # Strided interleave per frame (same invariant as the CPU
                    # path): even bytes low plane, odd bytes high plane.
                    for k, (_, blk, frame, out_pos) in enumerate(batch):
                        half = halves[k]
                        n_out = int(frame["n_out"])
                        seg = byte_blocks[blk].narrow(0, out_pos, n_out)
                        with torch.cuda.stream(stream):
                            seg[0::2].copy_(lo_dev[k][:half], non_blocking=True)
                            seg[1::2].copy_(hi_dev[k][:half], non_blocking=True)
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


def _read_fpz_into_storage(
    path: str, specs: list[MacroblockSpec], device: torch.device
) -> FlashTensorStorage:
    """Materialize an fpz (partially compressed) pack into ``device`` storage."""
    if device.type == "cpu":
        storage = _allocate_aligned_cpu_storage(specs)
        _fpz_read_into_cpu_storage(path, specs, storage.blocks)
        return storage
    if device.type == "cuda":
        storage = _allocate_empty_storage(specs, device)
        nvcomp = _load_nvcomp() if _fpz_gpu_decode_enabled() else None
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
