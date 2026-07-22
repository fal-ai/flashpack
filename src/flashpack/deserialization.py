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


def _fpz_read_frame_planes(
    fd: int, block_file_offset: int, frame: dict[str, Any], decompressor
) -> tuple[np.ndarray, np.ndarray]:
    """Read one fpz frame and return its ``(lo, hi)`` byte planes as uint8
    numpy arrays, each ``n_out // 2`` bytes.

    This is the shared decode step for both the CPU and CUDA read paths: it
    performs the file read (``preadv``) and the CPU zstd decode of the high
    plane. A future GPU decoder (nvcomp) can replace only the decompress call
    behind this same frame interface. The CPU path interleaves the planes with
    numpy; the CUDA path H2D-copies them and interleaves with strided copies.
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

    hi_bytes = decompressor.decompress(bytes(hi_raw), max_output_size=half)
    lo = np.frombuffer(lo_raw, dtype=np.uint8)
    hi = np.frombuffer(hi_bytes, dtype=np.uint8)
    if lo_len * 2 != n_out or lo.shape[0] != lo_len or hi.shape[0] != half:
        raise ValueError("fpz frame plane size mismatch")
    return lo, hi


def _fpz_decode_block_into_cpu(
    fd: int, spec: MacroblockSpec, dst_u8: np.ndarray, decompressor
) -> None:
    """Decode every frame of an fpz block into ``dst_u8`` (uint8 view of the
    uncompressed destination block). Interleaves the low/high planes so the
    reconstructed bytes are identical to the uncompressed pack:
    ``dst[0::2] = lo`` (low byte) and ``dst[1::2] = hi`` (high byte)."""
    assert spec.fpz is not None
    total = int(dst_u8.shape[0])
    out_pos = 0
    for frame in spec.fpz["frames"]:
        lo, hi = _fpz_read_frame_planes(fd, spec.offset_bytes, frame, decompressor)
        n_out = int(frame["n_out"])
        if out_pos + n_out > total:
            raise ValueError("fpz frames exceed the macroblock size")
        seg = dst_u8[out_pos : out_pos + n_out]
        seg[0::2] = lo
        seg[1::2] = hi
        out_pos += n_out
    if out_pos != total:
        raise ValueError(f"fpz frames cover {out_pos} bytes, expected {total}")


def _fpz_read_into_cpu_storage(
    path: str, specs: list[MacroblockSpec], blocks: list[torch.Tensor]
) -> None:
    """Fill pre-allocated (uncompressed-sized) CPU ``blocks`` from an fpz file.
    fpz blocks are decoded frame-by-frame; plain blocks are read verbatim."""
    zstandard = require_zstandard()
    decompressor = zstandard.ZstdDecompressor()
    fd = os.open(path, os.O_RDONLY)
    try:
        for spec, block in zip(specs, blocks):
            dst_u8 = block.view(torch.uint8).numpy()
            if spec.fpz is None:
                _pread_into(fd, spec.offset_bytes, memoryview(dst_u8))
            else:
                _fpz_decode_block_into_cpu(fd, spec, dst_u8, decompressor)
    finally:
        os.close(fd)


def _fpz_read_into_cuda_storage(
    path: str,
    specs: list[MacroblockSpec],
    blocks: list[torch.Tensor],
    device: torch.device,
) -> None:
    """Fill pre-allocated device ``blocks`` from an fpz file with a pool of
    reader threads (mirrors ``parallel_read_into_storage``).

    Each reader owns a file descriptor, a CUDA stream, and reusable pinned/
    device staging buffers. A work item is one fpz frame (or a whole plain
    block); frames write disjoint destination segments so the readers never
    collide. Per frame: read + CPU zstd-decode the planes (the shared
    ``_fpz_read_frame_planes``; zstd releases the GIL), H2D the contiguous low
    and decompressed-high planes, then two strided copies on the GPU
    (``dst_u8[0::2] = lo``, ``dst_u8[1::2] = hi``). A GPU decoder (nvcomp) would
    slot in by replacing the decompress inside ``_fpz_read_frame_planes``.

    GPU-untested locally (no CUDA device); the frame read, decode, and plane
    interleave are exercised by the CPU tests via the shared helpers above.
    """
    zstandard = require_zstandard()

    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError:
            return default

    n_threads = max(1, _env_int("FLASHPACK_READ_THREADS", 16))
    half_cap = FPZ_FRAME_UNCOMPRESSED_BYTES // 2

    byte_blocks = [b.view(torch.uint8) for b in blocks]

    # Order every reader stream after the destination allocation (same
    # wait_event pattern as parallel_read_into_storage).
    alloc_ready = torch.cuda.Event()
    alloc_ready.record(torch.cuda.current_stream(device))

    # Work items: ("frame", block_idx, frame, out_pos) or ("raw", block_idx, None, 0).
    work: queue.SimpleQueue = queue.SimpleQueue()
    n_tasks = 0
    for idx, spec in enumerate(specs):
        if spec.fpz is None:
            work.put(("raw", idx, None, 0))
            n_tasks += 1
        else:
            out_pos = 0
            for frame in spec.fpz["frames"]:
                work.put(("frame", idx, frame, out_pos))
                out_pos += int(frame["n_out"])
                n_tasks += 1
    n_threads = min(n_threads, max(1, n_tasks))
    for _ in range(n_threads):
        work.put(None)

    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            decompressor = zstandard.ZstdDecompressor()
            stream = torch.cuda.Stream(device=device)
            stream.wait_event(alloc_ready)
            lo_pin = torch.empty(half_cap, dtype=torch.uint8, pin_memory=True)
            hi_pin = torch.empty(half_cap, dtype=torch.uint8, pin_memory=True)
            lo_dev = torch.empty(half_cap, dtype=torch.uint8, device=device)
            hi_dev = torch.empty(half_cap, dtype=torch.uint8, device=device)
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
                        with torch.cuda.stream(stream):
                            dst.copy_(host, non_blocking=False)
                        continue
                    spec = specs[blk]
                    lo, hi = _fpz_read_frame_planes(
                        fd, spec.offset_bytes, frame, decompressor
                    )
                    n_out = int(frame["n_out"])
                    half = lo.shape[0]
                    lo_pin[:half].copy_(torch.from_numpy(lo))
                    hi_pin[:half].copy_(torch.from_numpy(hi))
                    seg = dst.narrow(0, out_pos, n_out)
                    with torch.cuda.stream(stream):
                        lo_dev[:half].copy_(lo_pin[:half], non_blocking=True)
                        hi_dev[:half].copy_(hi_pin[:half], non_blocking=True)
                        seg[0::2].copy_(lo_dev[:half], non_blocking=True)
                        seg[1::2].copy_(hi_dev[:half], non_blocking=True)
                    stream.synchronize()
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
