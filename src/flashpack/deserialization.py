import json
import math
import os
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
    DEFAULT_SHARD_BYTES,
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    MAGIC,
    SHARD_ALIGN_BYTES,
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
            specs.append(
                MacroblockSpec(
                    dtype=dtype,
                    offset_bytes=int(block["offset_bytes"]),
                    length_bytes=int(block["length_bytes"]),
                    length_elems=int(block["length_elems"]),
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
        raw = torch.empty(spec.length_bytes + align, dtype=torch.uint8)
        off = (-raw.data_ptr()) % align
        packing_dtype = get_packing_dtype(spec.dtype)
        block = raw.narrow(0, off, spec.length_bytes).view(packing_dtype)
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


@dataclass(frozen=True)
class _Window:
    """One superwindow of a macroblock.

    ``world`` equal shards of ``shard_bytes`` tile ``[base, base + span)`` of
    macroblock ``block``, and rank ``r`` owns shard ``r``. ``shard_bytes == 0``
    marks a sub-``world * align`` remainder that every rank reads itself --
    below that size a collective costs more than the duplicate read.
    """

    block: int
    base: int
    shard_bytes: int
    span: int


def _plan_windows(
    specs: list[MacroblockSpec],
    world: int,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    align: int = SHARD_ALIGN_BYTES,
) -> list[_Window]:
    """Tile every macroblock with interleaved superwindows.

    Rank ``r`` reads ``[base + r*C, base + (r+1)*C)`` of each window rather
    than one contiguous ``length/world`` slab. Interleaving keeps every rank's
    shard of a given window equal-sized, which is what lets the window be
    replicated with a single AllGather instead of ``world`` broadcasts, and it
    bounds the work in flight to one window at a time.

    Shard sizes step down geometrically (``C``, ``C/2``, ...) so a remainder
    never forces a multi-hundred-MB duplicate read on every rank; only the last
    sub-``world * align`` bytes are read redundantly.

    Every boundary is ``align``-aligned, so each shard starts on an element
    boundary for every dtype flashpack stores (``align`` is a multiple of all
    of them) and, when the macroblock's own file offset is aligned, on a 4096
    file offset too -- see ``_read_storage_sharded`` on the unaligned case.

    The plan is a pure function of the pack metadata and ``world``, so every
    rank derives an identical window list and therefore issues an identical
    collective sequence.
    """
    windows: list[_Window] = []
    for block_idx, spec in enumerate(specs):
        pos = 0
        n = int(spec.length_bytes)
        shard = max(align, shard_bytes - (shard_bytes % align))
        while shard >= align:
            span = world * shard
            while n - pos >= span:
                windows.append(_Window(block_idx, pos, shard, span))
                pos += span
            shard //= 2
            shard -= shard % align
        if pos < n:
            windows.append(_Window(block_idx, pos, 0, n - pos))
    return windows


def _all_gather_into(output: torch.Tensor, input_: torch.Tensor) -> None:
    """AllGather ``input_`` from every rank into ``output``.

    torch renamed this collective: ``all_gather_into_tensor`` is deprecated in
    favour of ``all_gather_single`` (which does not exist on the older torch
    this package still supports), so prefer the new name when present and fall
    back to the old one otherwise.
    """
    fn = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
    fn(output, input_)


_inplace_allgather_ok: dict[tuple[str, str], bool] = {}


def _supports_inplace_allgather(device: torch.device) -> bool:
    """Whether an in-place AllGather -- input aliasing the rank's own slice of
    the output -- both runs and produces the right bytes here.

    NCCL documents this rank-offset arrangement as in-place, which is what lets
    each rank read straight into its slot of the destination with no separate
    gather buffer. A backend that instead returns garbage would corrupt weights
    with nothing downstream to catch it, so this verifies the *result* on a tiny
    buffer once and falls back to staging the shard when it does not hold. The
    verdict is reduced across ranks so every rank takes the same path.

    Cached per (backend, device kind): one process can load over gloo/CPU and
    later over NCCL/CUDA, and the answer belongs to the transport, not the
    process.
    """
    cache_key = (dist.get_backend(), device.type)
    cached = _inplace_allgather_ok.get(cache_key)
    if cached is not None:
        return cached

    world = dist.get_world_size()
    rank = dist.get_rank()
    ok = True
    try:
        probe = torch.zeros(world * 8, dtype=torch.uint8, device=device)
        probe.narrow(0, rank * 8, 8).fill_(rank + 1)
        _all_gather_into(probe, probe.narrow(0, rank * 8, 8))
        expect = torch.arange(1, world + 1, dtype=torch.uint8, device=device)
        seen = probe.view(world, 8)
        ok = bool(
            torch.equal(seen.min(dim=1).values, expect)
            and torch.equal(seen.max(dim=1).values, expect)
        )
    except Exception:
        ok = False
    verdict = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    _inplace_allgather_ok[cache_key] = bool(verdict.item())
    return _inplace_allgather_ok[cache_key]


def _read_storage_sharded(
    path: str,
    specs: list[MacroblockSpec],
    storage: FlashTensorStorage,
    device: torch.device,
) -> None:
    """Every rank reads its shard of each superwindow, then one AllGather per
    window replicates it.

    Total disk bytes stay one pack-read, but the read wall drops toward
    ``read_time / world`` because the ranks read disjoint ranges concurrently --
    which also multiplies the request parallelism the filesystem sees, the part
    that matters most on a network FS whose cold throughput is per-client.

    Replication uses AllGather rather than ``world`` owner-broadcasts: it is the
    primitive for this pattern (every rank contributes one shard and receives
    the rest), it is one op per window instead of ``world`` ops, and it is what
    NCCL's copy-engine and NVLink-multicast fast paths are implemented for.

    A rank whose read fails still issues the rest of its collectives before
    raising, so a one-rank IO error surfaces as a synchronized exception instead
    of leaving the other ranks blocked in a collective forever.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    windows = _plan_windows(specs, world)

    # Each rank's own shards, as sub-specs the existing reader can consume. Its
    # chunk planner realigns to 4096 in *file* space (reading any sub-page head
    # buffered), so a macroblock whose own offset_bytes is unaligned -- fp8
    # static packs interleave a bf16 block ahead of the quantized one -- costs
    # one short buffered read per shard rather than losing O_DIRECT.
    sub_specs: list[MacroblockSpec] = []
    sub_blocks: list[torch.Tensor] = []
    for window in windows:
        spec = specs[window.block]
        block = storage.blocks[window.block]
        elem = block.element_size()
        lo = window.base + rank * window.shard_bytes
        length = window.shard_bytes or window.span
        sub_specs.append(
            MacroblockSpec(
                dtype=spec.dtype,
                offset_bytes=spec.offset_bytes + lo,
                length_bytes=length,
                length_elems=length // elem,
            )
        )
        sub_blocks.append(block.narrow(0, lo // elem, length // elem))

    read_error: BaseException | None = None
    if sub_specs:
        try:
            parallel_read_into_storage(path, sub_specs, sub_blocks, device)
        except BaseException as exc:  # noqa: BLE001 - re-raised, synchronized
            read_error = exc

    inplace = _supports_inplace_allgather(device)
    for window in windows:
        if not window.shard_bytes:
            continue  # read redundantly on every rank; nothing to replicate
        block = storage.blocks[window.block]
        elem = block.element_size()
        span = block.narrow(0, window.base // elem, window.span // elem)
        span = span.view(torch.uint8)
        shard = span.narrow(0, rank * window.shard_bytes, window.shard_bytes)
        _all_gather_into(span, shard if inplace else shard.clone())

    failed = torch.tensor(
        [1 if read_error is not None else 0], dtype=torch.int32, device=device
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        if read_error is not None:
            raise read_error
        raise RuntimeError(
            f"sharded flashpack load of {path} failed on another rank "
            f"(rank {rank} read its own shards successfully)"
        )


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
    sharded: bool = False,
) -> tuple[FlashTensorStorage, dict[str, Any]]:
    """Rank-``src`` reads the pack from disk; every rank returns the full
    storage, received via broadcast.

    With ``sharded=True`` every rank instead reads its own 1/N of each
    superwindow of each block, and one AllGather per window replicates it: the
    same one-pack-read total, but the disk wall drops toward ``read / world``
    because the ranks read disjoint ranges concurrently, and the replication
    runs at fabric rather than filesystem rates. Requires every rank to be able
    to read the pack payload from ``path`` (rank-``src`` mode only needs the
    footer on non-src ranks). Falls back to rank-``src`` mode for packs with
    compressed (fpz) blocks, whose on-disk bytes are not shard-addressable.

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
    if sharded and any("fpz" in block for block in meta.get("macroblocks", []) or []):
        sharded = False  # compressed payload bytes are not shard-addressable
    if sharded and dist.get_world_size() > 1:
        specs = _build_macroblock_specs(meta)
        storage = _allocate_empty_storage(specs, device)
        _read_storage_sharded(path, specs, storage, device)
        return storage, meta
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
    distributed_sharded: bool = False,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
    coerce_dtype: bool = False,
) -> None:
    """
    Assign the weights from a flashpack file to a model.

    ``distributed_sharded=True`` (with ``use_distributed_loading=True``) makes
    every rank read a 1/N shard instead of rank 0 reading the whole pack; see
    ``read_flashpack_file_distributed``.
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
            sharded=distributed_sharded,
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
