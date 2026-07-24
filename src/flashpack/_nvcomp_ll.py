"""ctypes bindings to libnvcomp's batched Zstd decompress C API (v5 ABI).

Why this exists: the ``nvidia.nvcomp`` pybind wrapper only exposes the
high-level Manager API (``Codec``/``Array``), which needs one Python Array
object per compressed chunk per decode call. At fpz chunk counts (tens of
thousands per pack) that per-object marshaling is the measured decode-wall
floor. The batched C API takes *device-resident tables* of chunk
pointers/sizes instead -- per batch, Python builds a few small tensors and
makes ONE foreign call, independent of chunk count. The pybind layer never
calls the batched C API at all, so going through ``libnvcomp.so`` directly
is the only route to it from Python.

ABI notes (verified against the headers shipped in nvidia-libnvcomp-cu12
5.3.0.16, the wheel the pybind package depends on):

* every ``device_*`` argument is a buffer in device memory, including the
  pointer tables themselves; only counts, the opts struct and the stream
  pass by host value;
* ``nvcompBatchedZstdDecompressOpts_t`` is a 64-byte struct passed BY VALUE
  (``backend`` enum + 60 reserved bytes; all-zero selects the default CUDA
  backend);
* v5 renamed the temp-size query to ``...GetTempSizeAsync`` (the widely
  documented ``...GetTempSizeEx`` is the v2-v4 name and does not exist in
  this .so);
* standard zstd frames (one independent frame per chunk) are accepted --
  the batched decompressor reads them directly, no nvcomp framing;
* for Zstd the per-chunk ``actual sizes`` and ``statuses`` output arrays
  must be real device buffers (unlike LZ4, where NULL is tolerated).

Load failures (missing wheel, missing symbol, non-Linux) degrade to
``load()`` returning ``None``; callers keep the pybind wrapper path as the
fallback.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading

logger = logging.getLogger(__name__)

NVCOMP_SUCCESS = 0


class _ZstdDecompressOpts(ctypes.Structure):
    """``nvcompBatchedZstdDecompressOpts_t``: 64 bytes, passed by value."""

    _fields_ = [("backend", ctypes.c_int), ("reserved", ctypes.c_char * 60)]


class _AlignmentRequirements(ctypes.Structure):
    """``nvcompAlignmentRequirements_t``: input/output/temp minimums."""

    _fields_ = [
        ("input", ctypes.c_size_t),
        ("output", ctypes.c_size_t),
        ("temp", ctypes.c_size_t),
    ]


def _find_libnvcomp() -> str | None:
    """Locate ``libnvcomp.so`` from the nvidia-libnvcomp wheel layout.

    The pip layout is ``site-packages/nvidia/libnvcomp/lib64/libnvcomp.so.5``;
    resolve it from the package rather than the linker path so the binding
    works without any LD_LIBRARY_PATH setup.
    """
    override = os.environ.get("FLASHPACK_LIBNVCOMP_PATH")
    if override:
        return override if os.path.exists(override) else None
    try:
        import nvidia.libnvcomp as _libnvcomp_pkg
    except ImportError:
        return None
    pkg_dir = os.path.dirname(_libnvcomp_pkg.__file__)
    for sub in ("lib64", "lib"):
        lib_dir = os.path.join(pkg_dir, sub)
        if not os.path.isdir(lib_dir):
            continue
        for name in sorted(os.listdir(lib_dir)):
            if name.startswith("libnvcomp.so"):
                return os.path.join(lib_dir, name)
    return None


class NvcompLL:
    """Thin, stateless handle over the batched Zstd decompress entry points.

    All methods are thread-safe: the underlying C functions are stateless
    launches, and ctypes releases the GIL for the duration of each call.
    """

    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib
        self._opts = _ZstdDecompressOpts()  # zero-filled = default backend

        self._get_alignments = lib.nvcompBatchedZstdDecompressGetRequiredAlignments
        self._get_alignments.restype = ctypes.c_int
        self._get_alignments.argtypes = [
            _ZstdDecompressOpts,
            ctypes.POINTER(_AlignmentRequirements),
        ]

        self._get_temp_size = lib.nvcompBatchedZstdDecompressGetTempSizeAsync
        self._get_temp_size.restype = ctypes.c_int
        self._get_temp_size.argtypes = [
            ctypes.c_size_t,  # num_chunks
            ctypes.c_size_t,  # max_uncompressed_chunk_bytes
            _ZstdDecompressOpts,
            ctypes.POINTER(ctypes.c_size_t),  # temp_bytes (host out)
            ctypes.c_size_t,  # max_total_uncompressed_bytes
        ]

        self._decompress = lib.nvcompBatchedZstdDecompressAsync
        self._decompress.restype = ctypes.c_int
        self._decompress.argtypes = [
            ctypes.c_void_p,  # device_compressed_chunk_ptrs (device void**)
            ctypes.c_void_p,  # device_compressed_chunk_bytes (device size_t*)
            ctypes.c_void_p,  # device_uncompressed_buffer_bytes (device size_t*)
            ctypes.c_void_p,  # device_uncompressed_chunk_bytes OUT (device size_t*)
            ctypes.c_size_t,  # num_chunks
            ctypes.c_void_p,  # device_temp_ptr
            ctypes.c_size_t,  # temp_bytes
            ctypes.c_void_p,  # device_uncompressed_chunk_ptrs (device void**)
            _ZstdDecompressOpts,  # by value
            ctypes.c_void_p,  # device_statuses (device nvcompStatus_t*)
            ctypes.c_void_p,  # cudaStream_t
        ]

        self._status_string = lib.nvcompGetStatusString
        self._status_string.restype = ctypes.c_char_p
        self._status_string.argtypes = [ctypes.c_int]

    def status_string(self, status: int) -> str:
        s = self._status_string(int(status))
        return s.decode() if s else f"nvcompStatus_t({status})"

    def _check(self, status: int, call: str) -> None:
        if status != NVCOMP_SUCCESS:
            raise RuntimeError(f"{call} failed: {self.status_string(status)}")

    def alignments(self) -> tuple[int, int, int]:
        """Required (input, output, temp) buffer alignments for decompression."""
        reqs = _AlignmentRequirements()
        self._check(
            self._get_alignments(self._opts, ctypes.byref(reqs)),
            "nvcompBatchedZstdDecompressGetRequiredAlignments",
        )
        return int(reqs.input), int(reqs.output), int(reqs.temp)

    def temp_size(
        self, num_chunks: int, max_chunk_bytes: int, max_total_bytes: int
    ) -> int:
        """Device scratch bytes needed for a decompress batch of this shape."""
        out = ctypes.c_size_t(0)
        self._check(
            self._get_temp_size(
                num_chunks,
                max_chunk_bytes,
                self._opts,
                ctypes.byref(out),
                max_total_bytes,
            ),
            "nvcompBatchedZstdDecompressGetTempSizeAsync",
        )
        return int(out.value)

    def decompress_async(
        self,
        src_ptrs_dev: int,
        src_sizes_dev: int,
        out_caps_dev: int,
        actual_sizes_dev: int,
        num_chunks: int,
        temp_ptr_dev: int,
        temp_bytes: int,
        dst_ptrs_dev: int,
        statuses_dev: int,
        cuda_stream: int,
    ) -> None:
        """Enqueue one batched decompress; all ``*_dev`` args are raw device
        addresses (``tensor.data_ptr()``). Raises on launch/validation errors;
        per-chunk data errors land in ``statuses_dev`` (device int32 array)
        for the caller to check stream-side."""
        self._check(
            self._decompress(
                src_ptrs_dev,
                src_sizes_dev,
                out_caps_dev,
                actual_sizes_dev,
                num_chunks,
                temp_ptr_dev,
                temp_bytes,
                dst_ptrs_dev,
                self._opts,
                statuses_dev,
                cuda_stream,
            ),
            "nvcompBatchedZstdDecompressAsync",
        )


_load_lock = threading.Lock()
_loaded: tuple[NvcompLL | None] | None = None


def load() -> NvcompLL | None:
    """Load and bind libnvcomp once; ``None`` (with a single warning) on any
    failure so callers can fall back to the pybind wrapper path."""
    global _loaded
    with _load_lock:
        if _loaded is not None:
            return _loaded[0]
        handle: NvcompLL | None = None
        path = _find_libnvcomp()
        if path is None:
            logger.warning(
                "flashpack: libnvcomp not found (pip install "
                "nvidia-libnvcomp-cu12); batched GPU decode unavailable, "
                "falling back to the nvcomp wrapper path."
            )
        else:
            try:
                handle = NvcompLL(ctypes.CDLL(path))
            except (OSError, AttributeError) as e:
                logger.warning(
                    "flashpack: could not bind batched nvcomp API from %s "
                    "(%s); falling back to the nvcomp wrapper path.",
                    path,
                    e,
                )
                handle = None
        _loaded = (handle,)
        return handle
