"""Unit tests for fpz split-plane zstd compression (``compress="fpz-bf16"``).

These exercise the CPU read path end to end: byte-identical round trips, the
mixed-dtype policy (only bf16 blocks compress), the multi-frame/alignment
logic, and the error surfaces. The CUDA decoder shares the frame read + decode
+ interleave helpers with the CPU path, so covering them here also covers the
GPU decoder's correctness-critical core (the GPU path itself needs a device and
is not run in CI).
"""

import os
import sys
import time

import numpy as np
import pytest
import torch
from flashpack import serialization
from flashpack.constants import (
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    FPZ_CODEC_SPLITPLANE_V1,
    FPZ_CODEC_SPLITPLANE_V2,
    FPZ_FRAME_ALIGN_BYTES,
)
from flashpack.deserialization import (
    assign_from_file,
    get_flashpack_file_metadata,
    iterate_from_flash_tensor,
    read_flashpack_file,
    revert_from_file,
)
from flashpack.serialization import pack_to_file
from flashpack.utils import require_zstandard

# The fpz read paths pull bytes with os.preadv (POSIX-only), matching the
# repo's O_DIRECT reader; the format targets Linux GPU fleets. Encoder and
# reader are exercised on Linux/macOS.
pytestmark = pytest.mark.skipif(
    not hasattr(os, "preadv"),
    reason="fpz read paths require os.preadv (POSIX-only)",
)


def _bf16_state_dict() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    # A genuinely high-entropy tensor (randn) and a low-entropy one (a smooth
    # ramp -- exponents change slowly, so the high byte plane compresses hard).
    random = torch.randn(1024, 512, generator=generator).to(torch.bfloat16)
    ramp = (torch.arange(1024 * 512, dtype=torch.float32) * 0.01).reshape(1024, 512)
    return {"block.random": random, "block.ramp": ramp.to(torch.bfloat16)}


def _uint16_view(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint16)


def _raw_bytes(t: torch.Tensor) -> torch.Tensor:
    # dtype-agnostic byte view for bit-exact (NaN-safe) comparison.
    return t.contiguous().view(torch.uint8)


def _pack(tmp_path, state_dict, name: str, **kwargs) -> str:
    path = str(tmp_path / name)
    kwargs.setdefault("target_dtype", None)
    pack_to_file(state_dict, path, **kwargs)
    return path


def test_bf16_roundtrip_is_bit_identical(tmp_path) -> None:
    source = _bf16_state_dict()
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    storage_p, meta_p = read_flashpack_file(plain, device="cpu")
    storage_c, meta_c = read_flashpack_file(comp, device="cpu")
    plain_tensors = dict(iterate_from_flash_tensor(storage_p, meta_p))
    comp_tensors = dict(iterate_from_flash_tensor(storage_c, meta_c))

    assert set(comp_tensors) == set(source)
    for name, original in source.items():
        # bit-exact against both the uncompressed pack and the source bytes
        assert torch.equal(_uint16_view(comp_tensors[name]), _uint16_view(original))
        assert torch.equal(
            _uint16_view(comp_tensors[name]), _uint16_view(plain_tensors[name])
        )


def test_compressed_file_is_v4_with_fpz_record(tmp_path) -> None:
    comp = _pack(tmp_path, _bf16_state_dict(), "comp.flashpack", compress="fpz-bf16")
    meta = get_flashpack_file_metadata(comp)
    assert meta["format"] == FILE_FORMAT_V4
    (block,) = meta["macroblocks"]
    assert block["fpz"]["codec"] == FPZ_CODEC_SPLITPLANE_V2
    assert len(block["fpz"]["frames"]) >= 1
    # v2 frames carry per-chunk compressed lengths instead of a single hi_len.
    assert block["fpz"]["frames"][0]["hi_chunks"]
    # length_elems stays logical; length_bytes is the smaller on-disk payload.
    assert block["length_bytes"] < block["length_elems"] * 2


def test_low_entropy_tensor_shrinks_file(tmp_path) -> None:
    ramp = (torch.arange(2048 * 1024, dtype=torch.float32) * 0.01).reshape(2048, 1024)
    source = {"w": ramp.to(torch.bfloat16)}
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    plain_size = os.path.getsize(plain)
    comp_size = os.path.getsize(comp)
    assert comp_size < plain_size
    assert plain_size / comp_size > 1.5


def test_mixed_dtype_only_bf16_compressed(tmp_path) -> None:
    source = {
        "bf16.big": torch.randn(512, 512).to(torch.bfloat16),
        "fp32.big": torch.randn(512, 512),
    }
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    meta = get_flashpack_file_metadata(comp)
    assert meta["format"] == FILE_FORMAT_V4
    by_dtype = {b["dtype"]: b for b in meta["macroblocks"]}
    assert "fpz" in by_dtype["bfloat16"]
    assert "fpz" not in by_dtype["float32"]

    # Only bf16 shrank, but the whole file must still round-trip bit-exactly.
    assert os.path.getsize(comp) < os.path.getsize(plain)
    storage, m = read_flashpack_file(comp, device="cpu")
    tensors = dict(iterate_from_flash_tensor(storage, m))
    assert torch.equal(tensors["fp32.big"], source["fp32.big"])
    assert torch.equal(
        _uint16_view(tensors["bf16.big"]), _uint16_view(source["bf16.big"])
    )


def test_fp32_only_compress_is_noop(tmp_path) -> None:
    # No bf16 block -> nothing to compress -> identical to the plain pack.
    source = {"a": torch.randn(64), "b": torch.randn(32)}
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    assert get_flashpack_file_metadata(comp)["format"] == FILE_FORMAT_V3
    with open(plain, "rb") as f:
        plain_bytes = f.read()
    with open(comp, "rb") as f:
        comp_bytes = f.read()
    assert plain_bytes == comp_bytes


def test_multi_frame_roundtrip_and_alignment(tmp_path, monkeypatch) -> None:
    # Shrink the frame step so a modest tensor spans several frames, exercising
    # the padding/alignment and multi-frame cover logic.
    monkeypatch.setattr(
        "flashpack.serialization.FPZ_FRAME_UNCOMPRESSED_BYTES", 8192, raising=True
    )
    ramp = (torch.arange(8192 * 6, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    source = {"w": ramp}
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    frames = get_flashpack_file_metadata(comp)["macroblocks"][0]["fpz"]["frames"]
    assert len(frames) >= 2
    for frame in frames:
        assert frame["payload_off"] % FPZ_FRAME_ALIGN_BYTES == 0
        assert frame["lo_len"] * 2 == frame["n_out"]

    storage, meta = read_flashpack_file(comp, device="cpu")
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, meta)]
    assert torch.equal(_uint16_view(w), _uint16_view(ramp))


def test_iterate_views_match_plain(tmp_path) -> None:
    source = _bf16_state_dict()
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    sp, mp = read_flashpack_file(plain, device="cpu")
    sc, mc = read_flashpack_file(comp, device="cpu")
    plain_items = list(iterate_from_flash_tensor(sp, mp))
    comp_items = list(iterate_from_flash_tensor(sc, mc))
    assert [n for n, _ in plain_items] == [n for n, _ in comp_items]
    for (_, tp), (_, tc) in zip(plain_items, comp_items):
        assert tp.shape == tc.shape
        assert torch.equal(_uint16_view(tp), _uint16_view(tc))


def test_revert_from_compressed_matches_source(tmp_path) -> None:
    source = _bf16_state_dict()
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    reverted = revert_from_file(comp)
    assert set(reverted) == set(source)
    for name, original in source.items():
        assert reverted[name].dtype is torch.bfloat16
        assert torch.equal(_uint16_view(reverted[name]), _uint16_view(original))


def test_assign_from_compressed_pack(tmp_path) -> None:
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(128, 128).to(torch.bfloat16)

    torch.manual_seed(0)
    source = Net()
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    torch.manual_seed(1)
    destination = Net()
    assign_from_file(destination, comp, device="cpu")
    assert torch.equal(
        _uint16_view(destination.linear.weight.detach()),
        _uint16_view(source.linear.weight.detach()),
    )


def test_unknown_compress_option_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported compress option"):
        pack_to_file(
            {"a": torch.randn(4).to(torch.bfloat16)},
            str(tmp_path / "pack.flashpack"),
            target_dtype=None,
            compress="gzip",
        )


def test_unknown_codec_rejected(tmp_path) -> None:
    comp = _pack(tmp_path, _bf16_state_dict(), "comp.flashpack", compress="fpz-bf16")
    meta = get_flashpack_file_metadata(comp)
    meta["macroblocks"][0]["fpz"]["codec"] = "bogus-codec-v9"
    with pytest.raises(ValueError, match="Unsupported fpz codec"):
        read_flashpack_file(comp, device="cpu", metadata=meta)


def test_truncated_frame_rejected(tmp_path) -> None:
    source = {"w": torch.randn(2048, 1024).to(torch.bfloat16)}
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    # Capture the full metadata, then physically truncate the payload so the
    # declared frame bytes run past EOF.
    meta = get_flashpack_file_metadata(comp)
    size = os.path.getsize(comp)
    with open(comp, "r+b") as f:
        f.truncate(size // 2)
    with pytest.raises(IOError, match="short read"):
        read_flashpack_file(comp, device="cpu", metadata=meta)


def test_corrupt_high_plane_rejected(tmp_path) -> None:
    source = {"w": torch.randn(1024, 512).to(torch.bfloat16)}
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    block = get_flashpack_file_metadata(comp)["macroblocks"][0]
    frame = block["fpz"]["frames"][0]
    hi_start = block["offset_bytes"] + frame["payload_off"] + frame["lo_len"]
    # Scribble over the first compressed high-plane chunk; zstd must reject it.
    first_chunk_len = int(frame["hi_chunks"][0])
    with open(comp, "r+b") as f:
        f.seek(hi_start)
        f.write(b"\xff" * min(64, first_chunk_len))
    zstandard = require_zstandard()
    with pytest.raises((zstandard.ZstdError, ValueError)):
        read_flashpack_file(comp, device="cpu")


def test_require_zstandard_message_when_missing(monkeypatch) -> None:
    # Simulate the optional dependency being absent.
    monkeypatch.setitem(sys.modules, "zstandard", None)
    with pytest.raises(ImportError, match="zstandard"):
        require_zstandard()


def test_interleave_byte_order_is_little_endian(tmp_path) -> None:
    # Confirm the plane split matches bf16's little-endian layout: even bytes
    # are the low (mantissa-LSB) plane, odd bytes are the high (sign+exp) plane.
    values = torch.tensor([1.5, -2.0, 0.0, 3.25], dtype=torch.bfloat16)
    comp = _pack(tmp_path, {"w": values}, "comp.flashpack", compress="fpz-bf16")
    storage, meta = read_flashpack_file(comp, device="cpu")
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, meta)]
    raw = _uint16_view(w).numpy().view(np.uint8)
    expected = _uint16_view(values).numpy().view(np.uint8)
    assert np.array_equal(raw, expected)


def test_streaming_decode_matches_uncompressed_pack(tmp_path) -> None:
    # Equivalence: the streaming compressed pack decodes bit-for-bit identically
    # to the plain uncompressed pack of the same mixed state dict. The odd sizes
    # force intra-block alignment padding (a zero gap between bf16.a and bf16.b),
    # exercising the streaming gap-fill against the memmap layout.
    source = {
        "bf16.a": torch.randn(301, 400).to(torch.bfloat16),
        "bf16.b": torch.randn(51).to(torch.bfloat16),
        "fp32.c": torch.randn(128, 64),
    }
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    sp, mp = read_flashpack_file(plain, device="cpu")
    sc, mc = read_flashpack_file(comp, device="cpu")
    plain_tensors = dict(iterate_from_flash_tensor(sp, mp))
    comp_tensors = dict(iterate_from_flash_tensor(sc, mc))
    assert set(plain_tensors) == set(comp_tensors) == set(source)
    for name in source:
        assert torch.equal(
            _raw_bytes(plain_tensors[name]), _raw_bytes(comp_tensors[name])
        )
        assert torch.equal(_raw_bytes(comp_tensors[name]), _raw_bytes(source[name]))


def test_streaming_creates_no_uncompressed_scratch(tmp_path, monkeypatch) -> None:
    # The whole point of streaming: never materialize the uncompressed payload.
    # Prove it two ways -- np.memmap (the only uncompressed-scratch allocator in
    # the write path) is never called, and exactly ONE tempfile is created (the
    # final compressed pack), not a scratch + final pair like the old post-pass.
    import flashpack.serialization as serialization

    # Small frames so the rolling-buffer multi-frame path runs under the guards.
    monkeypatch.setattr(
        serialization, "FPZ_FRAME_UNCOMPRESSED_BYTES", 1 << 16, raising=True
    )

    def no_memmap(*args, **kwargs):
        raise AssertionError("uncompressed scratch memmap must not be created")

    monkeypatch.setattr(serialization.np, "memmap", no_memmap)

    tempfiles: list[str] = []
    real_mkstemp = serialization.tempfile.mkstemp

    def counting_mkstemp(*args, **kwargs):
        result = real_mkstemp(*args, **kwargs)
        tempfiles.append(result[1])
        return result

    monkeypatch.setattr(serialization.tempfile, "mkstemp", counting_mkstemp)

    ramp = (torch.arange(1 << 18, dtype=torch.float32) * 0.001).to(torch.bfloat16)
    comp = str(tmp_path / "comp.flashpack")
    pack_to_file({"w": ramp}, comp, target_dtype=None, compress="fpz-bf16")

    assert len(tempfiles) == 1  # only the final compressed pack, no scratch
    frames = get_flashpack_file_metadata(comp)["macroblocks"][0]["fpz"]["frames"]
    assert len(frames) >= 2  # multi-frame rolling-buffer path exercised

    storage, meta = read_flashpack_file(comp, device="cpu")
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, meta)]
    assert torch.equal(_uint16_view(w), _uint16_view(ramp))


def test_streaming_gap_straddling_frame_boundary(tmp_path, monkeypatch) -> None:
    # Two bf16 tensors with alignment padding between them, and a frame step
    # small enough that the zero gap straddles a frame cut -- the case where
    # the rolling buffer must carry a partial gap across the flush boundary.
    import flashpack.serialization as serialization

    monkeypatch.setattr(serialization, "FPZ_FRAME_UNCOMPRESSED_BYTES", 64, raising=True)
    source = {
        "a": torch.randn(40).to(torch.bfloat16),
        "b": torch.randn(40).to(torch.bfloat16),
    }
    plain = _pack(tmp_path, source, "plain.flashpack", align_bytes=128)
    comp = _pack(
        tmp_path, source, "comp.flashpack", align_bytes=128, compress="fpz-bf16"
    )

    frames = get_flashpack_file_metadata(comp)["macroblocks"][0]["fpz"]["frames"]
    assert len(frames) >= 2

    sp, mp = read_flashpack_file(plain, device="cpu")
    sc, mc = read_flashpack_file(comp, device="cpu")
    plain_tensors = dict(iterate_from_flash_tensor(sp, mp))
    comp_tensors = dict(iterate_from_flash_tensor(sc, mc))
    for name in source:
        assert torch.equal(
            _raw_bytes(plain_tensors[name]), _raw_bytes(comp_tensors[name])
        )
        assert torch.equal(_raw_bytes(comp_tensors[name]), _raw_bytes(source[name]))


@pytest.mark.skipif(
    (os.cpu_count() or 1) < 4, reason="thread-scaling proof needs >=4 CPUs"
)
def test_cpu_decode_scales_with_threads(tmp_path, monkeypatch) -> None:
    # Regression guard for the serialized CPU decode: the fpz read path must
    # parallelize across FLASHPACK_READ_THREADS. zstd decompress releases the
    # GIL, so 8 threads must materially beat 1. Generous bound (<=0.6x wall)
    # with best-of-3 timing so CI jitter can't flake it.
    import flashpack.serialization as serialization

    monkeypatch.setattr(
        serialization, "FPZ_FRAME_UNCOMPRESSED_BYTES", 1 << 20, raising=True
    )
    # ~64 MB uncompressed, many frames; entropy high enough that decode (not
    # I/O from the warm page cache) dominates the wall.
    data = (torch.randn(32 * 1024 * 1024) * 0.05).to(torch.bfloat16)
    comp = str(tmp_path / "comp.flashpack")
    pack_to_file({"w": data}, comp, target_dtype=None, compress="fpz-bf16")

    frames = get_flashpack_file_metadata(comp)["macroblocks"][0]["fpz"]["frames"]
    assert len(frames) >= 16  # enough work to spread over 8 threads

    def best_wall(threads: int, reps: int = 3) -> float:
        monkeypatch.setenv("FLASHPACK_READ_THREADS", str(threads))
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            read_flashpack_file(comp, device="cpu")
            best = min(best, time.perf_counter() - t0)
        return best

    read_flashpack_file(comp, device="cpu")  # warm the page cache
    single = best_wall(1)
    multi = best_wall(8)

    # Sanity: decode is still correct under many threads.
    monkeypatch.setenv("FLASHPACK_READ_THREADS", "8")
    storage, meta = read_flashpack_file(comp, device="cpu")
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, meta)]
    assert torch.equal(_uint16_view(w), _uint16_view(data))

    assert multi <= 0.6 * single, (
        f"fpz CPU decode did not scale: 1-thread={single * 1e3:.1f} ms, "
        f"8-thread={multi * 1e3:.1f} ms (expected 8-thread <= 0.6x)"
    )


# --------------------------------------------------------------------------
# v2 (chunked high plane) format + v1 backward compatibility
# --------------------------------------------------------------------------


def test_v2_frame_splits_high_plane_into_chunks(tmp_path) -> None:
    # A 4096x1024 bf16 tensor has a 4 MiB high plane -> several 1 MiB chunks.
    source = {"w": torch.randn(4096, 1024).to(torch.bfloat16)}
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    block = get_flashpack_file_metadata(comp)["macroblocks"][0]
    assert block["fpz"]["codec"] == FPZ_CODEC_SPLITPLANE_V2
    (frame,) = block["fpz"]["frames"]
    assert len(frame["hi_chunks"]) >= 2  # multiple parallel-decodable chunks
    # Round-trips bit-exactly through the chunked decode.
    storage, meta = read_flashpack_file(comp, device="cpu")
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, meta)]
    assert torch.equal(_uint16_view(w), _uint16_view(source["w"]))


def test_v1_pack_still_reads(tmp_path, monkeypatch) -> None:
    # Backward compatibility: a pack written by the v1 encoder must still decode
    # bit-exactly (the reader supports both codecs).
    monkeypatch.setattr(serialization, "_DEFAULT_FPZ_VERSION", 1)
    source = _bf16_state_dict()
    comp = _pack(tmp_path, source, "comp_v1.flashpack", compress="fpz-bf16")
    block = get_flashpack_file_metadata(comp)["macroblocks"][0]
    assert block["fpz"]["codec"] == FPZ_CODEC_SPLITPLANE_V1
    assert "hi_len" in block["fpz"]["frames"][0]  # single-frame high plane

    storage, meta = read_flashpack_file(comp, device="cpu")
    tensors = dict(iterate_from_flash_tensor(storage, meta))
    for name, original in source.items():
        assert torch.equal(_uint16_view(tensors[name]), _uint16_view(original))


def test_v2_ratio_close_to_v1(tmp_path, monkeypatch) -> None:
    # v2 compresses each 64 KiB chunk independently, so it shrinks slightly less
    # than v1's single-frame high plane. Measure both on a realistic low-entropy
    # tensor; v2 must still compress and stay within a modest margin of v1.
    ramp = (torch.arange(2048 * 1024, dtype=torch.float32) * 0.01).reshape(2048, 1024)
    source = {"w": ramp.to(torch.bfloat16)}
    plain = _pack(tmp_path, source, "plain.flashpack")

    monkeypatch.setattr(serialization, "_DEFAULT_FPZ_VERSION", 1)
    v1 = _pack(tmp_path, source, "v1.flashpack", compress="fpz-bf16")
    monkeypatch.setattr(serialization, "_DEFAULT_FPZ_VERSION", 2)
    v2 = _pack(tmp_path, source, "v2.flashpack", compress="fpz-bf16")

    plain_sz = os.path.getsize(plain)
    v1_sz = os.path.getsize(v1)
    v2_sz = os.path.getsize(v2)
    assert v2_sz < plain_sz  # v2 still compresses
    assert v2_sz <= v1_sz * 1.25  # within a modest margin of v1


# --------------------------------------------------------------------------
# v2 parameterized chunk size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_bytes", [64 * 1024, 256 * 1024, 1024 * 1024])
def test_v2_roundtrip_at_various_chunk_sizes(tmp_path, chunk_bytes) -> None:
    # A 2048x1024 bf16 tensor has a 2 MiB high plane -> several chunks at each
    # size; every size must round-trip bit-exactly and record its chunk size.
    ramp = (torch.arange(2048 * 1024, dtype=torch.float32) * 0.01).reshape(2048, 1024)
    source = {"w": ramp.to(torch.bfloat16)}
    comp = _pack(
        tmp_path,
        source,
        f"c{chunk_bytes}.flashpack",
        compress="fpz-bf16",
        hi_chunk_bytes=chunk_bytes,
    )
    block = get_flashpack_file_metadata(comp)["macroblocks"][0]
    assert block["fpz"]["hi_chunk_usize"] == chunk_bytes  # footer field present
    assert len(block["fpz"]["frames"][0]["hi_chunks"]) >= 2

    storage, meta = read_flashpack_file(comp, device="cpu")
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, meta)]
    assert torch.equal(_uint16_view(w), _uint16_view(source["w"]))


def test_env_sets_v2_chunk_size(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLASHPACK_FPZ_CHUNK_BYTES", str(256 * 1024))
    source = {"w": torch.randn(1024, 512).to(torch.bfloat16)}
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    block = get_flashpack_file_metadata(comp)["macroblocks"][0]
    assert block["fpz"]["hi_chunk_usize"] == 256 * 1024


def test_v2_reader_defaults_chunk_size_when_field_absent(tmp_path) -> None:
    # Back-compat: a v2 pack written before hi_chunk_usize existed (field
    # absent) must decode with the 64 KiB default. The default pack uses 64 KiB,
    # so dropping the field and re-reading must still be bit-exact.
    source = {"w": torch.randn(1024, 512).to(torch.bfloat16)}
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")
    meta = get_flashpack_file_metadata(comp)
    del meta["macroblocks"][0]["fpz"]["hi_chunk_usize"]

    storage, m = read_flashpack_file(comp, device="cpu", metadata=meta)
    (w,) = [t for _, t in iterate_from_flash_tensor(storage, m)]
    assert torch.equal(_uint16_view(w), _uint16_view(source["w"]))


@pytest.mark.parametrize("bad", [4097, 1000, 64 * 1024 + 1])
def test_hi_chunk_bytes_must_be_multiple_of_align(tmp_path, bad) -> None:
    with pytest.raises(ValueError, match="multiple"):
        pack_to_file(
            {"w": torch.randn(64).to(torch.bfloat16)},
            str(tmp_path / "p.flashpack"),
            target_dtype=None,
            compress="fpz-bf16",
            hi_chunk_bytes=bad,
        )


def test_hi_chunk_bytes_too_large_rejected(tmp_path) -> None:
    from flashpack.constants import FPZ_FRAME_UNCOMPRESSED_BYTES

    with pytest.raises(ValueError, match="exceeds"):
        pack_to_file(
            {"w": torch.randn(64).to(torch.bfloat16)},
            str(tmp_path / "p.flashpack"),
            target_dtype=None,
            compress="fpz-bf16",
            hi_chunk_bytes=FPZ_FRAME_UNCOMPRESSED_BYTES,
        )
