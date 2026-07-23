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

import numpy as np
import pytest
import torch
from flashpack.constants import (
    FILE_FORMAT_V3,
    FILE_FORMAT_V4,
    FPZ_CODEC_SPLITPLANE_V1,
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
    assert block["fpz"]["codec"] == FPZ_CODEC_SPLITPLANE_V1
    assert len(block["fpz"]["frames"]) >= 1
    # length_elems stays logical; length_bytes is the smaller on-disk payload.
    assert block["length_bytes"] < block["length_elems"] * 2


def test_low_entropy_tensor_shrinks_file(tmp_path, capsys) -> None:
    ramp = (torch.arange(2048 * 1024, dtype=torch.float32) * 0.01).reshape(2048, 1024)
    source = {"w": ramp.to(torch.bfloat16)}
    plain = _pack(tmp_path, source, "plain.flashpack")
    comp = _pack(tmp_path, source, "comp.flashpack", compress="fpz-bf16")

    plain_size = os.path.getsize(plain)
    comp_size = os.path.getsize(comp)
    ratio = plain_size / comp_size
    with capsys.disabled():
        print(
            f"\n[fpz] low-entropy ramp: plain={plain_size} comp={comp_size} "
            f"ratio={ratio:.3f}x"
        )
    assert comp_size < plain_size
    assert ratio > 1.5


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
    # Scribble over the compressed high plane; zstd must reject it on read.
    with open(comp, "r+b") as f:
        f.seek(hi_start)
        f.write(b"\xff" * min(64, frame["hi_len"]))
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
