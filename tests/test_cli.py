"""Tests for the ``flashpack`` command-line interface."""

import json

import pytest
import safetensors.torch
import torch
from click.testing import CliRunner
from flashpack.__main__ import main
from flashpack.deserialization import is_flashpack_file
from flashpack.serialization import pack_to_file


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def packed_file(tmp_path) -> str:
    path = str(tmp_path / "model.flashpack")
    pack_to_file(
        {"weight": torch.randn(8, 4), "bias": torch.randn(8)},
        path,
        target_dtype=None,
    )
    return path


@pytest.fixture()
def packed_v4_file(tmp_path) -> str:
    path = str(tmp_path / "mixed.flashpack")
    pack_to_file(
        {"a": torch.randn(8), "b": torch.randn(8).to(torch.bfloat16)},
        path,
        target_dtype=None,
    )
    return path


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


class TestMetadataCommand:
    def test_plain_output(self, runner: CliRunner, packed_file: str) -> None:
        result = runner.invoke(main, ["metadata", packed_file])
        assert result.exit_code == 0
        assert "format" in result.output
        assert "index" not in result.output

    def test_json_output(self, runner: CliRunner, packed_file: str) -> None:
        result = runner.invoke(main, ["metadata", packed_file, "--json"])
        assert result.exit_code == 0
        meta = json.loads(result.output)
        assert meta["format"] == "flashpack_v3"
        assert "index" not in meta

    def test_json_output_with_index(self, runner: CliRunner, packed_file: str) -> None:
        result = runner.invoke(
            main, ["metadata", packed_file, "--json", "--show-index"]
        )
        assert result.exit_code == 0
        meta = json.loads(result.output)
        assert {rec["name"] for rec in meta["index"]} == {"weight", "bias"}

    def test_v4_macroblocks_output(
        self, runner: CliRunner, packed_v4_file: str
    ) -> None:
        result = runner.invoke(main, ["metadata", packed_v4_file, "--show-index"])
        assert result.exit_code == 0
        assert "macroblocks" in result.output

    def test_non_flashpack_file_fails(self, runner: CliRunner, tmp_path) -> None:
        garbage = tmp_path / "garbage.bin"
        garbage.write_bytes(b"\0" * 64)
        result = runner.invoke(main, ["metadata", str(garbage)])
        assert result.exit_code != 0

    def test_missing_file_fails(self, runner: CliRunner, tmp_path) -> None:
        result = runner.invoke(main, ["metadata", str(tmp_path / "nope.flashpack")])
        assert result.exit_code != 0


class TestConvertCommand:
    def test_safetensors_to_flashpack(self, runner: CliRunner, tmp_path) -> None:
        source_path = str(tmp_path / "model.safetensors")
        safetensors.torch.save_file({"weight": torch.randn(4, 4)}, source_path)
        destination = str(tmp_path / "model.flashpack")

        result = runner.invoke(main, ["convert", source_path, destination])
        assert result.exit_code == 0
        assert "Success" in result.output
        assert is_flashpack_file(destination)

    def test_repo_id_without_destination_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["convert", "not-a-file/not-a-repo"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestRevertCommand:
    def test_revert_to_safetensors(
        self, runner: CliRunner, packed_file: str, tmp_path
    ) -> None:
        destination = str(tmp_path / "reverted.safetensors")
        result = runner.invoke(main, ["revert", packed_file, destination])
        assert result.exit_code == 0
        assert "Success" in result.output
        state_dict = safetensors.torch.load_file(destination)
        assert set(state_dict) == {"weight", "bias"}

    def test_missing_file_fails(self, runner: CliRunner, tmp_path) -> None:
        result = runner.invoke(main, ["revert", str(tmp_path / "nope.flashpack")])
        assert result.exit_code != 0
