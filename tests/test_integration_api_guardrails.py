"""Dependency-free guards for distributed integration API plumbing.

Diffusers and Transformers are optional dependencies, so the hermetic unit-test
tier cannot import their FlashPack integrations.  These AST checks make sure the
public sharded-load options still reach the core mixin without adding either
large framework to FlashPack's required dependencies.
"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(path: str, name: str, *, class_name: str | None = None) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text())
    body: list[ast.stmt] = tree.body
    if class_name is not None:
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = cls.body
    return next(
        node
        for node in body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _keyword_names(function: ast.FunctionDef, called_name: str) -> set[str]:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == called_name:
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"call to {called_name} not found in {function.name}")


@pytest.mark.parametrize(
    ("path", "class_name", "called_name"),
    [
        (
            "src/flashpack/integrations/transformers/model.py",
            "FlashPackTransformersModelMixin",
            "from_flashpack",
        ),
        (
            "src/flashpack/integrations/diffusers/model.py",
            "FlashPackDiffusersModelMixin",
            "from_flashpack",
        ),
        (
            "src/flashpack/integrations/diffusers/pipeline.py",
            "FlashPackDiffusionPipeline",
            "load_sub_model_flashpack",
        ),
    ],
)
def test_model_integrations_forward_sharded_options(
    path: str, class_name: str, called_name: str
) -> None:
    function = _function(path, "from_pretrained_flashpack", class_name=class_name)
    argument_names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    expected = {"distributed_sharded", "distributed_shard_strategy"}
    assert expected <= argument_names
    assert expected <= _keyword_names(function, called_name)


def test_pipeline_component_loader_forwards_sharded_options() -> None:
    path = "src/flashpack/integrations/diffusers/pipeline.py"
    function = _function(path, "load_sub_model_flashpack")
    argument_names = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    expected = {"distributed_sharded", "distributed_shard_strategy"}
    assert expected <= argument_names
    assert expected <= _keyword_names(function, "from_pretrained_flashpack")
