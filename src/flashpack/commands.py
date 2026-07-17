from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def filter_kwargs_for_method(
    kwargs: dict[str, Any],
    method: Callable[[Any], Any],
) -> dict[str, Any]:
    """
    Filters the kwargs for a method.
    """
    params = inspect.signature(method).parameters

    has_var_kwargs = False
    for param in params.values():
        if param.kind == param.VAR_KEYWORD:
            has_var_kwargs = True
            break

    if has_var_kwargs:
        return kwargs

    return {k: v for k, v in kwargs.items() if k in params}


def convert_to_flashpack_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    destination_path: str,
    dtype: torch.dtype | str | None = None,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    silent: bool = True,
) -> str:
    """
    Converts a state dictionary to a flashpack file.
    """
    from .serialization import pack_to_file
    from .utils import filter_state_dict, string_to_dtype

    state_dict = filter_state_dict(
        state_dict,
        ignore_names,
        ignore_prefixes,
        ignore_suffixes,
    )

    if isinstance(dtype, str):
        dtype = string_to_dtype(dtype)

    pack_to_file(
        state_dict,
        destination_path,
        dtype,
        silent=silent,
    )

    return destination_path


def convert_to_flashpack_from_model(
    model: torch.nn.Module,
    destination_path: str,
    dtype: torch.dtype | str | None = None,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    silent: bool = True,
) -> str:
    """
    Converts a model to a flashpack file.
    """
    return convert_to_flashpack_from_state_dict(
        model.state_dict(),
        destination_path,
        dtype,
        ignore_names,
        ignore_prefixes,
        ignore_suffixes,
        silent,
    )


def convert_to_flashpack_from_state_dict_file(
    path: str,
    destination_path: str | None = None,
    dtype: torch.dtype | str | None = None,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    silent: bool = True,
) -> str:
    """
    Converts a state dictionary file to a flashpack file.
    """
    import torch

    _, ext = os.path.splitext(path)
    if ext == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, weights_only=True)

    return convert_to_flashpack_from_state_dict(
        state_dict,
        destination_path,
        dtype,
        ignore_names,
        ignore_prefixes,
        ignore_suffixes,
        silent,
    )


def convert_to_flashpack_from_diffusers_repo_id_or_dir(
    model_dir: str,
    destination_path: str,
    dtype: torch.dtype | str | None = None,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    silent: bool = True,
    **kwargs: Any,
) -> str:
    """
    Converts a diffusers model to a flashpack model.
    """
    from diffusers import AutoModel

    from .utils import string_to_dtype

    if isinstance(dtype, str):
        dtype = string_to_dtype(dtype)

    diffusers_model = AutoModel.from_pretrained(
        model_dir, **filter_kwargs_for_method(kwargs, AutoModel.from_pretrained)
    )
    diffusers_model.save_pretrained_flashpack(
        destination_path,
        target_dtype=dtype,
        ignore_names=ignore_names,
        ignore_prefixes=ignore_prefixes,
        ignore_suffixes=ignore_suffixes,
        silent=silent,
    )
    return destination_path


def convert_to_flashpack_from_transformers_repo_id_or_dir(
    model_dir: str,
    destination_path: str,
    dtype: torch.dtype | str | None = None,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    silent: bool = True,
    **kwargs: Any,
) -> str:
    """
    Converts a transformers model to a flashpack model.
    """
    from transformers import AutoModel

    from .utils import string_to_dtype

    if isinstance(dtype, str):
        dtype = string_to_dtype(dtype)

    transformers_model = AutoModel.from_pretrained(
        model_dir, **filter_kwargs_for_method(kwargs, AutoModel.from_pretrained)
    )
    transformers_model.save_pretrained_flashpack(
        destination_path,
        target_dtype=dtype,
        ignore_names=ignore_names,
        ignore_prefixes=ignore_prefixes,
        ignore_suffixes=ignore_suffixes,
        silent=silent,
    )
    return destination_path


def convert_to_flashpack(
    model_or_state_dict_or_path_or_repo_id: (
        str | torch.nn.Module | dict[str, torch.Tensor]
    ),
    destination_path: str | None = None,
    dtype: torch.dtype | str | None = None,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    ignore_suffixes: list[str] | None = None,
    use_transformers: bool = False,
    use_diffusers: bool = False,
    silent: bool = True,
    **kwargs: Any,
) -> str:
    """
    Converts a state dictionary, diffusers model, or transformers model to a flashpack file.
    """
    if os.path.isfile(model_or_state_dict_or_path_or_repo_id):
        if not destination_path:
            destination_path = model_or_state_dict_or_path_or_repo_id.replace(
                ".safetensors", ".flashpack"
            )

        return convert_to_flashpack_from_state_dict_file(
            model_or_state_dict_or_path_or_repo_id,
            destination_path,
            dtype,
            ignore_names,
            ignore_prefixes,
            ignore_suffixes,
            silent,
        )

    model_dir = model_or_state_dict_or_path_or_repo_id

    if os.path.isdir(model_or_state_dict_or_path_or_repo_id):
        if destination_path is None:
            destination_path = os.path.join(model_dir, "model.flashpack")
    else:
        assert (
            destination_path is not None
        ), "destination_path is required when model_or_state_dict_or_path_or_repo_id is repo_id"
        os.makedirs(destination_path, exist_ok=True)

        if not use_transformers and not use_diffusers:
            from huggingface_hub import hf_hub_download

            try:
                config_path = hf_hub_download(
                    repo_id=model_or_state_dict_or_path_or_repo_id,
                    filename="config.json",
                    **filter_kwargs_for_method(kwargs, hf_hub_download),
                )
            except Exception as e:
                raise ValueError(
                    "config.json not found in model directory. Explicitly specify use_transformers or use_diffusers."
                ) from e

            config = json.load(open(config_path))
            is_transformers = "transformers_version" in config
            is_diffusers = "_diffusers_version" in config
            if is_transformers and is_diffusers:
                raise ValueError(
                    "config.json contains both transformers_version and diffusers_version. Explicitly specify use_transformers or use_diffusers."
                )
            elif is_transformers:
                use_transformers = True
            elif is_diffusers:
                use_diffusers = True
            else:
                raise ValueError(
                    "config.json does not contain transformers_version or diffusers_version. Explicitly specify use_transformers or use_diffusers."
                )

    if use_transformers:
        return convert_to_flashpack_from_transformers_repo_id_or_dir(
            model_dir,
            destination_path,
            dtype,
            ignore_names,
            ignore_prefixes,
            ignore_suffixes,
            silent,
            **kwargs,
        )
    return convert_to_flashpack_from_diffusers_repo_id_or_dir(
        model_dir,
        destination_path,
        dtype,
        ignore_names,
        ignore_prefixes,
        ignore_suffixes,
        silent,
        **kwargs,
    )


def revert_from_flashpack(
    path: str,
    destination_path: str | None = None,
    silent: bool = True,
) -> str:
    """
    Reverts a flashpack file to a state dictionary.
    """
    from .deserialization import revert_from_file

    state_dict = revert_from_file(path, silent=silent)
    if not destination_path:
        destination_path = path.replace(".flashpack", ".safetensors")

    _, ext = os.path.splitext(destination_path)
    if ext == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, destination_path)
    else:
        import torch

        torch.save(state_dict, destination_path)

    return destination_path
