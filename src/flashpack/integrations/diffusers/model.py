from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import create_repo, snapshot_download

import diffusers
from diffusers import ModelMixin
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card

from ... import __version__
from ...constants import (
    DEFAULT_ALIGN_BYTES,
    DEFAULT_CHUNK_BYTES,
    DEFAULT_NUM_STREAMS,
    DEFAULT_NUM_WRITE_WORKERS,
)
from ...mixin import FlashPackMixin


class FlashPackDiffusersModelMixin(ModelMixin, FlashPackMixin):
    flashpack_init_method: str | None = "from_config"

    def save_pretrained_flashpack(
        self,
        save_directory: str | os.PathLike,
        *,
        is_main_process: bool = True,
        push_to_hub: bool = False,
        target_dtype: torch.dtype | None = None,
        align_bytes: int = DEFAULT_ALIGN_BYTES,
        silent: bool = True,
        num_workers: int = DEFAULT_NUM_WRITE_WORKERS,
        commit_message: str | None = None,
        create_pr: bool = False,
        token: str | None = None,
        repo_id: str | None = None,
        private: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if not os.path.isdir(save_directory):
            raise ValueError(f"Save directory {save_directory} is not a directory")

        model_path = os.path.join(save_directory, "model.flashpack")
        if is_main_process:
            self.save_config(save_directory)
            if os.path.exists(model_path):
                os.remove(model_path)

        self.save_flashpack(
            model_path,
            target_dtype=target_dtype,
            align_bytes=align_bytes,
            silent=silent,
            num_workers=num_workers,
        )

        if push_to_hub:
            repo_id = repo_id or save_directory.split(os.path.sep)[-1]
            repo_id = create_repo(
                repo_id, exist_ok=True, private=private, token=token
            ).repo_id
            model_card = load_or_create_model_card(repo_id, token=token)
            model_card = populate_model_card(model_card)
            model_card.save(Path(save_directory, "README.md").as_posix())

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

    @classmethod
    def from_pretrained_flashpack(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        device: str | torch.device | None = None,
        *,
        silent: bool = True,
        strict: bool | None = None,
        strict_params: bool = True,
        strict_buffers: bool = False,
        keep_flash_ref_on_model: bool = True,
        num_streams: int = DEFAULT_NUM_STREAMS,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        ignore_names: list[str] | None = None,
        ignore_prefixes: list[str] | None = None,
        ignore_suffixes: list[str] | None = None,
        subfolder: str | None = None,
        revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        local_dir_use_symlinks: bool = False,
        local_files_only: bool = False,
        cache_dir: str | os.PathLike | None = None,
        token: str | None = None,
        force_download: bool = False,
        proxies: dict[str, str] | None = None,
        use_distributed_loading: bool = False,
        rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        coerce_dtype: bool = False,
        **kwargs: Any,
    ) -> FlashPackDiffusersModelMixin:
        """
        Load a model from a flash pack file.
        """
        device = (
            torch.device(device)
            if isinstance(device, str)
            else torch.device("cpu") if device is None else device
        )

        user_agent = {
            "diffusers": diffusers.__version__,
            "flashpack": __version__,
            "framework": "pytorch",
            "file_type": "model",
        }

        if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
                proxies=proxies,
                user_agent=user_agent,
            )

        if subfolder:
            flashpack_path = os.path.join(
                pretrained_model_name_or_path, subfolder, "model.flashpack"
            )
        else:
            flashpack_path = os.path.join(
                pretrained_model_name_or_path, "model.flashpack"
            )

        if not os.path.exists(flashpack_path):
            raise FileNotFoundError(f"Flashpack file {flashpack_path} not found")

        config, unused_kwargs, _ = cls.load_config(
            pretrained_model_name_or_path,
            revision=revision,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            proxies=proxies,
            user_agent=user_agent,
        )

        if unused_kwargs:
            warnings.warn(f"Unused kwargs in config: {unused_kwargs}")

        return cls.from_flashpack(
            flashpack_path,
            config,
            device=device,
            silent=silent,
            strict=strict,
            strict_params=strict_params,
            strict_buffers=strict_buffers,
            keep_flash_ref_on_model=keep_flash_ref_on_model,
            num_streams=num_streams,
            chunk_bytes=chunk_bytes,
            ignore_names=ignore_names,
            ignore_prefixes=ignore_prefixes,
            ignore_suffixes=ignore_suffixes,
            use_distributed_loading=use_distributed_loading,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            coerce_dtype=coerce_dtype,
            **kwargs,
        )
