from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from accelerate import init_empty_weights

from .constants import (
    DEFAULT_ALIGN_BYTES,
    DEFAULT_CHUNK_BYTES,
    DEFAULT_NUM_STREAMS,
    DEFAULT_NUM_WRITE_WORKERS,
)
from .deserialization import assign_from_file
from .serialization import pack_to_file


class FlashPackMixin:
    flashpack_coerce_dtype: ClassVar[bool] = False
    flashpack_init_method: ClassVar[str | None] = None
    flashpack_ignore_names: ClassVar[list[str] | None] = None
    flashpack_ignore_prefixes: ClassVar[list[str] | None] = None
    flashpack_ignore_suffixes: ClassVar[list[str] | None] = None

    @classmethod
    def from_flashpack(
        cls,
        path: str,
        *args: Any,
        device: str | torch.device | None = None,
        silent: bool = False,
        strict: bool | None = None,
        strict_params: bool = True,
        strict_buffers: bool = False,
        keep_flash_ref_on_model: bool = True,
        ignore_names: list[str] | None = None,
        ignore_prefixes: list[str] | None = None,
        ignore_suffixes: list[str] | None = None,
        num_streams: int = DEFAULT_NUM_STREAMS,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        use_distributed_loading: bool = False,
        rank: int | None = None,
        local_rank: int | None = None,
        world_size: int | None = None,
        coerce_dtype: bool = False,
        init_fn: Callable[..., "FlashPackMixin"] | None = None,
        **kwargs: Any,
    ) -> FlashPackMixin:
        """
        Load a model from a flashpack file.
        """
        device = (
            torch.device(device)
            if isinstance(device, str)
            else torch.device("cpu")
            if device is None
            else device
        )

        with init_empty_weights():
            if init_fn is None:
                if cls.flashpack_init_method is not None and hasattr(
                    cls, cls.flashpack_init_method
                ):
                    init_fn = getattr(cls, cls.flashpack_init_method)
                else:
                    init_fn = cls

            parameters = inspect.signature(init_fn).parameters
            kwargs = {k: v for k, v in kwargs.items() if k in parameters}
            if "rank" in parameters:
                kwargs["rank"] = rank
            if "local_rank" in parameters:
                kwargs["local_rank"] = local_rank
            if "world_size" in parameters:
                kwargs["world_size"] = world_size

            model = init_fn(*args, **kwargs)

        assign_from_file(
            model=model,
            path=path,
            device=device,
            silent=silent,
            strict=strict,
            strict_params=strict_params,
            strict_buffers=strict_buffers,
            keep_flash_ref_on_model=keep_flash_ref_on_model,
            num_streams=num_streams,
            chunk_bytes=chunk_bytes,
            ignore_names=ignore_names or cls.flashpack_ignore_names,
            ignore_prefixes=ignore_prefixes or cls.flashpack_ignore_prefixes,
            ignore_suffixes=ignore_suffixes or cls.flashpack_ignore_suffixes,
            use_distributed_loading=use_distributed_loading,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            coerce_dtype=coerce_dtype or cls.flashpack_coerce_dtype,
        )
        return model

    def save_flashpack(
        self,
        destination_path: str,
        target_dtype: torch.dtype | None = None,
        name_order: list[str] | None = None,
        align_bytes: int = DEFAULT_ALIGN_BYTES,
        silent: bool = False,
        num_workers: int = DEFAULT_NUM_WRITE_WORKERS,
    ) -> None:
        """
        Save the model to a flashpack file.
        """
        pack_to_file(
            self,
            destination_path=destination_path,
            target_dtype=target_dtype,
            name_order=name_order,
            align_bytes=align_bytes,
            silent=silent,
            num_workers=num_workers,
        )
