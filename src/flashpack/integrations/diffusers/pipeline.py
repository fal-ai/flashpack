import importlib
import inspect
import os
import sys
import warnings
from typing import Any

import torch
import torch.distributed as dist
from diffusers import OnnxRuntimeModel
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, ModelMixin
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import (
    _get_detailed_type,
    _is_valid_type,
    is_accelerate_available,
    is_accelerate_version,
    is_torch_npu_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
from diffusers.utils.hub_utils import (
    _check_legacy_sharding_variant_format,
    load_or_create_model_card,
    populate_model_card,
)
from diffusers.utils.torch_utils import get_device, is_compiled_module
from huggingface_hub import DDUFEntry, create_repo, read_dduf_file, snapshot_download
from packaging import version
from typing_extensions import Self

from ...constants import (
    DEFAULT_ALIGN_BYTES,
    DEFAULT_CHUNK_BYTES,
    DEFAULT_NUM_STREAMS,
    DEFAULT_NUM_WRITE_WORKERS,
)
from ...utils import maybe_init_distributed
from .model import FlashPackDiffusersModelMixin

if is_torch_npu_available():
    import torch_npu  # noqa: F401

from diffusers.pipelines.pipeline_loading_utils import (
    ALL_IMPORTABLE_CLASSES,
    DUMMY_MODULES_FOLDER,
    LOADABLE_CLASSES,
    TRANSFORMERS_DUMMY_MODULES_FOLDER,
    _get_final_device_map,
    _get_load_method,
    _get_pipeline_class,
    _identify_model_variants,
    _maybe_raise_error_for_incorrect_transformers,
    _maybe_raise_warning_for_inpainting,
    _maybe_warn_for_wrong_component_in_quant_config,
    _resolve_custom_pipeline_and_cls,
    _unwrap_model,
    _update_init_kwargs_with_connected_pipeline,
    dispatch_model,
    get_class_obj_and_candidates,
    maybe_raise_or_warn,
    remove_hook_from_module,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

if is_accelerate_available():
    pass

if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel

LIBRARIES = []
for library in LOADABLE_CLASSES:
    LIBRARIES.append(library)

SUPPORTED_DEVICE_MAP = ["balanced"] + [get_device()]

try:
    from ..transformers import FlashPackTransformersModelMixin
except ImportError:
    FlashPackTransformersModelMixin = type(None)


class FlashPackDiffusionPipeline(DiffusionPipeline):
    """
    A flashpack-compatible pipeline mixin for diffusers.
    """

    def save_pretrained_flashpack(
        self,
        save_directory: str | os.PathLike,
        safe_serialization: bool = True,
        variant: str | None = None,
        max_shard_size: int | str | None = None,
        push_to_hub: bool = False,
        commit_message: str | None = None,
        create_pr: bool = False,
        token: str | None = None,
        repo_id: str | None = None,
        private: bool | None = None,
        is_main_process: bool = True,
        align_bytes: int = DEFAULT_ALIGN_BYTES,
        silent: bool = True,
        num_workers: int = DEFAULT_NUM_WRITE_WORKERS,
        target_dtype: torch.dtype | dict[str, torch.dtype] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Save the pipeline into a directory containing the pipeline components in their own subdirectories.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            repo_id = repo_id or save_directory.split(os.path.sep)[-1]
            repo_id = create_repo(
                repo_id, exist_ok=True, private=private, token=token
            ).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {
            k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)
        }
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            sub_model_dir = os.path.join(save_directory, pipeline_component_name)
            model_cls = sub_model.__class__

            sub_model_target_dtype = (
                target_dtype.get(pipeline_component_name, None)
                if isinstance(target_dtype, dict)
                else target_dtype
            )

            if isinstance(
                sub_model,
                (FlashPackDiffusersModelMixin, FlashPackTransformersModelMixin),
            ):
                os.makedirs(sub_model_dir, exist_ok=True)
                sub_model.save_pretrained_flashpack(
                    sub_model_dir,
                    is_main_process=is_main_process,
                    target_dtype=sub_model_target_dtype,
                    align_bytes=align_bytes,
                    silent=silent,
                    num_workers=num_workers,
                )
                continue

            # Dynamo wraps the original model in a private class.
            # I didn't find a public API to get the original class.
            if is_compiled_module(sub_model):
                sub_model = _unwrap_model(sub_model)
                model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(
                        model_cls, class_candidate
                    ):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                warnings.warn(
                    f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved."
                )
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = (
                "safe_serialization" in save_method_signature.parameters
            )
            save_method_accept_variant = "variant" in save_method_signature.parameters
            save_method_accept_max_shard_size = (
                "max_shard_size" in save_method_signature.parameters
            )

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant
            if save_method_accept_max_shard_size and max_shard_size is not None:
                # max_shard_size is expected to not be None in ModelMixin
                save_kwargs["max_shard_size"] = max_shard_size

            save_method(
                os.path.join(save_directory, pipeline_component_name), **save_kwargs
            )

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            # Create a new empty model card and eventually tag it
            model_card = load_or_create_model_card(
                repo_id, token=token, is_pipeline=True
            )
            model_card = populate_model_card(model_card)
            model_card.save(os.path.join(save_directory, "README.md"))

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
        pretrained_model_name_or_path: str | os.PathLike | None,
        device: str | torch.device | None = None,
        *,
        num_streams: int = DEFAULT_NUM_STREAMS,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        ignore_names: list[str] | None = None,
        ignore_prefixes: list[str] | None = None,
        silent: bool = True,
        use_distributed_loading: bool = False,
        coerce_dtype: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Load a pipeline from a directory containing the pipeline components in their own subdirectories.
        """
        # Copy the kwargs to re-use during loading connected pipeline.
        kwargs_copied = kwargs.copy()

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        provider_options = kwargs.pop("provider_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        dduf_file = kwargs.pop("dduf_file", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        quantization_config = kwargs.pop("quantization_config", None)

        rank = kwargs.get("rank", None)
        local_rank = kwargs.get("local_rank", None)
        world_size = kwargs.get("world_size", None)

        if (
            torch_dtype is not None
            and not isinstance(torch_dtype, dict)
            and not isinstance(torch_dtype, torch.dtype)
        ):
            torch_dtype = torch.float32
            warnings.warn(
                f"Passed `torch_dtype` {torch_dtype} is not a `torch.dtype`. Defaulting to `torch.float32`."
            )

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            warnings.warn(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if quantization_config is not None and not isinstance(
            quantization_config, PipelineQuantizationConfig
        ):
            raise ValueError(
                "`quantization_config` must be an instance of `PipelineQuantizationConfig`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`."
            )

        if device_map is not None and not isinstance(device_map, str):
            raise ValueError("`device_map` must be a string.")

        if device_map is not None and device_map not in SUPPORTED_DEVICE_MAP:
            raise NotImplementedError(
                f"{device_map} not supported. Supported strategies are: {', '.join(SUPPORTED_DEVICE_MAP)}"
            )

        if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
            if is_accelerate_version("<", "0.28.0"):
                raise NotImplementedError(
                    "Device placement requires `accelerate` version `0.28.0` or later."
                )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        if dduf_file:
            if custom_pipeline:
                raise NotImplementedError(
                    "Custom pipelines are not supported with DDUF at the moment."
                )
            if load_connected_pipeline:
                raise NotImplementedError(
                    "Connected pipelines are not supported with DDUF at the moment."
                )

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        # The variant filenames can have the legacy sharding checkpoint format that we check and throw
        # a warning if detected.
        if variant is not None and _check_legacy_sharding_variant_format(
            folder=cached_folder, variant=variant
        ):
            warn_msg = (
                f"Warning: The repository contains sharded checkpoints for variant '{variant}' maybe in a deprecated format. "
                "Please check your files carefully:\n\n"
                "- Correct format example: diffusion_pytorch_model.fp16-00003-of-00003.safetensors\n"
                "- Deprecated format example: diffusion_pytorch_model-00001-of-00002.fp16.safetensors\n\n"
                "If you find any files in the deprecated format:\n"
                "1. Remove all existing checkpoint files for this variant.\n"
                "2. Re-obtain the correct files by running `save_pretrained()`.\n\n"
                "This will ensure you're using the most up-to-date and compatible checkpoint format."
            )
            warnings.warn(warn_msg)

        dduf_entries = None
        if dduf_file:
            dduf_file_path = os.path.join(cached_folder, dduf_file)
            dduf_entries = read_dduf_file(dduf_file_path)
            # The reader contains already all the files needed, no need to check it again
            cached_folder = ""

        config_dict = cls.load_config(cached_folder, dduf_entries=dduf_entries)

        if dduf_file:
            _maybe_raise_error_for_incorrect_transformers(config_dict)

        # pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        # 2. Define which model components should load variants
        # We retrieve the information by matching whether variant model checkpoints exist in the subfolders.
        # Example: `diffusion_pytorch_model.safetensors` -> `diffusion_pytorch_model.fp16.safetensors`
        # with variant being `"fp16"`.
        model_variants = _identify_model_variants(
            folder=cached_folder, variant=variant, config=config_dict
        )
        if len(model_variants) == 0 and variant is not None:
            error_message = f"You are trying to load the model files of the `variant={variant}`, but no such modeling files are available."
            raise ValueError(error_message)

        # 3. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        custom_pipeline, custom_class_name = _resolve_custom_pipeline_and_cls(
            folder=cached_folder, config=config_dict, custom_pipeline=custom_pipeline
        )
        pipeline_class = _get_pipeline_class(
            cls,
            config=config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline,
            class_name=custom_class_name,
            cache_dir=cache_dir,
            revision=custom_revision,
        )

        if device_map is not None and pipeline_class._load_connected_pipes:
            raise NotImplementedError(
                "`device_map` is not yet supported for connected pipelines."
            )

        # DEPRECATED: To be removed in 1.0.0
        # we are deprecating the `StableDiffusionInpaintPipelineLegacy` pipeline which gets loaded
        # when a user requests for a `StableDiffusionInpaintPipeline` with `diffusers` version being <= 0.5.1.
        _maybe_raise_warning_for_inpainting(
            pipeline_class=pipeline_class,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config_dict,
        )

        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        expected_types = pipeline_class._get_signature_types()
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}
        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(
            config_dict, **kwargs
        )

        # define init kwargs and make sure that optional component modules are filtered out
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Special case: safety_checker must be loaded separately when using `from_flax`
        if (
            from_flax
            and "safety_checker" in init_dict
            and "safety_checker" not in passed_class_obj
        ):
            raise NotImplementedError(
                "The safety checker cannot be automatically loaded when loading weights `from_flax`."
                " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
                " separately if you need it."
            )

        # 5. Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            warnings.warn(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        # import it here to avoid circular import
        from diffusers import pipelines

        # 6. device map delegation
        final_device_map = None
        if device_map is not None:
            final_device_map = _get_final_device_map(
                device_map=device_map,
                pipeline_class=pipeline_class,
                passed_class_obj=passed_class_obj,
                init_dict=init_dict,
                library=library,
                max_memory=max_memory,
                torch_dtype=torch_dtype,
                cached_folder=cached_folder,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )

        if use_distributed_loading:
            maybe_init_distributed(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
            )

        # 7. Load each module in the pipeline
        current_device_map = None
        _maybe_warn_for_wrong_component_in_quant_config(init_dict, quantization_config)
        init_dict_keys = list(init_dict.keys())
        init_dict_keys.sort()
        for name in logging.tqdm(init_dict_keys, desc="Loading pipeline components..."):
            library_name, class_name = init_dict[name]

            # 7.1 device_map shenanigans
            if final_device_map is not None:
                if isinstance(final_device_map, dict) and len(final_device_map) > 0:
                    component_device = final_device_map.get(name, None)
                    if component_device is not None:
                        current_device_map = {"": component_device}
                    else:
                        current_device_map = None
                elif isinstance(final_device_map, str):
                    current_device_map = final_device_map

            # 7.2 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            class_name = class_name[4:] if class_name.startswith("Flax") else class_name

            # 7.3 Define all importable classes
            is_pipeline_module = hasattr(pipelines, library_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            loaded_sub_model = None

            # 7.4 Use passed sub model or load class_name from library_name
            if name in passed_class_obj:
                # if the model is in a pipeline module, then we load it from the pipeline
                # check that passed_class_obj has correct parent class
                maybe_raise_or_warn(
                    library_name,
                    library,
                    class_name,
                    importable_classes,
                    passed_class_obj,
                    name,
                    is_pipeline_module,
                )

                loaded_sub_model = passed_class_obj[name]
            else:
                # load sub model
                sub_model_dtype = (
                    torch_dtype.get(name, torch_dtype.get("default", torch.float32))
                    if isinstance(torch_dtype, dict)
                    else torch_dtype
                )
                try:
                    loaded_sub_model = load_sub_model_flashpack(
                        library_name=library_name,
                        class_name=class_name,
                        importable_classes=importable_classes,
                        pipelines=pipelines,
                        is_pipeline_module=is_pipeline_module,
                        pipeline_class=pipeline_class,
                        torch_dtype=sub_model_dtype,
                        provider=provider,
                        sess_options=sess_options,
                        device=device,
                        device_map=current_device_map,
                        max_memory=max_memory,
                        offload_folder=offload_folder,
                        offload_state_dict=offload_state_dict,
                        model_variants=model_variants,
                        name=name,
                        from_flax=from_flax,
                        variant=variant,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        cached_folder=cached_folder,
                        use_safetensors=use_safetensors,
                        dduf_entries=dduf_entries,
                        provider_options=provider_options,
                        quantization_config=quantization_config,
                        num_streams=num_streams,
                        chunk_bytes=chunk_bytes,
                        ignore_names=ignore_names,
                        ignore_prefixes=ignore_prefixes,
                        silent=silent,
                        use_distributed_loading=use_distributed_loading,
                        rank=rank,
                        local_rank=local_rank,
                        world_size=world_size,
                        coerce_dtype=coerce_dtype,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Error loading {name} from {library_name} {class_name}"
                    ) from e

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)
            if use_distributed_loading:
                dist.barrier()

        # 8. Handle connected pipelines.
        if pipeline_class._load_connected_pipes and os.path.isfile(
            os.path.join(cached_folder, "README.md")
        ):
            init_kwargs = _update_init_kwargs_with_connected_pipeline(
                init_kwargs=init_kwargs,
                passed_pipe_kwargs=passed_pipe_kwargs,
                passed_class_objs=passed_class_obj,
                folder=cached_folder,
                **kwargs_copied,
            )

        # 9. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(
            passed_modules + optional_modules
        ):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(
                list(init_kwargs.keys()) + list(passed_class_obj.keys())
            ) - set(optional_kwargs)
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 10. Type checking init arguments
        for kw, arg in init_kwargs.items():
            # Too complex to validate with type annotation alone
            if "scheduler" in kw:
                continue
            # Many tokenizer annotations don't include its "Fast" variant, so skip this
            # e.g T5Tokenizer but not T5TokenizerFast
            elif "tokenizer" in kw:
                continue
            elif isinstance(arg, list) and len(arg) == 2 and arg[0] is None:
                init_kwargs[kw] = arg[1]
            elif (
                arg is not None  # Skip if None
                and not expected_types[kw]
                == (inspect.Signature.empty,)  # Skip if no type annotations
                and not _is_valid_type(arg, expected_types[kw])  # Check type
            ):
                warnings.warn(
                    f"Expected types for {kw}: {expected_types[kw]}, got {_get_detailed_type(arg)}."
                )

        # 11. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 12. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        if device_map is not None:
            setattr(model, "hf_device_map", final_device_map)
        if quantization_config is not None:
            setattr(model, "quantization_config", quantization_config)
        return model


def load_sub_model_flashpack(
    library_name: str,
    class_name: str,
    importable_classes: list[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    torch_dtype: torch.dtype,
    provider: Any,
    sess_options: Any,
    device: str | torch.device | None,
    device_map: dict[str, torch.device] | str | None,
    max_memory: dict[int | str, int | str] | None,
    offload_folder: str | os.PathLike | None,
    offload_state_dict: bool,
    model_variants: dict[str, str],
    name: str,
    from_flax: bool,
    variant: str,
    low_cpu_mem_usage: bool,
    cached_folder: str | os.PathLike,
    use_safetensors: bool,
    dduf_entries: dict[str, DDUFEntry] | None,
    provider_options: Any,
    quantization_config: Any | None = None,
    num_streams: int = DEFAULT_NUM_STREAMS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ignore_names: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    silent: bool = False,
    use_distributed_loading: bool = False,
    rank: int | None = None,
    local_rank: int | None = None,
    world_size: int | None = None,
    coerce_dtype: bool = False,
) -> Any:
    """
    Helper method to load the module `name` from `library_name` and `class_name`.

    If the module is a FlashPackDiffusersModelMixin or FlashPackTransformersModelMixin,
    load it from a flash pack file.
    Otherwise, load it from the library using the appropriate load method.
    """
    from diffusers.quantizers import PipelineQuantizationConfig

    # retrieve class candidates
    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    # Check if flashpack
    if issubclass(class_obj, FlashPackDiffusersModelMixin) or issubclass(
        class_obj, FlashPackTransformersModelMixin
    ):
        component_dir = os.path.join(cached_folder, name)

        if device is None:
            if device_map in ["cuda", "auto", "balanced"] and torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                device = torch.device(f"cuda:{device_index}")
            else:
                device = torch.device("cpu")

        return class_obj.from_pretrained_flashpack(
            component_dir,
            device=device,
            num_streams=num_streams,
            chunk_bytes=chunk_bytes,
            ignore_names=ignore_names,
            ignore_prefixes=ignore_prefixes,
            silent=silent,
            use_distributed_loading=use_distributed_loading,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            coerce_dtype=coerce_dtype,
        ).to(device)

    load_method_name = None
    # retrieve load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(
            DUMMY_MODULES_FOLDER
        ) or none_module.startswith(TRANSFORMERS_DUMMY_MODULES_FOLDER)
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    load_method = _get_load_method(
        class_obj, load_method_name, is_dduf=dduf_entries is not None
    )

    # add kwargs to loading method
    loading_kwargs = {}
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs["torch_dtype"] = torch_dtype
    if issubclass(class_obj, OnnxRuntimeModel):
        loading_kwargs["provider"] = provider
        loading_kwargs["sess_options"] = sess_options
        loading_kwargs["provider_options"] = provider_options

    is_diffusers_model = issubclass(class_obj, ModelMixin)

    if is_transformers_available():
        transformers_version = version.parse(
            version.parse(transformers.__version__).base_version
        )
    else:
        transformers_version = "N/A"

    is_transformers_model = (
        is_transformers_available()
        and issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )

    # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
    # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
    # This makes sure that the weights won't be initialized which significantly speeds up loading.
    if is_diffusers_model or is_transformers_model:
        loading_kwargs["device_map"] = device_map
        loading_kwargs["max_memory"] = max_memory
        loading_kwargs["offload_folder"] = offload_folder
        loading_kwargs["offload_state_dict"] = offload_state_dict
        loading_kwargs["variant"] = model_variants.pop(name, None)
        loading_kwargs["use_safetensors"] = use_safetensors

        if from_flax:
            loading_kwargs["from_flax"] = True

        # the following can be deleted once the minimum required `transformers` version
        # is higher than 4.27
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            loading_kwargs.pop("variant")

        # if `from_flax` and model is transformer model, can currently not load with `low_cpu_mem_usage`
        if not (from_flax and is_transformers_model):
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            loading_kwargs["low_cpu_mem_usage"] = False

    if (
        quantization_config is not None
        and isinstance(quantization_config, PipelineQuantizationConfig)
        and issubclass(class_obj, torch.nn.Module)
    ):
        model_quant_config = quantization_config._resolve_quant_config(
            is_diffusers=is_diffusers_model, module_name=name
        )
        if model_quant_config is not None:
            loading_kwargs["quantization_config"] = model_quant_config

    # check if the module is in a subdirectory
    if dduf_entries:
        loading_kwargs["dduf_entries"] = dduf_entries
        loaded_sub_model = load_method(name, **loading_kwargs)
    elif os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(
            os.path.join(cached_folder, name), **loading_kwargs
        )
    else:
        # else load from the root directory
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    if isinstance(loaded_sub_model, torch.nn.Module) and isinstance(device_map, dict):
        # remove hooks
        remove_hook_from_module(loaded_sub_model, recurse=True)
        needs_offloading_to_cpu = device_map[""] == "cpu"

        if needs_offloading_to_cpu:
            dispatch_model(
                loaded_sub_model,
                state_dict=loaded_sub_model.state_dict(),
                device_map=device_map,
                force_hooks=True,
                main_device=0,
            )
        else:
            dispatch_model(loaded_sub_model, device_map=device_map, force_hooks=True)

    return loaded_sub_model
