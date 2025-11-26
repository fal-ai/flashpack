import copy
import json
import os
import warnings


def patch_transformers_auto_model() -> None:
    """
    Patch the transformers to add the FlashPackTransformersModelMixin to the model classes.
    """
    patch_auto_factory()


def patch_auto_factory() -> None:
    """
    Patch the auto factory to add the FlashPackTransformersModelMixin to the model classes.
    """
    from transformers.utils import (
        CONFIG_NAME,
        cached_file,
        extract_commit_hash,
        find_adapter_config_file,
        is_peft_available,
    )

    try:
        from transformers.configuration_utils import PreTrainedConfig
    except ImportError:
        from transformers.configuration_utils import (
            PretrainedConfig as PreTrainedConfig,
        )

    import transformers.models.auto.auto_factory
    from transformers.dynamic_module_utils import (
        get_class_from_dynamic_module,
        resolve_trust_remote_code,
    )
    from transformers.models.auto.configuration_auto import AutoConfig

    from flashpack.integrations.transformers.model import (
        FlashPackTransformersModelMixin,
    )

    def patched_from_config(cls, config, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        has_remote_code = (
            hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        )
        has_local_code = type(config) in cls._model_mapping
        if has_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code,
                config._name_or_path,
                has_local_code,
                has_remote_code,
                upstream_repo=upstream_repo,
            )

        if has_remote_code and trust_remote_code:
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                repo_id = config.name_or_path
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            # This block handles the case where the user is loading a model with `trust_remote_code=True`
            # but a library model exists with the same name. We don't want to override the autoclass
            # mappings in this case, or all future loads of that model will be the remote code model.
            if not has_local_code:
                cls.register(config.__class__, model_class, exist_ok=True)
                model_class.register_for_auto_class(auto_class=cls)
            _ = kwargs.pop("code_revision", None)
            model_class = transformers.models.auto.auto_factory.add_generation_mixin_to_remote_model(
                model_class
            )
            model_class = type(
                f"FlashPack{model_class.__name__}",
                (model_class, FlashPackTransformersModelMixin),
                {},
            )
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping:
            model_class = transformers.models.auto.auto_factory._get_model_class(
                config, cls._model_mapping
            )
            model_class = type(
                f"FlashPack{model_class.__name__}",
                (model_class, FlashPackTransformersModelMixin),
                {},
            )
            return model_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping)}."
        )

    def patched_from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.get("trust_remote_code")
        kwargs["_from_auto"] = True
        use_flashpack = kwargs.pop("use_flashpack", False)
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "use_auth_token",
            "token",
        ]
        hub_kwargs = {
            name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs
        }
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)
        use_auth_token = hub_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            hub_kwargs["token"] = token

        if commit_hash is None:
            if not isinstance(config, PreTrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    **hub_kwargs,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            if adapter_kwargs is None:
                adapter_kwargs = {}
                if token is not None:
                    adapter_kwargs["token"] = token

            maybe_adapter_path = find_adapter_config_file(
                pretrained_model_name_or_path,
                _commit_hash=commit_hash,
                **adapter_kwargs,
            )

            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, encoding="utf-8") as f:
                    adapter_config = json.load(f)

                    adapter_kwargs["_adapter_model_path"] = (
                        pretrained_model_name_or_path
                    )
                    pretrained_model_name_or_path = adapter_config[
                        "base_model_name_or_path"
                    ]

        if not isinstance(config, PreTrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype") == "auto":
                _ = kwargs.pop("torch_dtype")
            if kwargs.get("dtype") == "auto":
                _ = kwargs.pop("dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config") is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                code_revision=code_revision,
                _commit_hash=commit_hash,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"
            if kwargs_orig.get("dtype", None) == "auto":
                kwargs["dtype"] = "auto"
            if kwargs_orig.get("quantization_config", None) is not None:
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]

        has_remote_code = (
            hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        )
        has_local_code = type(config) in cls._model_mapping
        upstream_repo = None
        if has_remote_code:
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code,
            pretrained_model_name_or_path,
            has_local_code,
            has_remote_code,
            upstream_repo=upstream_repo,
        )
        kwargs["trust_remote_code"] = trust_remote_code

        # Set the adapter kwargs
        kwargs["adapter_kwargs"] = adapter_kwargs

        if has_remote_code and trust_remote_code:
            model_class = get_class_from_dynamic_module(
                class_ref,
                pretrained_model_name_or_path,
                code_revision=code_revision,
                **hub_kwargs,
                **kwargs,
            )
            _ = hub_kwargs.pop("code_revision", None)
            # This block handles the case where the user is loading a model with `trust_remote_code=True`
            # but a library model exists with the same name. We don't want to override the autoclass
            # mappings in this case, or all future loads of that model will be the remote code model.
            if not has_local_code:
                cls.register(config.__class__, model_class, exist_ok=True)
                model_class.register_for_auto_class(auto_class=cls)
            model_class = transformers.models.auto.auto_factory.add_generation_mixin_to_remote_model(
                model_class
            )
            model_class = type(
                f"FlashPack{model_class.__name__}",
                (model_class, FlashPackTransformersModelMixin),
                {},
            )
            if use_flashpack:
                return model_class.from_pretrained_flashpack(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **hub_kwargs,
                    **kwargs,
                )
            else:
                return model_class.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **hub_kwargs,
                    **kwargs,
                )
        elif type(config) in cls._model_mapping:
            model_class = transformers.models.auto.auto_factory._get_model_class(
                config, cls._model_mapping
            )
            model_class = type(
                f"FlashPack{model_class.__name__}",
                (model_class, FlashPackTransformersModelMixin),
                {},
            )
            if model_class.config_class == config.sub_configs.get("text_config", None):
                config = config.get_text_config()
            if use_flashpack:
                return model_class.from_pretrained_flashpack(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **hub_kwargs,
                    **kwargs,
                )
            else:
                return model_class.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=config,
                    **hub_kwargs,
                    **kwargs,
                )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping)}."
        )

    @classmethod
    def from_pretrained_flashpack(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args,
        **kwargs,
    ):
        kwargs["use_flashpack"] = True
        return cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    transformers.models.auto.auto_factory._BaseAutoModelClass.from_config = (
        patched_from_config
    )
    transformers.models.auto.auto_factory._BaseAutoModelClass.from_pretrained = (
        patched_from_pretrained
    )
    transformers.models.auto.auto_factory._BaseAutoModelClass.from_pretrained_flashpack = from_pretrained_flashpack
