from typing import Any


def patch_diffusers_auto_model() -> None:
    """
    Patch the diffusers to add the FlashPackDiffusersModelMixin to the model classes.
    """
    patch_diffusers_pipeline_loading_utils()

    from flashpack.integrations.diffusers.auto_model import AutoFlashPackModel

    import diffusers.models.auto_model

    diffusers.models.auto_model.AutoModel = AutoFlashPackModel


def patch_diffusers_pipeline_loading_utils() -> None:
    """
    Patch the pipeline loading utils to add the FlashPackDiffusersModelMixin to the model classes.
    """
    import diffusers.pipelines.pipeline_loading_utils

    _original_get_class_from_dynamic_module = (
        diffusers.pipelines.pipeline_loading_utils.get_class_from_dynamic_module
    )

    def patched_get_class_from_dynamic_module(
        *args: Any,
        **kwargs: Any,
    ) -> type:
        from flashpack.integrations.diffusers.model import FlashPackDiffusersModelMixin

        cls = _original_get_class_from_dynamic_module(*args, **kwargs)
        if cls is not None and not issubclass(cls, FlashPackDiffusersModelMixin):
            cls = type(
                f"FlashPack{cls.__name__}", (cls, FlashPackDiffusersModelMixin), {}
            )
        return cls

    diffusers.pipelines.pipeline_loading_utils.get_class_from_dynamic_module = (
        patched_get_class_from_dynamic_module
    )
    _original_get_class_obj_and_candidates = (
        diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates
    )

    def patched_get_class_obj_and_candidates(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[type, list[type]]:
        from flashpack.integrations.diffusers.model import FlashPackDiffusersModelMixin

        cls, candidates = _original_get_class_obj_and_candidates(*args, **kwargs)
        if cls is not None and not issubclass(cls, FlashPackDiffusersModelMixin):
            cls = type(
                f"FlashPack{cls.__name__}", (cls, FlashPackDiffusersModelMixin), {}
            )
        return cls, candidates

    diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates = (
        patched_get_class_obj_and_candidates
    )
