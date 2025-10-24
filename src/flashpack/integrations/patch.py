from ..utils import diffusers_is_available, transformers_is_available


def patch_integrations() -> None:
    """
    Patch the integrations.
    """
    if diffusers_is_available():
        from .diffusers import patch_diffusers_auto_model

        patch_diffusers_auto_model()

    if transformers_is_available():
        from .transformers import patch_transformers_auto_model

        patch_transformers_auto_model()
