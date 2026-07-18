"""Shared pytest configuration.

Markers split the suite into tiers:

* unmarked -- hermetic CPU-only unit tests; these run on every push/PR via
  ``.github/workflows/test.yaml``.
* ``gpu`` -- requires a CUDA device; skipped automatically when CUDA is
  unavailable.
* ``network`` -- downloads model weights from the Hugging Face Hub; excluded
  from CI, run manually on a suitable host.

IMPORTANT: run each ``network`` test FILE in its own pytest process
(``scripts/run_gpu_network_tests.sh`` does this). The integration tests call
``patch_diffusers_auto_model()`` / ``patch_transformers_auto_model()``, which
irreversibly swap module globals and class attributes; composed single-process
runs of the network files produce order-dependent failures (e.g. the Wan
accelerate-load test fails with "'FlashPackWanPipeline' object has no
attribute '_execution_device'" when test_integrations ran first).
"""

import pytest
import torch


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("gpu") is not None and not torch.cuda.is_available():
        pytest.skip("test requires a CUDA device")
