"""Shared pytest configuration.

Markers split the suite into tiers:

* unmarked -- hermetic CPU-only unit tests; these run on every push/PR via
  ``.github/workflows/test.yaml``.
* ``gpu`` -- requires a CUDA device; skipped automatically when CUDA is
  unavailable.
* ``network`` -- downloads model weights from the Hugging Face Hub; excluded
  from CI, run manually on a suitable host.
"""

import pytest
import torch


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("gpu") is not None and not torch.cuda.is_available():
        pytest.skip("test requires a CUDA device")
