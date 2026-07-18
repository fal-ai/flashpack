#!/usr/bin/env bash
# Run the manual `gpu`/`network` test tiers on a CUDA host, the way they are
# isolation-safe: the hermetic + gpu tests in one process, then ONE PYTEST
# PROCESS PER NETWORK TEST FILE.
#
# Why per-file: patch_diffusers_auto_model() / patch_transformers_auto_model()
# swap module globals (diffusers.models.auto_model.AutoModel, the
# pipeline_loading_utils class resolvers) and _BaseAutoModelClass loaders with
# no way to undo, and diffusers' lazy modules cache whatever they resolve
# first. Composing the network files in a single process therefore produces
# order-dependent failures — e.g. running test_integrations.py before
# test_wan_pipeline.py fails the Wan accelerate-load test with
# "'FlashPackWanPipeline' object has no attribute '_execution_device'" —
# while every file passes in its own process.
#
# Usage (from the repo root, in an env with the CUDA + network deps):
#   scripts/run_gpu_network_tests.sh [extra pytest args]

set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest tests/ -m "not network" -q "$@"

for f in tests/test_baseline.py tests/test_integrations.py \
         tests/test_speed_comparison.py tests/test_wan_pipeline.py; do
    python -m pytest "$f" -m network -q "$@"
done
