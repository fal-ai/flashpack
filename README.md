<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/flashpack-logo-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/flashpack-logo-black.png?raw=true">
  <img alt="FlashPack Logo" src="https://github.com/fal-ai/flashpack/blob/main/media/flashpack-logo-black.png?raw=true">
</picture>
<h2>Disk-to-GPU tensor loading at up to 18&nbsp;GB/s &mdash; no GDS required</h2>
</div>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/benchmark-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/benchmark-black.png?raw=true">
  <img alt="Benchmark Results" src="https://github.com/fal-ai/flashpack/blob/main/media/benchmark-black.png?raw=true">
</picture>
<em>Reproduce with <code>scripts/run_benchmark.py</code></em>
</div>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/load-state-dict-comparison-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/load-state-dict-comparison-black.png?raw=true">
  <img alt="load_state_dict Comparison" src="https://github.com/fal-ai/flashpack/blob/main/media/load-state-dict-comparison-black.png?raw=true">
</picture>
<em>Reproduce with <code>tests/test_speed_comparison.py</code></em>
</div>

## Updates

- **2026-08-11** (`v0.4.1`): **Sharded distributed read** — on multi-GPU loads every
  rank reads its own 1/N of the pack and the shards are replicated over NVLink/fabric,
  removing the world-size read amplification. See [Multi-GPU loading](#multi-gpu-loading).
- **2026-07-18** (`v0.4.0`): Opt-in **parallel CPU reads** (`FLASHPACK_CPU_PARALLEL_READ=1`),
  fixed the O_DIRECT warm-cache gate, CI load benchmark.
- **2026-07-17** (`v0.3.0`/`v0.3.1`): **Parallel reader** — 16 `preadv` threads into pinned
  staging with overlapped async H2D copies, O_DIRECT when the page cache is cold.
  True-cold loads from network filesystems got 14&ndash;20&times; faster.
- **2025-11-25**: Multiple data types per checkpoint with no regressions in speed.

## Installation

```bash
pip install flashpack
```

Requires Python &ge; 3.10 and PyTorch &ge; 2.0.

## Production numbers

FlashPack is the weight loader for fal's serverless model fleet. Measured there —
H100/H200/B200 nodes, pinned production packs on a network-backed (JuiceFS) weight
store, every load bit-verified against its source:

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/coldstart-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/coldstart-black.png?raw=true">
  <img alt="True-cold load comparison" src="https://github.com/fal-ai/flashpack/blob/main/media/coldstart-black.png?raw=true">
</picture>
<em>Data in <code>scripts/plot_fleet_coldstart.py</code></em>
</div>

| Weight-store state | FlashPack (v0.3+ parallel reader) | vs. single-threaded mmap loading |
|---|---|---|
| Page cache hot | 10&ndash;18 GB/s (micro-benchmark above) | ~3.5&times; |
| Node NVMe cache warm, page cache cold | 5&ndash;12 GB/s sustained (each pack within ~10% of its own page-cache-hot rate) | 3&ndash;8&times; |
| Fully cold (bytes still in the object store) | 15.35 GB in 12.5 s / 25.41 GB in 18 s | 14&ndash;20&times; |

The last row is the serverless cold-start case: a fault-driven mmap walk over the same
network filesystem ran at 0.06&ndash;0.10 GB/s (255 s and 245 s for those packs).
More context in the [fal FlashPack docs](https://fal.ai/docs/documentation/serverless/optimizations/flashpack).

## How it works

A `.flashpack` file lays every tensor out in large aligned contiguous macroblocks with
a small footer index — the whole file is read with a handful of long sequential I/Os
instead of one page fault per tensor. The reader runs 16 `preadv` threads, each with
its own pinned staging buffer and CUDA stream, so disk reads and host-to-device copies
overlap; tensors are reconstructed zero-copy on the GPU by aliasing the transferred
blocks. Reads use `O_DIRECT` when the page cache is actually cold (checked with
`mincore`) and buffered reads when it is warm, so hot loads run at memory speed while
cold loads skip the kernel page-cache copy entirely.

That is the same recipe GPUDirect Storage exists for — minus the parts that make GDS
hard to deploy: no kernel module, no filesystem allowlist, no PCIe topology tuning.
It works on any POSIX filesystem, including FUSE/network mounts, and degrades
gracefully (buffered parallel reads) where `O_DIRECT` is unavailable. The wins are
largest exactly where standard loaders are weakest: cold page caches, network-backed
storage, and many-shard checkpoints.

## Multi-GPU loading

Loading the same pack on N ranks normally costs N full reads (the reader bypasses the
page cache, and even with it, network filesystems often re-read per process).
`read_flashpack_file_distributed` removes that amplification:

```py
import flashpack

# rank 0 reads, everyone receives via broadcast
storage, metadata = flashpack.read_flashpack_file_distributed(
    "/path/to/model.flashpack", device="cuda",
)

# or: every rank reads 1/N of the payload, shards replicate over the fabric —
# the disk wall drops toward read_time / world_size
storage, metadata = flashpack.read_flashpack_file_distributed(
    "/path/to/model.flashpack", device="cuda", sharded=True,
)
```

The same paths are available from the model integrations, for ordinary BF16,
FP32, and quantized packs alike:

```py
model = MyFlashPackModel.from_pretrained_flashpack(
    "/path/to/repository",
    device="cuda",
    use_distributed_loading=True,
    distributed_sharded=True,  # omit for rank-0 read + broadcast
)
```

Two shard replication strategies ship (`contiguous` slabs with owner broadcasts, and
interleaved `windows` moved with one AllGather each); both are checksummed and
benchmarked in `scripts/bench_distributed_shard_strategy.py` — measure on your own
fabric before overriding the default. On an 8&times;H200 node a 14.5 GB fp8 pack loads
in ~0.65 s sharded, with the fabric moving 12.9 GB of replication in ~32&ndash;37 ms
depending on strategy.

## Tuning

The main knobs are environment variables; the defaults are the measured sweet spots
on H100/H200-class nodes.

| Variable | Default | Effect |
|---|---|---|
| `FLASHPACK_PARALLEL_READ` | `1` | Kill switch: `0` falls back to the legacy single-threaded loader and disables sharded distributed reads (resolved from rank 0's environment) |
| `FLASHPACK_READ_THREADS` | `16` | Reader threads (each with its own pinned buffer + CUDA stream) |
| `FLASHPACK_READ_CHUNK_BYTES` | `64 MiB` | Read/copy granularity |
| `FLASHPACK_DIRECT_IO` | auto | `0` forces buffered reads (escape hatch for filesystems where `O_DIRECT` misbehaves); auto uses `O_DIRECT` only when the page cache is cold |
| `FLASHPACK_CACHE_PINNED` | `1` | Keep pinned staging buffers cached between loads; `0` frees them |
| `FLASHPACK_CPU_PARALLEL_READ` | `0` | `1` extends the parallel reader to CPU-target loads (default CPU path is lazy mmap views) |
| `FLASHPACK_SHARD_BYTES` | `256 MiB` | Sharded distributed read, `windows` strategy only: bytes each rank reads per replication window |
| `FLASHPACK_SHARD_STRATEGY` | `contiguous` | `contiguous` or `windows` (see `src/flashpack/constants.py` for the trade-off) |

## Integration Guide
### Mixins
#### Diffusers/Transformers

```py
# Integration classes
from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin, FlashPackDiffusionPipeline
from flashpack.integrations.transformers import FlashPackTransformersModelMixin

# Base classes
from diffusers.models import MyModel, SomeOtherModel
from diffusers.pipelines import MyPipeline

# Define mixed classes
class FlashPackMyModel(MyModel, FlashPackDiffusersModelMixin):
    pass

class FlashPackMyPipeline(MyPipeline, FlashPackDiffusionPipeline):
    def __init__(
        self,
        my_model: FlashPackMyModel,
        other_model: SomeOtherModel,
    ) -> None:
        super().__init__()

# Load base pipeline
pipeline = FlashPackMyPipeline.from_pretrained("some/repository")

# Save flashpack pipeline
pipeline.save_pretrained_flashpack(
    "some_directory",
    push_to_hub=False,  # pass repo_id when using this
)

# Load directly from flashpack directory or repository
pipeline = FlashPackMyPipeline.from_pretrained_flashpack("my/flashpack-repository")
```

#### Vanilla PyTorch

```py
from flashpack import FlashPackMixin

class MyModule(nn.Module, FlashPackMixin):
    def __init__(self, some_arg: int = 4) -> None:
        ...

module = MyModule(some_arg = 4)
module.save_flashpack("model.flashpack")

loaded_module = module.from_flashpack("model.flashpack", some_arg=4)
```

### Direct Integration

```py
from flashpack import pack_to_file, assign_from_file

flashpack_path = "/path/to/model.flashpack"
model = nn.Module(...)

pack_to_file(model, flashpack_path, None)  # write state dict to file (None keeps source dtypes)
assign_from_file(model, flashpack_path)  # load state dict from file
```

## CLI Commands

FlashPack provides a command-line interface for converting, inspecting, and reverting flashpack files.

### `flashpack convert`

Convert a model to a flashpack file.

```bash
flashpack convert <path_or_repo_id> [destination_path] [options]
```

**Arguments:**
- `path_or_repo_id` - Local path or Hugging Face repository ID
- `destination_path` - (Optional) Output path for the flashpack file

**Options:**
| Option | Description |
|--------|-------------|
| `--subfolder` | Subfolder of the model (for repo_id) |
| `--variant` | Model variant (for repo_id) |
| `--dtype` | Target dtype for the flashpack file. When omitted, no type changes are made |
| `--ignore-names` | Tensor names to ignore (can be specified multiple times) |
| `--ignore-prefixes` | Tensor prefixes to ignore (can be specified multiple times) |
| `--ignore-suffixes` | Tensor suffixes to ignore (can be specified multiple times) |
| `--use-transformers` | Load the path as a transformers model |
| `--use-diffusers` | Load the path as a diffusers model |
| `-v, --verbose` | Enable verbose output |

**Examples:**
```bash
# Convert a local model
flashpack convert ./my_model ./my_model.flashpack

# Convert from Hugging Face
flashpack convert stabilityai/stable-diffusion-xl-base-1.0 --subfolder unet --use-diffusers

# Convert with specific dtype
flashpack convert ./my_model ./my_model.flashpack --dtype float16
```

### `flashpack revert`

Revert a flashpack file back to safetensors or torch format.

```bash
flashpack revert <path> [destination_path] [options]
```

**Arguments:**
- `path` - Path to the flashpack file
- `destination_path` - (Optional) Output path for the reverted file

**Options:**
| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |

**Example:**
```bash
flashpack revert ./my_model.flashpack ./my_model.safetensors
```

### `flashpack metadata`

Print the metadata of a flashpack file.

```bash
flashpack metadata <path> [options]
```

**Arguments:**
- `path` - Path to the flashpack file

**Options:**
| Option | Description |
|--------|-------------|
| `-i, --show-index` | Show the tensor index |
| `-j, --json` | Output metadata in JSON format |

**Examples:**
```bash
# View basic metadata
flashpack metadata ./my_model.flashpack

# View metadata with tensor index
flashpack metadata ./my_model.flashpack --show-index

# Output as JSON
flashpack metadata ./my_model.flashpack --json
```
