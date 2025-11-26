<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/flashpack-logo-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/flashpack-logo-black.png?raw=true">
  <img alt="FlashPack Logo" src="https://github.com/fal-ai/flashpack/blob/main/media/flashpack-logo-black.png?raw=true">
</picture>
<h2>Disk-to-GPU Tensor loading at up to 25Gbps without GDS</h2>
</div>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/benchmark-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/benchmark-black.png?raw=true">
  <img alt="Benchmark Results" src="https://github.com/fal-ai/flashpack/blob/main/media/benchmark-black.png?raw=true">
</picture>
<em>Run this benchmark in `scripts/run_benchmark.py`</em>
</div>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/load-state-dict-comparison-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai/flashpack/blob/main/media/load-state-dict-comparison-black.png?raw=true">
  <img alt="Benchmark Results" src="https://github.com/fal-ai/flashpack/blob/main/media/load-state-dict-comparison-black.png?raw=true">
</picture>
<em>Run this benchmark in `tests/test_speed_comparison.py`</em>
</div>

## Updates

- **2025-11-25**: Now supports **multiple data types per checkpoint** with no regressions in speed!


# Integration Guide
## Mixins
### Diffusers/Transformers

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

class FlashPackMyPipeline(MyPipeline, FlashPackDiffusionPipine):
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

### Vanilla PyTorch

```py
from flashpack import FlashPackMixin

class MyModule(nn.Module, FlashPackMixin):
    def __init__(self, some_arg: int = 4) -> None:
        ...

module = MyModule(some_arg = 4)
module.save_flashpack("model.flashpack")

loaded_module = module.from_flashpack("model.flashpack", some_arg=4)
```

## Direct Integration

```py
from flashpack import pack_to_file, assign_from_file

flashpack_path = "/path/to/model.flashpack"
model = nn.Module(...)

pack_to_file(model, flashpack_path)  # write state dict to file
assign_from_file(model, flashpack_path)  # load state dict from file
```

# CLI Commands

FlashPack provides a command-line interface for converting, inspecting, and reverting flashpack files.

## `flashpack convert`

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

## `flashpack revert`

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

## `flashpack metadata`

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
