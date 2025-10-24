<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal/flashpack/blob/main/media/flashpack-logo-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal/flashpack/blob/main/media/flashpack-logo-black.png?raw=true">
  <img alt="FlashPack Logo" src="https://github.com/painebenjamin/flatpack/blob/main/media/flashpack-logo-white.png?raw=true">
</picture>
<h2>Disk-to-GPU Tensor loading at up to 25Gbps without GDS</h2>
</div>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal/flashpack/blob/main/media/benchmark-white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal/flashpack/blob/main/media/benchmark-black.png?raw=true">
  <img alt="Benchmark Results" src="https://github.com/painebenjamin/flatpack/blob/main/media/benchmark-white.png?raw=true">
</picture>
</div>

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







