<div align="center">
<img src="https://github.com/painebenjamin/flatpack/blob/main/media/flashpack.png?raw=true" width="480" />
<h2>Distributed-Friendly Disk-to-GPU Tensor loading at up to 25Gbps</h2>
</div>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/painebenjamin/flatpack/blob/main/media/benchmark-dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/painebenjamin/flatpack/blob/main/media/benchmark-light.png?raw=true">
  <img alt="Benchmark Results" src="https://github.com/painebenjamin/flatpack/blob/main/media/benchmark-dark.png?raw=true">
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





