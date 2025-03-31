import dataclasses
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

@dataclasses.dataclass
class LoraConfig:
  r: int
  alpha: int

class Lora(nn.Module):
  def __init__(self, in_features: int, out_features: int, config: LoraConfig):
    super().__init__()
    self.scaling = config.alpha / config.r

  def define_parameters(self, in_features: int, out_features: int, config: LoraConfig) -> tuple[Tensor, Tensor]:
    lora_A = nn.Parameter((in_features, config.r), "float16")
    lora_B = nn.Parameter((config.r, out_features), "float16")
    return lora_A, lora_B

  def weight_application(self, x: Tensor, lora_A: Tensor, lora_B: Tensor) -> Tensor:
    return op.matmul(op.matmul(x, lora_A), lora_B)

  def post_computation(self, y: Tensor) -> Tensor:
    return y * self.scaling
