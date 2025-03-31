import torch
from peft.tuners.lora import Linear
from typing import Any

class GatherBMMLinear(Linear):
  def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
      if self.merged:
        self.unmerge()
      result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
      result = self.base_layer(x, *args, **kwargs)
      lora_A_gathered = torch.stack([self.lora_A[adapter_name].weight for adapter_name in adapter_names], dim=0).transpose(1, 2)
      lora_B_gathered = torch.stack([self.lora_B[adapter_name].weight * self.scaling[adapter_name] for adapter_name in adapter_names], dim=0).transpose(1, 2)
      x_dropped = torch.stack([self.lora_dropout[adapter_name](x[i].to(lora_A_gathered.dtype)) for i, adapter_name in enumerate(adapter_names)])
      lora_output_gathered = torch.bmm(torch.bmm(x_dropped, lora_A_gathered), lora_B_gathered)
      result = result + lora_output_gathered.to(result.dtype)
    elif self.merged:
      result = self.base_layer(x, *args, **kwargs)
    else:
      result = self.base_layer(x, *args, **kwargs)
      torch_result_dtype = result.dtype
      for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
          continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = x.to(lora_A.weight.dtype)

        if not self.use_dora[active_adapter]:
          result = result + lora_B(lora_A(dropout(x))) * scaling
        else:
          if isinstance(dropout, nn.Identity) or not self.training:
            base_result = result
          else:
            x = dropout(x)
            base_result = None

          result = result + self.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=self.get_base_layer(),
            base_result=base_result,
          )
      result = result.to(torch_result_dtype)

    return result
