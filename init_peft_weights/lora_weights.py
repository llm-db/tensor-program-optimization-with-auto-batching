import torch
from safetensors.torch import save_file
import os

NUM_LAYERS = 32
LORA_RANKS = [32, 64]
NUM_ADAPTERS = 64 

scaling_factor = 0.01

os.makedirs(f"/scratch/lucastr/lora_adapters", exist_ok=True)
for lora_rank in LORA_RANKS:
  for adapter_id in range(NUM_ADAPTERS):
    model_weights = {}
    for layer in range(NUM_LAYERS):
      lora_a_q_proj = scaling_factor * torch.randn((lora_rank, 4096), dtype=torch.float32)
      lora_b_q_proj = scaling_factor * torch.randn((4096, lora_rank), dtype=torch.float32)
      model_weights[f'layers.{layer}.self_attn.q_proj.lora_A.weight'] = lora_a_q_proj
      model_weights[f'layers.{layer}.self_attn.q_proj.lora_B.weight'] = lora_b_q_proj
      lora_a_v_proj = scaling_factor * torch.randn((lora_rank, 4096), dtype=torch.float32)
      lora_b_v_proj = scaling_factor * torch.randn((1024, lora_rank), dtype=torch.float32)
      model_weights[f'layers.{layer}.self_attn.v_proj.lora_A.weight'] = lora_a_v_proj
      model_weights[f'layers.{layer}.self_attn.v_proj.lora_B.weight'] = lora_b_v_proj
    output_file = f"/scratch/lucastr/lora_adapters/r{lora_rank}_a{adapter_id}.safetensors"
    save_file(model_weights, output_file)
