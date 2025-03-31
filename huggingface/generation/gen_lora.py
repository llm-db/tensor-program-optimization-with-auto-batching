import argparse
import gc
import numpy as np

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from peft.tuners.lora import Linear
from safetensors.torch import load_file

import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils
from custom_peft.lora_gatherbmm_linear_layer import GatherBMMLinear


def get_hf_model(
  model_name,
  cache_dir,
  attn_impl,
  gather_bmm,
  unique_adapters
):
  base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, attn_implementation=attn_impl)

  lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
  )
  if gather_bmm:
    module_mapping = {torch.nn.Linear: GatherBMMLinear}
    lora_config._register_custom_module(module_mapping)

  adapters = unique_adapters.copy()
  model = get_peft_model(base_model, lora_config, adapters.pop())
  for adapter in adapters:
    model = inject_adapter_in_model(lora_config, model, adapter)

  lora_weights = {}
  for adapter in unique_adapters:
    adapter_lora_weights = load_file(f"{cache_dir}/lora_adapters/r64_{adapter.replace("adapter", "a")}.safetensors")
    for k, v in adapter_lora_weights.items():
      lora_weights[f"base_model.model.model.{k[:-7]}.{adapter}.weight"] = v
  for name, param in model.named_parameters():
    if name in lora_weights.keys():
      param.data = lora_weights[name].to(dtype=param.dtype, device=param.device)

  model.to(torch.cuda.current_device())
  model = model.eval()
  return model

def run_generation(
  model_name,
  measure_full,
  measure_adapter,
  measure_dest,
  warmup_trials,
  trials,
  request_type,
  gather_bmm,
  batch_size,
  gen_len,
  cache_dir,
  prompt,
  attn_impl
):
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
  tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

  if request_type == "uniform":
    adapter_indices = np.sort(np.random.randint(0, int(np.ceil(np.sqrt(batch_size))), batch_size))
  elif request_type == "distinct":
    adapter_indices = np.arange(batch_size)
  elif request_type == "identical":
    adapter_indices = np.zeros(batch_size, dtype=int)
  else:
    print(f"unknown request_type {request_type}")
  adapters = [f"adapter{i}" for i in adapter_indices]

  with torch.no_grad():
    model = get_hf_model(
      model_name,
      cache_dir,
      attn_impl,
      gather_bmm,
      set(adapters)
    )

  prompt = open(f"{os.path.expanduser('~')}/{prompt}", "r").read()
  prompts = [prompt] * batch_size
  input_tokens = tokenizer(prompts, return_tensors="pt")
  input_tokens.to(torch.cuda.current_device())

  if measure_full or measure_adapter:
    measure_file = open(f"{os.path.expanduser('~')}/{measure_dest}", "w")
    for _ in range(warmup_trials):
      with torch.no_grad():
        output_ids = model.generate(**input_tokens, adapter_names=adapters, max_new_tokens=gen_len, do_sample=False)
  if measure_full:
    utils.add_model_hooks(model.base_model.model, measure_file)
  if measure_adapter:
    for _, module in model.named_modules():
      if (gather_bmm and type(module) == GatherBMMLinear) or \
         (not gather_bmm and type(module) == Linear):
        utils.add_model_hooks(module, measure_file)

  for _ in range(trials):
    with torch.no_grad():
      output_ids = model.generate(**input_tokens, adapter_names=adapters, max_new_tokens=gen_len, do_sample=False)

  if measure_full:
    utils.remove_model_hooks(model.base_model.model)
  if measure_adapter:
    for _, module in model.named_modules():
      if (gather_bmm and type(module) == GatherBMMLinear) or \
         (not gather_bmm and type(module) == Linear):
        utils.remove_model_hooks(module)

  for output in output_ids:
    print(tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))

if __name__ == "__main__":
  def str2bool(value):
    return True if (value.lower() == 'true') else False

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="model name or path")
  parser.add_argument("--measure_full", type=str2bool, default=False, help="whether or not to take measurements of prefill and decode")
  parser.add_argument("--measure_adapter", type=str2bool, default=False, help="whether or not to take measurements of the adapter part only")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--request_type", type=str, default="uniform")
  parser.add_argument("--gather_bmm", type=str2bool, default=False)
  parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
  parser.add_argument("--cache_dir", type=str, default="/scratch/lucastr", help="cache dir for model name")
  parser.add_argument("--prompt", type=str, default="fineinfer-autopeft/prompts/default.txt", help="path to prompt relative to ~")
  parser.add_argument("--attn_impl", type=str, default="flash_attention_2", help="the attention implementation to use in the model")
  args = parser.parse_args()

  gc.collect()

  run_generation(
    args.model_name,
    args.measure_full,
    args.measure_adapter,
    args.measure_dest,
    args.warmup_trials,
    args.trials,
    args.request_type,
    args.gather_bmm,
    args.batch_size,
    args.gen_len,
    args.cache_dir,
    args.prompt,
    args.attn_impl
  )
