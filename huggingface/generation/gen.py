import argparse
import gc

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
)

import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils


def get_hf_model(
  model_name,
  cache_dir,
  attn_impl
):
  model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, attn_implementation=attn_impl)
  model.to(torch.cuda.current_device())
  model = model.eval()
  return model

def run_generation(
  model_name,
  measure,
  measure_dest,
  warmup_trials,
  trials,
  batch_size,
  gen_len,
  cache_dir,
  prompt,
  attn_impl
):
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
  tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

  with torch.no_grad():
    model = get_hf_model(
      model_name,
      cache_dir,
      attn_impl
    )

  prompt = open(f"{os.path.expanduser('~')}/{prompt}", "r").read()
  prompts = [prompt] * batch_size
  input_tokens = tokenizer(prompts, return_tensors="pt")
  input_tokens.to(torch.cuda.current_device())

  if measure:
    measure_file = open(f"{os.path.expanduser('~')}/{measure_dest}", "w")
    for _ in range(warmup_trials):
      with torch.no_grad():
        output_ids = model.generate(**input_tokens, max_new_tokens=gen_len, do_sample=False)
    utils.add_model_hooks(model, measure_file)

  for _ in range(trials):
    with torch.no_grad():
      output_ids = model.generate(**input_tokens, max_new_tokens=gen_len, do_sample=False)

  if measure:
    utils.remove_model_hooks(model)

  for output in output_ids:
    print(tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))

if __name__ == "__main__":
  def str2bool(value):
    return True if (value.lower() == 'true') else False

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", "-m", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="model name or path")
  parser.add_argument("--measure", type=str2bool, default=False, help="whether or not to take measurements")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
  parser.add_argument("--cache_dir", type=str, default="/scratch/lucastr", help="cache dir for model name")
  parser.add_argument("--prompt", type=str, default="fineinfer-autopeft/prompts/default.txt", help="path to prompt relative to ~")
  parser.add_argument("--attn_impl", type=str, default="flash_attention_2", help="the attention implementation to use in the model")
  args = parser.parse_args()

  gc.collect()

  run_generation(
    args.model_name,
    args.measure,
    args.measure_dest,
    args.warmup_trials,
    args.trials,
    args.batch_size,
    args.gen_len,
    args.cache_dir,
    args.prompt,
    args.attn_impl
  )
