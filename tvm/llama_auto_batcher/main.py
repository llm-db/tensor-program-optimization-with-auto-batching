import argparse
import os
import numpy as np
from llama_auto_batcher import LlamaAutoBatcher
from lora import LoraConfig

def main(
  opt_config,
  prompt_path,
  measure_trials,
  measure_dest,
  warmup_trials,
  batch_size,
  num_seqs_per_adapter,
  lora_ranks,
  request_type,
  gen_len,
  fuse_decode
):
  LlamaAutoBatcher.set_opt_config_path(opt_config)
  LlamaAutoBatcher.set_measurement_vars(measure_trials, measure_dest, warmup_trials)
  prompt = open(f"{os.path.expanduser('~')}/{prompt_path}", "r").read()

  adapter_info = [rank for count, rank in zip(num_seqs_per_adapter, lora_ranks) for _ in range(count)]
  if request_type == "uniform":
    adapter_indices = [elem for num_seqs in num_seqs_per_adapter for elem in np.sort(np.random.randint(0, int(np.ceil(np.sqrt(num_seqs))), num_seqs))]
  elif request_type == "distinct":
    adapter_indices = [elem for num_seqs in num_seqs_per_adapter for elem in np.arange(num_seqs)] 
  elif request_type == "identical":
    adapter_indices = list(np.zeros(sum(num_seqs_per_adapter), dtype=int))
  else:
    print(f"unknown request_type {request_type}")
  front_padding = [-1] * (batch_size - sum(num_seqs_per_adapter))
  adapter_info = front_padding + adapter_info
  adapter_indices = front_padding + adapter_indices

  for i in range(batch_size):
    input_var = LlamaAutoBatcher.create_var(prompt)
    if adapter_info[i] != -1:
      LlamaAutoBatcher.set_adapter(LoraConfig(r=adapter_info[i], alpha=adapter_info[i]/2), wid=adapter_indices[i])
    result = LlamaAutoBatcher.generate(input_var, max_gen_len=gen_len)

  LlamaAutoBatcher.compile(fuse_decode)
  LlamaAutoBatcher.execute()


if __name__ == "__main__":
  def str2bool(value):
    return (value.lower() == 'true')

  def str2lst(value):
    return [int(x) for x in value.split(',')]

  parser = argparse.ArgumentParser()
  parser.add_argument("--opt_config", type=str, default="fineinfer-autopeft/tvm/opt_configs/default.json", help="path to optimization config relative to ~")
  parser.add_argument("--prompt", type=str, default="fineinfer-autopeft/prompts/default.txt", help="path to prompt relative to ~")
  parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
  parser.add_argument("--batch_size", type=int, default=1,  help="batch size")
  parser.add_argument("--lora_ranks", type=str2lst, default=[])
  parser.add_argument("--num_seqs_per_adapter", type=str2lst, default=[])
  parser.add_argument("--request_type", type=str, default="uniform")
  parser.add_argument("--measure", type=str2bool, default=False, help="whether or not to take measurements")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--measure_trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  parser.add_argument("--fuse_decode", type=str2bool, default=True)
  args = parser.parse_args()

  main(
    args.opt_config,
    args.prompt,
    args.measure_trials,
    args.measure_dest,
    args.warmup_trials,
    args.batch_size,
    args.num_seqs_per_adapter,
    args.lora_ranks,
    args.request_type,
    args.gen_len,
    args.fuse_decode
  )
