import argparse
import numpy as np
import os
import json
import torch
import sys

import tvm
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from custom.fuse_take_matmul import FuseTakeMatmul
from custom.matmul import Matmul
from custom.gemv import GEMV
from custom.reduction import Reduction


dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)


class SequentialLora(nn.Module):
  def __init__(self, batch_sizes, rs, num_adapters, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.batch_size = sum(batch_sizes)
    self.split_indices = np.cumsum(batch_sizes)[:-1].tolist()
    self.num_ranks = len(rs)
    for i, r in enumerate(rs):
      setattr(self, f"lora_A_{i}", nn.Parameter((num_adapters[i], in_features, r), "float16"))
      setattr(self, f"lora_B_{i}", nn.Parameter((num_adapters[i], r, out_features), "float16"))
  
  def flexible_forward(self, x: Tensor, *args):
    x_splitted = op.split(x, self.split_indices)
    y_lst = []
    for i, x_i in enumerate(x_splitted):
      y_lst.append(op.matmul(op.matmul(x_i, op.take(getattr(self, f"lora_A_{i}"), args[i], axis=0)), 
                            op.take(getattr(self, f"lora_B_{i}"), args[i], axis=0)) * 2.0)
    y = op.concat(y_lst, dim=0)
    return y

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, self.in_features], "float16"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    for i in range(self.num_ranks):
      if i == 0:
        length = self.split_indices[0]
      elif i == (self.num_ranks - 1):
        length = self.batch_size - self.split_indices[-1]
      else:
        length = self.split_indices[i] - self.split_indices[i-1]
      mod_spec["forward"][f"wids_{i}"] = nn.spec.Tensor([length], "int32")
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class PaddedLora(nn.Module):
  def __init__(self, batch_sizes, rs, num_adapters, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.batch_size = sum(batch_sizes)
    self.lora_A = nn.Parameter(
      (sum(num_adapters), in_features, max(rs)), 
      "float16"
    )
    self.lora_B = nn.Parameter(
      (sum(num_adapters), max(rs), out_features), 
      "float16"
    )
  
  def forward(self, x: Tensor, wids: Tensor):
    return op.matmul(op.matmul(x, op.take(self.lora_A, wids, axis=0)), 
                     op.take(self.lora_B, wids, axis=0)) * 2.0

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, self.in_features], "float16"),
        "wids": nn.spec.Tensor([self.batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class SplittedLora(nn.Module):
  def __init__(self, batch_sizes, rs, num_adapters, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.batch_size = sum(batch_sizes)

    split_lengths, self.sum_split = [], []
    for i, r in enumerate(rs):
      r_mult = int(r / min(rs))
      if r_mult == 1:
        self.sum_split.append(False)
        split_lengths.append(batch_sizes[i])
      else:
        for _ in range(batch_sizes[i]):
          self.sum_split.append(True)
          split_lengths.append(r_mult)
    self.split_indices = np.cumsum(split_lengths)[:-1].tolist()
    self.lora_batch_size = sum(split_lengths)
    num_splitted_adapters = sum([num_adapters[i] * int(rs[i] / min(rs)) for i in range(len(rs))])
    self.lora_A = nn.Parameter(
      (num_splitted_adapters, in_features, min(rs)), 
      "float16"
    )
    self.lora_B = nn.Parameter(
      (num_splitted_adapters, min(rs), out_features), 
      "float16"
    )
  
  def forward(self, x: Tensor, xids: Tensor, wids: Tensor):
    y = op.matmul(op.matmul(op.take(x, xids, axis=0), op.take(self.lora_A, wids, axis=0)), 
                  op.take(self.lora_B, wids, axis=0)) * 2.0
    y_splitted = op.split(y, self.split_indices)
    z_lst = []
    for i, y_i in enumerate(y_splitted):
      if self.sum_split[i]:
        z_lst.append(op.sum(y_i, axis=0, keepdims=True))
      else:
        z_lst.append(y_i)
    z = op.concat(z_lst, dim=0)
    return z

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, self.in_features], "float16"),
        "xids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "wids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


def create_sequential_forward(num_ranks):
  params_to_pass = ["x"] + [f"wids_{i}" for i in range(num_ranks)]
  params_for_arg = ["self"] + params_to_pass
  params_to_pass_str = ", ".join(params_to_pass)
  params_for_arg_str = ", ".join(params_for_arg)
  forward_code = f"\ndef forward({params_for_arg_str}):\n  return self.flexible_forward({params_to_pass_str})\n"
  local_dict = {}
  exec(forward_code, local_dict)
  setattr(SequentialLora, "forward", local_dict["forward"])

def get_params(rs, num_adapters, in_features, out_features):
  num_ranks = len(rs)
  max_r = max(rs)
  min_r = min(rs)

  # params
  A, B = [], []
  for i in range(num_ranks):
    A.append((0.01 * np.random.randn(num_adapters[i], in_features, rs[i])))
    B.append((0.01 * np.random.randn(num_adapters[i], rs[i], out_features)))

  # sequential params
  sequential_params = []
  for i in range(num_ranks):
    sequential_params.append(tvm.nd.array(A[i].astype("float16"), device=dev))
    sequential_params.append(tvm.nd.array(B[i].astype("float16"), device=dev))

  # padded params
  padded_A, padded_B = [], []
  for i in range(num_ranks):
    if rs[i] == max_r:
      padded_A.append(A[i])
      padded_B.append(B[i])
    else:
      A_pad = np.zeros((num_adapters[i], in_features, max_r - rs[i]))
      B_pad = np.zeros((num_adapters[i], max_r - rs[i], out_features))
      padded_A.append(np.concatenate((A[i], A_pad), axis=2))
      padded_B.append(np.concatenate((B[i], B_pad), axis=1))
  padded_params = [tvm.nd.array(np.concatenate(padded_A, axis=0).astype("float16"), device=dev), 
                   tvm.nd.array(np.concatenate(padded_B, axis=0).astype("float16"), device=dev)]

  # splitted params
  splitted_A, splitted_B = [], []
  for i in range(num_ranks):
    if rs[i] == min_r:
      splitted_A.append(A[i])
      splitted_B.append(B[i])
    else:
      A_splitted = np.concatenate([item for elem in np.split(A[i], len(A[i]), axis=0) for item in np.split(elem, int(rs[i] / min_r), axis=2)], axis=0)
      B_splitted = np.concatenate([item for elem in np.split(B[i], len(B[i])) for item in np.split(elem, int(rs[i] / min_r), axis=1)], axis=0)
      splitted_A.append(A_splitted)
      splitted_B.append(B_splitted)
  splitted_params = [tvm.nd.array(np.concatenate(splitted_A, axis=0).astype("float16"), device=dev),
                     tvm.nd.array(np.concatenate(splitted_B, axis=0).astype("float16"), device=dev)]
  
  return sequential_params, padded_params, splitted_params

def register_opt_pipeline(opt_config):
  with open(f"{os.path.expanduser('~')}/{opt_config}") as opt_config_file:
    opt_config_dict = json.load(opt_config_file)
  @register_pipeline("opt_llm")
  def _pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
      seq = tvm.transform.Sequential(
        [FuseTakeMatmul()] * opt_config_dict['FuseTakeMatmul'] +
        [relax.transform.FuseTransposeMatmul()] * opt_config_dict['FuseTransposeMatmul'] +
        [relax.transform.LegalizeOps()] +
        [relax.transform.AnnotateTIROpPattern()] * opt_config_dict['AnnotateTIROpPattern'] +
        [relax.transform.FoldConstant()] +
        [relax.transform.FuseOps()] * opt_config_dict['FuseOps'] +
        [relax.transform.FuseTIR()] * opt_config_dict['FuseTIR'] + 
        [relax.transform.DeadCodeElimination()] +
        [dlight.ApplyDefaultSchedule(
          *([Matmul()] * opt_config_dict['ApplyDefaultSchedule']['Matmul'] +
            [GEMV()] * opt_config_dict['ApplyDefaultSchedule']['GEMV'] +
            [Reduction()] * opt_config_dict['ApplyDefaultSchedule']['Reduction']),
          dlight.gpu.Fallback())] +
        [relax.transform.RewriteDataflowReshape()] * opt_config_dict['RewriteDataflowReshape'] +
        [relax.transform.ToNonDataflow()] +
        [relax.transform.RemovePurityChecking()] +
        [relax.transform.CallTIRRewrite()] +
        [relax.transform.StaticPlanBlockMemory()] * opt_config_dict['StaticPlanBlockMemory']  +
        [relax.transform.LowerAllocTensor()] +
        [relax.transform.KillAfterLastUse()] +
        [relax.transform.LowerRuntimeBuiltin()] +
        [relax.transform.VMShapeLower()] +
        [relax.transform.AttachGlobalSymbol()]
      )
      mod = seq(mod)
      return mod
    return _pipeline

def main(
  batch_sizes,
  request_type,
  rs,
  in_features,
  out_features,
  opt_config,
  measure,
  measure_dest,
  trials,
  warmup_trials
):
  adapter_indices, num_adapters = [], []
  for batch_size in batch_sizes:
    if request_type == "uniform":
      adapter_indices.append(np.sort(np.random.randint(0, int(np.ceil(np.sqrt(batch_size))), batch_size)))
    elif request_type == "distinct":
      adapter_indices.append(np.arange(batch_size))
    elif request_type == "identical":
      adapter_indices.append(np.zeros(batch_size, dtype=int))
    else:
      print(f"unknown request_type {request_type}")
    num_adapters.append(len(set(adapter_indices[-1])))
  sequential_params, padded_params, splitted_params = get_params(rs, num_adapters, in_features, out_features)
  register_opt_pipeline(opt_config)
  x = tvm.nd.array(np.random.uniform(-1e-2, 1e-2, size=(sum(batch_sizes), 1, in_features)).astype("float16"), device=dev)

  # sequential
  create_sequential_forward(len(rs))
  sequential_model = SequentialLora(batch_sizes, rs, num_adapters, in_features, out_features)
  sequential_model.to("float16")
  sequential_mod, _ = sequential_model.export_tvm(spec=sequential_model.get_default_spec())
  with target:
    sequential_vm = relax.VirtualMachine(relax.build(sequential_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)
  sequential_wids = [tvm.nd.array(adapter_indices[i].astype("int32"), device=dev) for i in range(len(adapter_indices))]

  # padded
  padded_model = PaddedLora(batch_sizes, rs, num_adapters, in_features, out_features)
  padded_model.to("float16")
  padded_mod, _ = padded_model.export_tvm(spec=padded_model.get_default_spec())
  with target:
    padded_vm = relax.VirtualMachine(relax.build(padded_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)
  start_wids = np.insert(np.cumsum(num_adapters), 0, 0)[:-1]
  padded_wids = tvm.nd.array(np.array([start_wids[i] + id for i, aids in enumerate(adapter_indices) for id in aids]).astype("int32"), device=dev)

  # splitted
  splitted_model = SplittedLora(batch_sizes, rs, num_adapters, in_features, out_features)
  splitted_model.to("float16")
  splitted_mod, _ = splitted_model.export_tvm(spec=splitted_model.get_default_spec())
  with target:
    splitted_vm = relax.VirtualMachine(relax.build(splitted_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)
  start_xids = np.insert(np.cumsum(batch_sizes), 0, 0)[:-1]
  splitted_xids = tvm.nd.array(np.array([start_xids[i] + id for i, r in enumerate(rs) for id in range(batch_sizes[i]) for _ in range(int(r / min(rs)))]).astype("int32"), device=dev)
  adapter_indices_splitted = [np.repeat(a, int(rs[i]/min(rs))) * int(rs[i]/min(rs)) + np.tile(np.arange(int(rs[i]/min(rs))), len(a)) for i, a in enumerate(adapter_indices)]
  start_wids_splitted = np.insert(np.cumsum([num_adapters[i] * int(r / min(rs)) for i, r in enumerate(rs)]), 0, 0)[:-1]
  splitted_wids = tvm.nd.array(np.array([start_wids_splitted[i] + elem for i, aids in enumerate(adapter_indices_splitted) for elem in aids]).astype("int32"), device=dev)

  if measure:
    measure_file = open(f"{os.path.expanduser('~')}/{measure_dest}", "w")

  # sequential
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      sequential_result = sequential_vm["forward"](x, *sequential_wids, sequential_params)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    sequential_result = sequential_vm["forward"](x, *sequential_wids, sequential_params)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # padded
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      padded_result = padded_vm["forward"](x, padded_wids, padded_params)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    padded_result = padded_vm["forward"](x, padded_wids, padded_params)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # splitted
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      splitted_result = splitted_vm["forward"](x, splitted_xids, splitted_wids, splitted_params)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    splitted_result = splitted_vm["forward"](x, splitted_xids, splitted_wids, splitted_params)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")
  
  if measure:
    measure_file.close()
  else:
    print(f"CORRECT: {all([np.allclose(sequential_result.numpy(), padded_result.numpy(), atol=1e-5), 
                           np.allclose(sequential_result.numpy(), splitted_result.numpy(), atol=1e-5)])}")

if __name__ == "__main__":
  def str2bool(value):
    return (value.lower() == 'true')

  def str2lst(value):
    return [int(elem) for elem in value.split(',')]

  parser = argparse.ArgumentParser()
  parser.add_argument("--opt_config", type=str, default="fineinfer-autopeft/tvm/opt_configs/default.json", help="path to optimization config relative to ~")
  parser.add_argument("--in_features", type=int, default=4096,  help="dimension of input features")
  parser.add_argument("--out_features", type=int, default=4096,  help="dimension of input features")
  parser.add_argument("--request_type", type=str, default="distinct")
  parser.add_argument("--rs", type=str2lst, default=[64, 32])
  parser.add_argument("--batch_sizes", type=str2lst, default=[1, 1])
  parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--measure", type=str2bool, default=False, help="whether or not to take measurements")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  args = parser.parse_args()

  assert len(args.batch_sizes) == len(args.rs), "batch_sizes and rs must have same length"
  for r in args.rs:
    assert (r % min(args.rs)) == 0, "all ranks must be dividable by the minimum rank at the moment"

  main(
    args.batch_sizes,
    args.request_type,
    args.rs,
    args.in_features,
    args.out_features,
    args.opt_config,
    args.measure,
    args.measure_dest,
    args.trials,
    args.warmup_trials
  )
