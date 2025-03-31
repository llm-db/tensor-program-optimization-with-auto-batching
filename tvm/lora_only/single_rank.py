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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from custom.fuse_take_matmul import FuseTakeMatmul
from custom.matmul import Matmul
from custom.gemv import GEMV
from custom.reduction import Reduction

dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)


class LoraSequential(nn.Module):
  def __init__(self, batch_size, in_features, lora_r, out_features, num_adapters, split_indices, with_basemodel, batch_size_fix):
    super().__init__()
    self.batch_size = batch_size
    self.in_features = in_features
    self.num_adapters = num_adapters
    self.split_indices = split_indices
    self.with_basemodel = with_basemodel
    self.batch_size_fix = batch_size_fix
    if with_basemodel:
      self.M = nn.Parameter((in_features, out_features), "float16")
    for i in range(self.num_adapters):
      setattr(self, f"lora_A_{i}", nn.Parameter((in_features, lora_r), "float16"))
      setattr(self, f"lora_B_{i}", nn.Parameter((lora_r, out_features), "float16"))
  
  def forward(self, x: Tensor):
    if len(self.split_indices) == 0:
      y = op.matmul(op.matmul(x, self.lora_A_0), self.lora_B_0) * 2.0
    else:
      x_splitted = op.split(x, self.split_indices)
      y_lst = []
      for i, x_i in enumerate(x_splitted):
        y_lst.append(op.matmul(op.matmul(x_i, getattr(self, f"lora_A_{i}")), getattr(self, f"lora_B_{i}")) * 2.0)
      y = op.concat(y_lst, dim=0)
    if self.with_basemodel:
      y += op.matmul(x, self.M)
    return y

  def get_default_spec(self):
    batch_size = 1 if (self.batch_size == 1) else (self.batch_size if self.batch_size_fix else "batch_size")
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([batch_size, 1, self.in_features], "float16"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class LoraGatherBMM(nn.Module):
  def __init__(self, batch_size, in_features, lora_r, out_features, num_adapters, with_basemodel, batch_size_fix):
    super().__init__()
    self.batch_size = batch_size
    self.in_features = in_features
    self.with_basemodel = with_basemodel
    self.batch_size_fix = batch_size_fix
    if with_basemodel:
      self.M = nn.Parameter((in_features, out_features), "float16")
    self.lora_A = nn.Parameter(
      (num_adapters, in_features, lora_r), 
      "float16"
    )
    self.lora_B = nn.Parameter(
      (num_adapters, lora_r, out_features), 
      "float16"
    )
  
  def forward(self, x: Tensor, wids: Tensor):
    y = op.matmul(op.matmul(x, op.take(self.lora_A, wids, axis=0)), 
                  op.take(self.lora_B, wids, axis=0)) * 2.0
    if self.with_basemodel:
      y += op.matmul(x, self.M)
    return y

  def get_default_spec(self):
    batch_size = 1 if (self.batch_size == 1) else (self.batch_size if self.batch_size_fix else "batch_size")
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([batch_size, 1, self.in_features], "float16"),
        "wids": nn.spec.Tensor([batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


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

def get_params(in_features, lora_r, out_features, num_adapters, with_basemodel):
  gatherbmm_params, seq_params = [], []
  if with_basemodel:
    M = tvm.nd.array((0.01 * np.random.randn(in_features, out_features)).astype("float16"), device=dev)
    gatherbmm_params.append(M)
    seq_params.append(M)
  A = (0.01 * np.random.randn(num_adapters, in_features, lora_r)).astype("float16")
  B = (0.01 * np.random.randn(num_adapters, lora_r, out_features)).astype("float16")
  gatherbmm_params.append(tvm.nd.array(A, device=dev))
  gatherbmm_params.append(tvm.nd.array(B, device=dev))
  for i in range(num_adapters):
    seq_params.append(tvm.nd.array(A[i], device=dev))
    seq_params.append(tvm.nd.array(B[i], device=dev))
  return gatherbmm_params, seq_params

def main(
  with_basemodel,
  batch_size_fix,
  opt_config,
  request_type,
  batch_size,
  in_features,
  out_features,
  lora_r,
  measure,
  measure_dest,
  warmup_trials,
  trials
):
  if request_type == "uniform":
    adapter_indices = np.sort(np.random.randint(0, int(np.ceil(np.sqrt(batch_size))), batch_size))
  elif request_type == "distinct":
    adapter_indices = np.arange(batch_size)
  elif request_type == "identical":
    adapter_indices = np.zeros(batch_size, dtype=int)
  else:
    print(f"unknown request_type {request_type}")
  num_adapters = len(set(adapter_indices))
  gatherbmm_params, sequential_params = get_params(in_features, lora_r, out_features, num_adapters, with_basemodel)
  register_opt_pipeline(opt_config)
  x = tvm.nd.array(np.random.uniform(-1e-2, 1e-2, size=(batch_size, 1, in_features)).astype("float16"), device=dev)

  if batch_size_fix:
    split_indices = [i for i in range(1, len(adapter_indices)) if adapter_indices[i] != adapter_indices[i - 1]]
    sequential_model = LoraSequential(batch_size, in_features, lora_r, out_features, num_adapters, split_indices, with_basemodel, batch_size_fix)
    sequential_model.to("float16")
    sequential_mod, _ = sequential_model.export_tvm(spec=sequential_model.get_default_spec())
    with target:
      sequential_vm = relax.VirtualMachine(relax.build(sequential_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  gatherbmm_model = LoraGatherBMM(batch_size, in_features, lora_r, out_features, num_adapters, with_basemodel, batch_size_fix)
  gatherbmm_model.to("float16")
  gatherbmm_mod, _ = gatherbmm_model.export_tvm(spec=gatherbmm_model.get_default_spec())
  with target:
    gatherbmm_vm = relax.VirtualMachine(relax.build(gatherbmm_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)
  gatherbmm_ids = tvm.nd.array(adapter_indices.astype("int32"), device=dev)

  if measure:
    measure_file = open(f"{os.path.expanduser('~')}/{measure_dest}", "w")

  # sequential
  if batch_size_fix:
    if measure:
      for _ in range(warmup_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        sequential_result = sequential_vm["forward"](x, sequential_params)
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
    for _ in range(trials):
      if measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
      sequential_result = sequential_vm["forward"](x, sequential_params)
      if measure:
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        measure_file.write(f"{time_ms}\n")

  # gatherbmm
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      gatherbmm_result = gatherbmm_vm["forward"](x, gatherbmm_ids, gatherbmm_params)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    gatherbmm_result = gatherbmm_vm["forward"](x, gatherbmm_ids, gatherbmm_params)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")
  
  if measure:
    measure_file.close()
  else:
    print(f"CORRECT: {np.allclose(sequential_result.numpy(), gatherbmm_result.numpy(), atol=1e-4)}")

if __name__ == "__main__":
  def str2bool(value):
    return True if (value.lower() == 'true') else False

  parser = argparse.ArgumentParser()
  parser.add_argument("--basemodel", type=str2bool, default=True)
  parser.add_argument("--batch_size_fix", type=str2bool, default=True)
  parser.add_argument("--opt_config", type=str, default="fineinfer-autopeft/tvm/opt_configs/default.json", help="path to optimization config relative to ~")
  parser.add_argument("--request_type", type=str, default="uniform")
  parser.add_argument("--batch_size", type=int, default=1,  help="batch size")
  parser.add_argument("--in_features", type=int, default=4096,  help="dimension of input features")
  parser.add_argument("--out_features", type=int, default=4096,  help="dimension of input features")
  parser.add_argument("--lora_r", type=int, default=64,  help="lora r")
  parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--measure", type=str2bool, default=False, help="whether or not to take measurements")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  args = parser.parse_args()

  main(
    args.basemodel,
    args.batch_size_fix,
    args.opt_config,
    args.request_type,
    args.batch_size,
    args.out_features,
    args.out_features,
    args.lora_r,
    args.measure,
    args.measure_dest,
    args.warmup_trials,
    args.trials
  )
