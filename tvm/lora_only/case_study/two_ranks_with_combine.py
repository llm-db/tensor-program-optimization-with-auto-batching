import argparse
import numpy as np
import os
import torch
import sys

import tvm
from tvm import dlight, relax
from tvm.script import tir as T
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


################################################################################
################################## SEQUENTIAL ##################################
################################################################################
class SequentialLora(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size_small = batch_size_small
    self.batch_size_large = batch_size_large
    self.split_indices = [batch_size_large]
    self.lora_A_large = nn.Parameter((num_adapters_large, 4096, r_large), "float16")
    self.lora_A_small = nn.Parameter((num_adapters_small, 4096, r_small), "float16")
    self.lora_B_large = nn.Parameter((num_adapters_large, r_large, 4096), "float16")
    self.lora_B_small = nn.Parameter((num_adapters_small, r_small, 4096), "float16")
  
  def forward(self, x: Tensor, wids_large: Tensor, wids_small: Tensor):
    x_splitted = op.split(x, self.split_indices)
    y_large = op.matmul(op.matmul(x_splitted[0], op.take(self.lora_A_large, wids_large, axis=0)),
                        op.take(self.lora_B_large, wids_large, axis=0)) * 2.0
    y_small = op.matmul(op.matmul(x_splitted[1], op.take(self.lora_A_small, wids_small, axis=0)),
                        op.take(self.lora_B_small, wids_small, axis=0)) * 2.0
    return op.concat([y_large, y_small], dim=0)

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size_large + self.batch_size_small, 1, 4096], "float16"),
        "wids_large": nn.spec.Tensor([self.batch_size_large], "int32"),
        "wids_small": nn.spec.Tensor([self.batch_size_small], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class SequentialLoraA(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size_small = batch_size_small
    self.batch_size_large = batch_size_large
    self.split_indices = [batch_size_large]
    self.lora_A_large = nn.Parameter((num_adapters_large, 4096, r_large), "float16")
    self.lora_A_small = nn.Parameter((num_adapters_small, 4096, r_small), "float16")
  
  def forward(self, x: Tensor, wids_large: Tensor, wids_small: Tensor):
    x_splitted = op.split(x, self.split_indices)
    y_large = op.matmul(x_splitted[0], op.take(self.lora_A_large, wids_large, axis=0))
    y_small = op.matmul(x_splitted[1], op.take(self.lora_A_small, wids_small, axis=0))
    return y_large, y_small

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size_large + self.batch_size_small, 1, 4096], "float16"),
        "wids_large": nn.spec.Tensor([self.batch_size_large], "int32"),
        "wids_small": nn.spec.Tensor([self.batch_size_small], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class SequentialLoraB(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size_small = batch_size_small
    self.batch_size_large = batch_size_large
    self.r_large = r_large
    self.r_small = r_small
    self.lora_B_large = nn.Parameter((num_adapters_large, r_large, 4096), "float16")
    self.lora_B_small = nn.Parameter((num_adapters_small, r_small, 4096), "float16")
  
  def forward(self, y_large: Tensor, y_small: Tensor, wids_large: Tensor, wids_small: Tensor):
    z_large = op.matmul(y_large, op.take(self.lora_B_large, wids_large, axis=0)) * 2.0
    z_small = op.matmul(y_small, op.take(self.lora_B_small, wids_small, axis=0)) * 2.0
    return op.concat([z_large, z_small], dim=0)

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "y_large": nn.spec.Tensor([self.batch_size_large, 1, self.r_large], "float16"),
        "y_small": nn.spec.Tensor([self.batch_size_small, 1, self.r_small], "float16"),
        "wids_large": nn.spec.Tensor([self.batch_size_large], "int32"),
        "wids_small": nn.spec.Tensor([self.batch_size_small], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


################################################################################
#################################### PADDED ####################################
################################################################################
class PaddedLora(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.lora_A = nn.Parameter(
      (num_adapters_small + num_adapters_large, 4096, r_large), 
      "float16"
    )
    self.lora_B = nn.Parameter(
      (num_adapters_small + num_adapters_large, r_large, 4096), 
      "float16"
    )
  
  def forward(self, x: Tensor, wids: Tensor):
    return op.matmul(op.matmul(x, op.take(self.lora_A, wids, axis=0)), 
                     op.take(self.lora_B, wids, axis=0)) * 2.0

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, 4096], "float16"),
        "wids": nn.spec.Tensor([self.batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class PaddedLoraA(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.lora_A = nn.Parameter(
      (num_adapters_small + num_adapters_large, 4096, r_large), 
      "float16"
    )
  
  def forward(self, x: Tensor, wids: Tensor):
    return op.matmul(x, op.take(self.lora_A, wids, axis=0))

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, 4096], "float16"),
        "wids": nn.spec.Tensor([self.batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class PaddedLoraB(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.r_large = r_large
    self.lora_B = nn.Parameter(
      (num_adapters_small + num_adapters_large, r_large, 4096), 
      "float16"
    )
  
  def forward(self, y: Tensor, wids: Tensor):
    return op.matmul(y, op.take(self.lora_B, wids, axis=0)) * 2.0

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "y": nn.spec.Tensor([self.batch_size, 1, self.r_large], "float16"),
        "wids": nn.spec.Tensor([self.batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


################################################################################
################################### SPLITTED ###################################
################################################################################
class SplittedLora(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small

    r_mult = int(r_large / r_small)
    split_lengths = [r_mult] * batch_size_large + [batch_size_small]
    self.sum_split = [True] * batch_size_large + [False]
    self.split_indices = np.cumsum(split_lengths)[:-1].tolist()
    self.lora_batch_size = sum(split_lengths)
    num_splitted_adapters = num_adapters_large * r_mult + num_adapters_small

    self.lora_A = nn.Parameter(
      (num_splitted_adapters, 4096, r_small), 
      "float16"
    )
    self.lora_B = nn.Parameter(
      (num_splitted_adapters, r_small, 4096), 
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
        "x": nn.spec.Tensor([self.batch_size, 1, 4096], "float16"),
        "xids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "wids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class SplittedLoraA(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small

    r_mult = int(r_large / r_small)
    split_lengths = [r_mult] * batch_size_large + [batch_size_small]
    self.lora_batch_size = sum(split_lengths)
    num_splitted_adapters = num_adapters_large * r_mult + num_adapters_small

    self.lora_A = nn.Parameter(
      (num_splitted_adapters, 4096, r_small), 
      "float16"
    )
  
  def forward(self, x: Tensor, xids: Tensor, wids: Tensor):
    y = op.matmul(op.take(x, xids, axis=0), op.take(self.lora_A, wids, axis=0))
    return y

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, 4096], "float16"),
        "xids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "wids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class SplittedLoraB(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.r_small = r_small

    r_mult = int(r_large / r_small)
    split_lengths = [r_mult] * batch_size_large + [batch_size_small]
    self.sum_split = [True] * batch_size_large + [False]
    self.split_indices = np.cumsum(split_lengths)[:-1].tolist()
    self.lora_batch_size = sum(split_lengths)
    num_splitted_adapters = num_adapters_large * r_mult + num_adapters_small

    self.lora_B = nn.Parameter(
      (num_splitted_adapters, r_small, 4096), 
      "float16"
    )
  
  def forward(self, y: Tensor, wids: Tensor):
    z = op.matmul(y, op.take(self.lora_B, wids, axis=0)) * 2.0
    z_splitted = op.split(z, self.split_indices)
    r_lst = []
    for i, z_i in enumerate(z_splitted):
      if self.sum_split[i]:
        r_lst.append(op.sum(z_i, axis=0, keepdims=True))
      else:
        r_lst.append(z_i)
    r = op.concat(r_lst, dim=0)
    return r

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "y": nn.spec.Tensor([self.lora_batch_size, 1, self.r_small], "float16"),
        "wids": nn.spec.Tensor([self.lora_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


################################################################################
################################### COMBINED ###################################
################################################################################
def get_take_matmul_combined(batch_size, combined_bs, large_r, num_combined_adapters):
  BATCH_SIZE = batch_size
  COMBINED_BS = combined_bs
  LARGE_R = large_r
  NUM_COMBINED_ADAPTERS = num_combined_adapters

  @T.prim_func
  def take_matmul_combined(
    lora_A: T.Buffer((T.int64(NUM_COMBINED_ADAPTERS), T.int64(4096), T.int64(LARGE_R)), "float16"), 
    xids: T.Buffer((T.int64(COMBINED_BS * LARGE_R),), "int32"), 
    wids: T.Buffer((T.int64(COMBINED_BS),), "int32"), 
    x: T.Buffer((T.int64(BATCH_SIZE), T.int64(1), T.int64(4096)), "float16"), 
    take_matmul: T.Buffer((T.int64(COMBINED_BS), T.int64(1), T.int64(LARGE_R)), "float16")
  ):
    T.func_attr({"op_pattern": 4, "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    take_matmul_rf_local = T.alloc_buffer((T.int64(16), T.int64(COMBINED_BS), T.int64(1), T.int64(LARGE_R)), "float16", scope="local")
    for ax0_ax1_fused_0 in T.thread_binding(T.int64(LARGE_R / 16 * COMBINED_BS), thread="blockIdx.x"):
      for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
        for ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
          with T.block("take_matmul_rf_init"):
            vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
            v0 = T.axis.spatial(T.int64(COMBINED_BS), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(LARGE_R))
            v1 = T.axis.spatial(T.int64(LARGE_R), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(LARGE_R))
            T.reads()
            T.writes(take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1])
            take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1] = T.float16(0.0)
          for ax2_fused_0, u in T.grid(T.int64(256), 1):
            with T.block("take_matmul_rf_update"):
              vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
              v0 = T.axis.spatial(T.int64(COMBINED_BS), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(LARGE_R))
              v1 = T.axis.spatial(T.int64(LARGE_R), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(LARGE_R))
              v2 = T.axis.spatial(T.int64(COMBINED_BS * LARGE_R), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1))
              vax2_fused_0 = T.axis.reduce(T.int64(256), ax2_fused_0)
              T.reads(
                take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1], 
                x[xids[v2], T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], 
                lora_A[wids[v0], vax2_fused_0 * T.int64(16) + vax2_fused_1, v1], 
                xids[v2], 
                wids[v0]
              )
              T.writes(take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1])
              take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1] = take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1] + \
                  x[xids[v2], T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * lora_A[wids[v0], vax2_fused_0 * T.int64(16) + vax2_fused_1, v1]
      for ax1_ax2_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
        for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
          with T.block("take_matmul"):
            vax2_fused_1 = T.axis.reduce(T.int64(16), ax0)
            v0 = T.axis.spatial(T.int64(COMBINED_BS), ax0_ax1_fused_0 // T.int64(LARGE_R / 16))
            v1 = T.axis.spatial(T.int64(LARGE_R), ax0_ax1_fused_0 % T.int64(LARGE_R / 16) * T.int64(16) + ax1_ax2_fused)
            T.reads(take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1])
            T.writes(take_matmul[v0, T.int64(0), v1])
            with T.init():
              take_matmul[v0, T.int64(0), v1] = T.float16(0.0)
            take_matmul[v0, T.int64(0), v1] = take_matmul[v0, T.int64(0), v1] + take_matmul_rf_local[vax2_fused_1, v0, T.int64(0), v1]
  return take_matmul_combined

def get_fused_take_matmul_multiply_combined(batch_size, combined_bs, large_r, num_combined_adapters):
  BATCH_SIZE = batch_size
  COMBINED_BS = combined_bs
  LARGE_R = large_r
  NUM_COMBINED_ADAPTERS = num_combined_adapters
  
  @T.prim_func
  def fused_take_matmul_multiply_combined(
    lora_B: T.Buffer((T.int64(NUM_COMBINED_ADAPTERS), T.int64(LARGE_R), T.int64(4096)), "float16"), 
    xids: T.Buffer((T.int64(COMBINED_BS * LARGE_R),), "int32"), 
    wids: T.Buffer((T.int64(COMBINED_BS),), "int32"), 
    lv: T.Buffer((T.int64(COMBINED_BS), T.int64(1), T.int64(LARGE_R)), "float16"), 
    T_multiply_intermediate: T.Buffer((T.int64(BATCH_SIZE), T.int64(1), T.int64(4096)), "float16")
  ):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    take_matmul_intermediate_local = T.alloc_buffer((T.int64(BATCH_SIZE), T.int64(1), T.int64(4096)), "float16", scope="local")
    take_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(16), T.int64(BATCH_SIZE), T.int64(1), T.int64(4096)), "float16", scope="local")
    for ax0_ax1_fused_0 in T.thread_binding(T.int64(4096 / 16 * BATCH_SIZE), thread="blockIdx.x"):
      for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
        for ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
          with T.block("take_matmul_rf_init"):
            vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
            v0 = T.axis.spatial(T.int64(BATCH_SIZE), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(4096))
            v1 = T.axis.spatial(T.int64(4096), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(4096))
            T.reads()
            T.writes(take_matmul_intermediate_rf_local[vax2_fused_1, v0, T.int64(0), v1])
            take_matmul_intermediate_rf_local[vax2_fused_1, v0, T.int64(0), v1] = T.float16(0.0)
    for ax0_ax1_fused_0 in T.thread_binding(T.int64(4096 / 16 * COMBINED_BS), thread="blockIdx.x"):
      for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
        for ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
          for ax2_fused_0, u in T.grid(T.int64(LARGE_R / 16), 1):
            with T.block("take_matmul_rf_update"):
              vax2_fused_1 = T.axis.spatial(T.int64(16), ax2_fused_1)
              v0 = T.axis.spatial(T.int64(COMBINED_BS), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(4096))
              v1 = T.axis.spatial(T.int64(4096), (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(4096))
              v2 = T.axis.spatial(T.int64(COMBINED_BS * LARGE_R), ((ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(4096)) * T.int64(LARGE_R) + ax2_fused_0 * T.int64(16) + ax2_fused_1)
              vax2_fused_0 = T.axis.reduce(T.int64(LARGE_R / 16), ax2_fused_0)
              T.reads(
                take_matmul_intermediate_rf_local[vax2_fused_1, xids[v2], T.int64(0), v1], 
                lv[wids[v0], T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], 
                lora_B[wids[v0], vax2_fused_0 * T.int64(16) + vax2_fused_1, v1], 
                wids[v0],
                xids[v2]
              )
              T.writes(take_matmul_intermediate_rf_local[vax2_fused_1, xids[v2], T.int64(0), v1])
              take_matmul_intermediate_rf_local[vax2_fused_1, xids[v2], T.int64(0), v1] = take_matmul_intermediate_rf_local[vax2_fused_1, xids[v2], T.int64(0), v1] + \
                  lv[wids[v0], T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * lora_B[wids[v0], vax2_fused_0 * T.int64(16) + vax2_fused_1, v1]
    for ax0_ax1_fused_0 in T.thread_binding(T.int64(4096 / 16 * BATCH_SIZE), thread="blockIdx.x"):
      for ax1_ax2_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
        for ax0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
          with T.block("take_matmul"):
            vax2_fused_1 = T.axis.reduce(T.int64(16), ax0)
            v0 = T.axis.spatial(T.int64(BATCH_SIZE), ax0_ax1_fused_0 // T.int64(256))
            v1 = T.axis.spatial(T.int64(4096), ax0_ax1_fused_0 % T.int64(256) * T.int64(16) + ax1_ax2_fused)
            T.reads(take_matmul_intermediate_rf_local[vax2_fused_1, v0, T.int64(0), v1])
            T.writes(take_matmul_intermediate_local[v0, T.int64(0), v1])
            with T.init():
              take_matmul_intermediate_local[v0, T.int64(0), v1] = T.float16(0.0)
            take_matmul_intermediate_local[v0, T.int64(0), v1] = take_matmul_intermediate_local[v0, T.int64(0), v1] + take_matmul_intermediate_rf_local[vax2_fused_1, v0, T.int64(0), v1]
      for ax0_ax1_fused_0_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
        for ax0_ax1_fused_1 in range(T.int64(1)):
          with T.block("T_multiply"):
            v0 = T.axis.spatial(T.int64(BATCH_SIZE), ax0_ax1_fused_0 // T.int64(256))
            v1 = T.axis.spatial(T.int64(4096), ax0_ax1_fused_0 % T.int64(256) * T.int64(16) + ax0_ax1_fused_0_1 + ax0_ax1_fused_1)
            T.reads(take_matmul_intermediate_local[v0, T.int64(0), v1])
            T.writes(T_multiply_intermediate[v0, T.int64(0), v1])
            T_multiply_intermediate[v0, T.int64(0), v1] = take_matmul_intermediate_local[v0, T.int64(0), v1] * T.float16(2.0)
  return fused_take_matmul_multiply_combined


class CombinedLora(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.r_large = r_large
    r_mult = int(r_large / r_small)
    self.combined_batch_size = batch_size_large + int(batch_size_small / r_mult)
    num_combined_adapters = num_adapters_large + int(num_adapters_small / r_mult)
    self.take_matmul_combined = get_take_matmul_combined(self.batch_size, self.combined_batch_size, self.r_large, num_combined_adapters)
    self.fused_take_matmul_multiply_combined = get_fused_take_matmul_multiply_combined(self.batch_size, self.combined_batch_size, self.r_large, num_combined_adapters)
    self.A = nn.Parameter((num_combined_adapters, 4096, self.r_large), "float16")
    self.B = nn.Parameter((num_combined_adapters, self.r_large, 4096), "float16")
  
  def forward(self, x: Tensor, xids: Tensor, wids: Tensor):
    y = nn.tensor_ir_op(
      self.take_matmul_combined,
      "take_matmul_combined",
      [self.A, xids, wids, x],
      out=nn.Tensor.placeholder([self.combined_batch_size, 1, self.r_large], "float16"),
    )
    z = nn.tensor_ir_op(
      self.fused_take_matmul_multiply_combined,
      "fused_take_matmul_multiply_combined",
      [self.B, xids, wids, y],
      out=nn.Tensor.placeholder([self.batch_size, 1, 4096], "float16"),
    )
    return z

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, 4096], "float16"),
        "xids": nn.spec.Tensor([self.combined_batch_size * self.r_large], "int32"),
        "wids": nn.spec.Tensor([self.combined_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class CombinedLoraA(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.r_large = r_large
    r_mult = int(r_large / r_small)
    self.combined_batch_size = batch_size_large + int(batch_size_small / r_mult)
    num_combined_adapters = num_adapters_large + int(num_adapters_small / r_mult)
    self.take_matmul_combined = get_take_matmul_combined(self.batch_size, self.combined_batch_size, self.r_large, num_combined_adapters)
    self.A = nn.Parameter((num_combined_adapters, 4096, self.r_large), "float16")
  
  def forward(self, x: Tensor, xids: Tensor, wids: Tensor):
    y = nn.tensor_ir_op(
      self.take_matmul_combined,
      "take_matmul_combined",
      [self.A, xids, wids, x],
      out=nn.Tensor.placeholder([self.combined_batch_size, 1, self.r_large], "float16"),
    )
    return y

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "x": nn.spec.Tensor([self.batch_size, 1, 4096], "float16"),
        "xids": nn.spec.Tensor([self.combined_batch_size * self.r_large], "int32"),
        "wids": nn.spec.Tensor([self.combined_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


class CombinedLoraB(nn.Module):
  def __init__(self, batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small):
    super().__init__()
    self.batch_size = batch_size_large + batch_size_small
    self.r_large = r_large
    r_mult = int(r_large / r_small)
    self.combined_batch_size = batch_size_large + int(batch_size_small / r_mult)
    num_combined_adapters = num_adapters_large + int(num_adapters_small / r_mult)
    self.take_matmul_combined = get_take_matmul_combined(self.batch_size, self.combined_batch_size, self.r_large, num_combined_adapters)
    self.fused_take_matmul_multiply_combined = get_fused_take_matmul_multiply_combined(self.batch_size, self.combined_batch_size, self.r_large, num_combined_adapters)
    self.B = nn.Parameter((num_combined_adapters, self.r_large, 4096), "float16")
  
  def forward(self, y: Tensor, xids: Tensor, wids: Tensor):
    z = nn.tensor_ir_op(
      self.fused_take_matmul_multiply_combined,
      "fused_take_matmul_multiply_combined",
      [self.B, xids, wids, y],
      out=nn.Tensor.placeholder([self.batch_size, 1, 4096], "float16"),
    )
    return z

  def get_default_spec(self):
    mod_spec = {
      "forward": {
        "y": nn.spec.Tensor([self.combined_batch_size, 1, self.r_large], "float16"),
        "xids": nn.spec.Tensor([self.combined_batch_size * self.r_large], "int32"),
        "wids": nn.spec.Tensor([self.combined_batch_size], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      }
    }
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)


def register_opt_pipeline():
  @register_pipeline("opt_llm")
  def _pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
      seq = tvm.transform.Sequential(
        [
          FuseTakeMatmul(),
          relax.transform.FuseTransposeMatmul(),
          relax.transform.LegalizeOps(),
          relax.transform.AnnotateTIROpPattern(),
          relax.transform.FoldConstant(),
          relax.transform.FuseOps(),
          relax.transform.FuseTIR(),
          relax.transform.DeadCodeElimination(),
          dlight.ApplyDefaultSchedule(
            Matmul(),
            GEMV(),
            Reduction(),
            dlight.gpu.Fallback()
          ),
          relax.transform.RewriteDataflowReshape(),
          relax.transform.ToNonDataflow(),
          relax.transform.RemovePurityChecking(),
          relax.transform.CallTIRRewrite(),
          relax.transform.StaticPlanBlockMemory(),
          relax.transform.LowerAllocTensor(),
          relax.transform.KillAfterLastUse(),
          relax.transform.LowerRuntimeBuiltin(),
          relax.transform.VMShapeLower(),
          relax.transform.AttachGlobalSymbol()
        ]
      )
      mod = seq(mod)
      return mod
    return _pipeline

def get_params(r_large, r_small, num_adapters_large, num_adapters_small):
  r_mult = int(r_large / r_small)

  # params
  A_large = (0.01 * np.random.randn(num_adapters_large, 4096, r_large))
  A_small = (0.01 * np.random.randn(num_adapters_small, 4096, r_small))
  B_large = (0.01 * np.random.randn(num_adapters_large, r_large, 4096))
  B_small = (0.01 * np.random.randn(num_adapters_small, r_small, 4096))

  # sequential
  A_sequential = [tvm.nd.array(A_large.astype("float16"), device=dev), 
                  tvm.nd.array(A_small.astype("float16"), device=dev)]
  B_sequential = [tvm.nd.array(B_large.astype("float16"), device=dev), 
                  tvm.nd.array(B_small.astype("float16"), device=dev)]

  # padded
  A_small_padded = np.concatenate((A_small, np.zeros((num_adapters_small, 4096, r_large - r_small))), axis=2)
  B_small_padded = np.concatenate((B_small, np.zeros((num_adapters_small, r_large - r_small, 4096))), axis=1)
  A_padded = [tvm.nd.array(np.concatenate((A_large, A_small_padded), axis=0).astype("float16"), device=dev)]
  B_padded = [tvm.nd.array(np.concatenate((B_large, B_small_padded), axis=0).astype("float16"), device=dev)]

  # splitted
  A_large_splitted = np.concatenate([item for elem in np.split(A_large, len(A_large), axis=0) for item in np.split(elem, r_mult, axis=2)], axis=0)
  B_large_splitted = np.concatenate([item for elem in np.split(B_large, len(B_large)) for item in np.split(elem, r_mult, axis=1)], axis=0)
  A_splitted = [tvm.nd.array(np.concatenate((A_large_splitted, A_small), axis=0).astype("float16"), device=dev)]
  B_splitted = [tvm.nd.array(np.concatenate((B_large_splitted, B_small), axis=0).astype("float16"), device=dev)]

  # combined
  A_small_combined = np.stack([np.concatenate([A_small[i * r_mult + j] for j in range(r_mult)], axis=1) for i in range(int(num_adapters_small / r_mult))])
  B_small_combined = np.stack([np.concatenate([B_small[i * r_mult + j] for j in range(r_mult)], axis=0) for i in range(int(num_adapters_small / r_mult))])
  A_combined = [tvm.nd.array(np.concatenate((A_large, A_small_combined), axis=0).astype("float16"), device=dev)]
  B_combined = [tvm.nd.array(np.concatenate((B_large, B_small_combined), axis=0).astype("float16"), device=dev)]

  return A_sequential, B_sequential, A_padded, B_padded, A_splitted, B_splitted, A_combined, B_combined

def main(
  request_type,
  r_large,
  r_small,
  batch_size_large,
  batch_size_small,
  trials,
  measure,
  measure_dest,
  warmup_trials
):
  if request_type == "uniform":
    adapter_indices_large = np.sort(np.random.randint(0, int(np.ceil(np.sqrt(batch_size_large))), batch_size_large))
    adapter_indices_small = np.sort(np.random.randint(0, int(np.ceil(np.sqrt(batch_size_small))), batch_size_small))
  elif request_type == "distinct":
    adapter_indices_large = np.arange(batch_size_large)
    adapter_indices_small = np.arange(batch_size_small)
  elif request_type == "identical":
    adapter_indices_large = np.zeros(batch_size_large, dtype=int)
    adapter_indices_small = np.zeros(batch_size_small, dtype=int)
  else:
    print(f"unknown request_type {request_type}")
  num_adapters_large = len(set(adapter_indices_large))
  num_adapters_small = len(set(adapter_indices_small))
  register_opt_pipeline()

  # sequential
  sequential_model = SequentialLora(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  sequential_model.to("float16")
  sequential_mod, _ = sequential_model.export_tvm(spec=sequential_model.get_default_spec())
  with target:
    sequential_vm = relax.VirtualMachine(relax.build(sequential_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  sequentialA_model = SequentialLoraA(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  sequentialA_model.to("float16")
  sequentialA_mod, _ = sequentialA_model.export_tvm(spec=sequentialA_model.get_default_spec())
  with target:
    sequentialA_vm = relax.VirtualMachine(relax.build(sequentialA_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  sequentialB_model = SequentialLoraB(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  sequentialB_model.to("float16")
  sequentialB_mod, _ = sequentialB_model.export_tvm(spec=sequentialB_model.get_default_spec())
  with target:
    sequentialB_vm = relax.VirtualMachine(relax.build(sequentialB_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  # padded
  padded_model = PaddedLora(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  padded_model.to("float16")
  padded_mod, _ = padded_model.export_tvm(spec=padded_model.get_default_spec())
  with target:
    padded_vm = relax.VirtualMachine(relax.build(padded_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  paddedA_model = PaddedLoraA(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  paddedA_model.to("float16")
  paddedA_mod, _ = paddedA_model.export_tvm(spec=paddedA_model.get_default_spec())
  with target:
    paddedA_vm = relax.VirtualMachine(relax.build(paddedA_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  paddedB_model = PaddedLoraB(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  paddedB_model.to("float16")
  paddedB_mod, _ = paddedB_model.export_tvm(spec=paddedB_model.get_default_spec())
  with target:
    paddedB_vm = relax.VirtualMachine(relax.build(paddedB_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  # splitted
  splitted_model = SplittedLora(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  splitted_model.to("float16")
  splitted_mod, _ = splitted_model.export_tvm(spec=splitted_model.get_default_spec())
  with target:
    splitted_vm = relax.VirtualMachine(relax.build(splitted_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  splittedA_model = SplittedLoraA(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  splittedA_model.to("float16")
  splittedA_mod, _ = splittedA_model.export_tvm(spec=splittedA_model.get_default_spec())
  with target:
    splittedA_vm = relax.VirtualMachine(relax.build(splittedA_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  splittedB_model = SplittedLoraB(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  splittedB_model.to("float16")
  splittedB_mod, _ = splittedB_model.export_tvm(spec=splittedB_model.get_default_spec())
  with target:
    splittedB_vm = relax.VirtualMachine(relax.build(splittedB_mod, target, pipeline=relax.get_pipeline("opt_llm")), dev)

  # combined
  combined_model = CombinedLora(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  combined_model.to("float16")
  combined_mod, _ = combined_model.export_tvm(spec=combined_model.get_default_spec())
  with target:
    combined_vm = relax.VirtualMachine(relax.build(combined_mod, target), dev)

  combinedA_model = CombinedLoraA(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  combinedA_model.to("float16")
  combinedA_mod, _ = combinedA_model.export_tvm(spec=combinedA_model.get_default_spec())
  with target:
    combinedA_vm = relax.VirtualMachine(relax.build(combinedA_mod, target), dev)

  combinedB_model = CombinedLoraB(batch_size_large, batch_size_small, r_large, r_small, num_adapters_large, num_adapters_small)
  combinedB_model.to("float16")
  combinedB_mod, _ = combinedB_model.export_tvm(spec=combinedB_model.get_default_spec())
  with target:
    combinedB_vm = relax.VirtualMachine(relax.build(combinedB_mod, target), dev)

  A_sequential, B_sequential, A_padded, B_padded, A_splitted, B_splitted, A_combined, B_combined = get_params(r_large, r_small, num_adapters_large, num_adapters_small)
  x = tvm.nd.array(np.random.uniform(-1e-2, 1e-2, size=(batch_size_large + batch_size_small, 1, 4096)).astype("float16"), device=dev)
  if measure:
    measure_file = open(f"{os.path.expanduser('~')}/{measure_dest}", "w")

  ################################################################################
  ################################## SEQUENTIAL ##################################
  ################################################################################
  sequential_wids_large = tvm.nd.array(adapter_indices_large.astype("int32"), device=dev)
  sequential_wids_small = tvm.nd.array(adapter_indices_small.astype("int32"), device=dev)

  # A
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      sequentialA_result_small, sequentialA_result_large = sequentialA_vm["forward"](x, sequential_wids_large, sequential_wids_small, A_sequential)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    sequentialA_result_small, sequentialA_result_large = sequentialA_vm["forward"](x, sequential_wids_large, sequential_wids_small, A_sequential)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # B
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      sequentialB_result = sequentialB_vm["forward"](sequentialA_result_small, sequentialA_result_large, sequential_wids_large, sequential_wids_small, B_sequential)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    sequentialB_result = sequentialB_vm["forward"](sequentialA_result_small, sequentialA_result_large, sequential_wids_large, sequential_wids_small, B_sequential)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # full
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      sequential_result = sequential_vm["forward"](x, sequential_wids_large, sequential_wids_small, A_sequential + B_sequential)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    sequential_result = sequential_vm["forward"](x, sequential_wids_large, sequential_wids_small, A_sequential + B_sequential)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  if not measure:
    assert np.allclose(sequentialB_result.numpy(), sequential_result.numpy(), atol=1e-5), "sequential separate A, B must be equal to full"

  ################################################################################
  #################################### PADDED ####################################
  ################################################################################
  padded_wids = tvm.nd.array(np.concatenate((adapter_indices_large, adapter_indices_small + num_adapters_large)).astype("int32"), device=dev)

  # A
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      paddedA_result = paddedA_vm["forward"](x, padded_wids, A_padded)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    paddedA_result = paddedA_vm["forward"](x, padded_wids, A_padded)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # B
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      paddedB_result = paddedB_vm["forward"](paddedA_result, padded_wids, B_padded)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    paddedB_result = paddedB_vm["forward"](paddedA_result, padded_wids, B_padded)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # full
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      padded_result = padded_vm["forward"](x, padded_wids, A_padded + B_padded)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    padded_result = padded_vm["forward"](x, padded_wids, A_padded + B_padded)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  if not measure:
    assert np.allclose(paddedB_result.numpy(), padded_result.numpy(), atol=1e-5), "padded separate A, B must be equal to full"
    assert np.allclose(padded_result.numpy(), sequential_result.numpy(), atol=1e-5), "padded must be equal to sequential"

  ################################################################################
  ################################### SPLITTED ###################################
  ################################################################################
  splitted_xids = tvm.nd.array(np.concatenate((np.array([id for id in range(batch_size_large) for _ in range(int(r_large / r_small))]), np.arange(batch_size_small) + batch_size_large)).astype("int32"), device=dev)
  splitted_wids_large = np.repeat(adapter_indices_large, int(r_large / r_small)) * int(r_large / r_small) + np.tile(np.arange(int(r_large / r_small)), batch_size_large)
  splitted_wids = tvm.nd.array(np.concatenate((splitted_wids_large, adapter_indices_small + max(splitted_wids_large) + 1)).astype("int32"), device=dev)
  
  # A
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      splittedA_result = splittedA_vm["forward"](x, splitted_xids, splitted_wids, A_splitted)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    splittedA_result = splittedA_vm["forward"](x, splitted_xids, splitted_wids, A_splitted)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # B
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      splittedB_result = splittedB_vm["forward"](splittedA_result, splitted_wids, B_splitted)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    splittedB_result = splittedB_vm["forward"](splittedA_result, splitted_wids, B_splitted)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # full
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      splitted_result = splitted_vm["forward"](x, splitted_xids, splitted_wids, A_splitted + B_splitted)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    splitted_result = splitted_vm["forward"](x, splitted_xids, splitted_wids, A_splitted + B_splitted)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  if not measure:
    assert np.allclose(splittedB_result.numpy(), splitted_result.numpy(), atol=1e-5), "splitted separate A, B must be equal to full"
    assert np.allclose(splitted_result.numpy(), sequential_result.numpy(), atol=1e-5), "splitted must be equal to sequential"

  ################################################################################
  ################################### COMBINED ###################################
  ################################################################################
  assert request_type == "distinct", "combined currently only works with distinct"
  combined_xids = tvm.nd.array(np.array([i for i in range(batch_size_large) for _ in range(r_large)] + [i for i in range(batch_size_large, batch_size_large + batch_size_small) for _ in range(r_small)]).astype("int32"), device=dev)
  combined_wids = tvm.nd.array(np.arange(batch_size_large + int(batch_size_small / int(r_large / r_small))).astype("int32"), device=dev)

  # A
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      combinedA_result = combinedA_vm["forward"](x, combined_xids, combined_wids, A_combined)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    combinedA_result = combinedA_vm["forward"](x, combined_xids, combined_wids, A_combined)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # B
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      combinedB_result = combinedB_vm["forward"](combinedA_result, combined_xids, combined_wids, B_combined)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    combinedB_result = combinedB_vm["forward"](combinedA_result, combined_xids, combined_wids, B_combined)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  # full
  if measure:
    for _ in range(warmup_trials):
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
      combined_result = combined_vm["forward"](x, combined_xids, combined_wids, A_combined + B_combined)
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
  for _ in range(trials):
    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    combined_result = combined_vm["forward"](x, combined_xids, combined_wids, A_combined + B_combined)
    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

  if not measure:
    assert np.allclose(combinedB_result.numpy(), combined_result.numpy(), atol=1e-5), "combined separate A, B must be equal to full"
    assert np.allclose(combined_result.numpy(), sequential_result.numpy(), atol=1e-5), "combined must be equal to sequential"

  if measure:
    measure_file.close()

if __name__ == "__main__":
  def str2bool(value):
    return (value.lower() == 'true')

  parser = argparse.ArgumentParser()
  parser.add_argument("--request_type", type=str, default="distinct")
  parser.add_argument("--r_large", type=int, default=64)
  parser.add_argument("--r_small", type=int, default=32)
  parser.add_argument("--batch_size_large", type=int, default=1)
  parser.add_argument("--batch_size_small", type=int, default=2)
  parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--measure", type=str2bool, default=False, help="whether or not to take measurements")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  args = parser.parse_args()

  assert (args.r_large % args.r_small) == 0, "r_large must be dividable by r_small at the moment"

  main(
    args.request_type,
    args.r_large,
    args.r_small,
    args.batch_size_large,
    args.batch_size_small,
    args.trials,
    args.measure,
    args.measure_dest,
    args.warmup_trials
  )

