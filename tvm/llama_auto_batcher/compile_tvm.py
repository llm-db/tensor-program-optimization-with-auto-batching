from llama_for_causal_lm import LlamaForCausalLM, LlamaAttention
from lora import Lora

from typing_extensions import get_origin, get_args
import types
import sys
import os
from itertools import combinations
import json
from transformers import AutoModelForCausalLM
import torch
from safetensors.torch import load_file
import numpy as np

import tvm
from tvm import dlight, relax
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from custom.write_tensor_part import write_tensor_part
from custom.fuse_take_matmul import FuseTakeMatmul
from custom.matmul import Matmul
from custom.gemv import GEMV
from custom.reduction import Reduction

def verify_adapter_def(adapter_class):
  def return_tensors_correct_with_count(return_var):
    if type(return_var) == types.GenericAlias:
      count = len(get_args(return_var))
      return_correct = ((get_origin(return_var) == tuple) and all(item == Tensor for item in get_args(return_var)))
      return return_correct, count
    elif type(return_var) == type:
      return (return_var == Tensor), 1
    else:
      return False, 0

  # define_parameters, weight_application and post_computation methods must be 
  # defined, weight_computation method is optional.
  assert hasattr(adapter_class, 'define_parameters'), "define_parameters method must be defined."
  assert hasattr(adapter_class, 'weight_application'), "weight_application method must be defined."
  assert hasattr(adapter_class, 'post_computation'), "post_computation method must be defined."
  weight_comp_defined = hasattr(adapter_class, 'weight_computation')

  # __init__ has to take in_features of type int, out_features of type int and a config.
  init_args = adapter_class.__init__.__annotations__
  assert ('in_features' in init_args and init_args['in_features'] == int), "__init__ must take `in_features: int` as input"
  assert ('out_features' in init_args and init_args['out_features'] == int), "__init__ must take `out_features: int` as input"
  assert ('config' in init_args), "__init__ must take `config: <Adapter>Config` as input"

  # define_parameters must take a subset of the inputs to __init__ as its input.
  # it returns a Tensor or a tuple of Tensors.
  define_params_args = adapter_class.define_parameters.__annotations__
  assert ('return' in define_params_args), "define_params must indicate the return type with `-> <return_type>`"
  define_params_return = define_params_args.pop('return')
  assert (define_params_args.items() <= init_args.items()), "define_params inputs must be a subset of __init__'s inputs"
  define_params_return_correct, define_params_num_rets = return_tensors_correct_with_count(define_params_return)
  assert define_params_return_correct, "define_params must return a Tensor (`-> Tensor`) or a tuple of Tensors (`-> tuple[Tensor, ...]`)"

  # weight_computation (if exists) takes as input as many Tensors as define_params
  # returns. like define_parameters it returns a Tensor or a tuple of Tensors.
  if weight_comp_defined:
    weight_comp_args = adapter_class.weight_computation.__annotations__
    assert ('return' in weight_comp_args), "weight_computation must indicate the return type with `-> <return_type>`"
    weight_comp_return = weight_comp_args.pop('return')
    weight_comp_input_correct = ((len(weight_comp_args) == define_params_num_rets) and 
        all(param_type == Tensor for param_type in weight_comp_args.values()))
    assert weight_comp_input_correct, "weight_computation must have exactly the input that define_params returns"
    weight_comp_return_correct, weight_comp_num_rets = return_tensors_correct_with_count(weight_comp_return)
    assert weight_comp_return_correct, "weight_computation must return a Tensor (`-> Tensor`) or a tuple of Tensors (`-> tuple[Tensor, ...]`)"
  
  # weight_application must return a single Tensor. as input it must take x and as many 
  # Tensors as weight_computation returns if weight_computation is defined. if it is 
  # not defined it must take as many Tensors as define_parameters returns.
  weight_appl_args = adapter_class.weight_application.__annotations__
  assert (('return' in weight_appl_args) and (weight_appl_args['return'] == Tensor)), "weight_application must return a Tensor and indicate it with `-> Tensor`"
  weight_appl_args.pop('return')
  assert (('x' in weight_appl_args) and (weight_appl_args['x'] == Tensor)), "weight_application must have `x: Tensor` as the first input argument"
  weight_appl_args.pop('x')
  weight_appl_input_correct = ((len(weight_appl_args) == (weight_comp_num_rets if weight_comp_defined else define_params_num_rets)) and 
      all(param_type == Tensor for param_type in weight_appl_args.values()))
  assert weight_appl_input_correct, f"weight_application must have as input what {'weight_computation' if weight_comp_defined else 'define_params'} returns"

  # post_computation only takes a single Tensor and returns a single Tensor
  # (it uses only members defined in __init__).
  post_comp_args = adapter_class.post_computation.__annotations__
  assert (('return' in post_comp_args) and (post_comp_args['return'] == Tensor)), "post_computation must return a Tensor and indicate it with `-> Tensor`"
  post_comp_args.pop('return')
  assert((len(post_comp_args) == 1) and (list(post_comp_args.values())[0] == Tensor)), "post_computation must have a single Tensor as input"

  return weight_comp_defined, define_params_num_rets

def update_adapter_def(adapter_class):
  weight_comp_defined, num_params = verify_adapter_def(adapter_class)

  def new_init(self, in_features, out_features, config, num_adapters):
    self.__original_init__(in_features, out_features, config)
    params = self.define_parameters(in_features, out_features, config)
    for i in range(num_params):
      setattr(self, f"p{i}", nn.Parameter([num_adapters] + params[i].shape, params[i].dtype))
  adapter_class.__original_init__ = adapter_class.__init__
  adapter_class.__init__ = new_init

  def forward(self, x: Tensor, wids: Tensor):
    params = [getattr(self, f"p{i}") for i in range(num_params)]
    if weight_comp_defined:
      params = self.weight_computation(*params)
    params = [op.take(param, wids, axis=0) for param in params]
    y = self.weight_application(x, *params)
    return self.post_computation(y)
  adapter_class.forward = forward

def update_llama_attention():
  def forward(
    self,
    hidden_states: Tensor, 
    paged_kv_cache: PagedKVCache,
    layer_id: int,
    *args
  ):
    q, k, v = self.qkv_projection(hidden_states)
    if len(args) == 0:
      pass
    elif len(args) == 2:
      # args[0]: aid
      # args[1]: wids
      aid = args[0]
      wids = args[1]
      q_ft_proj = getattr(self, f"q_ft_proj_{aid}")
      q += q_ft_proj(hidden_states, wids)
      v_ft_proj = getattr(self, f"v_ft_proj_{aid}")
      v += v_ft_proj(hidden_states, wids)
    elif (len(args) > 2) and (len(args) % 2 == 1):
      # args[0]: aids
      # args[1]: a0_sids
      # args[2]: a0_wids
      # args[3]: a1_sids
      # args[4]: a1_wids
      # ...
      q_a = op.zeros(q.shape, q.dtype)
      v_a = op.zeros(v.shape, v.dtype)
      for i, aid in enumerate(args[0]):
        sids = args[1 + i * 2]
        wids = args[1 + i * 2 + 1]
        hidden_states_lcl = op.take(hidden_states, sids, axis=0)
        q_ft_proj = getattr(self, f"q_ft_proj_{aid}")
        q_a_lcl = q_ft_proj(hidden_states_lcl, wids)
        q_a = nn.tensor_ir_op(write_tensor_part, "write_tensor_part", [q_a, q_a_lcl, sids], out=nn.Tensor.placeholder(q_a.shape, "float16"))
        v_ft_proj = getattr(self, f"v_ft_proj_{aid}")
        v_a_lcl = v_ft_proj(hidden_states_lcl, wids)
        v_a = nn.tensor_ir_op(write_tensor_part, "write_tensor_part", [v_a, v_a_lcl, sids], out=nn.Tensor.placeholder(v_a.shape, "float16"))
      q += q_a
      v += v_a
    else:
      sys.exit(f"args list in Attention forward must be of length 0 (no adapter), 2 (adapter type) or one of [3, 5, 7, ...] (mixed) but has length {len(args)}")
    attn_out = self.attention(q, k, v, paged_kv_cache, layer_id)
    out = self.o_proj(attn_out)
    return out
  LlamaAttention.forward = forward

def update_llama_for_causal_lm(batched_decode, seq_info):
  # no adapter functions
  def prefill_no_adapter(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    return self.flexible_prefill(input_embed, paged_kv_cache)
  LlamaForCausalLM.prefill_no_adapter = prefill_no_adapter

  def decode_no_adapter(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    return self.flexible_decode(input_embed, paged_kv_cache)
  LlamaForCausalLM.decode_no_adapter = decode_no_adapter

  adapter_configs = seq_info.get_adapter_configs()
  if len(adapter_configs) > 0:
    # adapter type functions
    num_adapters_per_type = seq_info.get_num_adapters_per_type()
    def llama_attention_init(self, config):
      self.__original_init__(config)
      for i in range(len(adapter_configs)):
        setattr(self, f"q_ft_proj_{i}", Lora(config.hidden_size, self.num_q_heads * self.head_dim, adapter_configs[i], num_adapters_per_type[i]))
        setattr(self, f"v_ft_proj_{i}", Lora(config.hidden_size, self.num_kv_heads * self.head_dim, adapter_configs[i], num_adapters_per_type[i]))
    LlamaAttention.__original_init__ = LlamaAttention.__init__
    LlamaAttention.__init__ = llama_attention_init

    for i in range(len(adapter_configs)):
      def prefill_a(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, wids: Tensor, i=i):
        return self.flexible_prefill(input_embed, paged_kv_cache, i, wids)
      setattr(LlamaForCausalLM, f"prefill_a{i}", prefill_a)
      def decode_a(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, wids: Tensor, i=i):
        return self.flexible_decode(input_embed, paged_kv_cache, i, wids)
      setattr(LlamaForCausalLM, f"decode_a{i}", decode_a)

    if batched_decode:
      # mixed adapter type functions
      default_params = ["input_embed", "paged_kv_cache"]
      for aid_tup in sum((list(combinations(range(len(adapter_configs)), r)) for r in range(1, len(adapter_configs) + 1)), []):
        aid_lst = list(aid_tup)
        method_tag = "_".join(f"a{i}" for i in aid_lst)
        aid_param = [str(list(aid_lst))]
        adapter_params = [item for aid in aid_lst for item in (f"a{aid}_sids", f"a{aid}_wids")]
        params_passed = ["self"] + default_params + adapter_params
        params_to_pass = default_params + aid_param + adapter_params
        params_passed_str = ", ".join(params_passed)
        params_to_pass_str = ", ".join(params_to_pass)
        decode_code = f"\ndef mixed_decode({params_passed_str}):\n  return self.flexible_decode({params_to_pass_str})\n"
        local_dict = {}
        exec(decode_code, local_dict)
        setattr(LlamaForCausalLM, f"mixed_decode_{method_tag}", local_dict["mixed_decode"])

  # get default spec method
  def get_default_spec_method(self):
    mod_spec = {
      "embed": {
        "input_ids": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      },
      "create_paged_kv_cache": {
        "max_batch_size": int,
        "max_total_seq_len": int,
        "prefill_chunk_size": int,
        "page_size": int,
        "$": {
          "param_mode": "none",
          "effect_mode": "none",
        },
      },
    }
    # no adapter
    mod_spec["prefill_no_adapter"] = {
      "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
      "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
      "$": {
        "param_mode": "packed",
        "effect_mode": "none",
      }
    }
    mod_spec["decode_no_adapter"] = {
      "input_embed": nn.spec.Tensor(["batch_size" if batched_decode else 1, 1, self.hidden_size], self.dtype),
      "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
      "$": {
        "param_mode": "packed",
        "effect_mode": "none",
      }
    }
    # adapter type
    for i in range(len(adapter_configs)):
      mod_spec[f"prefill_a{i}"] = {
        "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
        "wids": nn.spec.Tensor([1], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        }
      }
      mod_spec[f"decode_a{i}"] = {
        "input_embed": nn.spec.Tensor(["batch_size" if batched_decode else 1, 1, self.hidden_size], self.dtype),
        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
        "wids": nn.spec.Tensor(["batch_size" if batched_decode else 1], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        }
      }
    # mixed
    if batched_decode:
      for aid_tup in sum((list(combinations(range(len(adapter_configs)), r)) for r in range(1, len(adapter_configs) + 1)), []):
        aid_lst = list(aid_tup)
        method_tag = "_".join(f"a{i}" for i in aid_lst)
        mod_spec[f"mixed_decode_{method_tag}"] = {
          "input_embed": nn.spec.Tensor(["batch_size" if batched_decode else 1, 1, self.hidden_size], self.dtype),
          "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
          "$": {
            "param_mode": "packed",
            "effect_mode": "none",
          }
        }
        for aid in aid_lst:
          mod_spec[f"mixed_decode_{method_tag}"][f"a{aid}_sids"] = nn.spec.Tensor([f"a{aid}_len"], "int32")
          mod_spec[f"mixed_decode_{method_tag}"][f"a{aid}_wids"] = nn.spec.Tensor([f"a{aid}_len"], "int32")
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)
  setattr(LlamaForCausalLM, "get_default_spec", get_default_spec_method)

def prepare_model_weights(model_name, cache_dir, dev, named_params, seq_info):
  # base model
  model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16)
  param_dict = model.state_dict()
  param_dict = {
    k: v.half().numpy() if v.dtype == torch.bfloat16 else v.numpy()
    for k, v in param_dict.items()
  }

  # adapters
  adapter_configs = seq_info.get_adapter_configs()
  adapter_wids = seq_info.get_wids_per_type()
  for i, adapter_config in enumerate(adapter_configs):
    weights_lst = []
    for adapter_wid in adapter_wids[i]:
      weights_lst.append(load_file(f"{cache_dir}/lora_adapters/r{adapter_config.r}_a{adapter_wid}.safetensors"))
    weights_dict = {}
    for weights in weights_lst:
      for k, v in weights.items():
        v_new = v.half().numpy() if v.dtype == torch.bfloat16 else v.numpy()
        if k in weights_dict.keys():
          weights_dict[k].append(v_new)
        else:
          weights_dict[k] = [v_new]
    for k, v in weights_dict.items():
      param_k ='model.' + k.replace('proj', f'ft_proj_{i}').replace('lora_A', 'p0').replace('lora_B', 'p1')[:-7]
      param_v = np.stack(v, axis=0).transpose((0, 2, 1))
      param_dict[param_k] = param_v

  named_params = dict(named_params)
  model_params = []
  for k in named_params.keys():
    model_params.append(tvm.nd.array(param_dict[k].astype("float16"), device=dev))
  return model_params

def compile_tvm(
  batched_decode,
  seq_info,
  dev,
  model_config,
  opt_config_path,
  cache_dir
):
  if len(seq_info.get_adapter_configs()) > 0:
    update_adapter_def(Lora)
  update_llama_attention()
  update_llama_for_causal_lm(batched_decode, seq_info)

  target = tvm.target.Target.from_device(dev)
  model = LlamaForCausalLM(model_config, target)
  model.to("float16")
  mod, named_params = model.export_tvm(spec=model.get_default_spec())

  # Define Optimization Pipeline
  with open(opt_config_path) as opt_config_file:
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

  with target:
    ex = relax.build(mod, target, pipeline=relax.get_pipeline("opt_llm"))
    vm = relax.VirtualMachine(ex, dev)

  model_params = prepare_model_weights(model_config.name, cache_dir, dev, named_params, seq_info)

  return vm, model_params
