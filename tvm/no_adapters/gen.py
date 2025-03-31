import dataclasses
import argparse
import enum
from typing import Optional
import json
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import tvm
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache
from tvm import dlight, relax, te, tir
from tvm.relax import register_pipeline
from tvm.runtime import ShapeTuple

@dataclasses.dataclass
class LlamaConfig:
  hidden_size: int = 4096
  intermediate_size: int = 14336
  num_attention_heads: int = 32
  num_hidden_layers: int = 32
  rms_norm_eps: float = 1e-05
  vocab_size: int = 128256
  rope_theta: int = 500000.0 
  num_key_value_heads: int = 8
  head_dim: int = 128  # hidden_size // num_attention_heads


dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)


class RopeMode(enum.IntEnum):
  """The RoPE mode of the Paged KV cache.
  If it is none, the KV cache will not apply RoPE to q and k.
  If it is normal, RoPE will be applied to k before adding k to cache.
  Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
  """
  NONE = 0
  NORMAL = 1
  INLINE = 2


class LlamaFFN(nn.Module):
  def __init__(self, config: LlamaConfig):
    super().__init__()
    self.gate_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
    self.up_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
    self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

  def forward(self, x: Tensor):
    return self.down_proj(op.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
  def __init__(self, config: LlamaConfig):
    self.head_dim = config.head_dim
    self.num_q_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads

    self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

  def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
    d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
    b, s, _ = hidden_states.shape

    # QKV Projection
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    qkv = op.concat([q, k, v], -1)
    qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))

    # Attention
    output = op.reshape(
      paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_q_heads),
      (b, s, h_q * d),
    )

    # Output Projection
    return self.o_proj(output)


class LlamaDecoderLayer(nn.Module):
  def __init__(self, config: LlamaConfig):
    rms_norm_eps = config.rms_norm_eps
    self.self_attn = LlamaAttention(config)
    self.mlp = LlamaFFN(config)
    self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
    self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

  def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
    hidden_states += self.self_attn(
      self.input_layernorm(hidden_states), paged_kv_cache, layer_id
    )
    hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
    return hidden_states


class LlamaModel(nn.Module):
  def __init__(self, config: LlamaConfig):
    assert config.hidden_size % config.num_attention_heads == 0
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList(
      [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
    )
    self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

  def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    hidden_states = input_embed
    for layer_id, layer in enumerate(self.layers):
      hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
    hidden_states = self.norm(hidden_states)
    return hidden_states


class LlamaForCausalLM(nn.Module):
  def __init__(self, config: LlamaConfig):
    self.model = LlamaModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.num_hidden_layers = config.num_hidden_layers
    self.num_attention_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.head_dim = config.head_dim
    self.hidden_size = config.hidden_size
    self.vocab_size = config.vocab_size
    self.rope_theta = config.rope_theta
    self.dtype = "float32"

  def to(self, dtype: Optional[str] = None):
    super().to(dtype=dtype)
    if dtype is not None:
      self.dtype = dtype

  def embed(self, input_ids: Tensor):
    return self.model.embed_tokens(input_ids)

  def get_logits(self, hidden_states: Tensor):
    logits = self.lm_head(hidden_states)
    if logits.dtype != "float32":
      logits = logits.astype("float32")
    return logits

  def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    def _index(x: te.Tensor):  # x[:-1,:]
      b, s, d = x.shape
      return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

    hidden_states = self.model(input_embed, paged_kv_cache)
    hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
    logits = self.get_logits(hidden_states)
    return logits, paged_kv_cache

  def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    hidden_states = self.model(input_embed, paged_kv_cache)
    logits = self.get_logits(hidden_states)
    return logits, paged_kv_cache

  def create_paged_kv_cache(
    self,
    max_batch_size: tir.Var,
    max_total_seq_len: tir.Var,
    prefill_chunk_size: tir.Var,
    page_size: tir.Var,
  ) -> PagedKVCache:
    return TIRPagedKVCache(
      max_batch_size=max_batch_size,
      max_total_seq_len=max_total_seq_len,
      prefill_chunk_size=prefill_chunk_size,
      page_size=page_size,
      support_sliding_window=0,
      layer_partition=relax.ShapeExpr([0, self.num_hidden_layers]),
      num_hidden_layers=self.num_hidden_layers,
      num_attention_heads=self.num_attention_heads,
      num_key_value_heads=self.num_key_value_heads,
      head_dim=self.head_dim,
      rope_mode=RopeMode.INLINE,
      rope_scale=1,
      rope_theta=self.rope_theta,
      rope_scaling={},
      rope_ext_factors=relax.PrimValue(0),
      rotary_dim=self.head_dim,
      dtype=self.dtype,
      target=target,
    )

def set_default_spec(
  batch_size,
  shared_cache,
  batched_decode,
  batch_size_fix
):
  def get_default_spec_method(self):
    size = 1
    if shared_cache and batched_decode and (batch_size > 1):
      size = batch_size if batch_size_fix else "batch_size"
    mod_spec = {
      "embed": {
        "input_ids": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      },
      "prefill": {
        "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
        "$": {
          "param_mode": "packed",
          "effect_mode": "none",
        },
      },
      "decode": {
        "input_embed": nn.spec.Tensor([size, 1, self.hidden_size], self.dtype),
        "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
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
    return nn.spec.ModuleSpec.from_raw(mod_spec, self)
  setattr(LlamaForCausalLM, "get_default_spec", get_default_spec_method)

def compile(opt_config):
  model_config = LlamaConfig()
  model = LlamaForCausalLM(model_config)
  model.to("float16")
  mod, named_params = model.export_tvm(spec=model.get_default_spec())

  with open(f"{os.path.expanduser('~')}/{opt_config}") as opt_config_file:
    opt_config_dict = json.load(opt_config_file)
  @register_pipeline("opt_llm")
  def _pipeline():
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
      seq = tvm.transform.Sequential(
        [relax.transform.FuseTransposeMatmul()] * opt_config_dict['FuseTransposeMatmul'] +
        [relax.transform.LegalizeOps()] +
        [relax.transform.AnnotateTIROpPattern()] * opt_config_dict['AnnotateTIROpPattern'] +
        [relax.transform.FoldConstant()] +
        [relax.transform.FuseOps()] * opt_config_dict['FuseOps'] +
        [relax.transform.FuseTIR()]  * opt_config_dict['FuseTIR']+
        [relax.transform.DeadCodeElimination()] +
        [dlight.ApplyDefaultSchedule(
          *([dlight.gpu.Matmul()] * opt_config_dict['ApplyDefaultSchedule']['Matmul'] +
            [dlight.gpu.GEMV()] * opt_config_dict['ApplyDefaultSchedule']['GEMV'] +
            [dlight.gpu.Reduction()] * opt_config_dict['ApplyDefaultSchedule']['Reduction']),
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
  
  return vm, named_params

def prepare_model_weights(model_name, cache_dir, named_params):
  model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16)
  param_dict = model.state_dict()
  param_dict = {
    k: v.half().numpy() if v.dtype == torch.bfloat16 else v.numpy()
    for k, v in param_dict.items()
  }
  named_params = dict(named_params)
  params = [
    tvm.nd.array(param_dict[k].astype("float16"), device=dev) for k in named_params.keys()
  ]
  return params

def generate_cache_separate(
  input_tokens,
  vm,
  params,
  gen_len,
  measure,
  measure_file
):
  output_tokens = []
  for input in input_tokens:
    input_len = len(input)
    tokens = tvm.nd.array(np.asarray(input.unsqueeze(0), dtype="int32"), device=dev)
    kv_cache = vm["create_paged_kv_cache"](
      ShapeTuple([1]),
      ShapeTuple([input_len + gen_len]),
      ShapeTuple([input_len]),
      ShapeTuple([16]),
    )

    add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    add_sequence_func(kv_cache, 0)
    hidden_states = vm["embed"](tokens, params)

    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()

    begin_forward_func(kv_cache, ShapeTuple([0]), ShapeTuple([input_len]))
    logits, kv_cache = vm["prefill"](hidden_states, kv_cache, params)
    end_forward_func(kv_cache)

    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

    last_token = np.argmax(logits.numpy())
    output = [last_token]

    while len(output) < gen_len:
      tokens = tvm.nd.array(np.asarray([[last_token]]).astype("int32"), device=dev)
      hidden_states = vm["embed"](tokens, params)

      if measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

      begin_forward_func(kv_cache, ShapeTuple([0]), ShapeTuple([1]))
      logits, kv_cache = vm["decode"](hidden_states, kv_cache, params)
      end_forward_func(kv_cache)

      if measure:
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        measure_file.write(f"{time_ms}\n")

      last_token = np.argmax(logits.numpy())
      output.append(last_token)

    output_tokens.append(output)
    del kv_cache

  return output_tokens

def generate_cache_shared_decode_seq(
  input_tokens,
  vm,
  params,
  gen_len,
  measure,
  measure_file
):
  batch_size, input_len = input_tokens.shape
  kv_cache = vm["create_paged_kv_cache"](
    ShapeTuple([batch_size]),
    ShapeTuple([batch_size * input_len + batch_size * gen_len]),
    ShapeTuple([input_len]),
    ShapeTuple([16]),
  )
  add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
  begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
  end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")

  output_tokens = []
  for i, input in enumerate(input_tokens):
    tokens = tvm.nd.array(np.asarray(input.unsqueeze(0), dtype="int32"), device=dev)
    add_sequence_func(kv_cache, i)
    hidden_states = vm["embed"](tokens, params)

    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()

    begin_forward_func(kv_cache, ShapeTuple([i]), ShapeTuple([input_len]))
    logits, kv_cache = vm["prefill"](hidden_states, kv_cache, params)
    end_forward_func(kv_cache)

    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

    last_token = np.argmax(logits.numpy())
    output = [last_token]

    while len(output) < gen_len:
      tokens = tvm.nd.array(np.asarray([[last_token]]).astype("int32"), device=dev)
      hidden_states = vm["embed"](tokens, params)

      if measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

      begin_forward_func(kv_cache, ShapeTuple([i]), ShapeTuple([1]))
      logits, kv_cache = vm["decode"](hidden_states, kv_cache, params)
      end_forward_func(kv_cache)

      if measure:
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        measure_file.write(f"{time_ms}\n")

      last_token = np.argmax(logits.numpy())
      output.append(last_token)

    output_tokens.append(output)

  del kv_cache
  return output_tokens

def generate_cache_shared_decode_batched(
  input_tokens,
  vm,
  params,
  gen_len,
  measure,
  measure_file
):
  batch_size, input_len = input_tokens.shape
  kv_cache = vm["create_paged_kv_cache"](
    ShapeTuple([batch_size]),
    ShapeTuple([batch_size * input_len + batch_size * gen_len]),
    ShapeTuple([input_len]),
    ShapeTuple([16]),
  )
  add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
  begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
  end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")

  output_tokens = []
  for i, input in enumerate(input_tokens):
    tokens = tvm.nd.array(np.asarray(input.unsqueeze(0), dtype="int32"), device=dev)
    add_sequence_func(kv_cache, i)
    hidden_states = vm["embed"](tokens, params)

    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()

    begin_forward_func(kv_cache, ShapeTuple([i]), ShapeTuple([input_len]))
    logits, kv_cache = vm["prefill"](hidden_states, kv_cache, params)
    end_forward_func(kv_cache)

    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

    last_token = np.argmax(logits.numpy())
    output_tokens.append([last_token])

  decode_seq_ids = [i for i in range(batch_size)]
  decode_gen_lens = [1] * batch_size
  while len(output_tokens[0]) < gen_len:
    tokens = tvm.nd.array(np.asarray([[elem[-1]] for elem in output_tokens]).astype("int32"), device=dev)
    hidden_states = vm["embed"](tokens, params)

    if measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()

    begin_forward_func(kv_cache, ShapeTuple(decode_seq_ids), ShapeTuple(decode_gen_lens))
    logits, kv_cache = vm["decode"](hidden_states, kv_cache, params)
    end_forward_func(kv_cache)

    if measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      measure_file.write(f"{time_ms}\n")

    logits_np = logits.numpy()
    for i, logits_elem in enumerate(logits_np):
      output_tokens[i].append(np.argmax(logits_elem))

  del kv_cache
  return output_tokens

def generate(
  input_tokens, 
  vm, 
  params, 
  shared_cache, 
  batched_decode,
  trials,
  gen_len,
  measure,
  measure_dest,
  warmup_trials
):
  if shared_cache:
    if batched_decode:
      generate_func = generate_cache_shared_decode_batched
    else:
      generate_func = generate_cache_shared_decode_seq
  else:
    generate_func = generate_cache_separate

  if measure:
    measure_file = open(f"{os.path.expanduser('~')}/{measure_dest}", "w")
    for _ in range(warmup_trials):
      generate_func(
        input_tokens,
        vm,
        params,
        gen_len,
        False,
        measure_file
      )

  for _ in range(trials):
    output_tokens = generate_func(
      input_tokens,
      vm,
      params,
      gen_len,
      measure,
      measure_file if measure else None
    )
  
  if measure:
    measure_file.close()

  return output_tokens

def main(
  model_name,
  cache_dir,
  opt_config,
  prompt_path,
  batch_size,
  shared_cache,
  batched_decode,
  batch_size_fix,
  trials,
  gen_len,
  measure,
  measure_dest,
  warmup_trials
):
  # compile
  set_default_spec(batch_size, shared_cache, batched_decode, batch_size_fix)
  vm, named_params = compile(opt_config)
  params = prepare_model_weights(model_name, cache_dir, named_params)

  # prepare input tokens tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
  tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
  prompt = open(f"{os.path.expanduser('~')}/{prompt_path}", "r").read()
  prompts = [prompt] * batch_size
  input_tokens = tokenizer(prompts, return_tensors="pt")['input_ids']

  # generate
  output_tokens = generate(
    input_tokens, 
    vm, 
    params, 
    shared_cache, 
    batched_decode,
    trials,
    gen_len,
    measure,
    measure_dest,
    warmup_trials
  )

  # print output
  for output in output_tokens:
    print(tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))

if __name__ == "__main__":
  def str2bool(value):
    return True if (value.lower() == 'true') else False

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
  parser.add_argument("--cache_dir", type=str, default="/scratch/lucastr")
  parser.add_argument("--opt_config", type=str, default="fineinfer-autopeft/tvm/opt_configs/default.json", help="path to optimization config relative to ~")
  parser.add_argument("--prompt", type=str, default="fineinfer-autopeft/prompts/default.txt", help="path to prompt relative to ~")
  parser.add_argument("--batch_size", type=int, default=1,  help="batch size")
  parser.add_argument("--shared_cache", type=str2bool, default=True, help="whether or not to use a shared cache for the full batch")
  parser.add_argument("--batched_decode", type=str2bool, default=True, help="whether or not to batch the decode computation")
  parser.add_argument("--batch_size_fix", type=str2bool, default=False)
  parser.add_argument("--trials", type=int, default=1,  help="Number of token generation iterations")
  parser.add_argument("--gen_len", type=int, default=32,  help="number of tokens to generate")
  parser.add_argument("--measure", type=str2bool, default=False, help="whether or not to take measurements")
  parser.add_argument("--measure_dest", type=str, default="fineinfer-autopeft/measure.txt", help="where to store measurements")
  parser.add_argument("--warmup_trials", type=int, default=3, help="number of warmup trials")
  args = parser.parse_args()

  main(
    args.model_name,
    args.cache_dir,
    args.opt_config,
    args.prompt,
    args.batch_size,
    args.shared_cache,
    args.batched_decode,
    args.batch_size_fix,
    args.trials,
    args.gen_len,
    args.measure,
    args.measure_dest,
    args.warmup_trials
  ) 
