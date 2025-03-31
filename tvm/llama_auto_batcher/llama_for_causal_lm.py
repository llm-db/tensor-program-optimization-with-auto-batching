import enum
from typing import Optional
import dataclasses

from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache

@dataclasses.dataclass
class LlamaConfig:
  name: str = "meta-llama/Meta-Llama-3.1-8B"
  hidden_size: int = 4096
  intermediate_size: int = 14336
  num_attention_heads: int = 32
  num_hidden_layers: int = 32
  rms_norm_eps: float = 1e-05
  vocab_size: int = 128256
  rope_theta: int = 500000.0 
  num_key_value_heads: int = 8
  head_dim: int = 128  # hidden_size // num_attention_heads

class RopeMode(enum.IntEnum):
  NONE = 0
  NORMAL = 1
  INLINE = 2

class LlamaFFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.gate_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
    self.up_proj = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
    self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

  def forward(self, x: Tensor):
    return self.down_proj(op.silu(self.gate_proj(x)) * self.up_proj(x))
  
class LlamaAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
  def __init__(self, config):
    self.head_dim = config.head_dim
    self.num_q_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads

    self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

  def qkv_projection(self, hidden_states: Tensor):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    return q, k, v

  def attention(self, q: Tensor, k: Tensor, v: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
    d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
    b, s, _ = q.shape
    qkv = op.concat([q, k, v], -1)
    qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
    output = op.reshape(
      paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_q_heads),
      (b, s, h_q * d),
    )
    return output

  def o_projection(self, out: Tensor):
    return self.o_proj(out)

  # forward defined at compile time

class LlamaDecoderLayer(nn.Module):
  def __init__(self, config):
    rms_norm_eps = config.rms_norm_eps
    self.self_attn = LlamaAttention(config)
    self.mlp = LlamaFFN(config)
    self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
    self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

  def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int, *args):
    hidden_states += self.self_attn(
      self.input_layernorm(hidden_states), paged_kv_cache, layer_id, *args
    )
    hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
    return hidden_states

class LlamaModel(nn.Module):
  def __init__(self, config):
    assert config.hidden_size % config.num_attention_heads == 0
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList(
      [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
    )
    self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

  def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, *args):
    hidden_states = input_embed
    for layer_id, layer in enumerate(self.layers):
      hidden_states = layer(hidden_states, paged_kv_cache, layer_id, *args)
    hidden_states = self.norm(hidden_states)
    return hidden_states

class LlamaForCausalLM(nn.Module):
  def __init__(self, config, target):
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
    self.target = target

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

  def flexible_prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, *args):
    def _index(x: te.Tensor):  # x[:-1,:]
      b, s, d = x.shape
      return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")
    hidden_states = self.model(input_embed, paged_kv_cache, *args)
    hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
    logits = self.get_logits(hidden_states)
    return logits, paged_kv_cache
  
  # prefill functions with correct args are defined at compile time

  def flexible_decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache, *args):
    hidden_states = self.model(input_embed, paged_kv_cache, *args)
    logits = self.get_logits(hidden_states)
    return logits, paged_kv_cache

  # decode functions with correct args are defined at compile time

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
      target=self.target,
    )
