from exec_graph import ExecGraph
import torch
import numpy as np
import dataclasses
import os

import tvm
from tvm.runtime import ShapeTuple

@dataclasses.dataclass
class LlamaExecConfig:
  max_prompt_len: int
  max_gen_len: int
  page_size: int

class LlamaExecGraph(ExecGraph):
  def __init__(self, batchable_op_types=["decode"]):
    super().__init__(batchable_op_types)

  def set_exec_vars(
    self,
    vm,
    dev,
    params,
    tokenizer,
    seq_info,
    exec_config,
  ):
    self.vm = vm
    self.dev = dev
    self.params = params
    self.tokenizer = tokenizer
    self.seq_info = seq_info
    self.exec_config = exec_config
    self.measure = False
    self.set_kv_cache(False)

  def set_kv_cache(self, delete_prev=True):
    if delete_prev:
      del self.kv_cache
    self.kv_cache = self.vm["create_paged_kv_cache"](
      ShapeTuple([self.seq_info.get_num_sequences()]),
      ShapeTuple([self.seq_info.get_num_sequences() * (self.exec_config.max_prompt_len + self.exec_config.max_gen_len)]),
      ShapeTuple([self.exec_config.max_prompt_len]),
      ShapeTuple([self.exec_config.page_size])
    )

  def set_measuring(self, dest):
    self.measure = True
    self.measure_file = open(f"{os.path.expanduser('~')}/{dest}", "w")

  def execute_encode_prompt(self, op_node):
    assert len(op_node.input_vars) == 1, f"encode_prompt node must have exactly 1 input var but has {len(op_node.input_vars)}"
    assert len(op_node.output_vars) == 1, f"encode_prompt node must have exactly 1 output var but has {len(op_node.output_vars)}"
    assert len(op_node.seqids) == 1, f"encode_prompt node must have exactly 1 seqid but has {len(op_node.seqids)}"
    op_node.output_vars[0].var = self.tokenizer(op_node.input_vars[0].var, return_tensors="pt")['input_ids']

  def execute_prefill(self, op_node):
    assert len(op_node.input_vars) == 1, f"prefill node must have exactly 1 input var but has {len(op_node.input_vars)}"
    assert len(op_node.output_vars) == 1, f"prefill node must have exactly 1 output var but has {len(op_node.output_vars)}"
    assert len(op_node.seqids) == 1, f"prefill node must have exactly 1 seqid but has {len(op_node.seqids)}"

    sid = op_node.seqids[0]
    tokens = op_node.input_vars[0].var
    tokens_tvm = tvm.nd.array(np.asarray(tokens, dtype="int32"), device=self.dev)
    tvm.get_global_func("vm.builtin.kv_state_add_sequence")(self.kv_cache, sid)
    hidden_states = self.vm["embed"](tokens_tvm, self.params)

    adapter_info = self.seq_info.get_adapter_info(sid)
    if self.measure:
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      torch.cuda.synchronize()
      start_event.record()
    tvm.get_global_func("vm.builtin.kv_state_begin_forward")(
      self.kv_cache, 
      ShapeTuple([sid]), 
      ShapeTuple([tokens.shape[1]])
    )
    if adapter_info is None:
      logits, self.kv_cache = self.vm["prefill_no_adapter"](hidden_states, self.kv_cache, self.params)
    else:
      wid = tvm.nd.array(np.asarray([adapter_info["wid"]], dtype="int32"), device=self.dev)
      logits, self.kv_cache = self.vm[f"prefill_a{adapter_info['tid']}"](hidden_states, self.kv_cache, wid, self.params)
    tvm.get_global_func("vm.builtin.kv_state_end_forward")(self.kv_cache)
    if self.measure:
      end_event.record()
      torch.cuda.synchronize()
      time_ms = start_event.elapsed_time(end_event)
      self.measure_file.write(f"{time_ms}\n")

    op_node.output_vars[0].var = np.argmax(logits.numpy())

  def check_seqs_finished(self, seqs_2_decode, outputs):
    new_seqs_2_decode = []
    for sid in seqs_2_decode:
      if (outputs[sid][-1] != self.tokenizer.eos_token_id) and \
         (len(outputs[sid]) < self.seq_info.get_max_gen_len(sid)):
        new_seqs_2_decode.append(sid)
    return new_seqs_2_decode

  def execute_decode(self, op_node):
    assert len(op_node.input_vars) == len(op_node.output_vars), f"decode node has {len(op_node.input_vars)} input_vars and {len(op_node.output_vars)} ouput_vars, but must be equal"
    assert len(op_node.input_vars) == len(op_node.seqids), f"number of input variables ({len(op_node.input_vars)}) not equal to number of sequences to decode ({len(op_node.seqids)})"

    seqs_2_decode, original_positions = self.seq_info.order_seqs(op_node.seqids)
    outputs = {sid:[op_node.input_vars[original_positions[i]].var] for i, sid in enumerate(seqs_2_decode)}
    original_ordered_seqs_2_decode = seqs_2_decode.copy()

    while seqs_2_decode:
      tokens_tvm = tvm.nd.array(np.array([[outputs[sid][-1]] for sid in seqs_2_decode]).astype("int32"), device=self.dev)
      hidden_states = self.vm[f"embed"](tokens_tvm, self.params)

      method_name, param_list = self.seq_info.get_decode_method(seqs_2_decode)
      param_list = [tvm.nd.array(np.asarray(ids, dtype="int32"), device=self.dev) for ids in param_list]
      if self.measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
      tvm.get_global_func("vm.builtin.kv_state_begin_forward")(
        self.kv_cache, 
        ShapeTuple(seqs_2_decode), 
        ShapeTuple([1] * len(seqs_2_decode))
      )
      logits, self.kv_cache = self.vm[method_name](hidden_states, self.kv_cache, *param_list, self.params)
      tvm.get_global_func("vm.builtin.kv_state_end_forward")(self.kv_cache)
      if self.measure:
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        self.measure_file.write(f"{time_ms}\n")

      logits_np = logits.numpy()

      for i, sid in enumerate(seqs_2_decode):
        outputs[sid].append(np.argmax(logits_np[i]))
      seqs_2_decode = self.check_seqs_finished(seqs_2_decode, outputs)
    
    for i, sid in enumerate(original_ordered_seqs_2_decode):
      op_node.output_vars[original_positions[i]].var = torch.tensor(outputs[sid])

  def execute_decode_answer(self, op_node):
    assert len(op_node.input_vars) == 1, f"decode_answer node must have exactly 1 input var but has {len(op_node.input_vars)}"
    assert len(op_node.output_vars) == 1, f"decode_answer node must have exactly 1 output var but has {len(op_node.output_vars)}"
    assert len(op_node.seqids) == 1, f"decode_answer node must have exactly 1 seqid but has {len(op_node.seqids)}"
    op_node.output_vars[0].var = self.tokenizer.decode(op_node.input_vars[0].var)

  def execute_print(self, op_node):
    if self.measure:
      return
    assert len(op_node.input_vars) == 1, f"print node must have exactly 1 input var but has {len(op_node.input_vars)}"
    assert len(op_node.seqids) == 1, f"print node must have exactly 1 seqid but has {len(op_node.seqids)}"
    print(f"Answer for Sequence {op_node.seqids[0]} {op_node.input_vars[0].var}")

  def execute_op(self, op_node):
    if op_node.op_type == "encode_prompt":
      # print(f"encode prompt: {[v.id for v in op_node.input_vars]}")
      self.execute_encode_prompt(op_node)
    elif op_node.op_type == "prefill":
      # print(f"prefill: {[v.id for v in op_node.input_vars]}")
      self.execute_prefill(op_node)
    elif op_node.op_type == "decode":
      # print(f"decode: {[v.id for v in op_node.input_vars]}")
      self.execute_decode(op_node)
    elif op_node.op_type == "decode_answer":
      # print(f"decode_answer: {[v.id for v in op_node.input_vars]}")
      self.execute_decode_answer(op_node)
    elif op_node.op_type == "print":
      # print(f"print: {[v.id for v in op_node.input_vars]}")
      self.execute_print(op_node)
    else:
      print(f"[ERROR] unknown op_type: {op_node.op_type}")

