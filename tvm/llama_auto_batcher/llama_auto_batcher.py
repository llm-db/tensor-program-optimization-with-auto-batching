from compile_tvm import compile_tvm
from llama_exec_graph import LlamaExecGraph, LlamaExecConfig
from llama_seq_info import LlamaSeqInfo
from llama_for_causal_lm import LlamaConfig

from transformers import AutoTokenizer
import os

import tvm

class LlamaAutoBatcherClass:
  def __init__(self):
    self.graph = LlamaExecGraph()
    self.current = {'sid': -1, 'generated': True}

    # initialize with defaults. can be changed with functions below
    self.device = tvm.device("cuda", 0)
    self.model_config = LlamaConfig()
    self.opt_config_path = f"{os.path.expanduser('~')}/fineinfer-autopeft/tvm/opt_configs/default.json"
    self.cache_dir = "/scratch/lucastr"
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.name, padding_side='left', cache_dir=self.cache_dir)
    self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
    self.exec_config = LlamaExecConfig(max_prompt_len=512, max_gen_len=32, page_size=16)
    self.seq_info = LlamaSeqInfo(self.exec_config.max_gen_len)
    self.warmup_trials = 3
    self.measurement_trials = 0
    self.measurements_dest = f"{os.path.expanduser('~')}/fineinfer-autopeft/measure.txt"
  
  def set_device(self, device):
    self.device = device

  def set_model_config(self, model_config):
    self.model_config = model_config

  def set_opt_config_path(self, opt_config_path):
    self.opt_config_path = f"{os.path.expanduser('~')}/{opt_config_path}"

  def set_cache_dir(self, cache_dir):
    self.cache_dir = cache_dir

  def set_tokenizer(self, tokenizer):
    self.tokenizer = tokenizer

  def set_exec_config(self, exec_config):
    self.seq_info.update_max_gen_len(exec_config.max_gen_len)
    self.exec_config = exec_config

  def set_measurement_vars(self, trials, dest, warmup_trials=3):
    self.measurement_trials = trials
    self.measurements_dest = dest
    self.warmup_trials = warmup_trials

  # call graph class's method
  def create_var(self, var, source=True):
    return self.graph.variable(var, source)

  # add prefill and decode node to graph
  def generate(self, input_var, max_gen_len=None):
    # add sequence info
    if self.current['generated']:
      self.current['sid'] += 1
      self.seq_info.add_sequence(self.current['sid'])
    self.current['generated'] = True
    
    # set max gen len
    self.seq_info.set_max_gen_len(self.current['sid'], max_gen_len)

    # encode prompt
    encode_prompt_result = self.create_var(None, False)
    self.graph.operation(input_var, "encode_prompt", seqid=self.current['sid'], output_var=encode_prompt_result)

    # prefill
    prefill_result = self.create_var(None, False)
    self.graph.operation(encode_prompt_result, "prefill", seqid=self.current['sid'], output_var=prefill_result)

    # decode
    decode_result = self.create_var(None, False)
    self.graph.operation(prefill_result, "decode", seqid=self.current['sid'], output_var=decode_result)

    # decode answer
    decode_answer_result = self.create_var(None, False)
    self.graph.operation(decode_result, "decode_answer", seqid=self.current['sid'], output_var=decode_answer_result)

    return decode_answer_result
    
  # add print node to graph
  def print(self, var):
    self.graph.operation(var, "print", seqid=self.current['sid'])

  # config is constructed by user
  def set_adapter(self, config, wid):
    if self.current['generated']:
      self.current['sid'] += 1
      self.seq_info.add_sequence(self.current['sid'])
      self.current['generated'] = False
    self.seq_info.set_adapter(self.current['sid'], config, wid)

  def compile(self, fuse_decode=True):
    # operator fusion function to fuse decodes (multi-source bfs)
    if fuse_decode:
      self.graph.fuse_batchable_ops()

    vm, model_params = compile_tvm(
      (self.graph.get_max_batch_sizes()["decode"] > 1),
      self.seq_info,
      self.device,
      self.model_config,
      self.opt_config_path,
      self.cache_dir
    )

    # set execution variables
    self.graph.set_exec_vars(
      vm, 
      self.device, 
      model_params, 
      self.tokenizer, 
      self.seq_info,
      self.exec_config
    )

  def execute(self):
    if self.measurement_trials:
      # warmup
      for i in range(self.warmup_trials):
        if i != 0:
          self.graph.set_kv_cache()
        self.graph.execute()

      # measurements
      self.graph.set_measuring(self.measurements_dest)
      for _ in range(self.measurement_trials):
        self.graph.set_kv_cache()
        self.graph.execute()
    else:
      self.graph.execute()
  
LlamaAutoBatcher = LlamaAutoBatcherClass()
