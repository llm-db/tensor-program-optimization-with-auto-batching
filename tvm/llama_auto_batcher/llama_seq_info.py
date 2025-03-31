class LlamaSeqInfo():
  def __init__(self, max_gen_len):
    self.sid_2_info = {} # sid -> (adapter, max_gen_len)
    self.adapter_types = []
    self.max_gen_len = max_gen_len
  
  def add_sequence(self, sid):
    assert sid not in self.sid_2_info.keys(), f"sequence with id {sid} has already been added"
    self.sid_2_info[sid] = {"adapter": None, "max_gen_len": self.max_gen_len}

  def set_max_gen_len(self, sid, max_gen_len):
    if max_gen_len > self.max_gen_len:
      print(f"[WARN] you are trying to set a max_gen_len of {max_gen_len} for sequence {sid} which is bigger than the maximum {self.max_gen_len}. Setting max_gen_len to {self.max_gen_len}.")
      max_gen_len = self.max_gen_len
    self.sid_2_info[sid]["max_gen_len"] = max_gen_len

  def set_adapter(self, sid, config, wid):
    try:
      adapter_tid = self.adapter_types.index(config)
    except:
      adapter_tid = len(self.adapter_types)
      self.adapter_types.append(config)
    
    adapter_dict = {"tid": adapter_tid, "wid": wid}
    if self.sid_2_info[sid]["adapter"] is not None:
      print(f"[WARN] you are setting adapter {adapter_dict} for sequence {sid} but it has already been set to {self.sid_2_info[sid]["adapter"]}")
    self.sid_2_info[sid]["adapter"] = adapter_dict

  def get_adapter_configs(self):
    return self.adapter_types

  def get_num_adapters_per_type(self):
    wid_sets = [set() for _ in range(len(self.adapter_types))]
    for elem in self.sid_2_info.values():
      if elem["adapter"] == None:
        continue
      wid_sets[elem["adapter"]["tid"]].add(elem["adapter"]["wid"])
    out = [len(wid_set) for wid_set in wid_sets]
    return out

  def get_wids_per_type(self):
    out = [set() for _ in range(len(self.adapter_types))]
    for elem in self.sid_2_info.values():
      if elem["adapter"] == None:
        continue
      out[elem["adapter"]["tid"]].add(elem["adapter"]["wid"])
    return out

  def get_num_sequences(self):
    return len(self.sid_2_info)

  def update_max_gen_len(self, max_gen_len):
    self.max_gen_len = max_gen_len

  def get_max_gen_len(self, sid):
    return self.sid_2_info[sid]["max_gen_len"]

  def get_adapter_info(self, sid):
    return self.sid_2_info[sid]["adapter"]

  def order_seqs(self, sids):
    if len(self.adapter_types) == 0:
      return sids, [i for i in range(len(sids))]

    annotated_sids = [
      (i, sid, self.sid_2_info[sid]["adapter"], 
       self.sid_2_info[sid]["adapter"]["tid"] if self.sid_2_info[sid]["adapter"] else float('inf'), 
       self.sid_2_info[sid]["adapter"]["wid"] if self.sid_2_info[sid]["adapter"] else float('inf'))
      for i, sid in enumerate(sids)
    ]
    ordered = sorted(annotated_sids, key=lambda x: (x[2] is not None, x[3], x[4]))
    ordered_sid = [x[1] for x in ordered]
    original_positions = [x[0] for x in ordered]

    return ordered_sid, original_positions

  def get_decode_method(self, sids):
    seq_with_no_adapter = False
    adapter_sids = [[] for _ in range(len(self.adapter_types))]
    adapter_wids = [[] for _ in range(len(self.adapter_types))]
    for i, sid in enumerate(sids):
      if self.sid_2_info[sid]["adapter"] == None:
        if not seq_with_no_adapter:
          seq_with_no_adapter = True
        continue
      adapter_sids[self.sid_2_info[sid]["adapter"]["tid"]].append(i)
      adapter_wids[self.sid_2_info[sid]["adapter"]["tid"]].append(self.sid_2_info[sid]["adapter"]["wid"])
    non_empty_indices = [i for i, sublist in enumerate(adapter_sids) if sublist]
    if seq_with_no_adapter:
      if len(non_empty_indices) == 0:
        return "decode_no_adapter", []
      else:
        method_tag = "_".join(f"a{i}" for i in non_empty_indices)
        name = f"mixed_decode_{method_tag}"
        params = [item for pair in zip([adapter_sids[i] for i in non_empty_indices], [adapter_wids[i] for i in non_empty_indices]) for item in pair]
        return name, params
    else:
      if len(non_empty_indices) == 1:
        return f"decode_a{non_empty_indices[0]}", [adapter_wids[non_empty_indices[0]]]
      else:
        method_tag = "_".join(f"a{i}" for i in non_empty_indices)
        name = f"mixed_decode_{method_tag}"
        params = [item for pair in zip([adapter_sids[i] for i in non_empty_indices], [adapter_wids[i] for i in non_empty_indices]) for item in pair]
        return name, params


