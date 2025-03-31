import sys

class VarNode:
  def __init__(self, id, var):
    self.id = id
    self.var = var
    self.ops = []

class OpNode:
  def __init__(self, op_type):
    self.op_type = op_type
    self.input_vars = []
    self.output_vars = []
    self.seqids = []

class ExecGraph(object):
  def __init__(self, batchable_op_types):
    self.sources = []
    self.op_nodes = []
    self.batchable_op_types = batchable_op_types
    self.var_id = -1

  def add_source(self, var):
    self.sources.append(var)

  def variable(self, var, source):
    self.var_id += 1
    var_node = VarNode(self.var_id, var)
    if source:
      self.add_source(var_node)
    return var_node

  def operation(self, input_var, op_type, seqid=None, output_var=None):
    op_node = OpNode(op_type)
    input_var.ops.append(op_node)
    op_node.input_vars.append(input_var)
    if output_var:
      op_node.output_vars.append(output_var)
    if seqid is not None:
      op_node.seqids.append(seqid)
    self.op_nodes.append(op_node)
    return op_node

  def get_max_batch_sizes(self):
    batch_size_dict = {}
    for op in self.op_nodes:
      if op.op_type in self.batchable_op_types:
        if op.op_type in batch_size_dict.keys():
          if len(op.input_vars) > batch_size_dict[op.op_type]:
            batch_size_dict[op.op_type] = len(op.input_vars)
        else:
          batch_size_dict[op.op_type] = len(op.input_vars)
    return batch_size_dict

  def ms_bfs(self, f=lambda x: x):
    visit = set(self.sources)
    visit_next = set()
    while len(visit):
      for node in visit:
        if type(node) == VarNode:
          for child in node.ops:
            visit_next.add(child)
        elif type(node) == OpNode:
          for child in node.output_vars:
            visit_next.add(child)
        else:
          sys.exit(f"unknown node type {type(node)} - exiting")
      visit = f(visit_next)
      visit_next = set()

  def print(self):
    def print_f(node_set):
      print_newline = False
      for node in node_set:
        if type(node) == VarNode:
          print(f"Var(id={node.id})", end=" ")
          print_newline = True
        elif type(node) == OpNode:
          print(f"Op(type={node.op_type}, seqids={[id for id in node.seqids]}, in_ids={[var.id for var in node.input_vars]}, out_ids={[var.id for var in node.output_vars]})")
      if print_newline:
        print("")
      return node_set
    self.ms_bfs(print_f)

  def fuse_batchable_ops(self):
    def fuse_f(node_set):
      new_batchable_nodes = {}
      new_node_set = set()
      for node in node_set:
        # since all nodes in the set are of the same type, this happens
        # in the first iteration or not at all
        if type(node) == VarNode:
          return node_set
        if node.op_type in self.batchable_op_types:
          if node.op_type in new_batchable_nodes.keys():
            new_batchable_nodes[node.op_type].input_vars += node.input_vars
            new_batchable_nodes[node.op_type].output_vars += node.output_vars
            new_batchable_nodes[node.op_type].seqids += node.seqids
            for input_var in node.input_vars:
              input_var.ops[input_var.ops.index(node)] = new_batchable_nodes[node.op_type]
          else:
            new_batchable_nodes[node.op_type] = node
        else:
          new_node_set.add(node)
      new_node_set.update(set(new_batchable_nodes.values()))
      return new_node_set
    self.ms_bfs(fuse_f)

  def execute_op(self, op_node):
    pass

  def execute(self):
    def execute_f(node_set):
      for node in node_set:
        # since all nodes in the set are of the same type, this happens
        # in the first iteration or not at all
        if type(node) == VarNode:
          return node_set
        self.execute_op(node)
      return node_set
    self.ms_bfs(execute_f)
