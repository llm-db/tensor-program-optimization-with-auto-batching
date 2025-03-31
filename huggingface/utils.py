import torch

# add timing hooks
def add_model_hooks(model: torch.nn.Module, measure_file):

  def start_time_hook(module, input):
    module.__start_event__ = torch.cuda.Event(enable_timing=True)
    module.__end_event__ = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    module.__start_event__.record()

  def end_time_hook(module, input, output):
    module.__end_event__.record()
    torch.cuda.synchronize()
    time_ms = module.__start_event__.elapsed_time(module.__end_event__)
    module.__measurements_file__.write(f"{time_ms}\n")

  if not hasattr(model, '__measurements_file__'):
    model.__measurements_file__ = measure_file

  if not hasattr(model, '__start_time_hook_handle'):
    model.__start_time_hook_handle__ = model.register_forward_pre_hook(
      start_time_hook, )

  if not hasattr(model, '__end_time_hook_handle__'):
    model.__end_time_hook_handle__ = model.register_forward_hook(
      end_time_hook, )

# remove timing hooks
def remove_model_hooks(module):
  if hasattr(module, "__measurements_file__"):
    module.__measurements_file__.close()
    del module.__measurements_file__
  if hasattr(module, "__start_event__"):
    del module.__start_event__
  if hasattr(module, "__end_event__"):
    del module.__end_event__
  if hasattr(module, "__start_time_hook_handle__"):
    module.__start_time_hook_handle__.remove()
    del module.__start_time_hook_handle__
  if hasattr(module, "__end_time_hook_handle__"):
    module.__end_time_hook_handle__.remove()
    del module.__end_time_hook_handle__
