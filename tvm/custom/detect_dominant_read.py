from tvm import ir, tir
from tvm.dlight.base import collect_block_iter_vars_used_in_access_region

def detect_dominant_read(block: tir.Block) -> tir.PrimExpr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for buffer_region in block.reads:
        tir_vars = collect_block_iter_vars_used_in_access_region(block, buffer_region.region)
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = buffer_region
    assert dominant_read is not None
    
    if block.name_hint == "take_matmul":
      vars = []
      for i, e in enumerate(dominant_read.region):
        if i == 0:
            vars.extend(e.min.indices)
        else:
            vars.append(e.min)
    else:
        vars = [e.min for e in dominant_read.region]
    (result,) = dominant_read.buffer.offset_of(vars)

    return result