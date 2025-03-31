from tvm.script import tir as T

@T.prim_func
def write_tensor_part(x: T.handle, y: T.handle, ids: T.handle, z: T.handle):
  B, P, H, A = T.int64(), T.int64(), T.int64(), T.int64()
  X = T.match_buffer(x, (B, P, H), "float16")
  Y = T.match_buffer(y, (A, P, H), "float16")
  I = T.match_buffer(ids, (A,), "int32")
  Z = T.match_buffer(z, (B, P, H), "float16")

  for b, p, h in T.grid(B, P, H):
    with T.block("init"):
      vb, vp, vh = T.axis.remap("SSS", [b, p, h])
      Z[vb, vp, vh] = X[vb, vp, vh]

  for a, p, h in T.grid(A, P, H):
    with T.block("write"):
      va, vp, vh = T.axis.remap("SSS", [a, p, h])
      vb = I[va]
      Z[vb, vp, vh] = Y[va, vp, vh]