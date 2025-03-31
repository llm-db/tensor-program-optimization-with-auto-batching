# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""A compiler pass that fuses take + matmul and generate TIR function.
Note that
1. Please put the pass before LegalizeOps pass.
2. The pass would rewrite the relax ops into TIR functions. If you'd like to dispatch the
   ops into library (e.g. cuBLAS) calls, please run dispatch pass before this pass.
"""

import tvm
from tvm import IRModule, relax, te, tir
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator


@tvm.transform.module_pass(opt_level=0, name="FuseTakeMatmul")
class FuseTakeMatmul:  # pylint: disable=too-few-public-methods
    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        mod = relax.transform.FuseOpsByPattern(
            [
                (
                    "take_matmul_fuse",
                    *_pattern(),
                ),
            ]
        )(mod)
        take_matmul_codegen = _TakeMatmulFuser(mod)
        for g_var, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = take_matmul_codegen.visit_expr(func)
                take_matmul_codegen.builder_.update_func(g_var, func)
        return take_matmul_codegen.builder_.get()


def _pattern():
    # pylint: disable=invalid-name
    weights = wildcard()
    indices = wildcard()
    x = wildcard()
    weights_taken = is_op("relax.take")(weights, indices)
    out = is_op("relax.matmul")(x, weights_taken)
    # pylint: enable=invalid-name
    annotations = {"out": out, "weights": weights, "indices": indices, "x": x, "weights_taken": weights_taken}

    return out, annotations


# pylint: disable=missing-docstring,invalid-name


@mutator
class _TakeMatmulFuser(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(  # pylint: disable=arguments-renamed
        self,
        call: relax.Call,
    ) -> relax.Expr:
        out_dtype = None

        def te_take_matmul(weights: te.Tensor, indices: te.Tensor, x: te.Tensor) -> te.Tensor:
            nonlocal out_dtype
            weights_shape = list(weights.shape)
            x_shape = list(x.shape)
            output_shape = x_shape.copy()
            output_shape[-1] = weights_shape[-1]

            def matmul_compute(*idx_spatial):
                k = te.reduce_axis((0, x_shape[-1]), name="k")

                def multiply_compute(idx_reduce):
                    weights_indices = []
                    x_indices = []

                    x_indices.append(idx_spatial[0])
                    x_indices.append(idx_spatial[1])
                    x_indices.append(idx_reduce)

                    idx = indices[idx_spatial[0]]
                    weights_indices.append(idx)
                    weights_indices.append(idx_reduce)
                    weights_indices.append(idx_spatial[2])

                    dtype = out_dtype
  
                    return x(*x_indices).astype(dtype) * weights(*weights_indices).astype(dtype)

                return te.sum(multiply_compute(k), axis=k)

            return te.compute(
                output_shape,
                lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
                name="take_matmul",
            )

        if isinstance(call.op, relax.GlobalVar):
            function = self.builder_.get()[call.op]
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "take_matmul_fuse"
            ):
                out_dtype = function.ret_struct_info.dtype
                return self.builder_.call_te(
                    te_take_matmul,
                    call.args[0],
                    call.args[1],
                    call.args[2],
                    primfunc_name_hint="take_matmul",
                )

        return super().visit_call_(call)
