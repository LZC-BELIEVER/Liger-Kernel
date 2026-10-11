# AOT ID: ['1_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._inductor import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_grpo_20261011_pre_default_dev2/se/cseo4dsiwfj6fib2aub3jll4bzxa7ybahbiah3rs42h65prfvz6q.py
# Topologically Sorted Source Nodes: [sub_1, exp_1, log_ratio, coef_1, clipped_coef, unsqueeze, per_token_loss1, per_token_loss2], Original ATen: [aten.div, aten.expand, aten.mul, aten.add, aten.neg, aten.sub, aten.exp, aten.clamp, aten.unsqueeze, aten.eq, aten.where, aten.lt, aten.masked_fill, aten.gt, aten.ge, aten.le, aten.logical_and]
# Source node to ATen node mapping:
#   clipped_coef => clamp_max, clamp_min
#   coef_1 => exp
#   exp_1 => exp_1
#   log_ratio => sub
#   per_token_loss1 => mul
#   per_token_loss2 => mul_1
#   sub_1 => sub_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %primals_2 : Tensor "f32[1, 1024][1024, 1]npu:0" = PlaceHolder[target=primals_2]
#   %primals_4 : Tensor "bf16[1][1]npu:0" = PlaceHolder[target=primals_4]
#   %tangents_1 : Tensor "f32[][]npu:0" = PlaceHolder[target=tangents_1]
#   %clamp_min_1 : Tensor "f32[][]npu:0" = PlaceHolder[target=clamp_min_1]
#   %primals_5 : Tensor "f32[1, 1024][1024, 1]npu:0" = PlaceHolder[target=primals_5]
#   %tangents_2 : Tensor "f32[][]npu:0" = PlaceHolder[target=tangents_2]
#   %primals_3 : Tensor "f32[1, 1024][1024, 1]npu:0" = PlaceHolder[target=primals_3]
#   %where : Tensor "f32[1, 1024][1024, 1]npu:0" = PlaceHolder[target=where]
#   %div_3 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_2, %clamp_min_1), kwargs = {})
#   %expand : Tensor "f32[1, 1024][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%div_3, [1, 1024]), kwargs = {})
#   %mul_6 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_5), kwargs = {})
#   %div_4 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %clamp_min_1), kwargs = {})
#   %expand_1 : Tensor "f32[1, 1024][1, 0]npu:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%div_4, [1, 1024]), kwargs = {})
#   %mul_7 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, %primals_5), kwargs = {})
#   %mul_8 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 0.1), kwargs = {})
#   %add_1 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_8), kwargs = {})
#   %neg_1 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_1,), kwargs = {})
#   %neg_2 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%neg_1,), kwargs = {})
#   %sub_1 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %primals_2), kwargs = {})
#   %exp_1 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %mul_9 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %exp_1), kwargs = {})
#   %neg_3 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_9,), kwargs = {})
#   %add_2 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg_2, %neg_3), kwargs = {})
#   %neg_4 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_7,), kwargs = {})
#   %sub : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_2), kwargs = {})
#   %exp : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=5] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %clamp_min : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%exp, 0.8), kwargs = {})
#   %clamp_max : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.2), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 1][1, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_4, 1), kwargs = {})
#   %mul : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %unsqueeze), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, %unsqueeze), kwargs = {})
#   %eq : Tensor "b8[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %div_5 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%neg_4, 2), kwargs = {})
#   %where : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %div_5, %neg_4), kwargs = {})
#   %lt_2 : Tensor "b8[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %full_default : Tensor "f32[][]npu:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where_1 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_2, %full_default, %where), kwargs = {})
#   %gt_2 : Tensor "b8[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %where_3 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %full_default, %where), kwargs = {})
#   %mul_10 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %unsqueeze), kwargs = {})
#   %mul_11 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_3, %unsqueeze), kwargs = {})
#   %ge : Tensor "b8[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%exp, 0.8), kwargs = {})
#   %le : Tensor "b8[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%exp, 1.2), kwargs = {})
#   %logical_and : Tensor "b8[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%ge, %le), kwargs = {})
#   %where_4 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %mul_10, %full_default), kwargs = {})
#   %add_3 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %where_4), kwargs = {})
#   %mul_12 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %exp), kwargs = {})
#   %add_4 : Tensor "f32[1, 1024][1024, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_12), kwargs = {})
#   return %where,%add_4
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1')]

triton_unk_fused_add_clamp_div_eq_exp_expand_ge_gt_le_logical_and_lt_masked_fill_mul_neg_sub_unsqueeze_where_0 = async_compile.triton('triton_unk_fused_add_clamp_div_eq_exp_expand_ge_gt_le_logical_and_lt_masked_fill_mul_neg_sub_unsqueeze_where_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9382', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_clamp_div_eq_exp_expand_ge_gt_le_logical_and_lt_masked_fill_mul_neg_sub_unsqueeze_where_0', 'mutated_arg_names': ['in_out_ptr0'], 'backend_hash': '86e6f154485ae632a969d0568c37a91230b2a250bf8abf7d8e261c2c2553f087', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_add_clamp_div_eq_exp_expand_ge_gt_le_logical_and_lt_masked_fill_mul_neg_sub_unsqueeze_where_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp3 = tl.load(in_ptr1 + tl.arange(0,1) +  (0)).to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [1 ])
        tmp13 = tl.load(in_ptr2 + tl.arange(0,1) +  (0))
        tmp14 = tl.broadcast_to(tmp13, [1 ])
        tmp15 = tl.load(in_ptr3 + tl.arange(0,1) +  (0))
        tmp16 = tl.broadcast_to(tmp15, [1 ])
        tmp18 = tl.load(in_ptr4 + (x0), x0_mask)
        tmp24 = tl.load(in_ptr5 + tl.arange(0,1) +  (0))
        tmp25 = tl.broadcast_to(tmp24, [1 ])
        tmp33 = tl.load(in_ptr6 + (x0), x0_mask)
        tmp1 = tmp0 - tmp0
        tmp2 = tl_math.exp(tmp1)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp2 * tmp5
        tmp7 = 0.8
        tmp8 = triton_helpers.maximum(tmp2, tmp7)
        tmp9 = 1.2
        tmp10 = triton_helpers.minimum(tmp8, tmp9)
        tmp11 = tmp10 * tmp5
        tmp12 = tmp6 == tmp11
        tmp17 = (tmp14 / tmp16)
        tmp19 = tmp17 * tmp18
        tmp20 = -tmp19
        tmp21 = 0.5
        tmp22 = tmp20 * tmp21
        tmp23 = tl.where(tmp12, tmp22, tmp20)
        tmp26 = (tmp25 / tmp16)
        tmp27 = tmp26 * tmp18
        tmp28 = 0.1
        tmp29 = tmp19 * tmp28
        tmp30 = tmp27 + tmp29
        tmp31 = -tmp30
        tmp32 = -tmp31
        tmp34 = tmp33 - tmp0
        tmp35 = tl_math.exp(tmp34)
        tmp36 = tmp30 * tmp35
        tmp37 = -tmp36
        tmp38 = tmp32 + tmp37
        tmp39 = tmp6 > tmp11
        tmp40 = 0.0
        tmp41 = tl.where(tmp39, tmp40, tmp23)
        tmp42 = tmp41 * tmp5
        tmp43 = tmp2 >= tmp7
        tmp44 = tmp2 <= tmp9
        tmp45 = tmp43 & tmp44
        tmp46 = tmp6 < tmp11
        tmp47 = tl.where(tmp46, tmp40, tmp23)
        tmp48 = tmp47 * tmp5
        tmp49 = tl.where(tmp45, tmp48, tmp40)
        tmp50 = tmp42 + tmp49
        tmp51 = tmp50 * tmp2
        tmp52 = tmp38 + tmp51
        tl.store(in_out_ptr0 + (x0), tmp52, x0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_2, primals_3, primals_4, primals_5, clamp_min_1, tangents_1, tangents_2 = args
        args.clear()
        buf0 = empty_strided((1, 1024), (1024, 1), device='npu', dtype=torch.float32)
        buf1 = buf0;   # reuse
        # Topologically Sorted Source Nodes: [sub_1, exp_1, log_ratio, coef_1, clipped_coef, unsqueeze, per_token_loss1, per_token_loss2], Original ATen: [aten.div, aten.expand, aten.mul, aten.add, aten.neg, aten.sub, aten.exp, aten.clamp, aten.unsqueeze, aten.eq, aten.where, aten.lt, aten.masked_fill, aten.gt, aten.ge, aten.le, aten.logical_and]
        stream0 = get_raw_stream(0)
        triton_unk_fused_add_clamp_div_eq_exp_expand_ge_gt_le_logical_and_lt_masked_fill_mul_neg_sub_unsqueeze_where_0.run(buf1, primals_2, primals_4, tangents_1, clamp_min_1, primals_5, tangents_2, primals_3, 1024, stream=stream0)







        return (None, buf1, None, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((1, 1024), (1024, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 1024), (1024, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (1, ), device='npu:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((1, 1024), (1024, 1), device='npu:0', dtype=torch.float32)
    clamp_min_1 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    tangents_2 = rand_strided((), (), device='npu:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_3, primals_4, primals_5, clamp_min_1, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
