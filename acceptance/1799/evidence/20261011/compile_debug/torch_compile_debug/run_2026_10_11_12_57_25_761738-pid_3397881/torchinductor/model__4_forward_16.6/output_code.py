# AOT ID: ['4_forward']
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


# kernel path: /tmp/torchinductor_grpo_20261011_pytest_dev2/6c/c6csvwsib5w6sudt22nmuicjooe25hoku7eescrhwuwj6thokeum.py
# Topologically Sorted Source Nodes: [log_ratio, coef_1, is_lower_clipped, is_upper_clipped, unsqueeze, lt_1, and_, gt_1, and__1], Original ATen: [aten.sub, aten.exp, aten.lt, aten.gt, aten.unsqueeze, aten.bitwise_and]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   and__1 => bitwise_and_1
#   coef_1 => exp
#   gt_1 => gt_1
#   is_lower_clipped => lt
#   is_upper_clipped => gt
#   log_ratio => sub
#   lt_1 => lt_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %primals_2 : Tensor "f32[1, 47][47, 1]npu:0" = PlaceHolder[target=primals_2]
#   %primals_4 : Tensor "f32[1][1]npu:0" = PlaceHolder[target=primals_4]
#   %sub : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_2), kwargs = {})
#   %exp : Tensor "f32[1, 47][47, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %lt : Tensor "b8[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%exp, 0.8), kwargs = {})
#   %gt : Tensor "b8[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%exp, 1.2), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 1][1, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_4, 1), kwargs = {})
#   %lt_1 : Tensor "b8[1, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%unsqueeze, 0), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt, %lt_1), kwargs = {})
#   %gt_1 : Tensor "b8[1, 1][1, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%unsqueeze, 0), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %gt_1), kwargs = {})
#   return %bitwise_and,%bitwise_and_1
# SchedulerNodes: [SchedulerNode(name='op4'), SchedulerNode(name='op5')]

triton_unk_fused_bitwise_and_exp_gt_lt_sub_unsqueeze_0 = async_compile.triton('triton_unk_fused_bitwise_and_exp_gt_lt_sub_unsqueeze_0', '''
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
    size_hints=[47], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9382', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_bitwise_and_exp_gt_lt_sub_unsqueeze_0', 'mutated_arg_names': [], 'backend_hash': '86e6f154485ae632a969d0568c37a91230b2a250bf8abf7d8e261c2c2553f087', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bool, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_bitwise_and_exp_gt_lt_sub_unsqueeze_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp5 = tl.load(in_ptr1 + tl.arange(0,1) +  (0))
        tmp6 = tl.broadcast_to(tmp5, [1 ])
        tmp1 = tmp0 - tmp0
        tmp2 = tl_math.exp(tmp1)
        tmp3 = 0.8
        tmp4 = tmp2 < tmp3
        tmp7 = 0.0
        tmp8 = tmp6 < tmp7
        tmp9 = tmp4 & tmp8
        tmp10 = 1.2
        tmp11 = tmp2 > tmp10
        tmp12 = tmp6 > tmp7
        tmp13 = tmp11 & tmp12
        tl.store(out_ptr0 + (x0), tmp9, x0_mask)
        tl.store(out_ptr1 + (x0), tmp13, x0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_grpo_20261011_pytest_dev2/lt/clthdzucl4obyk4cznoyyh5sh7vkyeyhgpwtlyam55fi7w6nfweh.py
# Topologically Sorted Source Nodes: [log_ratio, coef_1, clipped_coef, unsqueeze, per_token_loss1, per_token_loss2, min_1, per_token_loss, sub_1, exp_1, sub_3, kl_div, mul_2, per_token_loss_1, mul_3, sum_1, mul_4, sum_3, mul_5, sum_5], Original ATen: [aten.sub, aten.exp, aten.clamp, aten.unsqueeze, aten.mul, aten.minimum, aten.neg, aten.add, aten.sum]
# Source node to ATen node mapping:
#   clipped_coef => clamp_max, clamp_min
#   coef_1 => exp
#   exp_1 => exp_1
#   kl_div => sub_4
#   log_ratio => sub
#   min_1 => minimum
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   per_token_loss => neg
#   per_token_loss1 => mul
#   per_token_loss2 => mul_1
#   per_token_loss_1 => add
#   sub_1 => sub_1
#   sub_3 => sub_3
#   sum_1 => sum_1
#   sum_3 => sum_3
#   sum_5 => sum_5
#   unsqueeze => unsqueeze
# Graph fragment:
#   %primals_2 : Tensor "f32[1, 47][47, 1]npu:0" = PlaceHolder[target=primals_2]
#   %primals_4 : Tensor "f32[1][1]npu:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "f32[1, 47][47, 1]npu:0" = PlaceHolder[target=primals_3]
#   %primals_5 : Tensor "f32[1, 47][47, 1]npu:0" = PlaceHolder[target=primals_5]
#   %bitwise_or : Tensor "b8[1, 47][47, 1]npu:0" = PlaceHolder[target=bitwise_or]
#   %sub : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_2), kwargs = {})
#   %exp : Tensor "f32[1, 47][47, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %clamp_min : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%exp, 0.8), kwargs = {})
#   %clamp_max : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.2), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 1][1, 1]npu:0"[num_users=4] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_4, 1), kwargs = {})
#   %mul : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %unsqueeze), kwargs = {})
#   %mul_1 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, %unsqueeze), kwargs = {})
#   %minimum : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%mul, %mul_1), kwargs = {})
#   %neg : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum,), kwargs = {})
#   %sub_1 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %primals_2), kwargs = {})
#   %exp_1 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sub_3 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%exp_1, %sub_1), kwargs = {})
#   %sub_4 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, 1.0), kwargs = {})
#   %mul_2 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, 0.04), kwargs = {})
#   %add : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%neg, %mul_2), kwargs = {})
#   %mul_3 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %primals_5), kwargs = {})
#   %sum_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_3,), kwargs = {})
#   %mul_4 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %primals_5), kwargs = {})
#   %sum_3 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : Tensor "f32[1, 47][47, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bitwise_or, %primals_5), kwargs = {})
#   %sum_5 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_5,), kwargs = {})
#   return %sum_1,%sum_3,%sum_5
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op3'), SchedulerNode(name='op8')]

triton_unk_fused_add_clamp_exp_minimum_mul_neg_sub_sum_unsqueeze_1 = async_compile.triton('triton_unk_fused_add_clamp_exp_minimum_mul_neg_sub_sum_unsqueeze_1', '''
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

@npu_triton_heuristics.persistent_reduction_npu_index(
    size_hints=[47],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'r0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9382', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_add_clamp_exp_minimum_mul_neg_sub_sum_unsqueeze_1', 'mutated_arg_names': [], 'backend_hash': '86e6f154485ae632a969d0568c37a91230b2a250bf8abf7d8e261c2c2553f087', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['r0'], 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False}
)
@triton.jit
def triton_unk_fused_add_clamp_exp_minimum_mul_neg_sub_sum_unsqueeze_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, r0_numel, R0BLOCK : tl.constexpr):
    R0BLOCK_SUB: tl.constexpr = 47
    r0 = tl.arange(0, R0BLOCK_SUB)
    r0_mask = r0 < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + tl.arange(0,1) +  (0))
    tmp4 = tl.broadcast_to(tmp3, [1 ])
    tmp13 = tl.load(in_ptr2 + (r0), r0_mask, other=0.0)
    tmp22 = tl.load(in_ptr3 + (r0), r0_mask, other=0.0)
    tmp33 = tl.load(in_ptr4 + (r0), r0_mask).to(tl.int1)
    tmp1 = tmp0 - tmp0
    tmp2 = tl_math.exp(tmp1)
    tmp5 = tmp2 * tmp4
    tmp6 = 0.8
    tmp7 = triton_helpers.maximum(tmp2, tmp6)
    tmp8 = 1.2
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 * tmp4
    tmp11 = triton_helpers.minimum(tmp5, tmp10)
    tmp12 = -tmp11
    tmp14 = tmp13 - tmp0
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp15 - tmp14
    tmp17 = 1.0
    tmp18 = tmp16 - tmp17
    tmp19 = 0.04
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 + tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tl.reshape(tmp23, [R0BLOCK_SUB])
    tmp26 = tl.where(r0_mask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp18 * tmp22
    tmp29 = tl.reshape(tmp28, [R0BLOCK_SUB])
    tmp31 = tl.where(r0_mask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp22
    tmp36 = tl.reshape(tmp35, [R0BLOCK_SUB])
    tmp38 = tl.where(r0_mask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tl.store(out_ptr0 + (tl.arange(0,1) ), tmp27, None)
    tl.store(out_ptr1 + (tl.arange(0,1) ), tmp32, None)
    tl.store(out_ptr2 + (tl.arange(0,1) ), tmp39, None)
''', device_str='npu')


# kernel path: /tmp/torchinductor_grpo_20261011_pytest_dev2/w3/cw3tpe7mgvr2vc6wzm4bbe3frbb4vjcwq57qfjnxeehmabwfgg2f.py
# Topologically Sorted Source Nodes: [sum_2, clamp_1, loss, truediv_1, truediv_2], Original ATen: [aten.sum, aten.clamp, aten.div]
# Source node to ATen node mapping:
#   clamp_1 => clamp_min_1
#   loss => div
#   sum_2 => sum_2
#   truediv_1 => div_1
#   truediv_2 => div_2
# Graph fragment:
#   %primals_1 : Tensor "f32[3, 47][47, 1]npu:0" = PlaceHolder[target=primals_1]
#   %sum_2 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_2]
#   %sum_1 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_1]
#   %clamp_min_1 : Tensor "f32[][]npu:0" = PlaceHolder[target=clamp_min_1]
#   %sum_3 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_3]
#   %sum_5 : Tensor "f32[][]npu:0" = PlaceHolder[target=sum_5]
#   %sum_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%primals_1,), kwargs = {})
#   %clamp_min_1 : Tensor "f32[][]npu:0"[num_users=4] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sum_2, 1.0), kwargs = {})
#   %div : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %clamp_min_1), kwargs = {})
#   %div_1 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_3, %clamp_min_1), kwargs = {})
#   %div_2 : Tensor "f32[][]npu:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_5, %clamp_min_1), kwargs = {})
#   return %sum_2,%clamp_min_1,%div,%div_1,%div_2
# SchedulerNodes: [SchedulerNode(name='op1'), SchedulerNode(name='op2'), SchedulerNode(name='op9'), SchedulerNode(name='op10'), SchedulerNode(name='op11')]

triton_unk_fused_clamp_div_sum_2 = async_compile.triton('triton_unk_fused_clamp_div_sum_2', '''
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

@npu_triton_heuristics.persistent_reduction_npu_index(
    size_hints=[141],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'r0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9382', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clamp_div_sum_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'backend_hash': '86e6f154485ae632a969d0568c37a91230b2a250bf8abf7d8e261c2c2553f087', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['r0'], 'low_dims': {0}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False}
)
@triton.jit
def triton_unk_fused_clamp_div_sum_2(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, r0_numel, R0BLOCK : tl.constexpr):
    R0BLOCK_SUB: tl.constexpr = 141
    r0 = tl.arange(0, R0BLOCK_SUB)
    r0_mask = r0 < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
    tmp7 = tl.load(in_out_ptr1 + tl.arange(0,1) +  (0))
    tmp8 = tl.broadcast_to(tmp7, [1 ])
    tmp10 = tl.load(in_out_ptr2 + tl.arange(0,1) +  (0))
    tmp11 = tl.broadcast_to(tmp10, [1 ])
    tmp13 = tl.load(in_out_ptr3 + tl.arange(0,1) +  (0))
    tmp14 = tl.broadcast_to(tmp13, [1 ])
    tmp1 = tl.reshape(tmp0, [R0BLOCK_SUB])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp5 = 1.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp9 = (tmp8 / tmp6)
    tmp12 = (tmp11 / tmp6)
    tmp15 = (tmp14 / tmp6)
    tl.store(in_out_ptr0 + (tl.arange(0,1)), tmp6, None)
    tl.store(in_out_ptr1 + (tl.arange(0,1)), tmp9, None)
    tl.store(in_out_ptr2 + (tl.arange(0,1)), tmp12, None)
    tl.store(in_out_ptr3 + (tl.arange(0,1)), tmp15, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        buf4 = empty_strided((1, 47), (47, 1), device='npu', dtype=torch.bool)
        buf5 = empty_strided((1, 47), (47, 1), device='npu', dtype=torch.bool)
        # Topologically Sorted Source Nodes: [log_ratio, coef_1, is_lower_clipped, is_upper_clipped, unsqueeze, lt_1, and_, gt_1, and__1], Original ATen: [aten.sub, aten.exp, aten.lt, aten.gt, aten.unsqueeze, aten.bitwise_and]
        stream0 = get_raw_stream(0)
        triton_unk_fused_bitwise_and_exp_gt_lt_sub_unsqueeze_0.run(primals_2, primals_4, buf4, buf5, 47, stream=stream0)
        # Topologically Sorted Source Nodes: [log_ratio, coef_1, is_lower_clipped, is_upper_clipped, unsqueeze, lt_1, and_, gt_1, and__1, is_clipped], Original ATen: [aten.sub, aten.exp, aten.lt, aten.gt, aten.unsqueeze, aten.bitwise_and, aten.bitwise_or]
        buf6 = torch.ops.aten.bitwise_or.Tensor(buf4, buf5)


        buf7 = buf6
        assert_size_stride(buf7, (1, 47), (47, 1), 'torch.ops.aten.bitwise_or.Tensor')
        assert_alignment(buf7, 16, 'torch.ops.aten.bitwise_or.Tensor')

        buf0 = empty_strided((), (), device='npu', dtype=torch.float32)
        buf3 = empty_strided((), (), device='npu', dtype=torch.float32)
        buf8 = empty_strided((), (), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [log_ratio, coef_1, clipped_coef, unsqueeze, per_token_loss1, per_token_loss2, min_1, per_token_loss, sub_1, exp_1, sub_3, kl_div, mul_2, per_token_loss_1, mul_3, sum_1, mul_4, sum_3, mul_5, sum_5], Original ATen: [aten.sub, aten.exp, aten.clamp, aten.unsqueeze, aten.mul, aten.minimum, aten.neg, aten.add, aten.sum]
        stream0 = get_raw_stream(0)
        triton_unk_fused_add_clamp_exp_minimum_mul_neg_sub_sum_unsqueeze_1.run(primals_2, primals_4, primals_3, primals_5, buf7, buf0, buf3, buf8, 47, stream=stream0)

        buf1 = empty_strided((), (), device='npu', dtype=torch.float32)
        buf2 = buf1;   # reuse
        buf9 = buf0;   # reuse
        buf10 = buf3;   # reuse
        buf11 = buf8;   # reuse
        # Topologically Sorted Source Nodes: [sum_2, clamp_1, loss, truediv_1, truediv_2], Original ATen: [aten.sum, aten.clamp, aten.div]
        stream0 = get_raw_stream(0)
        triton_unk_fused_clamp_div_sum_2.run(buf2, buf9, buf10, buf11, primals_1, 141, stream=stream0)

        return (buf9, buf10, buf11, primals_2, primals_3, primals_4, primals_5, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3, 47), (47, 1), device='npu:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 47), (47, 1), device='npu:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 47), (47, 1), device='npu:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (1, ), device='npu:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 47), (47, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
