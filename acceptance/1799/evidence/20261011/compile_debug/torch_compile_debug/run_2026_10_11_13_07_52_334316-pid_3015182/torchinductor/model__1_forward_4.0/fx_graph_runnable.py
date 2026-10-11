
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_grpo_20261011_pre_smoke_dev2'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_grpo_20261011_pre_smoke_dev2'
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.0+cpu
# torch cuda version: None
# torch git version: 0fabc3ba44823f257e70ce397d989c8de5e362c1


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
        sub = torch.ops.aten.sub.Tensor(primals_2, primals_2)
        exp = torch.ops.aten.exp.default(sub);  sub = None
        clamp_min = torch.ops.aten.clamp_min.default(exp, 0.8)
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 1.2);  clamp_min = None
        lt = torch.ops.aten.lt.Scalar(exp, 0.8)
        gt = torch.ops.aten.gt.Scalar(exp, 1.2)
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_4, 1)
        mul = torch.ops.aten.mul.Tensor(exp, unsqueeze);  exp = None
        mul_1 = torch.ops.aten.mul.Tensor(clamp_max, unsqueeze);  clamp_max = None
        minimum = torch.ops.aten.minimum.default(mul, mul_1);  mul = mul_1 = None
        neg = torch.ops.aten.neg.default(minimum);  minimum = None
        sub_1 = torch.ops.aten.sub.Tensor(primals_3, primals_2)
        exp_1 = torch.ops.aten.exp.default(sub_1)
        sub_3 = torch.ops.aten.sub.Tensor(exp_1, sub_1);  exp_1 = sub_1 = None
        sub_4 = torch.ops.aten.sub.Tensor(sub_3, 1.0);  sub_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_4, 0.1)
        add = torch.ops.aten.add.Tensor(neg, mul_2);  neg = mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(add, primals_5);  add = None
        sum_1 = torch.ops.aten.sum.default(mul_3);  mul_3 = None
        sum_2 = torch.ops.aten.sum.default(primals_1);  primals_1 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sum_2, 1.0);  sum_2 = None
        div = torch.ops.aten.div.Tensor(sum_1, clamp_min_1);  sum_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_4, primals_5);  sub_4 = None
        sum_3 = torch.ops.aten.sum.default(mul_4);  mul_4 = None
        div_1 = torch.ops.aten.div.Tensor(sum_3, clamp_min_1);  sum_3 = None
        lt_1 = torch.ops.aten.lt.Scalar(unsqueeze, 0)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(lt, lt_1);  lt = lt_1 = None
        gt_1 = torch.ops.aten.gt.Scalar(unsqueeze, 0);  unsqueeze = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(gt, gt_1);  gt = gt_1 = None
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(bitwise_and, bitwise_and_1);  bitwise_and = bitwise_and_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(bitwise_or, primals_5);  bitwise_or = None
        sum_5 = torch.ops.aten.sum.default(mul_5);  mul_5 = None
        div_2 = torch.ops.aten.div.Tensor(sum_5, clamp_min_1);  sum_5 = None
        return (div, div_1, div_2, primals_2, primals_3, primals_4, primals_5, clamp_min_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, device=device(type='npu', index=0))
    reader.tensor(buf0, (1, 1024), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4096, device=device(type='npu', index=0))
    reader.tensor(buf1, (1, 1024), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4096, device=device(type='npu', index=0))
    reader.tensor(buf2, (1, 1024), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 2, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (1,), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    reader.tensor(buf0, (1, 1024), is_leaf=True)  # primals_5
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)