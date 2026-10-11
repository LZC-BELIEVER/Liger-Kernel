
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_grpo_20261011_pytest_dev2'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_grpo_20261011_pytest_dev2'
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
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



isolate_fails_code_str = None




# torch version: 2.9.0+cpu
# torch cuda version: None
# torch git version: 0fabc3ba44823f257e70ce397d989c8de5e362c1


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_2, primals_3, primals_4, primals_5, clamp_min_1, tangents_1, tangents_2):
        div_3 = torch.ops.aten.div.Tensor(tangents_2, clamp_min_1);  tangents_2 = None
        expand = torch.ops.aten.expand.default(div_3, [1, 128]);  div_3 = None
        mul_6 = torch.ops.aten.mul.Tensor(expand, primals_5);  expand = None
        div_4 = torch.ops.aten.div.Tensor(tangents_1, clamp_min_1);  tangents_1 = clamp_min_1 = None
        expand_1 = torch.ops.aten.expand.default(div_4, [1, 128]);  div_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(expand_1, primals_5);  expand_1 = primals_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 0.04)
        add_1 = torch.ops.aten.add.Tensor(mul_6, mul_8);  mul_6 = mul_8 = None
        neg_1 = torch.ops.aten.neg.default(add_1)
        neg_2 = torch.ops.aten.neg.default(neg_1);  neg_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(primals_3, primals_2);  primals_3 = None
        exp_1 = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_1, exp_1);  add_1 = exp_1 = None
        neg_3 = torch.ops.aten.neg.default(mul_9);  mul_9 = None
        add_2 = torch.ops.aten.add.Tensor(neg_2, neg_3);  neg_2 = neg_3 = None
        neg_4 = torch.ops.aten.neg.default(mul_7);  mul_7 = None
        sub = torch.ops.aten.sub.Tensor(primals_2, primals_2);  primals_2 = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        clamp_min = torch.ops.aten.clamp_min.default(exp, 0.8)
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 1.2);  clamp_min = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_4, 1);  primals_4 = None
        mul = torch.ops.aten.mul.Tensor(exp, unsqueeze)
        mul_1 = torch.ops.aten.mul.Tensor(clamp_max, unsqueeze);  clamp_max = None
        eq = torch.ops.aten.eq.Tensor(mul, mul_1)
        div_5 = torch.ops.aten.div.Scalar(neg_4, 2)
        where = torch.ops.aten.where.self(eq, div_5, neg_4);  eq = div_5 = neg_4 = None
        lt_2 = torch.ops.aten.lt.Tensor(mul, mul_1)
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(lt_2, full_default, where);  lt_2 = None
        gt_2 = torch.ops.aten.gt.Tensor(mul, mul_1);  mul = mul_1 = None
        where_3 = torch.ops.aten.where.self(gt_2, full_default, where);  gt_2 = where = None
        mul_10 = torch.ops.aten.mul.Tensor(where_1, unsqueeze);  where_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(where_3, unsqueeze);  where_3 = unsqueeze = None
        ge = torch.ops.aten.ge.Scalar(exp, 0.8)
        le = torch.ops.aten.le.Scalar(exp, 1.2)
        logical_and = torch.ops.aten.logical_and.default(ge, le);  ge = le = None
        where_4 = torch.ops.aten.where.self(logical_and, mul_10, full_default);  logical_and = mul_10 = full_default = None
        add_3 = torch.ops.aten.add.Tensor(mul_11, where_4);  mul_11 = where_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(add_3, exp);  add_3 = exp = None
        add_4 = torch.ops.aten.add.Tensor(add_2, mul_12);  add_2 = mul_12 = None
        return (None, add_4, None, None, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 512, device=device(type='npu', index=0))
    reader.tensor(buf0, (1, 128), is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 512, device=device(type='npu', index=0))
    reader.tensor(buf1, (1, 128), is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 16, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (1,), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 4096, device=device(type='npu', index=0))
    reader.tensor(buf3, (1, 128), is_leaf=True)  # primals_5
    buf4 = reader.storage(None, 4, device=device(type='npu', index=0))
    reader.tensor(buf4, (), is_leaf=True)  # clamp_min_1
    buf5 = reader.storage(None, 4, device=device(type='npu', index=0))
    reader.tensor(buf5, (), is_leaf=True)  # tangents_1
    buf6 = reader.storage(None, 4, device=device(type='npu', index=0))
    reader.tensor(buf6, (), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)