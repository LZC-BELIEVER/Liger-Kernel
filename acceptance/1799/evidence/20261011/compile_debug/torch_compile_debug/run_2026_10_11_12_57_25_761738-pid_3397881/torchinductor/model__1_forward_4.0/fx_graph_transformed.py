class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 128]", primals_2: "f32[1, 128]", primals_3: "f32[1, 128]", primals_4: "bf16[1]", primals_5: "f32[1, 128]"):
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:140 in ppo_loss_fn, code: log_ratio = per_token_logps - old_per_token_logps
        sub: "f32[1, 128]" = torch.ops.aten.sub.Tensor(primals_2, primals_2)
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:155 in ppo_loss_fn, code: coef_1 = torch.exp(log_importance_weights)
        exp: "f32[1, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:93 in clip_coef_fn, code: clipped_coef = torch.clamp(coef, lower_bound, upper_bound)
        clamp_min: "f32[1, 128]" = torch.ops.aten.clamp_min.default(exp, 0.8)
        clamp_max: "f32[1, 128]" = torch.ops.aten.clamp_max.default(clamp_min, 1.2);  clamp_min = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:94 in clip_coef_fn, code: is_lower_clipped = coef < lower_bound
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(exp, 0.8)
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:95 in clip_coef_fn, code: is_upper_clipped = coef > upper_bound
        gt: "b8[1, 128]" = torch.ops.aten.gt.Scalar(exp, 1.2)
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:199 in ppo_loss_fn, code: per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        unsqueeze: "bf16[1, 1]" = torch.ops.aten.unsqueeze.default(primals_4, 1)
        mul: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp, unsqueeze);  exp = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:200 in ppo_loss_fn, code: per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        mul_1: "f32[1, 128]" = torch.ops.aten.mul.Tensor(clamp_max, unsqueeze);  clamp_max = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:201 in ppo_loss_fn, code: per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        minimum: "f32[1, 128]" = torch.ops.aten.minimum.default(mul, mul_1);  mul = mul_1 = None
        neg: "f32[1, 128]" = torch.ops.aten.neg.default(minimum);  minimum = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:13 in k3_loss_fn, code: return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0
        sub_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(primals_3, primals_2)
        exp_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_1)
        sub_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(exp_1, sub_1);  exp_1 = sub_1 = None
        sub_4: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_3, 1.0);  sub_3 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:218 in ppo_loss_fn, code: per_token_loss = per_token_loss + beta * kl_div
        mul_2: "f32[1, 128]" = torch.ops.aten.mul.Tensor(sub_4, 0.04)
        add: "f32[1, 128]" = torch.ops.aten.add.Tensor(neg, mul_2);  neg = mul_2 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:231 in ppo_loss_fn, code: loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        mul_3: "f32[1, 128]" = torch.ops.aten.mul.Tensor(add, primals_5);  add = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(mul_3);  mul_3 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(primals_1);  primals_1 = None
        clamp_min_1: "f32[]" = torch.ops.aten.clamp_min.default(sum_2, 1.0);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_1, clamp_min_1);  sum_1 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:252 in ppo_loss_fn, code: metrics.append(((kl_div * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)))
        mul_4: "f32[1, 128]" = torch.ops.aten.mul.Tensor(sub_4, primals_5);  sub_4 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(mul_4);  mul_4 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_3, clamp_min_1);  sum_3 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:256 in ppo_loss_fn, code: is_clipped = (is_lower_clipped & (advantages.unsqueeze(1) < 0)) | (
        lt_1: "b8[1, 1]" = torch.ops.aten.lt.Scalar(unsqueeze, 0)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(lt, lt_1);  lt = lt_1 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:257 in ppo_loss_fn, code: is_upper_clipped & (advantages.unsqueeze(1) > 0)
        gt_1: "b8[1, 1]" = torch.ops.aten.gt.Scalar(unsqueeze, 0);  unsqueeze = None
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(gt, gt_1);  gt = gt_1 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:256 in ppo_loss_fn, code: is_clipped = (is_lower_clipped & (advantages.unsqueeze(1) < 0)) | (
        bitwise_or: "b8[1, 128]" = torch.ops.aten.bitwise_or.Tensor(bitwise_and, bitwise_and_1);  bitwise_and = bitwise_and_1 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:266 in ppo_loss_fn, code: metrics.append((is_clipped * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0))
        mul_5: "f32[1, 128]" = torch.ops.aten.mul.Tensor(bitwise_or, primals_5);  bitwise_or = None
        sum_5: "f32[]" = torch.ops.aten.sum.default(mul_5);  mul_5 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(sum_5, clamp_min_1);  sum_5 = None
        return (div, div_1, div_2, primals_2, primals_3, primals_4, primals_5, clamp_min_1)
        