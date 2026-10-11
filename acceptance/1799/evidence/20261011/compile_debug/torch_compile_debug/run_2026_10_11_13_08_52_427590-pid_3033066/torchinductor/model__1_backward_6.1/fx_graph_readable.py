class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[1, 1024]", primals_3: "f32[1, 1024]", primals_4: "bf16[1]", primals_5: "f32[1, 1024]", clamp_min_1: "f32[]", tangents_1: "f32[]", tangents_2: "f32[]"):
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:252 in ppo_loss_fn, code: metrics.append(((kl_div * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)))
        div_3: "f32[]" = torch.ops.aten.div.Tensor(tangents_2, clamp_min_1);  tangents_2 = None
        expand: "f32[1, 1024]" = torch.ops.aten.expand.default(div_3, [1, 1024]);  div_3 = None
        mul_6: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(expand, primals_5);  expand = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:231 in ppo_loss_fn, code: loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        div_4: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, clamp_min_1);  tangents_1 = clamp_min_1 = None
        expand_1: "f32[1, 1024]" = torch.ops.aten.expand.default(div_4, [1, 1024]);  div_4 = None
        mul_7: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(expand_1, primals_5);  expand_1 = primals_5 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:218 in ppo_loss_fn, code: per_token_loss = per_token_loss + beta * kl_div
        mul_8: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(mul_7, 0.1)
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:218 in ppo_loss_fn, code: per_token_loss = per_token_loss + beta * kl_div
        add_1: "f32[1, 1024]" = torch.ops.aten.add.Tensor(mul_6, mul_8);  mul_6 = mul_8 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:13 in k3_loss_fn, code: return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0
        neg_1: "f32[1, 1024]" = torch.ops.aten.neg.default(add_1)
        neg_2: "f32[1, 1024]" = torch.ops.aten.neg.default(neg_1);  neg_1 = None
        sub_1: "f32[1, 1024]" = torch.ops.aten.sub.Tensor(primals_3, primals_2);  primals_3 = None
        exp_1: "f32[1, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul_9: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(add_1, exp_1);  add_1 = exp_1 = None
        neg_3: "f32[1, 1024]" = torch.ops.aten.neg.default(mul_9);  mul_9 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:13 in k3_loss_fn, code: return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0
        add_2: "f32[1, 1024]" = torch.ops.aten.add.Tensor(neg_2, neg_3);  neg_2 = neg_3 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:201 in ppo_loss_fn, code: per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        neg_4: "f32[1, 1024]" = torch.ops.aten.neg.default(mul_7);  mul_7 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:140 in ppo_loss_fn, code: log_ratio = per_token_logps - old_per_token_logps
        sub: "f32[1, 1024]" = torch.ops.aten.sub.Tensor(primals_2, primals_2);  primals_2 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:155 in ppo_loss_fn, code: coef_1 = torch.exp(log_importance_weights)
        exp: "f32[1, 1024]" = torch.ops.aten.exp.default(sub);  sub = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:93 in clip_coef_fn, code: clipped_coef = torch.clamp(coef, lower_bound, upper_bound)
        clamp_min: "f32[1, 1024]" = torch.ops.aten.clamp_min.default(exp, 0.8)
        clamp_max: "f32[1, 1024]" = torch.ops.aten.clamp_max.default(clamp_min, 1.2);  clamp_min = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:199 in ppo_loss_fn, code: per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        unsqueeze: "bf16[1, 1]" = torch.ops.aten.unsqueeze.default(primals_4, 1);  primals_4 = None
        mul: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(exp, unsqueeze)
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:200 in ppo_loss_fn, code: per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        mul_1: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(clamp_max, unsqueeze);  clamp_max = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:201 in ppo_loss_fn, code: per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        eq: "b8[1, 1024]" = torch.ops.aten.eq.Tensor(mul, mul_1)
        div_5: "f32[1, 1024]" = torch.ops.aten.div.Scalar(neg_4, 2)
        where: "f32[1, 1024]" = torch.ops.aten.where.self(eq, div_5, neg_4);  eq = div_5 = neg_4 = None
        lt_2: "b8[1, 1024]" = torch.ops.aten.lt.Tensor(mul, mul_1)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_1: "f32[1, 1024]" = torch.ops.aten.where.self(lt_2, full_default, where);  lt_2 = None
        gt_2: "b8[1, 1024]" = torch.ops.aten.gt.Tensor(mul, mul_1);  mul = mul_1 = None
        where_3: "f32[1, 1024]" = torch.ops.aten.where.self(gt_2, full_default, where);  gt_2 = where = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:200 in ppo_loss_fn, code: per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        mul_10: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(where_1, unsqueeze);  where_1 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:199 in ppo_loss_fn, code: per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        mul_11: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze);  where_3 = unsqueeze = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:93 in clip_coef_fn, code: clipped_coef = torch.clamp(coef, lower_bound, upper_bound)
        ge: "b8[1, 1024]" = torch.ops.aten.ge.Scalar(exp, 0.8)
        le: "b8[1, 1024]" = torch.ops.aten.le.Scalar(exp, 1.2)
        logical_and: "b8[1, 1024]" = torch.ops.aten.logical_and.default(ge, le);  ge = le = None
        where_4: "f32[1, 1024]" = torch.ops.aten.where.self(logical_and, mul_10, full_default);  logical_and = mul_10 = full_default = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:93 in clip_coef_fn, code: clipped_coef = torch.clamp(coef, lower_bound, upper_bound)
        add_3: "f32[1, 1024]" = torch.ops.aten.add.Tensor(mul_11, where_4);  mul_11 = where_4 = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:155 in ppo_loss_fn, code: coef_1 = torch.exp(log_importance_weights)
        mul_12: "f32[1, 1024]" = torch.ops.aten.mul.Tensor(add_3, exp);  add_3 = exp = None
        
         # File: /workspace3/lcw/liger-kernel-ref/src/liger_kernel/chunked_loss/grpo_loss.py:140 in ppo_loss_fn, code: log_ratio = per_token_logps - old_per_token_logps
        add_4: "f32[1, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_12);  add_2 = mul_12 = None
        return (None, add_4, None, None, None)
        