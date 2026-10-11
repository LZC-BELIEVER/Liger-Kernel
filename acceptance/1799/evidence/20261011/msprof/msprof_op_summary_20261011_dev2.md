# msProf op_summary analysis

- source: evidence/20261011/msprof/raw_20261011_dev2/PROF_000001_20261011131902494_02849883CQDRHHEI/mindstudio_profiler_output/op_summary_20261011131915.csv
- distinct kernels: 34; total task time: 487.115 ms
- loss-path kernels (`_grpo_loss*`, `_selective_log_softmax*`): 48.205 ms (9.9% of task time)

| kernel | calls | total ms | mean us | vec ratio | mte2 ratio (AIV) | mte1 ratio (AIC) | mac ratio (AIC) | fixpipe ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aclnnMatmul_MatMulV3Common_MatMulV3 | 24 | 350.952 | 14623.0 | 0.0012 | 0.0187 | 0.7341 | 0.8558 | 0.2765 |
| _selective_log_softmax_kernel | 9 | 20.335 | 2259.5 | 0.6493 | 0.2393 | 0.0000 | 0.0000 | 0.0000 |
| aclnnCat_ConcatD_ConcatD | 12 | 19.529 | 1627.4 | 0.0303 | 0.9627 | 0.0000 | 0.0000 | 0.0000 |
| aclnnMatmul_TransData_TransData | 12 | 11.836 | 986.3 | 0.0530 | 0.8838 | 0.0000 | 0.0000 | 0.0000 |
| aclnnAdd_TransposeAiCore_Transpose | 4 | 11.684 | 2921.0 | 0.0390 | 0.6819 | 0.0000 | 0.0000 | 0.0000 |
| aclnnAdd_AddAiCore_Add | 22 | 11.385 | 517.5 | 0.1971 | 0.9135 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceCopy_SliceV2AiCore_SliceV2 | 6 | 9.941 | 1656.9 | 0.0000 | 0.6732 | 0.0000 | 0.0000 | 0.0000 |
| aclnnMatmul_SliceV2AiCore_SliceV2 | 6 | 9.808 | 1634.6 | 0.0000 | 0.6837 | 0.0000 | 0.0000 | 0.0000 |
| _grpo_loss_bwd_kernel | 3 | 7.140 | 2380.1 | 0.6997 | 0.5223 | 0.0000 | 0.0000 | 0.0000 |
| _grpo_loss_bwd_kernel_seq | 3 | 6.979 | 2326.2 | 0.7187 | 0.5127 | 0.0000 | 0.0000 | 0.0000 |
| _grpo_loss_fwd_kernel | 3 | 6.892 | 2297.3 | 0.6443 | 0.2333 | 0.0000 | 0.0000 | 0.0000 |
| _grpo_loss_fwd_kernel_seq | 3 | 6.859 | 2286.3 | 0.6463 | 0.2333 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceNormal_DSARandomNormal_DSARandomNormal | 6 | 6.341 | 1056.9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 2 | 5.861 | 2930.7 | 0.0390 | 0.6835 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceZero_ZerosLikeAiCore_ZerosLike | 84 | 0.310 | 3.7 | 0.1947 | 0.0005 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceCopy_TensorMoveAiCore_TensorMove | 6 | 0.296 | 49.3 | 0.0000 | 0.6504 | 0.0000 | 0.0000 | 0.0000 |
| aclnnReduceSum_ReduceSumOpAiCore_ReduceSum | 57 | 0.285 | 5.0 | 0.0449 | 0.0499 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceNormal_CastAiCore_Cast | 6 | 0.241 | 40.1 | 0.0780 | 0.9340 | 0.0000 | 0.0000 | 0.0000 |
| aclnnMul_MulAiCore_Mul | 60 | 0.117 | 1.9 | 0.0278 | 0.2603 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceCopy_ViewCopyAiCore_ViewCopy | 6 | 0.067 | 11.2 | 0.0028 | 0.1253 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceCopy_BroadcastToAiCore_BroadcastTo | 6 | 0.058 | 9.6 | 0.0098 | 0.0130 | 0.0000 | 0.0000 | 0.0000 |
| aclnnDiv_RealDivAiCore_RealDiv | 39 | 0.054 | 1.4 | 0.0355 | 0.2665 | 0.0000 | 0.0000 | 0.0000 |
| aclnnClamp_ClipByValueV2AiCore_ClipByValueV2 | 24 | 0.044 | 1.8 | 0.0471 | 0.2907 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceFillScalar_FillAiCore_Fill | 12 | 0.019 | 1.6 | 0.0073 | 0.1568 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceOne_OnesLikeAiCore_OnesLike | 8 | 0.015 | 1.9 | 0.0313 | 0.0010 | 0.0000 | 0.0000 | 0.0000 |
| aclnnGtScalar_GreaterAiCore_Greater | 6 | 0.013 | 2.1 | 0.0581 | 0.1290 | 0.0000 | 0.0000 | 0.0000 |
| aclnnLtScalar_LessAiCore_Less | 6 | 0.012 | 2.0 | 0.0599 | 0.1706 | 0.0000 | 0.0000 | 0.0000 |
| aclnnBitwiseAndTensor_LogicalAndAiCore_LogicalAnd | 6 | 0.010 | 1.7 | 0.0572 | 0.1724 | 0.0000 | 0.0000 | 0.0000 |
| aclnnMul_BroadcastToAiCore_BroadcastTo | 6 | 0.010 | 1.7 | 0.0374 | 0.0976 | 0.0000 | 0.0000 | 0.0000 |
| aclnnBitwiseOrTensor_LogicalOrAiCore_LogicalOr | 3 | 0.005 | 1.8 | 0.0519 | 0.1659 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceRandom_CastAiCore_Cast | 2 | 0.005 | 2.7 | 0.0195 | 0.2496 | 0.0000 | 0.0000 | 0.0000 |
| aclnnExp_ExpAiCore_Exp | 3 | 0.004 | 1.5 | 0.0323 | 0.1109 | 0.0000 | 0.0000 | 0.0000 |
| aclnnCast_CastAiCore_Cast | 3 | 0.004 | 1.4 | 0.0450 | 0.1193 | 0.0000 | 0.0000 | 0.0000 |
| aclnnInplaceRandom_DSARandomUniform_DSARandomUniform | 2 | 0.003 | 1.6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
