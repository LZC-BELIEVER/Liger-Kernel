# Per-configuration comparison (default token_length sweep; POST a8c360c vs PRE generic-eager)

`pre_liger` = pre-optimization Liger; `post_liger` = final revision Liger.
`ratio` = Liger / Torch baseline (lower is better).

`*` marks a row whose PRE number is not the pre-optimization cost of that mode: the pre-PR upstream path performs its real backward inside the forward pass (`LigerFusedLinearPPOBase.forward` -> `torch.autograd.grad` per chunk) and its `backward` entry only replays the stored gradients.  PRE `forward` therefore includes the backward work and PRE `backward` measures only the replay, so only the `full` rows are comparable.  See `PROFILING_AND_OPTIMIZATION.md`.


## fused_linear_grpo_loss_sequence / forward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 1 | 70.2211 * | 9.9493 | 10.9537 | 6.4107 * | 0.9083 | n/a * |
| H4096/V128256/bfloat16/T1024/sequence | 2 | 140.3525 * | 19.1862 | 21.2298 | 6.6111 * | 0.9037 | n/a * |
| H4096/V128256/bfloat16/T1024/sequence | 4 | 278.6656 * | 37.7822 | 41.8046 | 6.6659 * | 0.9038 | n/a * |

## fused_linear_grpo_loss_sequence / backward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 1 | 8.0843 * | 15.5676 | 18.2228 | 0.4436 * | 0.8543 | n/a * |
| H4096/V128256/bfloat16/T1024/sequence | 2 | 8.1214 * | 25.2259 | 28.8353 | 0.2816 * | 0.8748 | n/a * |
| H4096/V128256/bfloat16/T1024/sequence | 4 | 8.1854 * | 44.4890 | 51.5469 | 0.1588 * | 0.8631 | n/a * |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 1 | 78.2856 | 25.5383 | 29.1585 | 2.6848 | 0.8758 | -67.38% |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 1 | 7062.0391 | 5292.7803 | 5536.0269 | 1.2757 | 0.9561 | -25.05% |
| H4096/V128256/bfloat16/T1024/sequence | 2 | 7094.0522 | 6130.0576 | 7087.5229 | 1.0009 | 0.8649 | -13.59% |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 2 | 148.4863 | 44.3685 | 50.1588 | 2.9603 | 0.8846 | -70.12% |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 4 | 7174.0552 | 8230.0625 | 11141.5547 | 0.6439 | 0.7387 | +14.72% |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/sequence | 4 | 287.0338 | 82.2840 | 93.2987 | 3.0765 | 0.8819 | -71.33% |

## fused_linear_grpo_loss_token / forward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/token | 1 | 70.3462 * | 9.2862 | 10.8972 | 6.4554 * | 0.8522 | n/a * |
| H4096/V128256/bfloat16/T1024/token | 2 | 140.1240 * | 18.0521 | 21.1882 | 6.6133 * | 0.8520 | n/a * |
| H4096/V128256/bfloat16/T1024/token | 4 | 278.5525 * | 35.5029 | 41.9293 | 6.6434 * | 0.8467 | n/a * |

## fused_linear_grpo_loss_token / backward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/token | 1 | 8.0724 * | 15.5821 | 18.1770 | 0.4441 * | 0.8572 | n/a * |
| H4096/V128256/bfloat16/T1024/token | 2 | 8.2181 * | 25.2147 | 28.8285 | 0.2851 * | 0.8746 | n/a * |
| H4096/V128256/bfloat16/T1024/token | 4 | 8.1508 * | 44.4714 | 51.5045 | 0.1583 * | 0.8634 | n/a * |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/token | 1 | 78.4752 | 24.8673 | 29.0901 | 2.6977 | 0.8548 | -68.31% |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/token | 1 | 7062.0391 | 5292.7793 | 5536.0381 | 1.2756 | 0.9561 | -25.05% |
| H4096/V128256/bfloat16/T1024/token | 2 | 7094.0522 | 6130.0566 | 7087.5229 | 1.0009 | 0.8649 | -13.59% |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/token | 2 | 148.1708 | 43.2694 | 50.0490 | 2.9605 | 0.8645 | -70.80% |
| H4096/V128256/bfloat16/T1024/token | 4 | 286.4800 | 79.9715 | 93.2012 | 3.0738 | 0.8581 | -72.08% |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H4096/V128256/bfloat16/T1024/token | 4 | 7174.0552 | 8230.0615 | 11141.5547 | 0.6439 | 0.7387 | +14.72% |

## Summary

- speed: 18 Liger/Torch pairs, max ratio 0.908310, min ratio 0.846733, regressions (>1.0): 0
- memory: 6 Liger/Torch pairs, max ratio 0.956061, min ratio 0.738682, regressions (>1.0): 0
- total comparisons: 24; regressions >1.0: 0
