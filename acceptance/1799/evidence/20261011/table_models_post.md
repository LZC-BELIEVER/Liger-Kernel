# Per-configuration comparison (POST-only model_config sweep (--bt 2048), revision a8c360c)

`pre_liger` = pre-optimization Liger; `post_liger` = final revision Liger.
`ratio` = Liger / Torch baseline (lower is better).

This is a **POST-only** table: the sweep has no PRE counterpart, so the pre-optimization columns are `n/a` by design.


## fused_linear_grpo_loss_sequence / forward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | deepseek_v2_lite | n/a | 5.1500 | 6.2071 | n/a | 0.8297 | n/a |
| T1024/sequence | deepseek_v3 | n/a | 15.4044 | 16.4815 | n/a | 0.9346 | n/a |
| T1024/sequence | llama_2_7b | n/a | 2.5815 | 2.7013 | n/a | 0.9556 | n/a |
| T1024/sequence | llama_3_8b | n/a | 9.8931 | 10.9349 | n/a | 0.9047 | n/a |
| T1024/sequence | qwen2.5_14b | n/a | 14.9087 | 16.4880 | n/a | 0.9042 | n/a |
| T1024/sequence | qwen2.5_72b | n/a | 22.8278 | 24.4054 | n/a | 0.9354 | n/a |
| T1024/sequence | qwen2.5_7b | n/a | 11.3934 | 12.9702 | n/a | 0.8784 | n/a |

## fused_linear_grpo_loss_sequence / backward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | deepseek_v2_lite | n/a | 6.8938 | 9.0524 | n/a | 0.7615 | n/a |
| T1024/sequence | deepseek_v3 | n/a | 26.3623 | 28.9952 | n/a | 0.9092 | n/a |
| T1024/sequence | llama_2_7b | n/a | 3.6215 | 4.1350 | n/a | 0.8758 | n/a |
| T1024/sequence | llama_3_8b | n/a | 15.5813 | 18.1697 | n/a | 0.8575 | n/a |
| T1024/sequence | qwen2.5_14b | n/a | 22.3299 | 25.4243 | n/a | 0.8783 | n/a |
| T1024/sequence | qwen2.5_72b | n/a | 35.1148 | 38.4131 | n/a | 0.9141 | n/a |
| T1024/sequence | qwen2.5_7b | n/a | 17.1654 | 20.3808 | n/a | 0.8422 | n/a |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | deepseek_v2_lite | n/a | 2380.4253 | 2830.3838 | n/a | 0.8410 | n/a |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | deepseek_v2_lite | n/a | 11.9971 | 15.3047 | n/a | 0.7839 | n/a |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | deepseek_v3 | n/a | 9146.2822 | 9384.5273 | n/a | 0.9746 | n/a |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | deepseek_v3 | n/a | 41.7641 | 45.4316 | n/a | 0.9193 | n/a |
| T1024/sequence | llama_2_7b | n/a | 6.2219 | 6.8455 | n/a | 0.9089 | n/a |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | llama_2_7b | n/a | 1344.5967 | 1399.0273 | n/a | 0.9611 | n/a |
| T1024/sequence | llama_3_8b | n/a | 5292.7803 | 5535.0273 | n/a | 0.9562 | n/a |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | llama_3_8b | n/a | 25.4029 | 29.1147 | n/a | 0.8725 | n/a |
| T1024/sequence | qwen2.5_14b | n/a | 37.2530 | 41.9433 | n/a | 0.8882 | n/a |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | qwen2.5_14b | n/a | 7765.3242 | 8052.0259 | n/a | 0.9644 | n/a |
| T1024/sequence | qwen2.5_72b | n/a | 12241.3262 | 12522.0273 | n/a | 0.9776 | n/a |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | qwen2.5_72b | n/a | 57.9944 | 62.7871 | n/a | 0.9237 | n/a |

## fused_linear_grpo_loss_sequence / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | qwen2.5_7b | n/a | 5523.8247 | 5813.5264 | n/a | 0.9502 | n/a |

## fused_linear_grpo_loss_sequence / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/sequence | qwen2.5_7b | n/a | 28.5262 | 33.2963 | n/a | 0.8567 | n/a |

## fused_linear_grpo_loss_token / forward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | deepseek_v2_lite | n/a | 4.6312 | 6.1961 | n/a | 0.7474 | n/a |
| T1024/token | deepseek_v3 | n/a | 14.7834 | 16.4465 | n/a | 0.8989 | n/a |
| T1024/token | llama_2_7b | n/a | 2.4359 | 2.7258 | n/a | 0.8936 | n/a |
| T1024/token | llama_3_8b | n/a | 9.2920 | 10.9182 | n/a | 0.8511 | n/a |
| T1024/token | qwen2.5_14b | n/a | 14.2431 | 16.4948 | n/a | 0.8635 | n/a |
| T1024/token | qwen2.5_72b | n/a | 22.0874 | 24.4845 | n/a | 0.9021 | n/a |
| T1024/token | qwen2.5_7b | n/a | 10.6892 | 12.9609 | n/a | 0.8247 | n/a |

## fused_linear_grpo_loss_token / backward / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | deepseek_v2_lite | n/a | 6.8747 | 9.0260 | n/a | 0.7617 | n/a |
| T1024/token | deepseek_v3 | n/a | 26.4165 | 28.9825 | n/a | 0.9115 | n/a |
| T1024/token | llama_2_7b | n/a | 3.6174 | 4.1159 | n/a | 0.8789 | n/a |
| T1024/token | llama_3_8b | n/a | 15.5828 | 18.2038 | n/a | 0.8560 | n/a |
| T1024/token | qwen2.5_14b | n/a | 22.2741 | 25.4585 | n/a | 0.8749 | n/a |
| T1024/token | qwen2.5_72b | n/a | 35.1643 | 38.4923 | n/a | 0.9135 | n/a |
| T1024/token | qwen2.5_7b | n/a | 17.1475 | 20.4019 | n/a | 0.8405 | n/a |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | deepseek_v2_lite | n/a | 11.4914 | 15.1942 | n/a | 0.7563 | n/a |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | deepseek_v2_lite | n/a | 2380.4243 | 2830.3838 | n/a | 0.8410 | n/a |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | deepseek_v3 | n/a | 41.0495 | 45.3525 | n/a | 0.9051 | n/a |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | deepseek_v3 | n/a | 9146.2812 | 9384.5391 | n/a | 0.9746 | n/a |
| T1024/token | llama_2_7b | n/a | 1344.5957 | 1399.0386 | n/a | 0.9611 | n/a |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | llama_2_7b | n/a | 6.0096 | 6.8498 | n/a | 0.8773 | n/a |
| T1024/token | llama_3_8b | n/a | 24.9027 | 29.0487 | n/a | 0.8573 | n/a |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | llama_3_8b | n/a | 5292.7793 | 5535.0386 | n/a | 0.9562 | n/a |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | qwen2.5_14b | n/a | 36.6039 | 41.9502 | n/a | 0.8726 | n/a |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | qwen2.5_14b | n/a | 7765.3232 | 8052.0371 | n/a | 0.9644 | n/a |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | qwen2.5_72b | n/a | 57.3462 | 62.7676 | n/a | 0.9136 | n/a |

## fused_linear_grpo_loss_token / full / memory

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | qwen2.5_72b | n/a | 12241.3242 | 12522.0391 | n/a | 0.9776 | n/a |
| T1024/token | qwen2.5_7b | n/a | 5523.8237 | 5813.5376 | n/a | 0.9502 | n/a |

## fused_linear_grpo_loss_token / full / speed

| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1024/token | qwen2.5_7b | n/a | 27.8225 | 33.2594 | n/a | 0.8365 | n/a |

## Summary

- speed: 42 Liger/Torch pairs, max ratio 0.955636, min ratio 0.747432, regressions (>1.0): 0
- memory: 14 Liger/Torch pairs, max ratio 0.977583, min ratio 0.841025, regressions (>1.0): 0
- total comparisons: 56; regressions >1.0: 0
