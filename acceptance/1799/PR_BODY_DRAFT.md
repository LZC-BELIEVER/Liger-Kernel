# Ascend fused linear GRPO loss

Related task: [triton-ascend#1799](https://github.com/triton-lang/triton-ascend/issues/1799)
Source upstream: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
Revision: `a8c360c` (`feat/ascend-fused-linear-grpo-loss`)

## Summary

Adds the Triton-Ascend path for `fused_linear_grpo_loss` and keeps the
forward/backward math aligned with the existing Torch reference for both
token-level (`GRPO`) and sequence-level (`GSPO`) importance sampling, including
LUSPO mask/padding and gradient handling.

## Implementation

* New Ascend fused-linear adapter
  (`src/liger_kernel/ops/backends/_ascend/ops/fused_linear_grpo.py`): owns the
  lm-head projection, bias and the extra pad row, drives the existing Ascend GRPO
  kernels per row chunk, and coalesces rows under a 2 GiB retained-logits budget.
* Dispatch: `liger_fused_linear_grpo` and `LigerFusedLinearGRPOLoss.forward` route
  NPU inputs to the adapter.
* Sequence-path correctness for LUSPO in the Ascend GRPO kernels.

## Local validation (Atlas A3, `Ascend910_9382`, CANN 9.1.0, device 2)

| Check | Result |
| --- | --- |
| `pytest test/chunked_loss/test_grpo_loss.py -q -rs` | `1024 passed, 566 skipped, 0 failed, 0 error` (995.18s) |
| Skip contract (unchanged `-rs` reasons) | 288 sequence-level unsupported + 96 token-level LUSPO + 144 `delta` unsupported + 24 bf16-VESPO guard + 12 large-V VESPO guard + 2 VLLM LUSPO = 566 |
| `python benchmark/scripts/benchmark_grpo_loss.py` | 48 rows, 24 Liger/Torch pairs, 0 regressions |
| Speed ratios (18 pairs) | max 0.908310, min 0.846733 (all ≤ 1.0) |
| Memory ratios (6 pairs) | max 0.956061, min 0.738682 (all ≤ 1.0) |
| Cross-model sweep (`--sweep-mode model_config --bt 2048`) | 112 rows, 56 pairs, 0 regressions; speed 0.747432–0.955636, memory 0.841025–0.977583 |

## Optimization before / after (per configuration)

“Before” = the pre-PR upstream generic fused-linear path
(`LigerFusedLinearGRPOFunction` / `LigerFusedLinearPPOBase.chunk_forward`, torch
matmul streaming with a 4096-wide vocabulary chunk, `compiled=False` because the
pre-PR default `compiled=True` aborts in TorchInductor/dynamo on this stack).
Same host, environment, device, benchmark script and sampling protocol; only the
source toggle differs.

| Kernel / metric | B | before (ms) | after (ms) | Torch (ms) | before ratio | after ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token / full | 1 | 78.475 | 24.867 | 29.090 | 2.6977 | 0.8548 |
| token / full | 2 | 148.171 | 43.269 | 50.049 | 2.9605 | 0.8645 |
| token / full | 4 | 286.480 | 79.971 | 93.201 | 3.0738 | 0.8581 |
| sequence / full | 1 | 78.286 | 25.538 | 29.159 | 2.6848 | 0.8758 |
| sequence / full | 2 | 148.486 | 44.368 | 50.159 | 2.9603 | 0.8846 |
| sequence / full | 4 | 287.034 | 82.284 | 93.299 | 3.0765 | 0.8819 |
| token / full memory (MB) | 1 | 7062.04 | 5292.78 | 5536.04 | 1.2756 | 0.9561 |
| token / full memory (MB) | 2 | 7094.05 | 6130.06 | 7087.52 | 1.0009 | 0.8649 |
| token / full memory (MB) | 4 | 7174.06 | 8230.06 | 11141.55 | 0.6439 | 0.7387 |

Notes for review:

* Only the `full` rows are comparable.  The pre-PR path performs its real
  backward inside the forward pass and its `backward` entry only replays stored
  gradients, so the `backward`-mode numbers are a measurement artifact for the
  “before” variant (verified by a gradient smoke run; see the local packet).
* At B=4 the “before” streaming path uses less memory (7.17 GB) than the
  coalesced “after” chunk (8.23 GB); the “after” variant is 71.3% faster there
  and still below the Torch baseline.

## Profiling analysis (CANN 9.1 msProf, retained raw export)

Measured on B=4 / T=1024 / H=4096 / V=128256 / bf16, token + sequence
(34 kernels, 487.1 ms device task time):

| Kernel | Calls | Total ms | Share | vec ratio | mte2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aclnnMatmul_MatMulV3Common_MatMulV3` (projection) | 24 | 350.952 | 72.0% | — | — (Cube `mac` 0.856) |
| `_selective_log_softmax_kernel` | 9 | 20.335 | 4.2% | 0.6493 | 0.2393 |
| `_grpo_loss_fwd_kernel` | 3 | 6.892 | 1.4% | 0.6443 | 0.2333 |
| `_grpo_loss_fwd_kernel_seq` | 3 | 6.859 | 1.4% | 0.6463 | 0.2333 |
| `_grpo_loss_bwd_kernel` | 3 | 7.140 | 1.5% | 0.6997 | 0.5223 |
| `_grpo_loss_bwd_kernel_seq` | 3 | 6.979 | 1.4% | 0.7187 | 0.5127 |

Root cause: the fused loss kernels (9.9% of device time) are Vector-pipe bound
(0.64–0.72) with the backward kernels additionally at ~0.51 MTE2 due to the
`dlogits` write-back; the dominant cost of the path is the lm-head projection
(72%) plus the `[B, T+1, V]` pad/layout chain (`Cat`/`TransData`/`Transpose`/
`SliceV2`, ≈16%), i.e. outside the loss kernels.  The measured before/after gain
comes from replacing the pre-PR 4096-wide vocabulary matmul streaming and
per-chunk gradient recomputation with the fused adapter, not from a faster loss
kernel.

## Platform boundary

All runtime numbers above are Atlas A3 (`Ascend910_9382`).  A2/910B and 950
devices were not available on this host, so their runtime/performance results are
not claimed; please confirm on matching hardware.  The Issue's published 910B3
numbers are not replaced by these A3 measurements.

## Reproduction

The exact commands, expected results, raw CSVs/logs and the msProf export are in
the local acceptance packet (`README.md`, `REPRODUCE.md`,
`EVIDENCE_MANIFEST.md`, `PROFILING_AND_OPTIMIZATION.md`); the PRE variant is the
retained `pre_variant/pre_variant.diff`.
