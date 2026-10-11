# Local Acceptance Checklist

Revision under acceptance: `a8c360c2381aa1b0bc86a214db3d3b6c1d0613ef`
(`feat/ascend-fused-linear-grpo-loss`).  Device: A3, `ASCEND_RT_VISIBLE_DEVICES=2`.

## Requirement Matrix

| Issue requirement | Local status | Evidence |
| --- | --- | --- |
| Liger Kernel 0.8.2 / Triton-Ascend 3.2.2 / torch-npu 2.9.0 / CANN 9.1.0 | PASS | `README.md` version table |
| Token-level and sequence-level importance sampling | PASS | pytest + both benchmark levels |
| Full pytest, 0 failed / 0 error, no assertion/tolerance/skip change | PASS | `evidence/20261011/pytest/pytest_full_20261011_dev2.log` |
| Skip reasons recorded with `-rs` | PASS | same log, `-rs` section (566 = 288+96+144+24+12+2) |
| Benchmark all configs (forward/backward/full × token/sequence) produce data | PASS | `evidence/20261011/bench_post/` (48 rows, no crash/timeout) |
| All measured speed ratios ≤ 1.0 | PASS | 18 pairs, 0.846733–0.908310 |
| All measured memory ratios ≤ 1.0 | PASS | 6 pairs, 0.738682–0.956061 |
| Cross-model sweep coverage (7 registry models, POST) | PASS | 112 rows, 56 pairs, 0 regressions; speed 0.747432–0.955636, memory 0.841025–0.977583; `table_models_post.*` |
| Independent optimization-before table, reproducible PRE | PASS | `pre_variant/pre_variant.diff`, `evidence/20261011/table_default_post_vs_pre.*` |
| PRE/POST gradients match the unchanged Torch reference | PASS | `tools/grad_smoke.py`, `evidence/20261011/grad_smoke_*` |
| msProf raw export (DB/structured tables, command, device, hashes) | PASS | `evidence/20261011/msprof/raw_20261011_dev2/` |
| Profiling root-cause analysis quoting real kernels + data | PASS | `PROFILING_AND_OPTIMIZATION.md` |
| Optimization-means explanation | PASS | same file, "Optimization measures" |
| A2/910B runtime/performance | EXEMPT for this host (goal scope) | Platform Boundary in `README.md` |
| 950 runtime/performance | EXEMPT for this host (goal scope) | Platform Boundary in `README.md` |
| Commit / push / PR | NOT PERFORMED | requires steward confirmation |

## Numeric Summary

| Run | Rows | Pairs | Regressions | Max ratio | Min ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| POST speed (default sweep) | 36 | 18 | 0 | 0.908310 | 0.846733 |
| POST memory (default sweep) | 12 | 6 | 0 | 0.956061 | 0.738682 |
| POST speed (cross-model sweep) | 84 | 42 | 0 | 0.955636 | 0.747432 |
| POST memory (cross-model sweep) | 28 | 14 | 0 | 0.977583 | 0.841025 |
| PRE speed (default sweep) | 36 | 18 | 6 comparable `full` rows all > 1.0; 12 rows carry `*` (6 forward + 6 backward replay) | 6.6659 | 0.1583 |
| PRE memory (default sweep) | 12 | 6 | 3 | 1.2757 | 0.6439 |
| POST full-mode speed vs baseline | 6 | 6 | 0 | 0.8846 | 0.8548 |
| PRE full-mode speed vs baseline | 6 | 6 | 6 | 3.0765 | 2.6848 |
| Full pytest | 1590 collected | — | 0 failed, 0 error | — | — |

## Acceptance Boundaries

1. **PRE-mode caveat.**  The pre-PR path runs its real backward inside the
   forward pass and its `backward` entry only replays stored gradients, so the
   PRE `backward` column (and the PRE `forward` column, which contains the
   backward work) are marked `*` and are not usable as pre-optimization
   performance.  Only the `full` rows are the before/after comparison:
   PRE 2.6848–3.0765 → POST 0.8548–0.8846 of the Torch baseline.  The replay
   behaviour is evidenced by `tools/grad_smoke.py`
   (flat ~7 ms timed backward across B, gradients still correct against the
   Torch reference).
2. **PRE runnability.**  The pre-PR default `compiled=True` cannot complete the
   sweep on this stack (dynamo shape-guard assertion); the retained failure log
   is `evidence/20261011/bench_pre/bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log`.
   The PRE numbers use `compiled=False`, the only runnable pre-PR configuration
   here, under the unchanged benchmark/test definitions.
3. **Memory trade-off.**  At B=4 the PRE streaming path uses less memory
   (7.17 GB) than the coalesced POST chunk (8.23 GB); POST is still ≤ 1.0 of the
   baseline (0.7387) and 71.3% faster in full mode.  Recorded, not hidden.
4. **Platform scope.**  Runtime evidence is A3-only.  A2/910B and 950 are exempt
   on this host per the goal; the boundary must be carried into the PR.
5. **Skipped tests** are contract/instability guards in the unchanged test file;
   they are not counted as passes.
