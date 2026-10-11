# Issue #1799 Local PR Acceptance Packet

## Scope

This packet records local evidence for `triton-lang/triton-ascend#1799`, the
`fused_linear_grpo_loss` Ascend adaptation in the sibling checkout:

`/workspace3/lcw/liger-kernel-ref`, branch `feat/ascend-fused-linear-grpo-loss`,
revision `a8c360c2381aa1b0bc86a214db3d3b6c1d0613ef`.

No test assertion, tolerance, skip contract, or benchmark definition was
modified, and no commit or push was made while preparing this packet.  The
measurement device was the local Atlas A3 host (`Ascend910_9382`); A2/910B and
950 runtime measurement are outside this machine's scope (see Platform
Boundary).

## Acceptance Position

**Locally COMPLETE for this machine's scope.**  All runnable Issue requirements
are closed on the single final revision `a8c360c`:

* `pytest test/chunked_loss/test_grpo_loss.py -q -rs` → `1024 passed,
  566 skipped, 0 failed, 0 error` (995.18s), with the `-rs` per-reason counts.
* `python benchmark/scripts/benchmark_grpo_loss.py` (default sweep) → 48 rows,
  24 Liger/Torch pairs, 0 regressions, every measured ratio ≤ 1.0.
* Cross-model sweep on the same revision → 112 rows, 56 pairs, 0 regressions.
* An independent, reproducible optimization-before/after per-configuration
  table (PRE variant = `pre_variant/pre_variant.diff`, POST = `a8c360c`).
* A fresh CANN 9.1.0 msProf raw export (DBs + structured tables) with
  root-cause analysis quoting the real kernel names and measured pipe ratios.
* This packet was updated in place; the previous round's evidence is retained
  under `evidence/archive_20260921/`.

Not performed by request: commit, push, PR submission (requires steward
confirmation).

## Version And Hardware

| Item | Value |
| --- | --- |
| Liger Kernel | `0.8.2` |
| Triton-Ascend | `3.2.2` |
| Triton core | `3.2.0` at runtime (`import triton; triton.__version__`); the `triton-3.5.0.dist-info` metadata left in site-packages is stale — the triton-ascend 3.2.2 overlay provides the `triton/` package |
| torch / torch-npu | `2.9.0` / `2.9.0` |
| CANN | `9.1.0` (`/workspace3/lcw/.local/Ascend-9.1.0/cann-9.1.0`) |
| Python env | `/workspace2/lanzichang/envs/grpo322/bin/python` (3.11.14) |
| Host | `Atlas-800T-A3-trainServer-29` |
| Runtime target reported by torch-npu | `Ascend910_9382` |
| Measurement device | `ASCEND_RT_VISIBLE_DEVICES=2` (npu-smi NPU 1 / Chip 0 / Phy-ID 2, Bus `0000:99:00.0`) |
| Required env | `ASCEND_UB_CAPACITY_BITS=1572864`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |

Device 2 was selected after inspecting all 16 chips: NPU 0 carries a live
process, NPUs 4/5 are held by the MSD/vLLM worker (~60 GB/chip), and NPU 6/7 are
held by the CANNBench round-8/round-9 workers.  Chips 1/2/3 were idle and
healthy; chip 2 passed a bf16 matmul health check.  The assignment is recorded
in `ACTIVE_STATUS.md`.

## Final Revision And Evidence Closure

All evidence was produced on one frozen revision.  Source state during every
measurement (POST = unmodified `a8c360c`):

```
3b2326626863af451e282e465b9379052f250afc8d7d235281020d10e7f43b83  src/liger_kernel/chunked_loss/functional.py
3509dc1996a95f88a321e508bcdf8e3939e69e3f5cb7ba1f1903660499285ddd  src/liger_kernel/chunked_loss/fused_linear_ppo.py
54141dc67bbb9a59f0a452417ddc794f305290136cb5614af94e54b215ce93b6  src/liger_kernel/chunked_loss/grpo_loss.py
9a7f989b4461493f001326f907246056015469f07968e7562468aba025400631  src/liger_kernel/ops/backends/_ascend/ops/__init__.py
78d9768746e66885173ca3bb063431d3827db827f1c631097526c966b4ae09d8  src/liger_kernel/ops/backends/_ascend/ops/grpo_loss.py
79d49ca9700c1ab7ee17570055dfa53e77bc789e867d6aeb1b78d0489011b3e3  src/liger_kernel/ops/backends/_ascend/ops/fused_linear_grpo.py
```

The PRE benchmark and PRE gradient smoke ran with `pre_variant/pre_variant.diff`
applied (two of these files differed); after reverting it the same six hashes
were re-verified and `git diff` was empty.  `git diff --quiet -- test/ benchmark/`
confirms the test and benchmark definitions are untouched.

## Functional Evidence

```bash
source /workspace3/lcw/.local/Ascend-9.1.0/cann-9.1.0/aarch64-linux/bin/setenv.bash
export ASCEND_RT_VISIBLE_DEVICES=2 TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_UB_CAPACITY_BITS=1572864
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_pytest_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_pytest_dev2
PYTHONPATH=src /workspace2/lanzichang/envs/grpo322/bin/python \
  -m pytest -q test/chunked_loss/test_grpo_loss.py --disable-warnings -rs
```

```
1024 passed, 566 skipped, 2 warnings in 995.18s (0:16:35)
```

Log: `evidence/20261011/pytest/pytest_full_20261011_dev2.log`.

Skip breakdown, taken from the `-rs` section of that log (it supersedes the
previous round's reconstructed table):

| Skip reason (unchanged test contract) | Count |
| --- | ---: |
| Sequence-level importance sampling unsupported for `cispo`/`sapo`/`vespo` (line 484) | 288 |
| Token-level importance sampling unsupported for `luspo` (line 486) | 96 |
| `delta` unsupported for `cispo`/`sapo`/`vespo` (line 488) | 144 |
| BF16 `vespo` numerical-instability guard (line 497) | 24 |
| Large-vocabulary `vespo` numerical-instability guard (line 501) | 12 |
| VLLM `luspo` unsupported (line 742) | 2 |
| **Total** | **566** |

Skipped cases are not counted as passes; the previous packet's reconstruction
(96 / 96 / 48 / 36) was slightly off and is replaced by the measured counts.

## Performance Evidence (POST, default sweep)

```bash
export LIGER_BENCH_TARGET=1799_post_default_20261011_dev2
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_post_default_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_post_default_dev2
PYTHONPATH=src:benchmark/scripts /workspace2/lanzichang/envs/grpo322/bin/python \
  benchmark/scripts/benchmark_grpo_loss.py
```

Result (13:00:38–13:06:34, device 2):

| Metric | Pairs | Ratio range | Regressions |
| --- | ---: | --- | ---: |
| speed (forward/backward/full × token/sequence × B=1,2,4) | 18 | 0.846733 – 0.908310 | 0 |
| memory (full × token/sequence × B=1,2,4) | 6 | 0.738682 – 0.956061 | 0 |

48 rows, no crash, no timeout, every configuration produced data.  CSV/log:
`evidence/20261011/bench_post/`.

### Cross-model sweep (POST-only)

```bash
export LIGER_BENCH_TARGET=1799_post_models_20261011_dev2
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_post_models_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_post_models_dev2
PYTHONPATH=src:benchmark/scripts /workspace2/lanzichang/envs/grpo322/bin/python \
  benchmark/scripts/benchmark_grpo_loss.py --sweep-mode model_config --bt 2048
```

Result (13:34:59–13:51:39, device 2): 112 rows, 56 Liger/Torch pairs
(42 speed + 14 memory), 0 regressions; speed ratios 0.747432–0.955636, memory
ratios 0.841025–0.977583.  This restores the previous packet's cross-model
coverage on the final revision (all seven registry models).  It is POST-only by
design: the before/after comparison uses the default sweep above, so this sweep
has no PRE counterpart (`--pre` omitted; the generated table says so).
CSV/log: `evidence/20261011/bench_post/all_benchmark_data_1799_post_models_20261011_dev2.csv`
and `bench_models_post_20261011_dev2.log`; table:
`evidence/20261011/table_models_post.{md,csv}`.

## Optimization Before / After

PRE is defined and reproduced by `pre_variant/pre_variant.diff` (see
`pre_variant/README.md`): the pre-PR upstream generic fused-linear path
(`LigerFusedLinearGRPOFunction` / `LigerFusedLinearPPOBase.chunk_forward`, torch
matmul streaming with a 4096-wide vocabulary chunk).  Its default
`compiled=True` mode **cannot complete the benchmark on this stack** —
TorchInductor/dynamo aborts while building the shape guard for the next swept
shape (`AssertionError: s37 … not in {...}`); that raw failure log is retained as
`evidence/20261011/bench_pre/bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log`.
The PRE numbers therefore come from the pre-PR implementation in its only
runnable configuration on this stack (`compiled=False`).

Sweep result (13:12:24–13:18:08, same device and protocol, only the diff
differs).  The comparable rows are `full` (forward + backward in one timed
region):

| Kernel / metric | B | PRE Liger | POST Liger | Torch | PRE ratio | POST ratio | POST vs PRE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token / speed | 1 | 78.475 | 24.867 | 29.090 | 2.6977 | 0.8548 | −68.31% |
| token / speed | 2 | 148.171 | 43.269 | 50.049 | 2.9605 | 0.8645 | −70.80% |
| token / speed | 4 | 286.480 | 79.971 | 93.201 | 3.0738 | 0.8581 | −72.08% |
| sequence / speed | 1 | 78.286 | 25.538 | 29.159 | 2.6848 | 0.8758 | −67.38% |
| sequence / speed | 2 | 148.486 | 44.368 | 50.159 | 2.9603 | 0.8846 | −70.12% |
| sequence / speed | 4 | 287.034 | 82.284 | 93.299 | 3.0765 | 0.8819 | −71.33% |
| token / memory (MB) | 1 | 7062.04 | 5292.78 | 5536.04 | 1.2756 | 0.9561 | −25.05% |
| token / memory (MB) | 2 | 7094.05 | 6130.06 | 7087.52 | 1.0009 | 0.8649 | −13.59% |
| token / memory (MB) | 4 | 7174.06 | 8230.06 | 11141.55 | 0.6439 | 0.7387 | +14.72% |

Full table: `evidence/20261011/table_default_post_vs_pre.md` / `.csv`
(24 rows; the `*`-marked rows are explained below).  Reading:

* Before the change the operator ran **2.68–3.08× slower than the Torch
  baseline** and missed the Issue's "not slower than baseline" requirement; on
  the final revision every measured configuration is **0.85–0.88×** of the
  baseline in full mode (0.85–0.88 for all six).
* Memory: PRE exceeded the baseline at B=1/B=2 (1.28× / 1.00×); POST is ≤ 1.0
  everywhere.  At B=4 the PRE streaming path happens to use *less* memory than
  the coalesced POST chunk (7.17 GB vs 8.23 GB) because row coalescing keeps a
  full `[4, T+1, V]` chunk live; that is a recorded trade-off, not a regression
  (POST is still 0.74× of the baseline and 71.3% faster in full mode).

### Measurement caveat on the PRE `forward` / `backward` rows

The pre-PR upstream path performs its *real* backward inside the forward pass
(`LigerFusedLinearPPOBase.forward` → `torch.autograd.grad` per chunk, results
stored with `ctx.save_for_backward`), and `LigerFusedLinearPPOBase.backward`
(`fused_linear_ppo.py:555-571`) only rescales those stored buffers by
`grad_output`.  Consequently the benchmark's `backward` mode — which times
`y.backward(do, retain_graph=True)` after one forward — measures only that
replay for the PRE variant, and the PRE `forward` column contains the backward
work.  Those two columns are marked `*` in the table and must not be read as
pre-optimization performance for those modes; the `full` rows are the
before/after numbers.

This was verified with a gradient smoke run along the same measurement path
(`tools/grad_smoke.py`, logs `evidence/20261011/grad_smoke_*_20261011_dev2.log`):

| Variant | dtype | B | loss vs Torch ref | x-grad rel-L2 | w-grad rel-L2 | tolerance allclose | timed backward |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| POST | bf16 | 1 / 2 / 4 | identical | 9.6e-06 / 5.8e-06 / 5.9e-06 | 2.5e-06 / 5.2e-07 / 1.2e-06 | pass | 14.4 / 24.1 / 43.3 ms |
| POST | fp32 | 1 / 2 / 4 | identical | 1.1e-07 / 1.2e-07 / 1.6e-07 | 1.0e-07 / 1.2e-07 / 1.6e-07 | pass | 39.0 / 68.2 / 135.6 ms |
| PRE | bf16 | 1 / 2 / 4 | identical | 3.2e-03 (flat) | 3.2e-03 (flat) | pass | 6.96 / 6.89 / 6.97 ms |
| PRE | fp32 | 1 / 2 / 4 | identical | ~7e-07 (flat) | ~5e-07 (flat) | pass | 6.75 / 6.84 / 7.03 ms |

Both variants produce gradients that match the unchanged Torch reference inside
the test file's own tolerance contract (bf16 `atol=0.1, rtol=0.5`; fp32
`atol=2e-5, rtol=1e-3`), and the repeated timed call is deterministic in both.
The PRE timed backward is flat at ~7 ms across batch sizes and is dominated by
the B-independent FP32 `[V, H]` weight-gradient buffer (2.1 GiB) that the replay
rescales — the documented measurement artifact, not a missing gradient.

## Profiling Root Cause And Optimization Measures

Full analysis: `PROFILING_AND_OPTIMIZATION.md`; raw export:
`evidence/20261011/msprof/raw_20261011_dev2/` (CANN 9.1 msProf, 2.7 MB,
88 files, structured CSVs + sqlite DBs, tree digest in `SHA256SUMS`).

Headline measured facts (B=4, T=1024, H=4096, V=128256, bf16, token+sequence,
warmup 1 + 2 timed iterations, device 2):

* 34 distinct kernels; 487.115 ms total task time.
* lm-head projection `aclnnMatmul_MatMulV3Common_MatMulV3`: 24 calls,
  350.952 ms (72.0%), Cube-bound (`aic_mac_ratio` 0.856).
* loss-path kernels: 48.205 ms (9.9%) — `_selective_log_softmax_kernel`
  20.335 ms, `_grpo_loss_fwd_kernel` 6.892 ms, `_grpo_loss_fwd_kernel_seq`
  6.859 ms, `_grpo_loss_bwd_kernel` 7.140 ms, `_grpo_loss_bwd_kernel_seq`
  6.979 ms.
* pad/layout plumbing (`aclnnCat_ConcatD`, `TransData`, `Transpose`, `Add`,
  `SliceV2`): ≈ 80.0 ms (16.4%) — the cost of presenting `[B, T+1, V]` in the
  registered kernels' layout.
* loss-kernel pipe ratios: forward vec 0.644/0.646 with MTE2 0.233; backward
  vec 0.700/0.719 with MTE2 0.522/0.513 — vector-bound, backward nearly
  balanced against its gradient write-back traffic.

Optimization measures actually implemented by `a8c360c` (source-level):

1. Ascend fused-linear adapter (`_ascend/ops/fused_linear_grpo.py`): owns the
   projection, bias and the extra zero pad row, and drives the existing Ascend
   GRPO kernels per row chunk, so the retained activation is
   `chunk_rows × (T+1) × V` instead of a materialised `[B, T, V]` FP32 buffer.
2. Row coalescing under a 2 GiB retained-logits budget for the public default
   `chunk_size=1`, removing per-row launch/allocation overhead.
3. Reuse of the UB-aware Ascend kernels (BLOCK_N 8192 forward/softmax, 4096
   backward; `(B, grid_seq)` grid capped by core count; dynamic `tl.range`
   vocabulary traversal) instead of streaming 4096-wide vocabulary matmuls.
4. Chunk-local reduction with per-loss-type normalizers, so chunked
   accumulation reproduces the reference reduction.
5. Sequence-path correctness: `LUSPO` constexpr handling of mask/padding and the
   valid-token weighted backward in the Ascend sequence kernels.

## Platform Boundary

The only runtime evidence here is A3 (`Ascend910_9382`).  A2/910B and 950
physical devices are not present on this host, so no runtime or performance
claim is made for them; per the current goal, A2/950 local measurement is exempt
for this machine scope and the boundary must be carried into the PR text.  The
previous round's offline `Ascend910B3` / `Ascend910_9579` `CompiledKernel`
probes were target-lowering compatibility checks only and remain archived, not
runtime evidence.

## Delivery Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Ascend implementation registered and runnable | PASS | POST benchmark CSV/log |
| Token and sequence paths | PASS | pytest + benchmark, both levels |
| Full pytest, 0 failed / 0 error, `-rs` reasons | PASS | `evidence/20261011/pytest/` |
| Tests and benchmark definitions unchanged | PASS | `git diff --quiet -- test/ benchmark/` |
| Benchmark all configs produce data, no crash/timeout | PASS | default sweep 48 rows / 24 pairs + cross-model sweep 112 rows / 56 pairs, device 2 |
| Liger ≤ Torch baseline on every measured config | PASS | default speed 0.847–0.908 / memory 0.739–0.956; models speed 0.747–0.956 / memory 0.841–0.978 |
| Independent optimization-before/after table, reproducible PRE | PASS | `pre_variant/`, `evidence/20261011/table_default_post_vs_pre.*` |
| PRE gradient correctness along the measured path | PASS | `evidence/20261011/grad_smoke_*` |
| Fresh CANN 9.1 msProf raw export (DB/tables/hashes) | PASS | `evidence/20261011/msprof/` |
| Profiling root cause + optimization explanation | PASS | `PROFILING_AND_OPTIMIZATION.md` |
| A2/950 runtime/performance | EXEMPT (this host) | Platform Boundary above |
| Commit / push / PR submission | NOT DONE | requires steward confirmation |

## Evidence Index

| Path | Content |
| --- | --- |
| `ACTIVE_STATUS.md` | resume handoff, NPU assignment, active commands |
| `REPRODUCE.md` | exact commands with expected results |
| `EVIDENCE_MANIFEST.md` | file/provenance/sha256 table |
| `PROFILING_AND_OPTIMIZATION.md` | msProf root cause + optimization measures |
| `GAPS_AND_NEXT_STEPS.md` | closed gaps and remaining boundaries |
| `PR_BODY_DRAFT.md` | PR material |
| `SHA256SUMS` | hashes of the packet artifacts + tree digests |
| `tools/` | comparison-table, msProf-analysis, gradient-smoke helpers |
| `pre_variant/pre_variant.diff` | reproducible PRE variant (4 toggles) |
| `evidence/20261011/` | this round's raw evidence |
| `evidence/archive_20260921/` | previous round's retained evidence (archived) |
