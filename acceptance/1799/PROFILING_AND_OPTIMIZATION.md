# Profiling Root Cause And Optimization Measures

Status: **VERIFIED with retained raw evidence** (this file replaces the previous
round's `UNVERIFIED / NOT READY` historical notes, which are archived under
`evidence/archive_20260921/`).

## Raw export provenance

| Item | Value |
| --- | --- |
| Collector | CANN 9.1.0 `msprof` (`/workspace3/lcw/.local/Ascend-9.1.0/cann-9.1.0/aarch64-linux/bin/msprof`) |
| Device | `ASCEND_RT_VISIBLE_DEVICES=2` (A3, Phy-ID 2, `Ascend910_9382`) |
| Export root | `evidence/20261011/msprof/raw_20261011_dev2/PROF_000001_20261011131902494_02849883CQDRHHEI/` |
| Structured tables | `mindstudio_profiler_output/{op_summary,op_statistic,task_time,api_statistic}_20261011131915.csv` |
| Databases | `msprof_20261011131914.db`, `device_2/sqlite/{ai_core_op_summary,op_counter,ascend_task,metric_summary,…}.db`, `host/sqlite/*.db` |
| Size / files | 2.7 MB / 88 files (tree digest in `SHA256SUMS`) |
| Application log | `evidence/20261011/msprof/msprof_collect_20261011_dev2.log` |
| Analysis output | `evidence/20261011/msprof/msprof_op_summary_20261011_dev2.{md,csv}` |

Exact command (application wrapper `evidence/20261011/msprof/run_profile_grpo.sh`
drives `profile_grpo_path.py`, which mirrors the benchmark setup
`LigerLMHeadGRPO` with llama_3_8b shapes: B=4, T=1024, H=4096, V=128256, bf16,
forward + backward, token and sequence levels, 1 warm-up + 2 timed iterations):

```bash
export ASCEND_RT_VISIBLE_DEVICES=2
msprof --output=…/evidence/20261011/msprof/raw_20261011_dev2 \
  --ai-core=on --aic-metrics=PipeUtilization --ascendcl=off \
  --application=…/evidence/20261011/msprof/run_profile_grpo.sh \
  -- --B 4 --iters 2 --warmup 1 --levels token,sequence
```

The application itself reported 80.209 / 80.163 ms per timed iteration for the
token level and 82.517 / 82.476 ms for the sequence level, i.e. the profiled run
reproduces the benchmark's `full` numbers (79.97 ms / 82.28 ms at B=4 in
`evidence/20261011/bench_post/`).

## Measured kernel breakdown

34 distinct kernels, 487.115 ms total task time (device-side task duration as
reported by `op_summary`).  Duration-weighted pipe ratios are shown for the
loss-path kernels; the projection is a Cube op (`aic_*`).

| Kernel | Calls | Total ms | Share | AIV vec ratio | AIV MTE2 ratio | AIC mac ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aclnnMatmul_MatMulV3Common_MatMulV3` (lm-head projection) | 24 | 350.952 | 72.0% | — | — | 0.856 (MTE1 0.734) |
| `_selective_log_softmax_kernel` | 9 | 20.335 | 4.2% | 0.6493 | 0.2393 | — |
| `aclnnCat_ConcatD_ConcatD` (pad row cat) | 12 | 19.529 | 4.0% | 0.0303 | 0.9627 | — |
| `aclnnMatmul_TransData_TransData` (layout) | 12 | 11.836 | 2.4% | 0.0530 | 0.8838 | — |
| `aclnnAdd_TransposeAiCore_Transpose` | 4 | 11.684 | 2.4% | 0.0390 | 0.6819 | — |
| `aclnnAdd_AddAiCore_Add` (bias add) | 22 | 11.385 | 2.3% | 0.1971 | 0.9135 | — |
| `aclnnInplaceCopy_SliceV2AiCore_SliceV2` | 6 | 9.941 | 2.0% | 0.0000 | 0.6732 | — |
| `aclnnMatmul_SliceV2AiCore_SliceV2` | 6 | 9.808 | 2.0% | 0.0000 | 0.6837 | — |
| `_grpo_loss_bwd_kernel` | 3 | 7.140 | 1.5% | 0.6997 | 0.5223 | — |
| `_grpo_loss_bwd_kernel_seq` | 3 | 6.979 | 1.4% | 0.7187 | 0.5127 | — |
| `_grpo_loss_fwd_kernel` | 3 | 6.892 | 1.4% | 0.6443 | 0.2333 | — |
| `_grpo_loss_fwd_kernel_seq` | 3 | 6.859 | 1.4% | 0.6463 | 0.2333 | — |
| `aclnnInplaceNormal_DSARandomNormal` (benchmark `do`) | 6 | 6.341 | 1.3% | — | — | — |
| `aclnnInplaceCopy_TransposeAiCore_Transpose` | 2 | 5.861 | 1.2% | 0.0390 | 0.6835 | — |
| remaining 20 kernels (ZerosLike ×84, ReduceSum ×57, Mul ×60, Div ×39, …) | — | ≈ 3.0 | 0.6% | — | — | — |

Aggregates:

* **Loss path** (`_grpo_loss*`, `_selective_log_softmax*`): **48.205 ms = 9.9%**
  of device task time.
* **Pad / layout plumbing** (`ConcatD`, `TransData`, `Transpose`, `Add`,
  `SliceV2`): **≈ 80.0 ms = 16.4%** — the cost of presenting `[B, T+1, V]` in
  the registered kernels' required layout and format.
* **lm-head projection**: 72% — the operator's own math, irreducible by the
  adapter (it is the `[B·T, H] × [H, V]` GEMM plus the reference-model
  projection).

## Root cause reading

1. The fused loss kernels are **Vector-pipe bound**: forward vec 0.644/0.646
   with MTE2 0.233, backward vec 0.700/0.719 with MTE2 0.522/0.513.  The
   backward kernels carry roughly 2× the MTE2 pressure of the forward kernels
   because they stream the `dlogits [B, T+1, V]` gradient back out, and at
   0.51–0.52 MTE2 they are close to balanced against the Vector pipe — the
   remaining headroom inside the loss kernels is UB/Vector efficiency, not
   launch count.
2. The dominant cost of the measured path is the lm-head projection (72%) and
   the pad/layout conversions (16%), both of which sit *outside* the fused loss
   kernels.  Any further end-to-end work on this operator has to attack the
   `[B, T+1, V]` materialisation/layout chain (for example avoiding the extra
   pad row or the NZ conversions), not the loss kernels.
3. Consequently the before/after gain measured in the benchmark comes from
   **how much the pre-PR path over-pays relative to this fused adapter**, not
   from a faster loss kernel: the pre-PR upstream path streams the vocabulary in
   4096-wide torch matmul chunks (32 passes over the vocabulary per sequence
   chunk) and recomputes gradients per chunk inside its forward pass, which is
   why it lands at 2.68–3.08× the Torch baseline while the adapter lands at
   0.85–0.88×.

Historical cross-check: the previous round's unverifiable notes claimed forward
vector ratio ≈0.63–0.68, backward vector ≈0.72–0.79 and backward MTE2 ≈0.43–0.51.
The retained raw export now reproduces 0.644/0.646 (forward vec), 0.700/0.719
(backward vec) and 0.522/0.513 (backward MTE2) — same order, now evidence-backed.

## Optimization measures (source-level, in `a8c360c`)

1. **Ascend fused-linear adapter** —
   `src/liger_kernel/ops/backends/_ascend/ops/fused_linear_grpo.py` (new file,
   320 lines).  The registered Ascend GRPO kernels consume transformer-style
   logits with one extra row (`[B, T+1, V]`), while Liger's fused-linear API
   starts from hidden states.  The adapter owns the projection, the bias and the
   zero pad row, and drives `fused_selective_log_softmax` + `GrpoLossFunction`
   per row chunk, so the retained activation is `chunk_rows × (T+1) × V`
   instead of a materialised `[B, T, V]` FP32 buffer with per-chunk
   recomputation.
2. **Row coalescing under a retained-logits budget** — for the public default
   `chunk_size=1`, rows are coalesced while
   `chunk_rows × (T+1) × V × dtype_bytes ≤ 2 GiB`, removing per-row launch and
   allocation overhead (and the per-row `_project_and_pad` cat allocations).
   Measured trade-off: at B=4 the coalesced chunk retains 8.23 GB vs the
   streaming PRE path's 7.17 GB, while time drops from 286.5 ms to 80.0 ms
   (token level, full mode).
3. **Reuse of the UB-aware Ascend kernels instead of vocabulary streaming** —
   `BLOCK_N` 8192 (forward / selective log-softmax) and 4096 (backward) from
   `compute_default_tiling_strategy`, a `(B, grid_seq)` grid capped by the core
   count, dynamic `tl.range` vocabulary traversal and host-side token windowing;
   the pre-PR path instead performed 32 torch matmuls per sequence chunk.
4. **Chunk-local reduction with per-loss-type normalizers** (`_loss_normalizer`,
   `_chunk_reduced_loss`) so chunked accumulation reproduces the reference
   reduction (`grpo`/`bnpo`/`dr_grpo`/`dapo`/`cispo`/`vespo`/`sapo`/`luspo`).
5. **Sequence-path correctness** — `LUSPO` constexpr handling in
   `_grpo_loss_fwd_kernel_seq` / `_grpo_loss_bwd_kernel_seq` (mask/padding
   semantics and valid-token weighted backward), which the unchanged pytest
   suite exercises at both importance-sampling levels.

## Supporting correctness evidence for the profiling path

`tools/grad_smoke.py` (logs `evidence/20261011/grad_smoke_*_20261011_dev2.log`)
replays the benchmark's backward call for both variants: losses are bit-identical
to `TorchLMHeadGRPO`, gradients agree with the reference inside the test file's
own tolerances (bf16 rel-L2 ≈1e-6 POST / 3.2e-3 PRE; fp32 ≈1e-7), and the
POST timed backward scales with batch size (14.4 → 43.3 ms bf16) whereas the PRE
timed backward is flat at ~7 ms because it only replays stored gradients — see
`README.md` "Measurement caveat".
