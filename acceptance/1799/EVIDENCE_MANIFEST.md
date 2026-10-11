# Evidence Manifest

All entries were produced on 2026-10-11 on device 2 (A3, Phy-ID 2) at revision
`a8c360c` with the grpo322/CANN 9.1.0 environment; commands are in
`REPRODUCE.md`.  `sha256sum -c SHA256SUMS` verifies the hashes below.

## This round (2026-10-11)

| File | sha256 | Role / provenance |
| --- | --- | --- |
| `evidence/20261011/pytest/pytest_full_20261011_dev2.log` | `e04c1704…b9719b0` | Full `pytest -q -rs`; 1024 passed / 566 skipped / 0 failed / 0 error |
| `evidence/20261011/bench_post/all_benchmark_data_1799_post_default_20261011_dev2.csv` | `8db7d726…672560` | POST default sweep CSV (48 rows) |
| `evidence/20261011/bench_post/bench_default_post_20261011_dev2.log` | `49aff4b5…831cfe` | POST benchmark stdout |
| `evidence/20261011/bench_post/all_benchmark_data_1799_post_models_20261011_dev2.csv` | `9f2502c7…1c7c78` | POST cross-model sweep CSV (112 rows) |
| `evidence/20261011/bench_post/bench_models_post_20261011_dev2.log` | `269fd1ab…c93a645` | POST cross-model sweep stdout |
| `evidence/20261011/table_models_post.md` | `99f199f4…4d244c` | POST-only cross-model per-config table |
| `evidence/20261011/table_models_post.csv` | `074ccb02…0a13f3` | Same table, machine-readable |
| `evidence/20261011/bench_pre/all_benchmark_data_1799_pre_default_20261011_dev2.csv` | `6ad6b057…326e1` | PRE default sweep CSV (48 rows, `compiled=False`) |
| `evidence/20261011/bench_pre/bench_default_pre_20261011_dev2.log` | `9bf0aede…6760a` | PRE benchmark stdout |
| `evidence/20261011/bench_pre/bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log` | `65af40d3…e64b7` | Pre-PR default (`compiled=True`) aborting in TorchInductor/dynamo (renamed from `bench_default_pre_20261011_dev2.log`) |
| `evidence/20261011/grad_smoke_post_bfloat16_20261011_dev2.log` | `d64cf1dc…387276` | POST gradient smoke, bf16, B=1/2/4 |
| `evidence/20261011/grad_smoke_post_float32_20261011_dev2.log` | `2a5f092b…960197` | POST gradient smoke, fp32, B=1/2/4 |
| `evidence/20261011/grad_smoke_pre_bfloat16_20261011_dev2.log` | `faa253bb…f31543` | PRE gradient smoke, bf16 (flat ~7 ms timed backward) |
| `evidence/20261011/grad_smoke_pre_float32_20261011_dev2.log` | `9e2ca962…965ea` | PRE gradient smoke, fp32 |
| `evidence/20261011/table_default_post_vs_pre.md` | `f387319a…a13fc1` | Optimization-before/after per-config table (with `*` caveat marks) |
| `evidence/20261011/table_default_post_vs_pre.csv` | `7eea5e0b…dec7a7` | Same table, machine-readable |
| `evidence/20261011/msprof/raw_20261011_dev2/` | tree digest `bc444aa3…f1ac59` (`find -type f \| sort \| xargs sha256sum \| sha256sum`) | msProf raw export: structured CSVs + sqlite DBs, 88 files, 2.7 MB |
| `evidence/20261011/msprof/msprof_collect_20261011_dev2.log` | `96eab9c9…f94ef8` | msprof collector log incl. application timings |
| `evidence/20261011/msprof/msprof_op_summary_20261011_dev2.md` | `41a08fe0…726c15` | Kernel/pipe-ratio analysis |
| `evidence/20261011/msprof/msprof_op_summary_20261011_dev2.csv` | `8f0bc68b…597282` | Same analysis, machine-readable |
| `pre_variant/pre_variant.diff` | `11748ef3…059b96` | PRE variant: 4 toggles (2 dispatch + 2 `compiled`), verified byte-identical to the working-tree `git diff` during the PRE run |
| `tools/make_comparison_table.py` | `28ad2b0a…823849` | Paired/post-only table generator (no NPU) |
| `tools/analyze_msprof.py` | `e9cd0386…9fe0c7` | msProf `op_summary` aggregator (no NPU) |
| `tools/grad_smoke.py` | `8e96c7bb…01ce0e` | Gradient smoke along the benchmark measurement path |
| `evidence/20261011/compile_debug/` | (directory, informational) | TorchInductor debug dumps for the three `torch.compile` runs of this session (pytest pid 3397881, PRE smoke pid 3015182, and the aborted `compiled=True` PRE benchmark pid 3033066); moved here so the accepted checkout stays clean |

The packet documents (`README.md`, `ACCEPTANCE_CHECKLIST.md`, `REPRODUCE.md`,
`PROFILING_AND_OPTIMIZATION.md`, `GAPS_AND_NEXT_STEPS.md`, `PR_BODY_DRAFT.md`,
`ACTIVE_STATUS.md`) are the claim layer and are intentionally outside
`SHA256SUMS`.

## Archived previous round (2026-09-21/23)

`evidence/archive_20260921/` retains the previous packet copies
(`benchmark_default_a3_20260921_npu4.{csv,log}`, `benchmark_models_a3_20260921_npu4.{csv,log}`,
`pytest_a3_20260921_npu4.log`, `offline_target_compile_20260920.log`) plus the
original `SHA256SUMS_20260921_original.txt`, whose hashes were re-verified
against these copies (tree digest
`9f307897…b21a6dff`).  Their `/tmp` source paths no longer exist, so they are
archive-only provenance, not live evidence; the previous round's skip-count
reconstruction and profiling notes are superseded by this round's measured
numbers.

## Renames and deletions

* `bench_default_pre_20261011_dev2.log` → `bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log`
  (the name was reused by the successful eager PRE run).
* `grad_smoke_post_{bf16,fp32}_20261011_dev2.log` → `evidence/20261011/superseded/`
  (first smoke iteration gave each variant its own random `do` scalar; kept for
  audit, superseded by the shared-`do` logs).
* `SHA256SUMS` → `evidence/archive_20260921/SHA256SUMS_20260921_original.txt`.
* The two generated `benchmark/data/all_benchmark_data_1799_{post,pre}_default_20261011_dev2.csv`
  files were byte-verified against the copies above (`8db7d726…`, `6ad6b057…`)
  and then moved out of the accepted checkout to `/tmp/liger_repo_cleanup_20261011/`;
  `git status --porcelain` in `/workspace3/lcw/liger-kernel-ref` is empty.

## Source-state evidence

* POST revision: `a8c360c2381aa1b0bc86a214db3d3b6c1d0613ef`, `git diff` empty;
  six file hashes in `README.md`.
* PRE revision: same two files changed exactly as in
  `pre_variant/pre_variant.diff` (`functional.py` `b083217d…0e9c10`,
  `grpo_loss.py` `8492c634…9fcb817` during the PRE run); the other four files
  unchanged.
* `git diff --quiet -- test/chunked_loss/test_grpo_loss.py` and
  `git diff --quiet -- benchmark/` both pass.
