# Reproduction Commands

All commands run from `/workspace3/lcw/liger-kernel-ref` (accepted revision
`a8c360c`, working tree clean) with the grpo322 environment and CANN 9.1.0.
`PACKET=/workspace3/lcw/SLAI-Ascend-OpForge/docs/lzc/1799`.

Common prefix:

```bash
source /workspace3/lcw/.local/Ascend-9.1.0/cann-9.1.0/aarch64-linux/bin/setenv.bash
export ASCEND_RT_VISIBLE_DEVICES=2          # A3 device 2 (Phy-ID 2)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_UB_CAPACITY_BITS=1572864
```

## 1. Full correctness suite

```bash
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_pytest_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_pytest_dev2
PYTHONPATH=src /workspace2/lanzichang/envs/grpo322/bin/python \
  -m pytest -q test/chunked_loss/test_grpo_loss.py --disable-warnings -rs
```

Expected (measured 2026-10-11, device 2): `1024 passed, 566 skipped,
0 failed, 0 error` in 995.18s, with the `-rs` breakdown
288 + 96 + 144 + 24 + 12 + 2 = 566.  Log:
`$PACKET/evidence/20261011/pytest/pytest_full_20261011_dev2.log`.

## 2. Default benchmark (POST = accepted revision)

```bash
cd /workspace3/lcw/liger-kernel-ref
export LIGER_BENCH_TARGET=1799_post_default_20261011_dev2
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_post_default_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_post_default_dev2
export TORCH_COMPILE_DEBUG=0
PYTHONPATH=src:benchmark/scripts /workspace2/lanzichang/envs/grpo322/bin/python \
  benchmark/scripts/benchmark_grpo_loss.py
cp benchmark/data/all_benchmark_data_${LIGER_BENCH_TARGET}.csv $PACKET/evidence/20261011/bench_post/
```

Expected: 48 rows, 24 Liger/Torch pairs, 0 regressions; speed 0.846733–0.908310,
memory 0.738682–0.956061.  `LIGER_BENCH_TARGET` keeps the run out of the
repository's tracked `benchmark/data/all_benchmark_data.csv`.

## 3. PRE variant (optimization before)

```bash
cd /workspace3/lcw/liger-kernel-ref
git apply $PACKET/pre_variant/pre_variant.diff      # POST -> PRE (4 toggles)
git diff                                            # verify against the diff file
export LIGER_BENCH_TARGET=1799_pre_default_20261011_dev2
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_pre_default_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_pre_default_dev2
PYTHONPATH=src:benchmark/scripts /workspace2/lanzichang/envs/grpo322/bin/python \
  benchmark/scripts/benchmark_grpo_loss.py
cp benchmark/data/all_benchmark_data_${LIGER_BENCH_TARGET}.csv $PACKET/evidence/20261011/bench_pre/
git apply -R $PACKET/pre_variant/pre_variant.diff    # PRE -> POST (restore)
git diff --quiet && echo restored
```

Expected PRE result: the upstream generic path at 2.68–3.08× the Torch baseline
in `full` mode (see `README.md`).  The default `compiled=True` form of this path
aborts in TorchInductor/dynamo on the second swept shape; that failure log is
retained as
`evidence/20261011/bench_pre/bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log`.

## 4. Gradient smoke (same measurement path)

```bash
cd /workspace3/lcw/liger-kernel-ref
# POST (revision unchanged):
PYTHONPATH=src /workspace2/lanzichang/envs/grpo322/bin/python \
  $PACKET/tools/grad_smoke.py --variant POST --dtype bfloat16 --levels token --B 1,2,4
PYTHONPATH=src /workspace2/lanzichang/envs/grpo322/bin/python \
  $PACKET/tools/grad_smoke.py --variant POST --dtype float32  --levels token --B 1,2,4
# PRE: re-apply pre_variant.diff first, repeat with --variant PRE, then revert.
```

Expected: loss identical to `TorchLMHeadGRPO`; gradient relative-L2 differences
~1e-6 (POST bf16) / ~1e-7 (fp32 both variants) / ~3e-3 (PRE bf16), all passing
the test file's tolerances; POST timed backward scales with B (14→43 ms bf16),
PRE timed backward is flat at ~7 ms.

## 5. msProf raw collection (CANN 9.1)

```bash
export TRITON_CACHE_DIR=/tmp/triton_grpo_20261011_msprof_dev2
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_grpo_20261011_msprof_dev2
/workspace3/lcw/.local/Ascend-9.1.0/cann-9.1.0/aarch64-linux/bin/msprof \
  --output=$PACKET/evidence/20261011/msprof/raw_20261011_dev2 \
  --ai-core=on --aic-metrics=PipeUtilization --ascendcl=off \
  --application=$PACKET/evidence/20261011/msprof/run_profile_grpo.sh \
  -- --B 4 --iters 2 --warmup 1 --levels token,sequence
python3 $PACKET/tools/analyze_msprof.py \
  $PACKET/evidence/20261011/msprof/raw_20261011_dev2/PROF_*/mindstudio_profiler_output/op_summary_*.csv \
  --out-md $PACKET/evidence/20261011/msprof/msprof_op_summary_20261011_dev2.md \
  --out-csv $PACKET/evidence/20261011/msprof/msprof_op_summary_20261011_dev2.csv
```

`run_profile_grpo.sh` drives `profile_grpo_path.py`, which mirrors the benchmark
setup (`LigerLMHeadGRPO`, llama_3_8b shapes, forward+backward) for both
importance-sampling levels.  Expected: 34 kernels, 487.1 ms task time, 9.9% in
the `_grpo_loss*`/`_selective_log_softmax*` kernels; the app reports ~80.2 ms
and ~82.5 ms per timed iteration for token and sequence.

## 6. Comparison table

```bash
python3 $PACKET/tools/make_comparison_table.py \
  --post $PACKET/evidence/20261011/bench_post/all_benchmark_data_1799_post_default_20261011_dev2.csv \
  --pre  $PACKET/evidence/20261011/bench_pre/all_benchmark_data_1799_pre_default_20261011_dev2.csv \
  --label "default token_length sweep; POST a8c360c vs PRE generic-eager" \
  --pre-caveat-modes backward,forward \
  --out-md  $PACKET/evidence/20261011/table_default_post_vs_pre.md \
  --out-csv $PACKET/evidence/20261011/table_default_post_vs_pre.csv
```

## 7. Evidence integrity

```bash
cd $PACKET && sha256sum -c SHA256SUMS
```
