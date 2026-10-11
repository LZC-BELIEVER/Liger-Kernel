# PRE variant (optimization-before) for Issue #1799

## Definition

`pre_variant.diff` converts revision `a8c360c` into the pre-optimization state
for this operator on Ascend with exactly four toggles, two per dispatch site:

| File | Toggle |
| --- | --- |
| `src/liger_kernel/chunked_loss/functional.py` (`liger_fused_linear_grpo`) | `if _input.device.type == "npu":` → `if False:` |
| `src/liger_kernel/chunked_loss/functional.py` (generic call) | `compiled,` → `False,` |
| `src/liger_kernel/chunked_loss/grpo_loss.py` (`LigerFusedLinearGRPOLoss.forward`) | `if _input.device.type == "npu":` → `if False:` |
| `src/liger_kernel/chunked_loss/grpo_loss.py` (generic call) | `self.compiled,` → `False,` |

With the two NPU branches disabled the run falls back to the code path that
existed before this PR — `LigerFusedLinearGRPOFunction` /
`LigerFusedLinearPPOBase.chunk_forward`, i.e. the upstream generic fused-linear
path: torch matmul streaming with a 4096-wide vocabulary chunk plus the
`torch.compile`-d PPO loss math.

The `compiled` toggle is part of the PRE definition because the pre-PR default
(`compiled=True`) **cannot complete the Issue benchmark on this stack**:
TorchInductor/dynamo aborts while building the shape guard for the next swept
shape (`AssertionError: s37 … not in {...}` raised from
`torch/fx/experimental/symbolic_shapes.py::_print_Symbol`, after guard
allocation at `_inductor/codegen/simd.py::can_use_32bit_indexing`).  The raw
failure log is retained in the packet:

```
docs/lzc/1799/evidence/20261011/bench_pre/bench_default_pre_generic_compiled_true_FAILED_20261011_dev2.log
```

(that file was produced as `bench_default_pre_20261011_dev2.log` and renamed
when the successful eager PRE run reused the name; the rename is recorded in
`EVIDENCE_MANIFEST.md`).  The PRE numbers in the comparison table therefore
measure the pre-PR implementation in its only runnable configuration on this
stack, under the unchanged benchmark/test definitions.

No test or benchmark definition is touched by this diff.

## Apply / revert

```bash
cd /workspace3/lcw/liger-kernel-ref          # must be clean at a8c360c
git apply docs/…/pre_variant.diff            # POST -> PRE   (verify with git diff)
git apply -R docs/…/pre_variant.diff         # PRE  -> POST  (restore)
git diff --quiet && echo restored
```

## Reproducibility contract

* Base revision: `a8c360c2381aa1b0bc86a214db3d3b6c1d0613ef` (branch
  `feat/ascend-fused-linear-grpo-loss`).
* `pre_variant.diff` is the only source change.  During the PRE run the working
  tree `git diff` was verified to contain exactly the same changed lines as
  this file, and the PRE-tagged source sha256 values (4 toggles ⇒ two files
  differ from POST) are recorded in `EVIDENCE_MANIFEST.md` together with the
  file's own `sha256`.
* Post-optimization source hashes before/after the PRE benchmark are identical;
  both are listed in `EVIDENCE_MANIFEST.md`, which is the proof that the final
  evidence set closes on one revision.
* The PRE and POST benchmarks use the same host, CANN 9.1.0 environment, device,
  benchmark script, and sampling protocol; only this diff differs.
