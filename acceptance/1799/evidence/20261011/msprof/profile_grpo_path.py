"""Reproduce the Issue #1799 benchmark GRPO configuration for msProf collection.

This script mirrors ``benchmark/scripts/benchmark_grpo_loss.py`` (its
``setup_grpo_loss`` + ``LigerLMHeadGRPO`` usage) for the default
``llama_3_8b`` benchmark configuration, and runs forward + backward for both
``token`` and ``sequence`` importance-sampling levels so the profiler captures
the real loss-path kernels:

  * ``_selective_log_softmax_kernel``
  * ``_grpo_loss_fwd_kernel`` / ``_grpo_loss_fwd_kernel_seq``
  * ``_grpo_loss_bwd_kernel`` / ``_grpo_loss_bwd_kernel_seq``

It only drives the public Liger API with random inputs of the benchmark shapes;
it does not implement, replace, or fall back to any other operator.

Run it from the liger-kernel-ref checkout with ``PYTHONPATH=src``:

    PYTHONPATH=src python profile_grpo_path.py --B 4 --iters 2 --warmup 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# The benchmark imports ``test.chunked_loss.test_grpo_loss`` from the
# liger-kernel-ref checkout; reproduce that import root here (cwd is the
# checkout when driven through run_profile_grpo.sh).
sys.path.insert(0, os.environ.get("LIGER_REPO_ROOT", os.getcwd()))


def _build_inputs(B, T, H, V, dtype):
    x = torch.randn(B, T, H, requires_grad=True, dtype=dtype, device="npu")
    selected_token_ids = torch.randint(0, V, (B, T), dtype=torch.long, device="npu")
    attention_mask = torch.ones(B, T, device="npu")
    advantages = torch.randn(B, dtype=dtype, device="npu")
    ref_input = torch.randn(B, T, H, dtype=dtype, device="npu")
    return x, selected_token_ids, attention_mask, advantages, ref_input


def _run_level(level, args):
    from test.chunked_loss.test_grpo_loss import LigerLMHeadGRPO

    dtype = getattr(torch, args.dtype)
    module = LigerLMHeadGRPO(
        H=args.H,
        V=args.V,
        dtype=dtype,
        importance_sampling_level=level,
    ).to("npu")
    x, ids, mask, adv, ref_input = _build_inputs(args.B, args.T, args.H, args.V, dtype)

    def step():
        if x.grad is not None:
            x.grad = None
        out = module(x, ids, mask, adv, ref_input=ref_input)[0]
        out.backward()
        return out

    for _ in range(args.warmup):
        step()
    torch.npu.synchronize()

    times = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = step()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    print(
        f"[profile_grpo_path] level={level} B={args.B} T={args.T} H={args.H} V={args.V} "
        f"dtype={args.dtype} iters={args.iters} loss={float(out.detach().cpu()):.6f} "
        f"ms={[round(t, 3) for t in times]}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--V", type=int, default=128256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--levels", type=str, default="token,sequence")
    args = parser.parse_args()

    for level in args.levels.split(","):
        _run_level(level.strip(), args)


if __name__ == "__main__":
    main()
