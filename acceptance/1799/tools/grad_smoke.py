#!/usr/bin/env python3
"""Gradient smoke test along the Issue #1799 benchmark measurement path.

``benchmark/scripts/utils.py::run_speed_benchmark`` measures the ``backward``
mode as: build the graph once, then time ``y.backward(do, retain_graph=True)``
with ``grad_to_none`` clearing ``.grad`` between repetitions.  This helper
replays exactly that call and reports:

* the loss, ``x.grad`` and ``lin.weight.grad`` magnitudes per batch size,
* whether the repeated timed call is deterministic (pure replay) or recomputes,
* agreement with ``TorchLMHeadGRPO`` (the unchanged reference) under the test
  file's own per-dtype tolerance contract,
* wall time of one timed ``backward`` call.

Run from the liger-kernel-ref checkout with ``PYTHONPATH=src``:

    python grad_smoke.py --variant POST --dtype bfloat16 --B 1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  (registers the npu device)

sys.path.insert(0, os.environ.get("LIGER_REPO_ROOT", os.getcwd()))

# Tolerances come from test/chunked_loss/test_grpo_loss.py::test_correctness.
TOLERANCE = {
    "bfloat16": (1e-1, 5e-1),
    "float32": (2e-5, 1e-3),
}


def _grad_stats(tensor):
    flat = tensor.detach().float()
    return {
        "absmax": float(flat.abs().max()),
        "l2": float(flat.norm()),
        "nonzero": int(torch.count_nonzero(flat)),
        "numel": int(flat.numel()),
    }


def _rel_diff(cand, ref):
    a = cand.detach().float()
    b = ref.detach().float()
    denom = b.abs().max().clamp(min=1e-12)
    return float((a - b).abs().max() / denom)


def _rel_l2(cand, ref):
    a = cand.detach().float()
    b = ref.detach().float()
    denom = b.norm().clamp(min=1e-12)
    return float((a - b).norm() / denom)


def run_case(B, level, H, V, T, dtype, seed):
    from test.chunked_loss.test_grpo_loss import LigerLMHeadGRPO
    from test.chunked_loss.test_grpo_loss import TorchLMHeadGRPO
    from test.utils import set_seed

    set_seed(seed)
    liger = LigerLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=level).to("npu")
    ref_mod = TorchLMHeadGRPO(H=H, V=V, dtype=dtype, importance_sampling_level=level).to("npu")
    ref_mod.load_state_dict({k: v.clone() for k, v in liger.state_dict().items()})

    set_seed(seed + 1)
    x0 = torch.randn(B, T, H, dtype=dtype, device="npu")
    ref_x = torch.randn(B, T, H, dtype=dtype, device="npu")
    ids = torch.randint(0, V, (B, T), device="npu")
    mask = torch.ones(B, T, device="npu")
    adv = torch.randn(B, dtype=dtype, device="npu")

    atol, rtol = TOLERANCE[str(dtype).replace("torch.", "")]
    result = {"B": B, "level": level, "dtype": str(dtype).replace("torch.", "")}
    grads = {}

    # One shared upstream gradient scalar for both implementations so that the
    # comparison isolates the operator, not the random `do` draw.  The benchmark
    # draws its own `do` per provider; for correctness both sides must see the
    # same one.
    set_seed(seed + 2)
    do_ref = torch.randn((), dtype=torch.float32, device="npu")

    for name, module in (("candidate", liger), ("reference", ref_mod)):
        x = x0.detach().clone().requires_grad_(True)
        y = module(x, ids, mask, adv, ref_input=ref_x)[0]
        do = do_ref.to(y.dtype)
        torch.npu.synchronize()

        def one_timed_backward():
            x.grad = None
            module.lin.weight.grad = None
            y.backward(do, retain_graph=True)
            torch.npu.synchronize()
            return x.grad.clone(), module.lin.weight.grad.clone()

        one_timed_backward()  # warm-up (first call may compile)
        t0 = time.perf_counter()
        gx1, gw1 = one_timed_backward()
        dt_ms = (time.perf_counter() - t0) * 1e3
        gx2, gw2 = one_timed_backward()

        grads[name] = {"x": gx1, "w": gw1}
        result[f"{name}_loss"] = float(y.detach().cpu())
        result[f"{name}_backward_ms"] = round(dt_ms, 3)
        result[f"{name}_xgrad"] = _grad_stats(gx1)
        result[f"{name}_wgrad"] = _grad_stats(gw1)
        result[f"{name}_timed_call_deterministic"] = bool(
            torch.equal(gx1, gx2) and torch.equal(gw1, gw2)
        )

    for tag, key in (("xgrad", "x"), ("wgrad", "w")):
        cand = grads["candidate"][key]
        ref = grads["reference"][key]
        result[f"{tag}_rel_max_diff"] = _rel_diff(cand, ref)
        result[f"{tag}_rel_l2_diff"] = _rel_l2(cand, ref)
        result[f"{tag}_allclose"] = bool(
            torch.allclose(cand.float(), ref.float(), atol=atol, rtol=rtol)
        )
    result["tolerance"] = {"atol": atol, "rtol": rtol}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="POST")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(TOLERANCE))
    ap.add_argument("--levels", default="token")
    ap.add_argument("--B", default="1,2,4")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--V", type=int, default=128256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    for level in args.levels.split(","):
        for B in [int(b) for b in args.B.split(",")]:
            res = run_case(B, level.strip(), args.H, args.V, args.T, dtype, args.seed)
            res["variant"] = args.variant
            print("SMOKE " + json.dumps(res, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
