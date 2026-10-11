#!/usr/bin/env python3
"""Build the Issue #1799 per-configuration comparison table from benchmark CSVs.

Inputs are raw ``all_benchmark_data*.csv`` files produced by
``benchmark/scripts/benchmark_grpo_loss.py`` (``--sweep-mode token_length`` or
``model_config``).  Every row is paired on
``(kernel_name, kernel_operation_mode, metric_name, x_value,
extra_benchmark_config_str)`` between the ``liger`` and ``torch`` providers, and
across the pre-/post-optimization runs.

Example:

    python make_comparison_table.py \
        --post  evidence/20261011/bench_post/all_benchmark_data_..._post.csv \
        --pre   evidence/20261011/bench_pre/all_benchmark_data_..._pre.csv \
        --label default \
        --out-md  evidence/20261011/table_default.md \
        --out-csv evidence/20261011/table_default.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict

MODE_ORDER = {"forward": 0, "backward": 1, "full": 2}


def _load(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def _key(row):
    return (
        row["kernel_name"],
        row["kernel_operation_mode"],
        row["metric_name"],
        row["x_value"],
        row["extra_benchmark_config_str"],
    )


def _index(rows, provider):
    out = OrderedDict()
    for row in rows:
        if row["kernel_provider"] != provider:
            continue
        out[_key(row)] = row
    return out


def _cfg_label(cfg_str):
    cfg = json.loads(cfg_str)
    parts = []
    if "hidden_size" in cfg:
        parts.append(f"H{cfg['hidden_size']}")
    if "vocab_size" in cfg:
        parts.append(f"V{cfg['vocab_size']}")
    if "dtype" in cfg:
        parts.append(str(cfg["dtype"]).replace("torch.", ""))
    if "T" in cfg:
        parts.append(f"T{cfg['T']}")
    if "importance_sampling_level" in cfg:
        parts.append(cfg["importance_sampling_level"])
    return "/".join(parts)


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True, help="post-optimization benchmark CSV")
    ap.add_argument(
        "--pre",
        default=None,
        help="pre-optimization benchmark CSV; omit for a POST-only table (e.g. a sweep "
        "that has no PRE counterpart)",
    )
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument(
        "--pre-caveat-modes",
        default="",
        help=(
            "Comma-separated modes whose PRE number is a measurement artifact and must "
            "not be read as pre-optimization performance (rendered with '*' and n/a delta)."
        ),
    )
    ap.add_argument(
        "--pre-caveat-text",
        default=(
            "`*` marks a row whose PRE number is not the pre-optimization cost of that "
            "mode: the pre-PR upstream path performs its real backward inside the forward "
            "pass (`LigerFusedLinearPPOBase.forward` -> `torch.autograd.grad` per chunk) "
            "and its `backward` entry only replays the stored gradients.  PRE `forward` "
            "therefore includes the backward work and PRE `backward` measures only the "
            "replay, so only the `full` rows are comparable.  See "
            "`PROFILING_AND_OPTIMIZATION.md`."
        ),
    )
    args = ap.parse_args()
    replay_only_modes = {m.strip() for m in args.pre_caveat_modes.split(",") if m.strip()}

    post_liger = _index(_load(args.post), "liger")
    post_torch = _index(_load(args.post), "torch")
    pre_liger = _index(_load(args.pre), "liger") if args.pre else {}

    def _sort_key(k):
        raw_x = k[3]
        try:
            x_ord = (0, float(raw_x), "")
        except (TypeError, ValueError):
            x_ord = (1, 0.0, str(raw_x))
        return (k[0], MODE_ORDER.get(k[1], 9), x_ord, k[4])

    keys = sorted(set(post_liger) | set(post_torch) | set(pre_liger), key=_sort_key)

    records = []
    for key in keys:
        kernel, mode, metric, x_value, cfg = key
        post_l = float(post_liger[key]["y_value_50"]) if key in post_liger else None
        post_t = float(post_torch[key]["y_value_50"]) if key in post_torch else None
        pre_l = float(pre_liger[key]["y_value_50"]) if key in pre_liger else None
        ratio = None if (post_l is None or post_t is None or post_t == 0) else post_l / post_t
        pre_ratio = None if (pre_l is None or post_t is None or post_t == 0) else pre_l / post_t
        delta = None if (pre_l in (None, 0) or post_l is None) else (post_l / pre_l - 1.0) * 100.0
        records.append(
            {
                "kernel_name": kernel,
                "mode": mode,
                "metric": metric,
                "config": _cfg_label(cfg),
                "x": x_value,
                "pre_liger": pre_l,
                "post_liger": post_l,
                "torch": post_t,
                "post_ratio": ratio,
                "pre_ratio": pre_ratio,
                "pre_to_post_pct": delta,
                "pre_note": (
                    "PRE measurement is replay-only for this mode; not interpretable as "
                    "pre-optimization performance"
                    if mode in replay_only_modes
                    else ""
                ),
            }
        )

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = [f"# Per-configuration comparison ({args.label})", ""]
    lines.append("`pre_liger` = pre-optimization Liger; `post_liger` = final revision Liger.")
    lines.append("`ratio` = Liger / Torch baseline (lower is better).")
    if not args.pre:
        lines.append("")
        lines.append(
            "This is a **POST-only** table: the sweep has no PRE counterpart, so the "
            "pre-optimization columns are `n/a` by design."
        )
    if replay_only_modes:
        lines.append("")
        lines.append(args.pre_caveat_text)
    lines.append("")
    current = None
    for rec in records:
        head = (rec["kernel_name"], rec["mode"], rec["metric"])
        if head != current:
            current = head
            lines.append("")
            lines.append(f"## {head[0]} / {head[1]} / {head[2]}")
            lines.append("")
            lines.append("| config | x | pre Liger | post Liger | Torch | pre ratio | post ratio | post vs pre |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append(
            "| {cfg} | {x} | {pre} | {post} | {torch} | {pr} | {po} | {d} |".format(
                cfg=rec["config"],
                x=rec["x"],
                pre=(
                    "n/a" if rec["pre_liger"] is None else f"{rec['pre_liger']:.4f}" + (" *" if rec["pre_note"] else "")
                ),
                post=_fmt(rec["post_liger"]),
                torch=_fmt(rec["torch"]),
                pr=(
                    "n/a" if rec["pre_ratio"] is None else f"{rec['pre_ratio']:.4f}" + (" *" if rec["pre_note"] else "")
                ),
                po=_fmt(rec["post_ratio"]),
                d=(
                    "n/a *"
                    if rec["pre_note"]
                    else ("n/a" if rec["pre_to_post_pct"] is None else f"{rec['pre_to_post_pct']:+.2f}%")
                ),
            )
        )

    speed = [r for r in records if r["metric"] == "speed" and r["post_ratio"] is not None]
    memory = [r for r in records if r["metric"] == "memory" and r["post_ratio"] is not None]
    regressions = [r for r in records if r["post_ratio"] is not None and r["post_ratio"] > 1.0]
    lines += ["", "## Summary", ""]
    for name, group in (("speed", speed), ("memory", memory)):
        if not group:
            continue
        ratios = [r["post_ratio"] for r in group]
        lines.append(
            f"- {name}: {len(group)} Liger/Torch pairs, max ratio {max(ratios):.6f}, "
            f"min ratio {min(ratios):.6f}, regressions (>1.0): {sum(1 for r in ratios if r > 1.0)}"
        )
    lines.append(f"- total comparisons: {len(records)}; regressions >1.0: {len(regressions)}")
    if regressions:
        lines.append("")
        lines.append("REGRESSIONS:")
        for rec in regressions:
            lines.append(
                f"  - {rec['kernel_name']}/{rec['mode']}/{rec['config']}/x={rec['x']} "
                f"liger={rec['post_liger']:.4f} torch={rec['torch']:.4f} ratio={rec['post_ratio']:.6f}"
            )

    with open(args.out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines[-6:]))
    return 0 if not regressions else 1


if __name__ == "__main__":
    sys.exit(main())
