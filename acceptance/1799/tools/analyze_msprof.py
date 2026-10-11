#!/usr/bin/env python3
"""Summarize a CANN msProf ``op_summary`` export for Issue #1799.

The export is produced by::

    msprof --output=<dir> --ai-core=on --aic-metrics=PipeUtilization \
           --ascendcl=off --application=<run_profile_grpo.sh> -- --B <B> ...

``<dir>/PROF_*/mindstudio_profiler_output/op_summary_*.csv`` carries one row per
executed kernel task with the AI Core / AI Vector pipe ratios.  This helper
aggregates them per kernel so the loss-path kernels can be ranked against the
aclnn projection/layout operators that surround them.

Usage::

    python analyze_msprof.py <op_summary.csv> [<op_summary.csv> ...] \
        --out-md out.md --out-csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

LOSS_KERNEL_MARKERS = ("_grpo_loss", "_selective_log_softmax")
PIPE_COLUMNS = (
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
    "aic_mac_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_fixpipe_ratio",
    "aic_scalar_ratio",
)


def _f(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def analyze(paths):
    stats = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0, "ratios": defaultdict(float), "weight": 0.0}
    )
    for path in paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = row.get("Op Name", "")
                if not name:
                    continue
                acc = stats[name]
                dur = _f(row, "Task Duration(us)")
                acc["count"] += 1
                acc["duration_us"] += dur
                # duration-weighted pipe ratios
                w = dur if dur > 0 else 1.0
                acc["weight"] += w
                for col in PIPE_COLUMNS:
                    acc["ratios"][col] += _f(row, col) * w
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op_summary", nargs="+")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    stats = analyze(args.op_summary)
    rows = []
    for name, acc in stats.items():
        rec = {
            "kernel": name,
            "count": acc["count"],
            "total_ms": acc["duration_us"] / 1000.0,
            "mean_us": acc["duration_us"] / max(1, acc["count"]),
            "is_loss_path": any(m in name for m in LOSS_KERNEL_MARKERS),
        }
        for col in PIPE_COLUMNS:
            rec[col] = acc["ratios"][col] / acc["weight"] if acc["weight"] else 0.0
        rows.append(rec)
    rows.sort(key=lambda r: -r["total_ms"])
    total_ms = sum(r["total_ms"] for r in rows)
    loss_ms = sum(r["total_ms"] for r in rows if r["is_loss_path"])

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# msProf op_summary analysis",
        "",
        f"- source: {', '.join(args.op_summary)}",
        f"- distinct kernels: {len(rows)}; total task time: {total_ms:.3f} ms",
        f"- loss-path kernels (`_grpo_loss*`, `_selective_log_softmax*`): {loss_ms:.3f} ms "
        f"({100.0 * loss_ms / total_ms if total_ms else 0:.1f}% of task time)",
        "",
        "| kernel | calls | total ms | mean us | vec ratio | mte2 ratio (AIV) | mte1 ratio (AIC) | mac ratio (AIC) | fixpipe ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rec in rows:
        lines.append(
            "| {k} | {c} | {t:.3f} | {m:.1f} | {v:.4f} | {m2:.4f} | {m1:.4f} | {mac:.4f} | {fp:.4f} |".format(
                k=rec["kernel"],
                c=rec["count"],
                t=rec["total_ms"],
                m=rec["mean_us"],
                v=rec["aiv_vec_ratio"],
                m2=rec["aiv_mte2_ratio"],
                m1=rec["aic_mte1_ratio"],
                mac=rec["aic_mac_ratio"],
                fp=rec["aic_fixpipe_ratio"],
            )
        )
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines[:6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
