"""Build performance tables from experiments_manifest.csv.

The manifest fixes which experiment directory belongs to each paper condition.
This script only applies the declared selection rule to logs/train.jsonl and
writes run-level and group-level CSV files.



  uv run python scripts/build_performance_table.py \
    --manifest experiments_manifest.csv \
    --output-dir runs/performance

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

RUN_COLUMNS = [
    "group_id",
    "role",
    "method",
    "selection_rule",
    "realized_selection_rule",
    "seed",
    "p",
    "test_acc",
    "test_loss",
    "val_acc",
    "val_loss",
    "epoch",
    "sparsity",
    "hard_density",
    "topology_freeze_reason",
    "topology_frozen_epoch",
    "topology_best_epoch",
    "topology_best_metric_value",
    "topology_rollback_applied",
    "experiment_dir",
    "notes",
]

SUMMARY_COLUMNS = [
    "group_id",
    "role",
    "method",
    "selection_rule",
    "n",
    "test_acc_mean",
    "test_acc_std",
    "test_acc_median",
    "test_acc_worst",
    "test_acc_best",
    "seed_values",
    "p_values",
    "realized_selection_rules",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build main performance CSV tables.")
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument("--output-dir", default="runs/performance")
    parser.add_argument(
        "--main-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only rows with main_table=yes.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def finite_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def has_metric(row: dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def choose_by_metric(
    rows: list[dict[str, Any]], metric: str, realized_rule: str
) -> tuple[dict[str, Any], str]:
    candidates = [row for row in rows if has_metric(row, metric)]
    if not candidates:
        raise ValueError(f"No rows contain metric {metric!r}")
    # Tie-break by earliest epoch so the rule is deterministic and conservative.
    selected = max(
        candidates, key=lambda row: (float(row[metric]), -int(row.get("epoch", 0)))
    )
    return selected, realized_rule


def select_row(
    log_rows: list[dict[str, Any]], selection_rule: str
) -> tuple[dict[str, Any], str]:
    rule = selection_rule.strip().lower()
    if rule in {"test@best-val", "test@best-val+val_rollback"}:
        return choose_by_metric(log_rows, "val_acc", "test@best-val")
    if rule in {"historical_best-test/no-val", "best-test/no-val"}:
        return choose_by_metric(log_rows, "test_acc", "best-test/no-val")
    if rule == "fixed_baseline":
        if any(has_metric(row, "val_acc") for row in log_rows):
            return choose_by_metric(log_rows, "val_acc", "test@best-val")
        return choose_by_metric(log_rows, "test_acc", "best-test/no-val")
    raise ValueError(f"Unsupported selection_rule: {selection_rule!r}")


def build_run_row(manifest_row: dict[str, str]) -> dict[str, Any]:
    exp_dir = Path(manifest_row["experiment_dir"])
    log_path = exp_dir / "logs" / "train.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    selected, realized_rule = select_row(
        read_jsonl(log_path), manifest_row["selection_rule"]
    )
    freeze_reason = selected.get("topology_freeze_reason") or selected.get(
        "theta_freeze_reason"
    )

    return {
        "group_id": manifest_row["group_id"],
        "role": manifest_row["role"],
        "method": manifest_row["method"],
        "selection_rule": manifest_row["selection_rule"],
        "realized_selection_rule": realized_rule,
        "seed": manifest_row["seed"],
        "p": manifest_row["p"],
        "test_acc": selected.get("test_acc"),
        "test_loss": selected.get("test_loss"),
        "val_acc": selected.get("val_acc"),
        "val_loss": selected.get("val_loss"),
        "epoch": selected.get("epoch"),
        "sparsity": selected.get("sparsity"),
        "hard_density": selected.get("hard_density"),
        "topology_freeze_reason": freeze_reason,
        "topology_frozen_epoch": selected.get("topology_frozen_epoch"),
        "topology_best_epoch": selected.get("topology_best_epoch"),
        "topology_best_metric_value": selected.get("topology_best_metric_value"),
        "topology_rollback_applied": selected.get("topology_rollback_applied"),
        "experiment_dir": manifest_row["experiment_dir"],
        "notes": manifest_row.get("notes", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    group_keys = sorted(
        {(r["group_id"], r["role"], r["method"], r["selection_rule"]) for r in rows}
    )
    for group_id, role, method, selection_rule in group_keys:
        group = [
            r
            for r in rows
            if (r["group_id"], r["role"], r["method"], r["selection_rule"])
            == (group_id, role, method, selection_rule)
        ]
        vals = [finite_float(r["test_acc"]) for r in group]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        p_values = sorted({r["p"] for r in group if r["p"] != ""})
        seed_values = sorted({r["seed"] for r in group if r["seed"] != ""})
        rules = sorted({r["realized_selection_rule"] for r in group})
        summaries.append(
            {
                "group_id": group_id,
                "role": role,
                "method": method,
                "selection_rule": selection_rule,
                "n": len(vals),
                "test_acc_mean": statistics.fmean(vals),
                "test_acc_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "test_acc_median": statistics.median(vals),
                "test_acc_worst": min(vals),
                "test_acc_best": max(vals),
                "seed_values": ";".join(seed_values),
                "p_values": ";".join(p_values),
                "realized_selection_rules": ";".join(rules),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    with manifest_path.open() as f:
        manifest_rows = list(csv.DictReader(f))
    if args.main_only:
        manifest_rows = [
            row for row in manifest_rows if row.get("main_table", "").lower() == "yes"
        ]

    run_rows = [build_run_row(row) for row in manifest_rows]
    summary_rows = summarize(run_rows)

    write_csv(output_dir / "main_performance_runs.csv", run_rows, RUN_COLUMNS)
    write_csv(
        output_dir / "main_performance_summary.csv", summary_rows, SUMMARY_COLUMNS
    )

    print(f"Wrote {output_dir / 'main_performance_runs.csv'}")
    print(f"Wrote {output_dir / 'main_performance_summary.csv'}")


if __name__ == "__main__":
    main()
