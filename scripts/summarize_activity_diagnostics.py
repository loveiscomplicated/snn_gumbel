"""Summarize diagnose_liquid.py activity CSV by manifest group."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any


ACTIVITY_METRICS = [
    "firing_rate_mean",
    "firing_rate_max",
    "dead_neurons",
    "active_gt001",
    "active_gt005",
    "overactive_gt020",
    "recurrent_input_ratio",
    "cosine_mean",
    "cosine_min",
    "cosine_max",
    "readout_margin_mean",
    "readout_margin_min",
    "readout_margin_max",
    "readout_class_mean_accuracy",
    "readout_accuracy",
    "readout_sample_margin_mean",
    "readout_correct_margin_mean",
    "readout_incorrect_margin_mean",
    "readout_num_samples",
    "readout_num_correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize activity diagnostics.")
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument(
        "--activity-csv",
        default="runs/diagnostics/main_g1_g3_g4/activity/activity_diagnostics.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="runs/diagnostics/main_g1_g3_g4/activity/activity_group_summary.csv",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse exact duplicate rows with the same exp before summarizing.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def mean_std(rows: list[dict[str, str]], metric: str) -> tuple[float, float]:
    vals = [to_float(row.get(metric)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.stdev(vals)


def dedupe_exact_repeated_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop exact repeated rows for the same experiment.

    run_activity_diagnostics.py appends to an existing CSV, so rerunning the same
    command can duplicate rows. Exact duplicates are not independent diagnostic
    observations and should not inflate group n.
    """
    seen: set[tuple[tuple[str, str], ...]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    args = parse_args()
    manifest = read_csv(Path(args.manifest))
    activity_rows = read_csv(Path(args.activity_csv))
    if args.dedupe:
        activity_rows = dedupe_exact_repeated_runs(activity_rows)

    manifest_by_run = {Path(row["experiment_dir"]).name: row for row in manifest}
    joined: list[dict[str, str]] = []
    for row in activity_rows:
        manifest_row = manifest_by_run.get(row["exp"])
        if manifest_row is None:
            continue
        joined.append({**row, **{
            "group_id": manifest_row["group_id"],
            "role": manifest_row["role"],
            "method": manifest_row["method"],
            "selection_rule": manifest_row["selection_rule"],
            "p": manifest_row["p"],
        }})

    group_keys = sorted({(r["group_id"], r["role"], r["method"], r["selection_rule"]) for r in joined})
    output_rows: list[dict[str, Any]] = []
    for group_id, role, method, selection_rule in group_keys:
        group = [
            row for row in joined
            if (row["group_id"], row["role"], row["method"], row["selection_rule"])
            == (group_id, role, method, selection_rule)
        ]
        out: dict[str, Any] = {
            "group_id": group_id,
            "role": role,
            "method": method,
            "selection_rule": selection_rule,
            "n_runs": len({row["exp"] for row in group}),
            "n_rows": len(group),
            "seeds": ";".join(sorted({row["seed"] for row in group if row["seed"]})),
            "p_values": ";".join(sorted({row["p"] for row in group if row["p"]})),
        }
        for metric in ACTIVITY_METRICS:
            mean, std = mean_std(group, metric)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        output_rows.append(out)

    columns = [
        "group_id",
        "role",
        "method",
        "selection_rule",
        "n_runs",
        "n_rows",
        "seeds",
        "p_values",
    ]
    for metric in ACTIVITY_METRICS:
        columns.extend([f"{metric}_mean", f"{metric}_std"])

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
