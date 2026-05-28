"""Fill topology_diagnostics activity_metrics.csv from diagnose_liquid output.

topology_diagnostics.py writes a graph-first placeholder activity_metrics.csv.
run_activity_diagnostics.py writes the actual per-run activity diagnostics under
an activity/ subdirectory. This script joins the two by manifest/config identity
and overwrites the placeholder with populated activity rows.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import yaml


ACTIVITY_COLUMNS = [
    "experiment_name",
    "method_label",
    "recurrent_mode",
    "seed",
    "activity_diagnostics_available",
    "num_batches",
    "num_samples",
    "mean_firing_rate",
    "max_firing_rate",
    "active_neurons_gt_005",
    "rec_input_abs_ratio",
    "class_mean_cosine_mean",
    "mean_logit_margin",
    "accuracy_on_diagnostic_batches",
    "skip_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate activity_metrics.csv from activity_diagnostics.csv."
    )
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument(
        "--activity-csv",
        default=None,
        help="Defaults to <diagnostics-dir>/activity/activity_diagnostics.csv.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Diagnostic batches used by run_activity_diagnostics.py.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def config_experiment_name(experiment_dir: str) -> str:
    config_path = Path(experiment_dir) / "config.yaml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return str(cfg.get("experiment_name") or Path(experiment_dir).name)


def dedupe_activity_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def populated_row(
    placeholder: dict[str, str],
    activity: dict[str, str] | None,
    num_batches: int,
) -> dict[str, Any]:
    if activity is None:
        row = {column: placeholder.get(column, "") for column in ACTIVITY_COLUMNS}
        row["activity_diagnostics_available"] = False
        return row

    num_samples = as_float(activity.get("readout_num_samples"))
    if not math.isfinite(num_samples):
        num_samples = float("nan")

    return {
        "experiment_name": placeholder["experiment_name"],
        "method_label": placeholder["method_label"],
        "recurrent_mode": placeholder["recurrent_mode"],
        "seed": placeholder["seed"],
        "activity_diagnostics_available": True,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "mean_firing_rate": as_float(activity.get("firing_rate_mean")),
        "max_firing_rate": as_float(activity.get("firing_rate_max")),
        "active_neurons_gt_005": as_float(activity.get("active_gt005")),
        "rec_input_abs_ratio": as_float(activity.get("recurrent_input_ratio")),
        "class_mean_cosine_mean": as_float(activity.get("cosine_mean")),
        # Sample-level diagnostic logit margin; class-mean margins remain in
        # activity/activity_diagnostics.csv as readout_margin_*.
        "mean_logit_margin": as_float(activity.get("readout_sample_margin_mean")),
        "accuracy_on_diagnostic_batches": as_float(activity.get("readout_accuracy")),
        "skip_reason": "",
    }


def main() -> None:
    args = parse_args()
    diagnostics_dir = Path(args.diagnostics_dir)
    placeholder_path = diagnostics_dir / "activity_metrics.csv"
    activity_path = (
        Path(args.activity_csv)
        if args.activity_csv
        else diagnostics_dir / "activity" / "activity_diagnostics.csv"
    )

    placeholders = read_csv(placeholder_path)
    manifest = read_csv(Path(args.manifest))
    activity_rows = dedupe_activity_rows(read_csv(activity_path))
    activity_by_dir = {row["exp"]: row for row in activity_rows}

    dir_by_experiment_name: dict[str, str] = {}
    for row in manifest:
        exp_dir = row["experiment_dir"]
        if not Path(exp_dir).exists():
            continue
        dir_by_experiment_name[config_experiment_name(exp_dir)] = Path(exp_dir).name

    output_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for placeholder in placeholders:
        experiment_name = placeholder["experiment_name"]
        exp_dir_name = dir_by_experiment_name.get(experiment_name)
        activity = activity_by_dir.get(exp_dir_name or "")
        if activity is None:
            missing.append(experiment_name)
        output_rows.append(populated_row(placeholder, activity, args.num_batches))

    write_csv(placeholder_path, output_rows, ACTIVITY_COLUMNS)
    available = sum(
        str(row["activity_diagnostics_available"]).lower() == "true"
        for row in output_rows
    )
    print(f"Wrote {placeholder_path}")
    print(f"activity_diagnostics_available: {available}/{len(output_rows)}")
    if missing:
        print("Missing activity rows:", ", ".join(missing))


if __name__ == "__main__":
    main()
