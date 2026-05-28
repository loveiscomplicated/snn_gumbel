"""Summarize topology diagnostics using the experiment manifest and performance table."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any


SUMMARY_METRICS = [
    "test_acc",
    "density",
    "n_active_edges",
    "in_gini",
    "out_gini",
    "in_degree_std",
    "out_degree_std",
    "edge_EE_frac",
    "edge_EI_frac",
    "edge_IE_frac",
    "edge_II_frac",
    "reciprocity",
    "three_cycle_density",
    "clustering_undirected",
    "largest_weak_component",
    "largest_strong_component",
    "reachability_ratio",
    "avg_shortest_path_reachable",
    "effective_diameter_p90",
    "readout_in_degree_corr",
    "readout_out_degree_corr",
    "readout_total_degree_corr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize diagnostics by manifest group.")
    parser.add_argument("--manifest", default="experiments_manifest.csv")
    parser.add_argument("--diagnostics-dir", default="runs/diagnostics/main_g1_g3_g4")
    parser.add_argument(
        "--performance-runs",
        default="runs/performance/main_performance_runs.csv",
        help="Run-level performance table from build_performance_table.py.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def exp_key(path_or_name: str) -> str:
    return Path(path_or_name).name


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.fmean(vals), statistics.stdev(vals)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = read_csv(Path(args.manifest))
    perf_rows = read_csv(Path(args.performance_runs))
    diag_dir = Path(args.diagnostics_dir)
    graph_rows = read_csv(diag_dir / "graph_metrics.csv")

    manifest_by_exp = {exp_key(r["experiment_dir"]): r for r in manifest}
    perf_by_exp = {exp_key(r["experiment_dir"]): r for r in perf_rows}

    joined: list[dict[str, Any]] = []
    for row in graph_rows:
        key = exp_key(row["experiment_dir"])
        manifest_row = manifest_by_exp.get(key)
        if manifest_row is None:
            continue
        perf_row = perf_by_exp.get(key, {})
        joined_row = {
            **row,
            "group_id": manifest_row["group_id"],
            "role": manifest_row["role"],
            "method": manifest_row["method"],
            "selection_rule": manifest_row["selection_rule"],
            "manifest_seed": manifest_row["seed"],
            "p": manifest_row["p"],
            "test_acc": perf_row.get("test_acc", ""),
            "selected_epoch": perf_row.get("epoch", ""),
            "realized_selection_rule": perf_row.get("realized_selection_rule", ""),
        }
        joined.append(joined_row)

    if not joined:
        raise RuntimeError("No graph rows matched manifest entries.")

    group_keys = sorted({(r["group_id"], r["role"], r["method"], r["selection_rule"]) for r in joined})
    summary_rows: list[dict[str, Any]] = []
    for group_id, role, method, selection_rule in group_keys:
        group = [
            r for r in joined
            if (r["group_id"], r["role"], r["method"], r["selection_rule"])
            == (group_id, role, method, selection_rule)
        ]
        out: dict[str, Any] = {
            "group_id": group_id,
            "role": role,
            "method": method,
            "selection_rule": selection_rule,
            "n": len(group),
            "seeds": ";".join(sorted({r["manifest_seed"] for r in group if r["manifest_seed"]})),
            "p_values": ";".join(sorted({r["p"] for r in group if r["p"]})),
        }
        for metric in SUMMARY_METRICS:
            values = [to_float(r.get(metric)) for r in group]
            mean, std = mean_std(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        summary_rows.append(out)

    joined_columns = [
        "group_id",
        "role",
        "method",
        "selection_rule",
        "realized_selection_rule",
        "manifest_seed",
        "p",
        "test_acc",
        "selected_epoch",
        *graph_rows[0].keys(),
    ]
    summary_columns = [
        "group_id",
        "role",
        "method",
        "selection_rule",
        "n",
        "seeds",
        "p_values",
    ]
    for metric in SUMMARY_METRICS:
        summary_columns.extend([f"{metric}_mean", f"{metric}_std"])

    write_csv(diag_dir / "diagnostics_joined_runs.csv", joined, joined_columns)
    write_csv(diag_dir / "diagnostics_group_summary.csv", summary_rows, summary_columns)
    print(f"Wrote {diag_dir / 'diagnostics_joined_runs.csv'}")
    print(f"Wrote {diag_dir / 'diagnostics_group_summary.csv'}")


if __name__ == "__main__":
    main()
