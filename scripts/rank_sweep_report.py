"""Build CSV summaries and plots for low-rank sweep experiments.

Examples:
  uv run python scripts/rank_sweep_report.py \
    --experiment-prefix lsm_shd_alif_lowrank_readout_spike_adaptation_concat_2606010 \
    --output-dir runs/rank_sweep/alif_spike_adapt_seed42

  uv run python scripts/rank_sweep_report.py \
    --experiments experiments/exp_a experiments/exp_b \
    --selection-rule test@best-val
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_liquid import compute_graph_metrics
from src.lsm.trainer import build_model, get_device
from src.utils.config import load_config


RUN_COLUMNS = [
    "experiment_dir",
    "timestamp",
    "experiment_name",
    "rank",
    "seed",
    "selection_rule",
    "selected_epoch",
    "selected_test_acc",
    "selected_test_loss",
    "selected_val_acc",
    "selected_val_loss",
    "best_val_epoch",
    "best_val_acc",
    "best_val_test_acc",
    "final_epoch",
    "final_test_acc",
    "final_val_acc",
    "hard_density",
    "topology_frozen_epoch",
    "topology_freeze_reason",
    "topology_rollback_applied",
    "theta_param_count",
    "n_liquid",
    "recurrent_mode",
    "in_degree_gini",
    "out_degree_gini",
    "avg_degree_gini",
    "clustering",
    "graph_metrics_source",
]

SUMMARY_COLUMNS = [
    "rank",
    "n",
    "seed_values",
    "theta_param_count",
    "selected_test_acc_mean",
    "selected_test_acc_std",
    "selected_test_acc_worst",
    "selected_test_acc_best",
    "best_val_acc_mean",
    "hard_density_mean",
    "hard_density_std",
    "final_test_acc_mean",
    "in_degree_gini_mean",
    "in_degree_gini_std",
    "out_degree_gini_mean",
    "out_degree_gini_std",
    "avg_degree_gini_mean",
    "avg_degree_gini_std",
    "clustering_mean",
    "clustering_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize rank sweep runs into CSV tables and plots."
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[],
        help="Explicit experiment directories to include.",
    )
    parser.add_argument(
        "--experiment-prefix",
        action="append",
        default=[],
        help="Experiment directory name prefix to match under experiments/.",
    )
    parser.add_argument(
        "--experiment-name",
        action="append",
        default=[],
        help="Match runs by config.yaml experiment_name instead of directory naming.",
    )
    parser.add_argument(
        "--config-equals",
        action="append",
        default=[],
        help="Extra config filter in dotted.path=value form, e.g. liquid.recurrent_mode=learned_lowrank",
    )
    parser.add_argument(
        "--experiments-root",
        default="experiments",
        help="Root directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write CSV and PNG outputs.",
    )
    parser.add_argument(
        "--selection-rule",
        choices=["auto", "test@best-val", "best-test/no-val"],
        default="auto",
        help="How to select the representative test score for each run.",
    )
    parser.add_argument(
        "--expected-ranks",
        nargs="*",
        type=int,
        default=[],
        help="Expected rank grid. If provided, missing ranks/seeds are reported.",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="*",
        type=int,
        default=[],
        help="Expected seed grid. If provided, missing ranks/seeds are reported.",
    )
    parser.add_argument(
        "--timestamp-from",
        default="",
        help="Only include experiment dirs whose trailing YYMMDDHHMMSS timestamp is >= this value.",
    )
    parser.add_argument(
        "--timestamp-to",
        default="",
        help="Only include experiment dirs whose trailing YYMMDDHHMMSS timestamp is <= this value.",
    )
    parser.add_argument(
        "--dedupe-rank-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse duplicate runs with the same (rank, seed).",
    )
    parser.add_argument(
        "--include-graph-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach graph metrics from diagnose_topology.json or checkpoint topology.",
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


def _has_metric(row: dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _choose_by_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    candidates = [row for row in rows if _has_metric(row, metric)]
    if not candidates:
        raise ValueError(f"No rows contain metric {metric!r}")
    return max(candidates, key=lambda row: (float(row[metric]), -int(row["epoch"])))


def choose_selected_row(
    rows: list[dict[str, Any]], selection_rule: str
) -> tuple[dict[str, Any], str]:
    if selection_rule == "test@best-val":
        return _choose_by_metric(rows, "val_acc"), "test@best-val"
    if selection_rule == "best-test/no-val":
        return _choose_by_metric(rows, "test_acc"), "best-test/no-val"
    if any(_has_metric(row, "val_acc") for row in rows):
        return _choose_by_metric(rows, "val_acc"), "test@best-val"
    return _choose_by_metric(rows, "test_acc"), "best-test/no-val"


def _cfg_get(cfg: dict[str, Any], dotted_path: str) -> Any:
    value: Any = cfg
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _dir_timestamp(exp_dir: Path) -> str | None:
    match = re.search(r"(\d{12})$", exp_dir.name)
    return match.group(1) if match else None


def _matches_filters(exp_dir: Path, args: argparse.Namespace) -> bool:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        return False
    cfg = yaml.safe_load(cfg_path.read_text())

    if args.experiment_name and cfg.get("experiment_name") not in args.experiment_name:
        return False

    for item in args.config_equals:
        if "=" not in item:
            raise ValueError(f"--config-equals must be dotted.path=value, got {item!r}")
        dotted_path, raw_value = item.split("=", 1)
        expected = yaml.safe_load(raw_value)
        if _cfg_get(cfg, dotted_path) != expected:
            return False

    ts = _dir_timestamp(exp_dir)
    if args.timestamp_from:
        if ts is None or ts < args.timestamp_from:
            return False
    if args.timestamp_to:
        if ts is None or ts > args.timestamp_to:
            return False
    return True


def resolve_experiments(args: argparse.Namespace) -> list[Path]:
    resolved = [Path(exp) for exp in args.experiments]
    root = Path(args.experiments_root)
    for prefix in args.experiment_prefix:
        matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
        resolved.extend(matches)
    if args.experiment_name or args.config_equals or args.timestamp_from or args.timestamp_to:
        matches = sorted(
            p for p in root.iterdir() if p.is_dir() and _matches_filters(p, args)
        )
        resolved.extend(matches)
    unique = sorted({p.resolve() for p in resolved})
    if not unique:
        raise ValueError("No experiments matched. Pass --experiments or --experiment-prefix.")
    return [Path(p) for p in unique]


def _load_graph_metrics_from_checkpoint(exp_dir: Path) -> tuple[dict[str, float], str]:
    diag_path = exp_dir / "diagnose_topology.json"
    if diag_path.exists():
        obj = json.loads(diag_path.read_text())
        graph = obj.get("graph_metrics")
        if isinstance(graph, dict):
            return graph, "diagnose_topology.json"

    cfg = load_config(exp_dir / "config.yaml")
    device = get_device()
    model = build_model(cfg, device)
    ckpt = torch.load(exp_dir / "checkpoints" / "best.pt", map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    torch.manual_seed(cfg.seed)
    model.liquid.unlock_epoch_mask()
    model.liquid.sample_mask(tau=cfg.tau_end)
    mask = model.liquid.get_binary_mask().detach().cpu().bool()
    dale_sign = model.liquid.dale_sign.detach().cpu().reshape(-1)
    graph = compute_graph_metrics(
        mask,
        dale_sign,
        skip_cycle_metrics=True,
        skip_clustering=False,
    )
    return graph, "checkpoint"


def build_run_row(
    exp_dir: Path, selection_rule: str, include_graph_metrics: bool
) -> dict[str, Any]:
    cfg_path = exp_dir / "config.yaml"
    log_path = exp_dir / "logs" / "train.jsonl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log: {log_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    rows = read_jsonl(log_path)
    if not rows:
        raise ValueError(f"Empty log file: {log_path}")
    selected, realized_rule = choose_selected_row(rows, selection_rule)
    final = rows[-1]
    best_val_row = None
    if any(_has_metric(row, "val_acc") for row in rows):
        best_val_row = _choose_by_metric(rows, "val_acc")

    liquid = cfg["liquid"]
    n_liquid = int(liquid["n_liquid"])
    rank = int(liquid["theta_rank"])
    graph_metrics: dict[str, float] = {}
    graph_source = ""
    if include_graph_metrics:
        graph_metrics, graph_source = _load_graph_metrics_from_checkpoint(exp_dir)
    in_gini = graph_metrics.get("in_degree_gini")
    out_gini = graph_metrics.get("out_degree_gini")
    avg_gini = None
    if in_gini is not None and out_gini is not None:
        avg_gini = 0.5 * (float(in_gini) + float(out_gini))

    return {
        "experiment_dir": exp_dir.name,
        "experiment_name": cfg.get("experiment_name", exp_dir.name),
        "rank": rank,
        "seed": cfg.get("seed"),
        "selection_rule": realized_rule,
        "selected_epoch": selected.get("epoch"),
        "selected_test_acc": selected.get("test_acc"),
        "selected_test_loss": selected.get("test_loss"),
        "selected_val_acc": selected.get("val_acc"),
        "selected_val_loss": selected.get("val_loss"),
        "best_val_epoch": None if best_val_row is None else best_val_row.get("epoch"),
        "best_val_acc": None if best_val_row is None else best_val_row.get("val_acc"),
        "best_val_test_acc": None if best_val_row is None else best_val_row.get("test_acc"),
        "final_epoch": final.get("epoch"),
        "final_test_acc": final.get("test_acc"),
        "final_val_acc": final.get("val_acc"),
        "hard_density": selected.get("hard_density"),
        "topology_frozen_epoch": selected.get("topology_frozen_epoch"),
        "topology_freeze_reason": selected.get("topology_freeze_reason")
        or selected.get("theta_freeze_reason"),
        "topology_rollback_applied": selected.get("topology_rollback_applied"),
        "theta_param_count": 2 * n_liquid * rank + 1,
        "n_liquid": n_liquid,
        "recurrent_mode": liquid.get("recurrent_mode"),
        "in_degree_gini": in_gini,
        "out_degree_gini": out_gini,
        "avg_degree_gini": avg_gini,
        "clustering": graph_metrics.get("clustering"),
        "graph_metrics_source": graph_source,
        "timestamp": _dir_timestamp(exp_dir) or "",
    }


def summarize(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank in sorted({int(row["rank"]) for row in run_rows}):
        group = [row for row in run_rows if int(row["rank"]) == rank]
        selected_scores = [float(row["selected_test_acc"]) for row in group]
        densities = [float(row["hard_density"]) for row in group]
        best_vals = [
            float(row["best_val_acc"]) for row in group if row["best_val_acc"] is not None
        ]
        final_scores = [float(row["final_test_acc"]) for row in group]
        in_ginis = [float(row["in_degree_gini"]) for row in group if row["in_degree_gini"] is not None]
        out_ginis = [float(row["out_degree_gini"]) for row in group if row["out_degree_gini"] is not None]
        avg_ginis = [float(row["avg_degree_gini"]) for row in group if row["avg_degree_gini"] is not None]
        clusterings = [float(row["clustering"]) for row in group if row["clustering"] is not None]
        out.append(
            {
                "rank": rank,
                "n": len(group),
                "seed_values": ";".join(str(row["seed"]) for row in sorted(group, key=lambda r: int(r["seed"]))),
                "theta_param_count": group[0]["theta_param_count"],
                "selected_test_acc_mean": statistics.fmean(selected_scores),
                "selected_test_acc_std": statistics.stdev(selected_scores)
                if len(selected_scores) > 1
                else 0.0,
                "selected_test_acc_worst": min(selected_scores),
                "selected_test_acc_best": max(selected_scores),
                "best_val_acc_mean": statistics.fmean(best_vals) if best_vals else None,
                "hard_density_mean": statistics.fmean(densities),
                "hard_density_std": statistics.stdev(densities) if len(densities) > 1 else 0.0,
                "final_test_acc_mean": statistics.fmean(final_scores),
                "in_degree_gini_mean": statistics.fmean(in_ginis) if in_ginis else None,
                "in_degree_gini_std": statistics.stdev(in_ginis) if len(in_ginis) > 1 else 0.0,
                "out_degree_gini_mean": statistics.fmean(out_ginis) if out_ginis else None,
                "out_degree_gini_std": statistics.stdev(out_ginis) if len(out_ginis) > 1 else 0.0,
                "avg_degree_gini_mean": statistics.fmean(avg_ginis) if avg_ginis else None,
                "avg_degree_gini_std": statistics.stdev(avg_ginis) if len(avg_ginis) > 1 else 0.0,
                "clustering_mean": statistics.fmean(clusterings) if clusterings else None,
                "clustering_std": statistics.stdev(clusterings) if len(clusterings) > 1 else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_constant_fields(run_rows: list[dict[str, Any]]) -> None:
    constant_fields = ["experiment_name", "n_liquid", "recurrent_mode", "selection_rule"]
    varying = []
    for field in constant_fields:
        values = {row[field] for row in run_rows}
        if len(values) > 1:
            varying.append((field, sorted(values)))
    if varying:
        print("[warn] Sweep mixes settings beyond rank/seed:")
        for field, values in varying:
            print(f"  - {field}: {values}")


def audit_rank_seed_grid(
    run_rows: list[dict[str, Any]], expected_ranks: list[int], expected_seeds: list[int]
) -> None:
    pairs: dict[tuple[int, int], list[str]] = {}
    for row in run_rows:
        key = (int(row["rank"]), int(row["seed"]))
        pairs.setdefault(key, []).append(str(row["experiment_dir"]))

    duplicates = {key: value for key, value in pairs.items() if len(value) > 1}
    if duplicates:
        print("[warn] Duplicate (rank, seed) entries detected:")
        for (rank, seed), exp_dirs in sorted(duplicates.items()):
            print(f"  - rank={rank}, seed={seed}: {exp_dirs}")

    if expected_ranks and expected_seeds:
        missing = [
            (rank, seed)
            for rank in sorted(expected_ranks)
            for seed in sorted(expected_seeds)
            if (rank, seed) not in pairs
        ]
        if missing:
            print("[warn] Missing expected (rank, seed) entries:")
            for rank, seed in missing:
                print(f"  - rank={rank}, seed={seed}")


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_rank_seed(run_rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (int(row["rank"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)

    deduped: list[dict[str, Any]] = []
    for (rank, seed), group in sorted(grouped.items()):
        if len(group) == 1:
            deduped.append(group[0])
            continue

        fingerprints = {}
        for row in group:
            exp_dir = root / row["experiment_dir"]
            cfg_hash = _sha1(exp_dir / "config.yaml")
            log_hash = _sha1(exp_dir / "logs" / "train.jsonl")
            ckpt_path = exp_dir / "checkpoints" / "best.pt"
            ckpt_hash = _sha1(ckpt_path) if ckpt_path.exists() else ""
            fingerprints[row["experiment_dir"]] = (cfg_hash, log_hash, ckpt_hash)

        unique_fps = set(fingerprints.values())
        if len(unique_fps) == 1:
            chosen = max(group, key=lambda row: row["timestamp"])
            print(
                "[warn] Duplicate identical runs collapsed for "
                f"rank={rank}, seed={seed}: {[row['experiment_dir'] for row in group]} -> {chosen['experiment_dir']}"
            )
            deduped.append(chosen)
            continue

        chosen = max(
            group,
            key=lambda row: (
                float(row["selected_val_acc"]) if row["selected_val_acc"] is not None else float("-inf"),
                float(row["selected_test_acc"]),
                row["timestamp"],
            ),
        )
        print(
            "[warn] Duplicate non-identical runs detected for "
            f"rank={rank}, seed={seed}; keeping best selected run {chosen['experiment_dir']}"
        )
        deduped.append(chosen)
    return deduped


def save_plots(run_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    run_df = pd.DataFrame(run_rows).sort_values(["rank", "seed"])
    summary_df = pd.DataFrame(summary_rows).sort_values("rank")

    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, seed_df in run_df.groupby("seed"):
        ax.plot(
            seed_df["rank"],
            seed_df["selected_test_acc"],
            marker="o",
            linewidth=1.2,
            alpha=0.5,
            label=f"seed {seed}",
        )
    ax.plot(
        summary_df["rank"],
        summary_df["selected_test_acc_mean"],
        color="black",
        linewidth=2.0,
        marker="o",
        label="mean",
    )
    if len(summary_df) > 1:
        mean = summary_df["selected_test_acc_mean"]
        std = summary_df["selected_test_acc_std"]
        ax.fill_between(summary_df["rank"], mean - std, mean + std, alpha=0.15, color="black")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Selected test accuracy")
    ax.set_title("Rank sweep performance")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_vs_test_acc.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        summary_df["rank"],
        summary_df["selected_test_acc_mean"],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Rank r")
    ax1.set_ylabel("Selected test accuracy", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["rank"],
        summary_df["hard_density_mean"],
        color="#d62728",
        marker="s",
        linewidth=1.8,
    )
    ax2.set_ylabel("Hard density", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Performance and learned density by rank")
    fig.tight_layout()
    fig.savefig(output_dir / "rank_vs_test_acc_and_density.png", dpi=180)
    plt.close(fig)

    graph_cols = [
        "in_degree_gini_mean",
        "out_degree_gini_mean",
        "avg_degree_gini_mean",
        "clustering_mean",
    ]
    if all(col in summary_df.columns for col in graph_cols) and not summary_df[graph_cols].isna().all().all():
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        ax = axes[0]
        ax.plot(
            summary_df["rank"],
            summary_df["selected_test_acc_mean"],
            color="black",
            linewidth=2.0,
            marker="o",
            label="mean test acc",
        )
        if len(summary_df) > 1:
            mean = summary_df["selected_test_acc_mean"]
            std = summary_df["selected_test_acc_std"]
            ax.fill_between(summary_df["rank"], mean - std, mean + std, alpha=0.15, color="black")
        ax2 = ax.twinx()
        ax2.plot(
            summary_df["rank"],
            summary_df["hard_density_mean"],
            color="#d62728",
            linewidth=1.6,
            marker="s",
            label="mean density",
        )
        ax.set_ylabel("Test accuracy")
        ax2.set_ylabel("Hard density", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.grid(True, alpha=0.25)
        ax.set_title("Rank sweep: performance, density, and graph diagnostics")

        ax = axes[1]
        ax.plot(summary_df["rank"], summary_df["in_degree_gini_mean"], marker="o", label="in-degree gini")
        ax.plot(summary_df["rank"], summary_df["out_degree_gini_mean"], marker="o", label="out-degree gini")
        ax.plot(summary_df["rank"], summary_df["clustering_mean"], marker="o", label="clustering")
        ax.set_ylabel("Raw graph metric")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=9)

        ax = axes[2]
        overlay = pd.DataFrame(
            {
                "rank": summary_df["rank"],
                "test_acc": summary_df["selected_test_acc_mean"],
                "avg_gini": summary_df["avg_degree_gini_mean"],
                "clustering": summary_df["clustering_mean"],
            }
        )
        for col, label, color in [
            ("test_acc", "test acc (norm)", "black"),
            ("avg_gini", "avg gini (norm)", "#1f77b4"),
            ("clustering", "clustering (norm)", "#2ca02c"),
        ]:
            vals = overlay[col].astype(float)
            vmin = vals.min()
            vmax = vals.max()
            norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else vals * 0.0
            ax.plot(overlay["rank"], norm, marker="o", linewidth=2.0, label=label, color=color)
        ax.set_ylabel("Min-max normalized")
        ax.set_xlabel("Rank r")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=9)

        left_confounded_max = 16
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.axvspan(summary_df["rank"].min(), left_confounded_max, color="#f4d9d9", alpha=0.18)

        axes[2].text(
            0.02,
            0.04,
            "Caution: left-half / low-r region co-varies with learned density; interpret mechanism there with density-confound caution.",
            transform=axes[2].transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        fig.tight_layout()
        fig.savefig(output_dir / "rank_vs_performance_and_graph_metrics.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    experiments = resolve_experiments(args)
    run_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for exp_dir in experiments:
        try:
            run_rows.append(
                build_run_row(
                    exp_dir,
                    args.selection_rule,
                    args.include_graph_metrics,
                )
            )
        except ValueError as exc:
            skipped.append(f"{exp_dir.name}: {exc}")
    if not run_rows:
        raise ValueError("No completed runs were available to summarize.")
    if args.dedupe_rank_seed:
        run_rows = dedupe_rank_seed(run_rows, Path(args.experiments_root))
    run_rows.sort(key=lambda row: (int(row["rank"]), int(row["seed"]), row["experiment_dir"]))
    validate_constant_fields(run_rows)
    audit_rank_seed_grid(run_rows, args.expected_ranks, args.expected_seeds)
    summary_rows = summarize(run_rows)

    write_csv(output_dir / "rank_sweep_runs.csv", run_rows, RUN_COLUMNS)
    write_csv(output_dir / "rank_sweep_summary.csv", summary_rows, SUMMARY_COLUMNS)
    save_plots(run_rows, summary_rows, output_dir)

    print(f"Wrote {output_dir / 'rank_sweep_runs.csv'}")
    print(f"Wrote {output_dir / 'rank_sweep_summary.csv'}")
    print(f"Wrote {output_dir / 'rank_vs_test_acc.png'}")
    print(f"Wrote {output_dir / 'rank_vs_test_acc_and_density.png'}")
    if args.include_graph_metrics:
        print(f"Wrote {output_dir / 'rank_vs_performance_and_graph_metrics.png'}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete runs:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
