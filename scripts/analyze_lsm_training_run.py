"""Analyze one LSM training run without using test accuracy for selection.

Examples:
  uv run python scripts/analyze_lsm_training_run.py \
    --experiment_dir experiments/my_lsm_run

  uv run python scripts/analyze_lsm_training_run.py \
    --experiment_dir experiments/my_lsm_run --stdout_log copied_stdout.txt

  uv run python scripts/analyze_lsm_training_run.py \
    --experiment_dir experiments/my_lsm_run --log_file copied_stdout.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


TOPOLOGY_GRAD_EXPLOSION_THRESHOLD = 50.0
FIRING_RUNAWAY_THRESHOLD = 0.9
LOW_TEST_GAP_THRESHOLD = 0.03
DEFAULT_TIE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize best-val, test-at-best-val, topology, and firing diagnostics for one LSM run."
    )
    parser.add_argument(
        "--experiment_dir",
        required=True,
        help="Experiment directory containing logs/train.jsonl, diagnostics/, or a copied stdout log.",
    )
    parser.add_argument(
        "--stdout_log",
        default="",
        help="Optional copied stdout log text file. Used when JSONL metrics are unavailable.",
    )
    parser.add_argument(
        "--log_file",
        default="",
        help="Alias for --stdout_log.",
    )
    parser.add_argument(
        "--tie_epsilon",
        type=float,
        default=DEFAULT_TIE_EPSILON,
        help="Treat val_acc values within this epsilon as tied.",
    )
    parser.add_argument(
        "--topology_grad_threshold",
        type=float,
        default=TOPOLOGY_GRAD_EXPLOSION_THRESHOLD,
        help="Threshold for first topology gradient explosion epoch.",
    )
    parser.add_argument(
        "--low_test_gap",
        type=float,
        default=LOW_TEST_GAP_THRESHOLD,
        help="Warn when test@best_val trails nearby/final test by at least this amount.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


METRIC_ALIASES = {
    "topology_grad_pre_clip": [
        "topology_grad_pre_clip",
        "topology_grad_norm_pre_clip",
        "theta_grad_norm_pre_clip",
        "theta_grad_norm",
    ],
    "topology_grad_post_clip": [
        "topology_grad_post_clip",
        "topology_grad_norm_post_clip",
        "theta_grad_norm_post_clip",
    ],
}


def metric(row: dict[str, Any], name: str) -> float | None:
    for key in METRIC_ALIASES.get(name, [name]):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def epoch(row: dict[str, Any]) -> int | None:
    return safe_int(row.get("epoch"))


def has_metric(row: dict[str, Any], name: str) -> bool:
    return metric(row, name) is not None


def merge_rows_by_epoch(*row_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epoch: dict[int, dict[str, Any]] = {}
    without_epoch: list[dict[str, Any]] = []
    for rows in row_sets:
        for row in rows:
            ep = epoch(row)
            if ep is None:
                without_epoch.append(dict(row))
                continue
            merged = by_epoch.setdefault(ep, {"epoch": ep})
            for key, value in row.items():
                if value is not None or key not in merged:
                    merged[key] = value
    merged_rows = list(by_epoch.values()) + without_epoch
    return sorted(merged_rows, key=lambda row: epoch(row) or 10**12)


EPOCH_LINE_RE = re.compile(r"^\[(?P<epoch>\d+)/(?:\d+)\|(?P<phase>[^\]]+)\]\s+(?P<body>.*)$")
KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[-+A-Za-z0-9_:.@/]+)"
)
TOPOLOGY_FREEZE_EPOCH_RE = re.compile(
    r"\[TopologyFreeze\].*?triggered at epoch\s+(?P<freeze_epoch>\d+)"
)
TOPOLOGY_ROLLBACK_RE = re.compile(
    r"Rolling back topology to epoch\s+(?P<rollback_epoch>\d+)\s+"
    r"with\s+val_acc=(?P<rollback_val>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _assign_stdout_value(row: dict[str, Any], key: str, raw_value: str) -> None:
    if key == "loss":
        row["train_loss"] = safe_float(raw_value)
    elif key == "train":
        row["train_acc"] = safe_float(raw_value)
    elif key == "val":
        row["val_acc"] = safe_float(raw_value)
    elif key == "test":
        row["test_acc"] = safe_float(raw_value)
    elif key == "select":
        if ":" in raw_value:
            name, value = raw_value.split(":", 1)
            row["selection_metric"] = name
            row["selection_acc"] = safe_float(value)
    elif key == "topo_best":
        match = re.match(r"(?P<name>[^:]+):(?P<value>[-+0-9.eE]+)@(?P<epoch>\d+)", raw_value)
        if match:
            row["topology_best_metric_name"] = match.group("name")
            row["topology_best_metric_value"] = safe_float(match.group("value"))
            row["topology_best_epoch"] = safe_int(match.group("epoch"))
    elif key == "topo_bad":
        row["topology_bad_count"] = safe_int(raw_value)
    elif key == "topo_frozen":
        row["topology_frozen"] = raw_value.lower() == "true"
    elif key == "sp":
        row["sparsity"] = safe_float(raw_value)
        row["hard_density"] = safe_float(raw_value)
    elif key == "grad":
        row["grad_norm"] = safe_float(raw_value)
    elif key == "fr":
        if "/" in raw_value:
            mean_value, max_value = raw_value.split("/", 1)
            row["mean_firing_rate"] = safe_float(mean_value)
            row["max_firing_rate"] = safe_float(max_value)
    elif key == "adapt":
        if "/" in raw_value:
            mean_value, max_value = raw_value.split("/", 1)
            row["mean_adaptation"] = safe_float(mean_value)
            row["max_adaptation"] = safe_float(max_value)
    elif key == "topo_grad":
        if "/" in raw_value:
            pre_value, post_value = raw_value.split("/", 1)
            row["topology_grad_pre_clip"] = safe_float(pre_value)
            row["topology_grad_post_clip"] = safe_float(post_value)
    elif key in {"lr", "tau", "topo_lr", "bias_lr"}:
        row[key] = safe_float(raw_value)


def parse_stdout_log(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    if not path.exists():
        return rows, metadata
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        freeze_match = TOPOLOGY_FREEZE_EPOCH_RE.search(line)
        if freeze_match:
            metadata["topology_frozen_epoch"] = safe_int(
                freeze_match.group("freeze_epoch")
            )
            rollback_match = TOPOLOGY_ROLLBACK_RE.search(line)
            if rollback_match:
                metadata["rollback_target_epoch"] = safe_int(
                    rollback_match.group("rollback_epoch")
                )
                metadata["rollback_target_val_acc"] = safe_float(
                    rollback_match.group("rollback_val")
                )
            continue

        epoch_match = EPOCH_LINE_RE.match(line)
        if not epoch_match:
            continue
        row: dict[str, Any] = {
            "epoch": safe_int(epoch_match.group("epoch")),
            "phase": epoch_match.group("phase"),
        }
        for match in KEY_VALUE_RE.finditer(epoch_match.group("body")):
            _assign_stdout_value(row, match.group("key"), match.group("value"))
        rows.append(row)
    return rows, metadata


def find_stdout_log(experiment_dir: Path, explicit_path: str) -> Path | None:
    if explicit_path:
        return Path(explicit_path)
    if experiment_dir.is_file():
        return experiment_dir
    candidates = [
        experiment_dir / "stdout.log",
        experiment_dir / "stdout.txt",
        experiment_dir / "train.log",
        experiment_dir / "train.txt",
        experiment_dir / "log.txt",
    ]
    candidates.extend(sorted(experiment_dir.glob("*.log")))
    candidates.extend(sorted(experiment_dir.glob("*.txt")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_rows(experiment_dir: Path, stdout_log: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    sources: list[str] = []
    stdout_metadata: dict[str, Any] = {}
    train_rows = read_jsonl(experiment_dir / "logs" / "train.jsonl")
    if train_rows:
        sources.append(str(experiment_dir / "logs" / "train.jsonl"))

    diag_rows = read_jsonl(experiment_dir / "diagnostics" / "epoch_metrics.jsonl")
    if diag_rows:
        sources.append(str(experiment_dir / "diagnostics" / "epoch_metrics.jsonl"))

    if train_rows or diag_rows:
        return merge_rows_by_epoch(train_rows, diag_rows), stdout_metadata, sources

    stdout_path = find_stdout_log(experiment_dir, stdout_log)
    if stdout_path is None:
        return [], stdout_metadata, sources

    stdout_rows, stdout_metadata = parse_stdout_log(stdout_path)
    if stdout_rows:
        sources.append(str(stdout_path))
    return stdout_rows, stdout_metadata, sources


def best_by_val(rows: list[dict[str, Any]], tie_epsilon: float) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = [row for row in rows if has_metric(row, "val_acc")]
    if not candidates:
        return None, []
    best_value = max(metric(row, "val_acc") for row in candidates)
    assert best_value is not None
    tied = [
        row
        for row in candidates
        if metric(row, "val_acc") is not None
        and abs(metric(row, "val_acc") - best_value) <= tie_epsilon
    ]
    tied_with_loss = [row for row in tied if has_metric(row, "val_loss")]
    if tied_with_loss:
        selected = min(tied_with_loss, key=lambda row: (metric(row, "val_loss"), epoch(row) or 10**12))
    else:
        selected = min(tied, key=lambda row: epoch(row) or 10**12)
    return selected, sorted(tied, key=lambda row: epoch(row) or 10**12)


def top_rows(
    rows: list[dict[str, Any]], metric_name: str, limit: int = 10
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if has_metric(row, metric_name)]
    return sorted(
        candidates,
        key=lambda row: (metric(row, metric_name), -(epoch(row) or 10**12)),
        reverse=True,
    )[:limit]


def first_epoch_where(
    rows: list[dict[str, Any]], metric_name: str, threshold: float
) -> dict[str, Any] | None:
    for row in sorted(rows, key=lambda item: epoch(item) or 10**12):
        value = metric(row, metric_name)
        if value is not None and value > threshold:
            return row
    return None


def max_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if has_metric(row, metric_name)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (metric(row, metric_name), -(epoch(row) or 10**12)))


def compact_row(row: dict[str, Any] | None, keys: list[str]) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key in keys:
        if key in row:
            out[key] = row.get(key)
        elif key in METRIC_ALIASES:
            value = metric(row, key)
            if value is not None:
                out[key] = value
    return out


def final_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: epoch(row) or -1)


def infer_topology_freeze_epoch(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> int | None:
    meta_epoch = safe_int(metadata.get("topology_frozen_epoch"))
    if meta_epoch is not None:
        return meta_epoch
    values = [safe_int(row.get("topology_frozen_epoch")) for row in rows]
    values = [value for value in values if value is not None]
    if values:
        return min(values)
    frozen_rows = [
        row
        for row in rows
        if row.get("topology_frozen") is True and epoch(row) is not None
    ]
    return epoch(frozen_rows[0]) if frozen_rows else None


def infer_rollback_target_epoch(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> int | None:
    meta_epoch = safe_int(metadata.get("rollback_target_epoch"))
    if meta_epoch is not None:
        return meta_epoch
    explicit_values = [
        safe_int(row.get("topology_rollback_target_epoch")) for row in rows
    ]
    explicit_values = [value for value in explicit_values if value is not None]
    if explicit_values:
        return explicit_values[-1]
    rollback_rows = [row for row in rows if row.get("topology_rollback_applied") is True]
    for row in rollback_rows:
        target = safe_int(row.get("topology_runaway_rollback_epoch"))
        if target is not None:
            return target
        target = safe_int(row.get("topology_best_epoch"))
        if target is not None:
            return target
    return None


def nearby_test_values(rows: list[dict[str, Any]], selected_epoch: int) -> list[float]:
    values: list[float] = []
    for row in rows:
        ep = epoch(row)
        test = metric(row, "test_acc")
        if ep is not None and test is not None and abs(ep - selected_epoch) <= 2:
            values.append(test)
    return values


def build_report(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    sources: list[str],
    tie_epsilon: float,
    topology_grad_threshold: float,
    low_test_gap: float,
) -> dict[str, Any]:
    selected_best_val, tied_best_val_rows = best_by_val(rows, tie_epsilon)
    final = final_row(rows)
    top_val = top_rows(rows, "val_acc", 10)
    top_test = top_rows(rows, "test_acc", 10)
    first_grad_explosion = first_epoch_where(
        rows, "topology_grad_pre_clip", topology_grad_threshold
    )
    max_grad = max_row(rows, "topology_grad_pre_clip")
    first_firing_runaway = first_epoch_where(
        rows, "max_firing_rate", FIRING_RUNAWAY_THRESHOLD
    )
    max_firing = max_row(rows, "max_firing_rate")
    freeze_epoch = infer_topology_freeze_epoch(rows, metadata)
    rollback_target_epoch = infer_rollback_target_epoch(rows, metadata)

    warnings: list[str] = []
    if len(tied_best_val_rows) > 1:
        tied_epochs = [epoch(row) for row in tied_best_val_rows]
        warnings.append("best val_acc tie detected")
        warnings.append(
            f"best val_acc has multiple tied epochs within epsilon: {tied_epochs}"
        )
        warnings.append(
            "recommend enabling validation-loss tie-break for val_acc ties; do not use test accuracy for selection."
        )
        tied_with_loss = [row for row in tied_best_val_rows if has_metric(row, "val_loss")]
        if tied_with_loss:
            earliest = tied_best_val_rows[0]
            lower_later = [
                row
                for row in tied_with_loss
                if (epoch(row) or -1) > (epoch(earliest) or -1)
                and metric(row, "val_loss") is not None
                and metric(earliest, "val_loss") is not None
                and metric(row, "val_loss") < metric(earliest, "val_loss")
            ]
            if lower_later:
                warnings.append(
                    "later tied best-val epoch has lower val_loss: "
                    + ", ".join(
                        f"epoch {epoch(row)} val_loss={metric(row, 'val_loss'):.6g}"
                        for row in lower_later
                    )
                )
        selected_epoch = epoch(selected_best_val) if selected_best_val else None
        later_equal = [
            row
            for row in tied_best_val_rows
            if selected_epoch is not None and (epoch(row) or -1) > selected_epoch
        ]
        if later_equal:
            warnings.append(
                "later equal-val epoch exists: "
                + ", ".join(f"epoch {epoch(row)}" for row in later_equal)
            )

    if selected_best_val is not None:
        selected_epoch = epoch(selected_best_val)
        selected_test = metric(selected_best_val, "test_acc")
        if selected_epoch is not None and selected_test is not None:
            nearby = nearby_test_values(rows, selected_epoch)
            final_test = metric(final or {}, "test_acc")
            comparison_values = nearby + ([final_test] if final_test is not None else [])
            if comparison_values and max(comparison_values) - selected_test >= low_test_gap:
                warnings.append(
                    "test@best_val is much lower than nearby/final test; report this, "
                    "but do not select by test accuracy."
                )

    if rollback_target_epoch is not None and tied_best_val_rows:
        later_equal = [
            row
            for row in tied_best_val_rows
            if (epoch(row) or -1) > rollback_target_epoch
        ]
        if later_equal:
            warnings.append(
                "topology rollback target differs from later equal-val candidate(s): "
                + ", ".join(f"epoch {epoch(row)}" for row in later_equal)
            )

    if first_grad_explosion is not None:
        warnings.append(
            "topo_grad_norm_pre_clip exceeds threshold: "
            f"epoch {epoch(first_grad_explosion)} value={metric(first_grad_explosion, 'topology_grad_pre_clip'):.6g}"
        )
    if first_firing_runaway is not None:
        warnings.append(
            "max_firing_rate exceeds 0.9: "
            f"epoch {epoch(first_firing_runaway)} value={metric(first_firing_runaway, 'max_firing_rate'):.6g}"
        )

    row_keys = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "selection_metric",
        "selection_acc",
        "topology_grad_pre_clip",
        "topology_grad_post_clip",
        "topology_grad_norm_pre_clip",
        "topology_grad_norm_post_clip",
        "theta_grad_norm_pre_clip",
        "theta_grad_norm_post_clip",
        "topology_best_epoch",
        "topology_frozen_epoch",
        "topology_rollback_applied",
        "topology_rollback_reason",
        "topology_rollback_target_epoch",
        "topology_runaway_guard_triggered",
        "topology_runaway_freeze_remaining",
        "topology_runaway_rollback_epoch",
        "hard_density_delta",
        "topology_logit_delta_l2",
        "topology_logit_delta_mean_abs",
        "topology_sigmoid_delta_mean_abs",
        "topology_entropy",
        "mean_firing_rate",
        "max_firing_rate",
        "mean_adaptation",
        "max_adaptation",
        "sparsity",
        "hard_density",
    ]

    return {
        "sources": sources,
        "num_epochs": len(rows),
        "selection_policy": {
            "uses_test_accuracy": False,
            "primary_metric": "val_acc if available",
            "tie_epsilon": tie_epsilon,
            "recommended_tie_break": "lower val_loss when val_acc ties; keep current behavior if val_loss is unavailable",
        },
        "best_val_epoch": epoch(selected_best_val) if selected_best_val else None,
        "best_val_acc": metric(selected_best_val or {}, "val_acc"),
        "test_at_best_val": metric(selected_best_val or {}, "test_acc"),
        "test_at_best_val_epoch": metric(selected_best_val or {}, "test_acc"),
        "best_val_row": compact_row(selected_best_val, row_keys),
        "top_10_val_epochs_with_test": [
            compact_row(row, ["epoch", "val_acc", "val_loss", "test_acc", "test_loss"])
            for row in top_val
        ],
        "top_10_test_epochs_with_val": [
            compact_row(row, ["epoch", "test_acc", "test_loss", "val_acc", "val_loss"])
            for row in top_test
        ],
        "val_acc_ties_at_best_value": [
            compact_row(row, ["epoch", "val_acc", "val_loss", "test_acc"])
            for row in tied_best_val_rows
        ],
        "all_epochs_tied_at_best_val_acc": [
            epoch(row) for row in tied_best_val_rows
        ],
        "topology_freeze_epoch": freeze_epoch,
        "rollback_target_epoch": rollback_target_epoch,
        "first_topology_gradient_explosion_epoch": compact_row(
            first_grad_explosion,
            ["epoch", "topology_grad_pre_clip", "topology_grad_post_clip"],
        ),
        "max_topology_gradient_epoch": compact_row(
            max_grad, ["epoch", "topology_grad_pre_clip", "topology_grad_post_clip"]
        ),
        "max_topology_gradient_value": metric(max_grad or {}, "topology_grad_pre_clip"),
        "first_max_firing_rate_gt_0_9_epoch": compact_row(
            first_firing_runaway, ["epoch", "max_firing_rate", "mean_firing_rate"]
        ),
        "max_firing_epoch": compact_row(
            max_firing, ["epoch", "max_firing_rate", "mean_firing_rate"]
        ),
        "max_firing_value": metric(max_firing or {}, "max_firing_rate"),
        "final_epoch_metrics": compact_row(final, row_keys),
        "warnings": warnings,
    }


def print_table(title: str, rows: list[dict[str, Any] | None]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  none")
        return
    for row in rows:
        print("  " + json.dumps(row, sort_keys=True))


def print_text_report(report: dict[str, Any]) -> None:
    print("# LSM Training Run Analysis")
    print("\nSources:")
    if report["sources"]:
        for source in report["sources"]:
            print(f"- {source}")
    else:
        print("- none")

    print("\nSelection:")
    print("- test accuracy used for selection: false")
    print(
        "- recommended tie-break: lower val_loss when val_acc ties; "
        "if val_loss is unavailable, keep current behavior and log a tie warning"
    )

    print("\nSummary:")
    summary_keys = [
        "num_epochs",
        "best_val_epoch",
        "best_val_acc",
        "test_at_best_val",
        "test_at_best_val_epoch",
        "topology_freeze_epoch",
        "rollback_target_epoch",
    ]
    for key in summary_keys:
        print(f"- {key}: {report.get(key)}")

    print("\nBest Val Row:")
    print(json.dumps(report.get("best_val_row"), indent=2, sort_keys=True))

    print_table("Top 10 Val Epochs With Test Values:", report["top_10_val_epochs_with_test"])
    print_table("Top 10 Test Epochs With Val Values:", report["top_10_test_epochs_with_val"])
    print_table("Val Acc Ties At Best Value:", report["val_acc_ties_at_best_value"])

    print("\nTopology / Gradient / Firing:")
    for key in [
        "first_topology_gradient_explosion_epoch",
        "max_topology_gradient_epoch",
        "first_max_firing_rate_gt_0_9_epoch",
        "max_firing_epoch",
    ]:
        print(f"- {key}: {json.dumps(report.get(key), sort_keys=True)}")

    print("\nFinal Epoch Metrics:")
    print(json.dumps(report.get("final_epoch_metrics"), indent=2, sort_keys=True))

    print("\nWarnings:")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"- {warning}")
    else:
        print("- none")


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    rows, metadata, sources = load_rows(
        experiment_dir, args.log_file or args.stdout_log
    )
    if not rows:
        raise SystemExit(
            f"No metrics found in {experiment_dir}. Expected logs/train.jsonl, "
            "diagnostics/epoch_metrics.jsonl, or a stdout .log/.txt file."
        )
    report = build_report(
        rows=rows,
        metadata=metadata,
        sources=sources,
        tie_epsilon=max(args.tie_epsilon, 0.0),
        topology_grad_threshold=args.topology_grad_threshold,
        low_test_gap=args.low_test_gap,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
