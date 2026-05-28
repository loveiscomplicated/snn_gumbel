"""
Topology analysis tool for learned liquid checkpoints.

Focus:
  1. E/I in-degree and out-degree
  2. Hub neuron analysis
  3. Seed-to-seed edge overlap
  4. Loop / motif summaries
  5. Readout importance vs topology centrality

Usage:
    python scripts/analyze_topology.py experiments/exp_a experiments/exp_b


for p in 040 045 050; do           
    for s in 42 43 44 45; do
      dir=$(ls -d experiments/lsm_shd_rs_density_control_p${p}_s${s}_* 2>/dev/null | head -1)
      if [ -n "$dir" ]; then
        python scripts/diagnose_liquid.py "$dir/config.yaml" \
          --checkpoint "$dir/checkpoints/best.pt" \
          --batches 4 --classes 20 --samples-per-class 10 \
          --out-json "$dir/diagnose_topology.json" \
          --out-csv runs/diagnostics/topology_summary.csv \
          --save-embeddings "$dir/embedding_pca.csv" \
          batch_size=16
      fi
    done
  done



for s in 42 43 44 45; do                     
    dir=$(ls -d experiments/lsm_shd_lowrank_r16_valrollback_m50p10_s${s}_* 2>/dev/null | head -1)         
    if [ -n "$dir" ]; then
      python scripts/diagnose_liquid.py "$dir/config.yaml" \
        --checkpoint "$dir/checkpoints/best.pt" \
        --batches 4 --classes 20 --samples-per-class 10 \                                                  
        --out-json "$dir/diagnose_topology.json" \   
        --out-csv runs/diagnostics/topology_summary.csv \                            
        --save-embeddings "$dir/embedding_pca.csv" \
        batch_size=16
    fi
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lsm.trainer import build_model, get_device
from src.utils.config import load_config

try:
    from diagnose_liquid import compute_graph_metrics, print_graph_topology_metrics
except ImportError:
    from scripts.diagnose_liquid import (
        compute_graph_metrics,
        print_graph_topology_metrics,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse learned liquid topology")
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment directories containing config.yaml and checkpoints/best.pt",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top hubs / important neurons to print",
    )
    parser.add_argument(
        "--compare-seed",
        type=str,
        default="44",
        help="Seed label to treat as the failure-side reference in comparisons",
    )
    return parser.parse_args()


def tensor_summary(x: torch.Tensor) -> str:
    x = x.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        return "mean=0.0000  std=0.0000  min=0.0000  max=0.0000"
    std = x.std().item() if x.numel() > 1 else 0.0
    return (
        f"mean={x.mean().item():.4f}  std={std:.4f}  "
        f"min={x.min().item():.4f}  max={x.max().item():.4f}"
    )


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom.item() == 0.0:
        return float("nan")
    return torch.dot(a, b).item() / denom.item()


def print_header(title: str) -> None:
    print(f"\n[{title}]")


@dataclass
class ExperimentTopology:
    label: str
    experiment_dir: Path
    checkpoint_path: Path
    mask: torch.Tensor
    readout_weight: torch.Tensor
    dale_sign: torch.Tensor
    in_degree: torch.Tensor
    out_degree: torch.Tensor
    readout_importance: torch.Tensor


def infer_label(exp_dir: Path, cfg) -> str:
    seed = getattr(cfg, "seed", None)
    mode = cfg.liquid.recurrent_mode
    tau_end = getattr(cfg, "tau_end", None)
    suffix = f"seed{seed}"
    if tau_end is not None:
        suffix += f"_tau{tau_end:g}"
    return f"{mode}:{suffix}"


def load_experiment(exp_dir: Path, device: torch.device) -> ExperimentTopology:
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    cfg = load_config(str(config_path))
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.liquid.unlock_epoch_mask()
    model.liquid.sample_mask(tau=cfg.tau_end)

    mask = model.liquid.get_binary_mask().detach().cpu().bool()
    dale_sign = model.liquid.dale_sign.detach().cpu().reshape(-1)
    in_degree = mask.float().sum(dim=0)
    out_degree = mask.float().sum(dim=1)
    readout_weight = model.readout.weight.detach().cpu()
    readout_importance = readout_weight.norm(dim=0)

    return ExperimentTopology(
        label=infer_label(exp_dir, cfg),
        experiment_dir=exp_dir,
        checkpoint_path=checkpoint_path,
        mask=mask,
        readout_weight=readout_weight,
        dale_sign=dale_sign,
        in_degree=in_degree,
        out_degree=out_degree,
        readout_importance=readout_importance,
    )


def topk_pairs(values: torch.Tensor, k: int) -> str:
    topk = values.topk(min(k, values.numel()))
    return ", ".join(
        f"{idx}:{val:.0f}"
        for idx, val in zip(topk.indices.tolist(), topk.values.tolist())
    )


def sign_masks(dale_sign: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    exc = dale_sign > 0
    inh = dale_sign < 0
    return exc, inh


def subset_summary(mask: torch.Tensor, neuron_mask: torch.Tensor) -> str:
    vals = mask.float()[neuron_mask]
    return tensor_summary(vals)


def reciprocal_counts(
    mask: torch.Tensor, exc: torch.Tensor, inh: torch.Tensor
) -> dict[str, int]:
    reciprocal = mask & mask.T
    counts = {
        "total_pairs": int(reciprocal.triu(diagonal=1).sum().item()),
        "ee_pairs": int(
            (reciprocal & exc[:, None] & exc[None, :]).triu(diagonal=1).sum().item()
        ),
        "ei_pairs": int((reciprocal & exc[:, None] & inh[None, :]).sum().item()),
        "ii_pairs": int(
            (reciprocal & inh[:, None] & inh[None, :]).triu(diagonal=1).sum().item()
        ),
    }
    return counts


def directed_3cycle_count(mask: torch.Tensor) -> int:
    a = mask.float()
    tri = torch.trace(a @ a @ a).item()
    return int(round(tri / 3.0))


def feedforward_triplet_count(mask: torch.Tensor) -> int:
    # Count i->j->k with i!=k and no direct feedback k->i. Crude transitive motif proxy.
    a = mask.float()
    two_step = a @ a
    no_feedback = 1.0 - a.T
    candidate = two_step * no_feedback
    candidate.fill_diagonal_(0.0)
    return int(candidate.sum().item())


def print_single_experiment_report(exp: ExperimentTopology, topk: int) -> None:
    print("=" * 72)
    print(f"[Experiment] {exp.label}")
    print(f"  dir        : {exp.experiment_dir}")
    print(f"  checkpoint : {exp.checkpoint_path}")

    exc, inh = sign_masks(exp.dale_sign)
    print_header("1. Degree decomposition")
    print(f"  in-degree (all)         : {tensor_summary(exp.in_degree)}")
    print(f"  out-degree (all)        : {tensor_summary(exp.out_degree)}")
    print(f"  exc in-degree           : {subset_summary(exp.in_degree, exc)}")
    print(f"  exc out-degree          : {subset_summary(exp.out_degree, exc)}")
    print(f"  inh in-degree           : {subset_summary(exp.in_degree, inh)}")
    print(f"  inh out-degree          : {subset_summary(exp.out_degree, inh)}")

    print_graph_topology_metrics(compute_graph_metrics(exp.mask, exp.dale_sign))

    total_degree = exp.in_degree + exp.out_degree
    print_header("2. Hub neurons")
    print(f"  top in-degree hubs      : {topk_pairs(exp.in_degree, topk)}")
    print(f"  top out-degree hubs     : {topk_pairs(exp.out_degree, topk)}")
    print(f"  top total-degree hubs   : {topk_pairs(total_degree, topk)}")
    print(f"  top readout importance  : {topk_pairs(exp.readout_importance, topk)}")

    print_header("3. Loop / motif summary")
    rec = reciprocal_counts(exp.mask, exc, inh)
    print(
        f"  reciprocal pairs        : total={rec['total_pairs']}  "
        f"EE={rec['ee_pairs']}  EI={rec['ei_pairs']}  II={rec['ii_pairs']}"
    )
    print(f"  directed 3-cycles       : {directed_3cycle_count(exp.mask)}")
    print(f"  feedforward triplets    : {feedforward_triplet_count(exp.mask)}")

    print_header("4. Readout vs centrality")
    print(
        f"  corr(readout, in)       : {pearson_corr(exp.readout_importance, exp.in_degree):.4f}"
    )
    print(
        f"  corr(readout, out)      : {pearson_corr(exp.readout_importance, exp.out_degree):.4f}"
    )
    print(
        f"  corr(readout, total)    : {pearson_corr(exp.readout_importance, total_degree):.4f}"
    )

    top_degree = set(
        total_degree.topk(min(topk, total_degree.numel())).indices.tolist()
    )
    top_readout = set(
        exp.readout_importance.topk(
            min(topk, exp.readout_importance.numel())
        ).indices.tolist()
    )
    overlap = sorted(top_degree & top_readout)
    print(
        f"  top-degree ∩ top-readout: {len(overlap)} / {min(topk, total_degree.numel())}  "
        f"{overlap}"
    )


def jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> tuple[float, int, int, int]:
    inter = int((mask_a & mask_b).sum().item())
    union = int((mask_a | mask_b).sum().item())
    only_a = int((mask_a & ~mask_b).sum().item())
    only_b = int((mask_b & ~mask_a).sum().item())
    score = inter / union if union > 0 else 0.0
    return score, inter, only_a, only_b


def shared_topk_fraction(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    set_a = set(a.topk(min(k, a.numel())).indices.tolist())
    set_b = set(b.topk(min(k, b.numel())).indices.tolist())
    denom = max(min(k, a.numel()), 1)
    return len(set_a & set_b) / denom


def print_cross_seed_report(
    experiments: list[ExperimentTopology], topk: int, compare_seed: str
) -> None:
    if len(experiments) < 2:
        return

    print("=" * 72)
    print("[Cross-seed comparison]")
    print_header("5. Edge overlap")
    for a, b in itertools.combinations(experiments, 2):
        score, inter, only_a, only_b = jaccard(a.mask, b.mask)
        print(
            f"  {a.label} vs {b.label}: jaccard={score:.4f}  "
            f"shared={inter}  only_a={only_a}  only_b={only_b}"
        )

    success = [exp for exp in experiments if compare_seed not in exp.label]
    failure = [exp for exp in experiments if compare_seed in exp.label]
    if success and failure:
        print_header("6. Success vs failure aggregate")
        failure_ref = failure[0]
        success_pair_scores = []
        failure_scores = []
        for a, b in itertools.combinations(success, 2):
            score, _, _, _ = jaccard(a.mask, b.mask)
            success_pair_scores.append(score)
        for exp in success:
            score, _, _, _ = jaccard(exp.mask, failure_ref.mask)
            failure_scores.append(score)
            print(
                f"  success {exp.label} vs failure {failure_ref.label}: "
                f"jaccard={score:.4f}"
            )
        if success_pair_scores:
            print(
                f"  success-success jaccard: mean={sum(success_pair_scores)/len(success_pair_scores):.4f}  "
                f"min={min(success_pair_scores):.4f}  max={max(success_pair_scores):.4f}"
            )
        if failure_scores:
            print(
                f"  success-failure jaccard: mean={sum(failure_scores)/len(failure_scores):.4f}  "
                f"min={min(failure_scores):.4f}  max={max(failure_scores):.4f}"
            )

        print_header("7. Failure-side ranking shifts")
        for exp in success:
            hub_overlap = shared_topk_fraction(
                exp.in_degree + exp.out_degree,
                failure_ref.in_degree + failure_ref.out_degree,
                topk,
            )
            readout_overlap = shared_topk_fraction(
                exp.readout_importance,
                failure_ref.readout_importance,
                topk,
            )
            print(
                f"  {exp.label} vs {failure_ref.label}: "
                f"top-hub overlap={hub_overlap:.3f}  top-readout overlap={readout_overlap:.3f}"
            )


def main():
    args = parse_args()
    device = get_device()
    experiments = [load_experiment(Path(exp), device) for exp in args.experiments]
    for exp in experiments:
        print_single_experiment_report(exp, args.topk)
    print_cross_seed_report(experiments, args.topk, args.compare_seed)
    print("=" * 72)


if __name__ == "__main__":
    main()
