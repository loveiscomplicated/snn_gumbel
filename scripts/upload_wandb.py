"""
Upload a completed experiment directory to Weights & Biases.

What gets uploaded:
  - config.yaml        → wandb run config
  - logs/train.jsonl   → per-epoch metrics (each row = one step)
  - figures/*.png      → wandb.Image media
  - checkpoints/best.pt → wandb Artifact

Usage:
    python scripts/upload_wandb.py --exp ablation_random_sparse_2603231847
    python scripts/upload_wandb.py --exp ablation_random_sparse_2603231847 --project my_project
    python scripts/upload_wandb.py --exp experiments/ablation_random_sparse_2603231847
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import wandb


def load_config_yaml(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {exp_dir}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(exp_dir: Path) -> list[dict]:
    log_path = exp_dir / "logs" / "train.jsonl"
    if not log_path.exists():
        print(f"  [warn] train.jsonl not found, skipping metrics.")
        return []
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def upload(exp_dir: Path, project: str, entity: str | None):
    cfg = load_config_yaml(exp_dir)
    rows = load_jsonl(exp_dir)

    run_name = exp_dir.name  # e.g. ablation_random_sparse_2603231847

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=cfg,
        reinit=True,
    )

    # --- epoch metrics ---
    if rows:
        print(f"  Logging {len(rows)} epochs...")
        for row in rows:
            epoch = row.pop("epoch")
            wandb.log(row, step=epoch)
        print("  Done.")

    # --- figures ---
    figures_dir = exp_dir / "figures"
    if figures_dir.exists():
        pngs = sorted(figures_dir.glob("*.png"))
        if pngs:
            print(f"  Uploading {len(pngs)} figures...")
            wandb.log({p.stem: wandb.Image(str(p)) for p in pngs})

    # --- checkpoint artifact ---
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        print("  Uploading checkpoint artifact...")
        artifact = wandb.Artifact(
            name=f"{run_name}_checkpoint",
            type="model",
            description=f"Best checkpoint for {run_name}",
        )
        artifact.add_file(str(ckpt_path), name="best.pt")
        run.log_artifact(artifact)

    run.finish()
    print(f"\nUploaded to wandb: {run.url}")


def main():
    parser = argparse.ArgumentParser(description="Upload experiment results to wandb")
    parser.add_argument(
        "--exp", required=True,
        help="Experiment directory name or path (e.g. ablation_full_2603231840 or experiments/ablation_full_2603231840)",
    )
    parser.add_argument(
        "--project", default="snn_gumbel",
        help="wandb project name (default: snn_gumbel)",
    )
    parser.add_argument(
        "--entity", default=None,
        help="wandb entity (team/user). Omit to use default.",
    )
    args = parser.parse_args()

    exp_path = Path(args.exp)
    # 경로가 존재하지 않으면 experiments/ 아래에서 찾기
    if not exp_path.exists():
        exp_path = Path("experiments") / args.exp
    if not exp_path.exists():
        print(f"Error: experiment directory not found: {args.exp}")
        sys.exit(1)

    print(f"Experiment : {exp_path}")
    print(f"Project    : {args.project}")
    upload(exp_path, project=args.project, entity=args.entity)


if __name__ == "__main__":
    main()
