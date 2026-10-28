"""PerSPL training entry point — two phases driven by a single JSON config.

Usage:
    python train.py --config configs/<NAME>.json

What it does:
    1. Phase 1 (seen-user training): trains a full PerSPL head with V + W
       on the seen-user split, using phase-1 hyperparameters.
    2. Phase 2 (unseen-user fine-tune): freezes the phase-1 last-epoch V,
       initialises a fresh W for the unseen users, and trains W using
       phase-2 hyperparameters.

Output:
    outputs/<dataset>/<TS>_seed<S>/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import TwoPhaseConfig, load_config
from data import EmbeddingDataset
from loaders import load_helpsteer2, load_personalllm, load_reddit_tldr
from model import PerSPLRewardHead
from trainer import PerSPLIterTrainer, run_unseen_eval
from utils import seed_everything


# ============================================================================
# Dataset dispatch
# ============================================================================


def _load_dataset(cfg: TwoPhaseConfig, cache_dir: str | None) -> EmbeddingDataset:
    """Dispatch to the right loader given cfg.dataset.

    All shared per-user sizing args (n_seen / n_unseen / *_labeled_per_user /
    max_unlabeled / seed) come from cfg. Dataset-specific extras (alpha,
    val_fraction, score_diff_threshold, label_flip_ratio) also come from
    cfg's optional top-level fields.
    """
    common = dict(
        n_seen=cfg.n_seen,
        n_unseen=cfg.n_unseen,
        seen_labeled_per_user=cfg.seen_labeled_per_user,
        unseen_labeled_per_user=cfg.unseen_labeled_per_user,
        max_unlabeled=cfg.max_unlabeled,
        seed=cfg.seed,
    )
    if cfg.dataset == "reddit_tldr":
        return load_reddit_tldr(
            cache_dir=cache_dir or "./cache/reddit_tldr",
            **common,
        )
    if cfg.dataset == "personalllm":
        return load_personalllm(
            cache_dir=cache_dir or "./cache/personalllm",
            alpha=cfg.alpha,
            val_fraction=cfg.val_fraction,
            label_flip_ratio=cfg.label_flip_ratio,
            **common,
        )
    if cfg.dataset == "helpsteer2":
        return load_helpsteer2(
            cache_dir=cache_dir or "./cache/helpsteer2",
            alpha=cfg.alpha,
            val_fraction=cfg.val_fraction,
            label_flip_ratio=cfg.label_flip_ratio,
            score_diff_threshold=cfg.score_diff_threshold,
            **common,
        )
    raise SystemExit(
        f"unknown dataset {cfg.dataset!r}; supported: reddit_tldr, "
        f"personalllm, helpsteer2"
    )


# ============================================================================
# JSON-safe serialiser
# ============================================================================


def _json_default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"non-serializable: {type(o).__name__}")


# ============================================================================
# Main
# ============================================================================


def main():
    cli = argparse.ArgumentParser(
        description="PerSPL two-phase training (single-config)."
    )
    cli.add_argument("--config", required=True,
                     help="Path to the unified JSON config "
                          "(see configs/ for examples).")
    cli.add_argument("--seed", type=int, default=None,
                     help="Random seed. Overrides 'seed' field in JSON. "
                          "Required when JSON has no 'seed' field "
                          "(the canonical release configs omit it so "
                          "you can sweep seeds without editing the JSON).")
    cli.add_argument("--save-dir", default=None,
                     help="Output directory. Default: "
                          "outputs/<dataset>/<TS>_seed<S>/")
    cli.add_argument("--cache-dir", default=None,
                     help="Override the per-dataset cache directory "
                          "(default ./cache/<dataset>/).")
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()

    cfg = load_config(args.config, seed=args.seed)

    save_dir = Path(args.save_dir) if args.save_dir else (
        Path("outputs") / cfg.dataset
        / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_seed{cfg.seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    seed_everything(cfg.seed)

    # Step 1: load dataset.
    print("=" * 60)
    print(f"Step 1: Loading {cfg.dataset} (seed={cfg.seed})")
    print("=" * 60)
    dataset = _load_dataset(cfg, args.cache_dir)
    print(f"Dataset: {len(dataset.user_ids)} seen users, "
          f"{len(dataset.unseen_user_ids)} unseen users, "
          f"emb_dim={dataset.emb_dim}")

    if len(dataset.unseen_user_ids) == 0:
        print("[fatal] dataset has 0 unseen users — phase 2 cannot run. "
              "Increase n_unseen in the config.", file=sys.stderr)
        sys.exit(1)

    num_users = len(dataset.user_ids)

    # Step 2: phase 1 — seen-user training.
    print("\n" + "=" * 60)
    print("Phase 1: seen-user training (phase-1 hp)")
    print("=" * 60)
    head = PerSPLRewardHead(
        emb_dim=dataset.emb_dim, num_users=num_users,
        rank=cfg.rank, temperature=cfg.temperature,
    )
    trainer = PerSPLIterTrainer(
        head=head, dataset=dataset, config=cfg.phase1, device=device,
    )
    phase1_summary = trainer.train()
    last_epoch_state = phase1_summary.pop("last_epoch_state")

    # Free phase-1 trainer state before phase 2 allocates its own.
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: phase 2 — unseen-user fine-tune from phase-1 last-epoch V.
    print("\n" + "=" * 60)
    print("Phase 2: unseen-user fine-tune (phase-2 hp, V from phase-1 last-epoch)")
    print("=" * 60)
    # Re-seed before phase 2 so the trajectory is independent of phase-1
    # RNG advancement — matches dev-repo standalone phase-2 behavior.
    seed_everything(cfg.seed)
    last_head = PerSPLRewardHead(
        emb_dim=dataset.emb_dim, num_users=num_users,
        rank=cfg.rank, temperature=cfg.temperature,
    ).to(device)
    last_head.load_state_dict(last_epoch_state)

    _, phase2_out = run_unseen_eval(
        stage1_head=last_head, dataset=dataset,
        config=cfg.phase2, device=device,
    )
    accuracy = phase2_out["accuracy"]
    phase2_summary = phase2_out["phase2_summary"]

    # Step 4: write the only output.
    results = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "config": str(Path(args.config).resolve()),
        "n_seen": len(dataset.user_ids),
        "n_unseen": len(dataset.unseen_user_ids),
        "accuracy": accuracy,
        "phase1_summary": phase1_summary,
        "phase2_summary": phase2_summary,
    }
    out_path = save_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    print("\n" + "=" * 60)
    print(f"Done! {out_path}")
    print(f"  seen   best-epoch acc = {accuracy['seen_best_epoch_acc']:.4f}")
    print(f"  seen   last-epoch acc = {accuracy['seen_last_epoch_acc']:.4f}")
    print(f"  unseen best-epoch acc = {accuracy['unseen_best_epoch_acc']:.4f}")
    print(f"  unseen last-epoch acc = {accuracy['unseen_last_epoch_acc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
