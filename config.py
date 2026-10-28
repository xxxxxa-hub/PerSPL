"""Two-phase config schema.

A single JSON file drives both phase 1 (seen-user training) and phase 2
(unseen-user fine-tune). Shared parameters (dataset, model shape,
per-user data sizes) live at the top level; everything that can differ
between phases lives inside the ``phase1`` / ``phase2`` blocks.

Schema (see ``configs/*.json`` for examples):

    {
      "dataset": "reddit_tldr",
      "seed": 42,
      "n_seen": 20, "n_unseen": 20,
      "seen_labeled_per_user": 100, "unseen_labeled_per_user": 50,
      "max_unlabeled": -1,
      "rank": 2, "temperature": 100.0,
      "phase1": { ... per-phase hp ... },
      "phase2": { ... per-phase hp ... }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerSPLPhaseConfig:
    """Per-phase training hyperparameters.

    Fields here can differ between phase 1 and phase 2. Anything that
    must agree across phases lives on ``TwoPhaseConfig`` instead.
    """
    # Training schedule
    lr: float
    num_epochs: int
    n_iter: int
    mini_batch_size: int
    unlabeled_mini_batch_size: int
    n_unlabeled_per_user: int
    tasks_per_batch: int

    # Loss
    labeled_loss_weight: float       # alpha
    iso_filter_threshold: float      # tau
    reward_reg_lambda: float
    threshold_method: str = "isotonic"

    # LR schedule
    lr_schedule: str = "constant"
    lr_min: float = 0.0
    warmup_frac: float = 0.05
    linear_start_factor: float = 1.0
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    onecycle_anneal_strategy: str = "cos"
    onecycle_no_cycle_momentum: bool = False
    onecycle_three_phase: bool = False

    # Optimizer
    weight_decay: float = 0.0
    weight_decay_target: str = "all"   # all | user | shared

    # Logging
    log_interval: int = 1


@dataclass
class TwoPhaseConfig:
    """Full config — shared params at top level + two phase blocks."""
    # Identity
    dataset: str
    seed: int

    # Per-user data sizing (must match across both phases)
    n_seen: int
    n_unseen: int
    seen_labeled_per_user: int
    unseen_labeled_per_user: int
    max_unlabeled: int

    # Model shape (must match across both phases — V is reused)
    rank: int
    temperature: float

    phase1: PerSPLPhaseConfig
    phase2: PerSPLPhaseConfig

    # Dataset-specific extras (defaults apply when absent from JSON;
    # match dev psrm/train.py argparse defaults so cfg-driven runs are
    # bit-comparable across the two repos).
    alpha: float = 0.1                  # personalllm, helpsteer2
    val_fraction: float = 0.0           # personalllm, helpsteer2
    label_flip_ratio: float = 0.0       # personalllm, helpsteer2
    score_diff_threshold: float = 1.0   # helpsteer2


def load_config(path: str, *, seed: Optional[int] = None) -> TwoPhaseConfig:
    """Load and validate a TwoPhaseConfig from a JSON file.

    If ``seed`` is given, it overrides any ``seed`` field in the JSON.
    If neither the JSON nor the CLI provides a seed, raises ValueError.
    """
    with open(path) as f:
        d = json.load(f)
    if seed is not None:
        d["seed"] = seed
    if "seed" not in d:
        raise ValueError(
            f"config {path}: no 'seed' field in JSON and no --seed CLI "
            f"override given; one of the two must be set."
        )
    try:
        p1 = PerSPLPhaseConfig(**d.pop("phase1"))
        p2 = PerSPLPhaseConfig(**d.pop("phase2"))
    except KeyError as e:
        raise ValueError(
            f"config {path}: missing required 'phase1' or 'phase2' block"
        ) from e
    try:
        return TwoPhaseConfig(phase1=p1, phase2=p2, **d)
    except TypeError as e:
        raise ValueError(
            f"config {path}: invalid top-level field — {e}"
        ) from e
