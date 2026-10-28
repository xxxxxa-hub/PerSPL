"""Small shared utilities: seeding, LR schedules, threshold fitting, accuracy."""

from __future__ import annotations

import math
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    OneCycleLR,
)


# ============================================================================
# Seeding
# ============================================================================


def seed_everything(seed: int = 42, strict: bool = True) -> None:
    """Seed Python / NumPy / PyTorch (CPU + all CUDA devices) for reproducibility.

    With ``strict=True`` (default) also enables cuDNN deterministic mode and
    deterministic PyTorch algorithms, making runs bit-identical across
    launches on the same machine + CUDA driver. Pass ``strict=False`` only
    if you hit a non-deterministic op.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        # cuBLAS matmul needs this env var to be deterministic on CUDA >= 10.2;
        # must be set before cuBLAS is first invoked.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


# ============================================================================
# LR schedules
# ============================================================================


VALID_SCHEDULES = ("constant", "cosine", "linear", "warmup_cosine", "onecycle")


def build_lr_scheduler(
    optimizer: Optimizer,
    schedule: str,
    total_steps: int,
    lr: float,
    *,
    lr_min: float = 0.0,
    warmup_frac: float = 0.05,
    linear_start_factor: float = 1.0,
    onecycle_pct_start: float = 0.3,
    onecycle_div_factor: float = 25.0,
    onecycle_final_div_factor: float = 1e4,
    onecycle_anneal_strategy: str = "cos",
    onecycle_cycle_momentum: bool = True,
    onecycle_three_phase: bool = False,
):
    """Per-step LR scheduler factory.

    Returns ``None`` for ``constant`` or degenerate cases — callers must
    guard ``scheduler.step()`` with ``if scheduler is not None``.
    """
    if schedule == "constant" or total_steps <= 0:
        return None
    if schedule not in VALID_SCHEDULES:
        raise ValueError(
            f"Unknown lr_schedule: {schedule!r}. Options: {list(VALID_SCHEDULES)}"
        )

    if schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_min)

    if schedule == "linear":
        end_factor = (lr_min / lr) if lr > 0 else 0.0
        return LinearLR(
            optimizer,
            start_factor=linear_start_factor,
            end_factor=end_factor,
            total_iters=total_steps,
        )

    if schedule == "warmup_cosine":
        warmup_steps = max(1, int(total_steps * warmup_frac))
        lr_min_ratio = (lr_min / lr) if lr > 0 else 0.0

        def _fn(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            t = min(t, 1.0)
            return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

        return LambdaLR(optimizer, lr_lambda=_fn)

    if schedule == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=onecycle_pct_start,
            div_factor=onecycle_div_factor,
            final_div_factor=onecycle_final_div_factor,
            anneal_strategy=onecycle_anneal_strategy,
            cycle_momentum=onecycle_cycle_momentum,
            three_phase=onecycle_three_phase,
        )

    raise AssertionError("unreachable")


# ============================================================================
# Threshold fitting (isotonic + KDE-Bayes)
# ============================================================================


def compute_threshold(
    norm_chosen: torch.Tensor,
    norm_rejected: torch.Tensor,
    method: str = "isotonic",
    prior: float = 0.5,
    ir=None,
) -> float:
    """Per-user threshold from chosen/rejected reward scores.

    Methods:
      - ``min_chosen``: ``min(chosen)``.
      - ``midpoint``: ``(mean(chosen) + mean(rejected)) / 2``.
      - ``kde_bayes``: minimum-Bayes-risk threshold from Gaussian KDE.
      - ``isotonic``: midpoint of the ``P(chosen | r) = 0.5`` region under
        a sklearn isotonic regression fit. Pass a pre-fit ``ir`` to skip
        the second fit when caller already has one.
    """
    if method == "min_chosen":
        return norm_chosen.min().item()
    if method == "midpoint":
        return (norm_chosen.mean().item() + norm_rejected.mean().item()) / 2.0
    if method == "kde_bayes":  # TODO delete
        return _kde_bayes_threshold(norm_chosen, norm_rejected, prior)
    if method == "isotonic":
        return _isotonic_threshold(norm_chosen, norm_rejected, ir=ir)
    raise ValueError(f"Unknown threshold method: {method}")


def _silverman_bandwidth(data: torch.Tensor) -> torch.Tensor:
    n = data.numel()
    return data.std() * (4.0 / (3.0 * n)) ** 0.2


def _kde_eval(data: torch.Tensor, grid: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    diff = grid.unsqueeze(1) - data.unsqueeze(0)
    kernel = torch.exp(-0.5 * (diff / h) ** 2)
    return kernel.mean(dim=1) / (h * math.sqrt(2.0 * math.pi))


def _kde_bayes_threshold(
    norm_chosen: torch.Tensor,
    norm_rejected: torch.Tensor,
    prior: float = 0.5,
    n_grid: int = 500,
) -> float:
    c = norm_chosen.detach()
    r = norm_rejected.detach()

    if c.numel() < 5:
        return ((c.mean() + r.mean()) / 2.0).item()

    h_c = _silverman_bandwidth(c)
    h_r = _silverman_bandwidth(r)

    lo = min(c.min().item(), r.min().item())
    hi = max(c.max().item(), r.max().item())
    margin = 0.1 * (hi - lo + 1e-8)
    grid = torch.linspace(lo - margin, hi + margin, n_grid, device=c.device)
    dr = (grid[1] - grid[0]).item()

    density_c = _kde_eval(c, grid, h_c)
    density_r = _kde_eval(r, grid, h_r)

    # cumsum has no deterministic CUDA impl — fall back to CPU for the CDF.
    cdf_c = torch.cumsum(density_c.cpu(), dim=0) * dr
    cdf_r = torch.cumsum(density_r.cpu(), dim=0) * dr
    cdf_c = (cdf_c / (cdf_c[-1] + 1e-12)).clamp(0, 1)
    cdf_r = (cdf_r / (cdf_r[-1] + 1e-12)).clamp(0, 1)

    risk = prior * cdf_c + (1 - prior) * (1 - cdf_r)
    return grid.cpu()[risk.argmin()].item()


def _isotonic_threshold(
    norm_chosen: torch.Tensor,
    norm_rejected: torch.Tensor,
    n_grid: int = 500,
    ir=None,
) -> float:
    """Midpoint of ``P(chosen | r) = 0.5`` under isotonic regression."""
    c = norm_chosen.detach()
    r = norm_rejected.detach()

    midpoint = ((c.mean() + r.mean()) / 2.0).item()
    if c.numel() < 5 or r.numel() < 5:
        return midpoint

    if ir is None:
        from sklearn.isotonic import IsotonicRegression
        X = torch.cat([c, r]).cpu().numpy().astype(np.float64)
        y = np.concatenate([
            np.ones(c.numel(), dtype=np.float64),
            np.zeros(r.numel(), dtype=np.float64),
        ])
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(X, y)

    lo = min(c.min().item(), r.min().item())
    hi = max(c.max().item(), r.max().item())
    if hi <= lo:
        return midpoint
    grid = np.linspace(lo, hi, n_grid)
    ps = ir.predict(grid)

    if not (ps > 0.5).any() and not (ps < 0.5).any():
        # IR pooled flat at 0.5 across the whole range.
        return midpoint

    left = int(np.argmax(ps >= 0.5))
    right = len(ps) - 1 - int(np.argmax((ps <= 0.5)[::-1]))

    if left <= right:
        return float((grid[left] + grid[right]) / 2.0)

    p0, p1 = float(ps[right]), float(ps[left])
    x0, x1 = float(grid[right]), float(grid[left])
    t = (0.5 - p0) / (p1 - p0)
    return x0 + t * (x1 - x0)


# ============================================================================
# Per-user accuracy
# ============================================================================


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def per_user_accuracy(
    score_fn: ScoreFn,
    dataset,
    device: str = "cuda",
) -> dict:
    """Per-user accuracy on held-out test pairs.

    Processes users one-at-a-time so peak memory is bounded by a single
    user's test set, not the concatenated tensor over all users.

    Returns ``{"mean", "std", "per_user", "num_users", "total_pairs"}``.
    Users with zero test pairs are excluded.
    """
    accs: list[float] = []
    total_pairs = 0
    with torch.no_grad():
        for uid in dataset.user_ids:
            ud = dataset.user_data[uid]
            n_test = ud.test_chosen_embs.shape[0]
            if n_test == 0:
                continue
            chosen = ud.test_chosen_embs.to(device)
            rejected = ud.test_rejected_embs.to(device)
            uid_t = torch.full((n_test,), uid, dtype=torch.long, device=device)
            r_c = score_fn(chosen, uid_t)
            r_r = score_fn(rejected, uid_t)
            acc = (r_c > r_r).float().mean().item()
            accs.append(acc)
            total_pairs += n_test

    return {
        "mean": float(np.mean(accs)) if accs else 0.0,
        "std": float(np.std(accs)) if accs else 0.0,
        "per_user": accs,
        "num_users": len(accs),
        "total_pairs": total_pairs,
    }
