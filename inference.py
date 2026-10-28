"""Test-set evaluation entry point."""

from __future__ import annotations

from data import EmbeddingDataset
from model import PerSPLRewardHead
from utils import per_user_accuracy

__all__ = ["evaluate_per_user"]


def evaluate_per_user(
    head: PerSPLRewardHead,
    dataset: EmbeddingDataset,
    device: str = "cuda",
) -> dict:
    """Per-user test accuracy. Returns mean/std/per_user/num_users/total_pairs."""
    head = head.to(device).eval()
    return per_user_accuracy(head.score_single, dataset, device)
