"""Per-user embedding containers + flat-tensor builders for training.

This module owns everything that turns raw per-user embedding data into
GPU-ready tensors. Loaders in ``loaders.py`` produce ``EmbeddingDataset``
objects; the trainer in ``trainer.py`` consumes them via
``build_training_tensors`` and ``PerUserDataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ============================================================================
# Per-user containers
# ============================================================================


class LazyIndexedEmbedding:
    """Memory-efficient proxy: stores indices into a shared embedding tensor.

    Many users share a single ``(n_prompts, n_responses, emb_dim)`` tensor;
    each user's view is a 1-D index tensor selecting one response per
    prompt. The full per-user matrix is materialised on demand via ``.to()``
    or ``.float()`` — typically right before a forward pass.
    """

    def __init__(self, shared_embs: torch.Tensor, indices: torch.Tensor):
        # shared_embs: (n_prompts, n_responses, emb_dim)
        # indices:     (n_prompts,)  – which response per prompt
        self._shared_embs = shared_embs
        self._indices = indices

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self._indices.shape[0], self._shared_embs.shape[2]])

    def _materialize(self) -> torch.Tensor:
        p = torch.arange(self._indices.shape[0])
        return self._shared_embs[p, self._indices]

    def to(self, device) -> torch.Tensor:
        return self._materialize().to(device)

    def float(self) -> torch.Tensor:
        return self._materialize().float()

    def __sub__(self, other):
        if isinstance(other, LazyIndexedEmbedding):
            return self._materialize() - other._materialize()
        return self._materialize() - other

    def __rsub__(self, other):
        return other - self._materialize()


@dataclass
class UserData:
    """Per-user embedding data container."""
    labeled_chosen_embs: torch.Tensor   # [n_train, emb_dim]
    labeled_rejected_embs: torch.Tensor  # [n_train, emb_dim]
    unlabeled_embs: torch.Tensor        # [n_unlabeled, emb_dim]
    test_chosen_embs: "torch.Tensor | LazyIndexedEmbedding"     # [n_test, emb_dim]
    test_rejected_embs: "torch.Tensor | LazyIndexedEmbedding"   # [n_test, emb_dim]
    val_chosen_embs: Optional[torch.Tensor] = None
    val_rejected_embs: Optional[torch.Tensor] = None


@dataclass
class EmbeddingDataset:
    """Unified dataset of pre-computed embeddings."""
    user_data: dict[int, UserData]
    user_ids: list[int]
    unseen_user_data: dict[int, UserData]
    unseen_user_ids: list[int]
    emb_dim: int
    V_ref: Optional[torch.Tensor] = None
    shared_unlabeled_embs: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None


# ============================================================================
# Training-tensor builders
# ============================================================================


@dataclass
class TrainingTensors:
    """Flat device-resident tensors + per-user views consumed by the trainer."""
    labeled_chosen: torch.Tensor
    labeled_rejected: torch.Tensor
    labeled_uids: torch.Tensor
    labeled_dataset: Optional[TensorDataset]
    per_user_labeled: dict[int, tuple[torch.Tensor, torch.Tensor]]
    per_user_val: dict[int, tuple[torch.Tensor, torch.Tensor]]
    val_chosen: Optional[torch.Tensor]
    val_rejected: Optional[torch.Tensor]
    val_uids: Optional[torch.Tensor]
    test_chosen: Optional[torch.Tensor]
    test_rejected: Optional[torch.Tensor]
    test_uids: Optional[torch.Tensor]

    @property
    def n_labeled(self) -> int:
        return self.labeled_chosen.shape[0]


def build_training_tensors(dataset: EmbeddingDataset,
                           device: torch.device) -> TrainingTensors:
    """Flatten ``EmbeddingDataset.user_data`` into device tensors."""
    labeled_c_list, labeled_r_list, labeled_u_list = [], [], []
    per_user_labeled: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    per_user_val: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for uid in dataset.user_ids:
        ud = dataset.user_data[uid]
        n_labeled = ud.labeled_chosen_embs.shape[0]
        if n_labeled > 0:
            labeled_c_list.append(ud.labeled_chosen_embs)
            labeled_r_list.append(ud.labeled_rejected_embs)
            labeled_u_list.append(torch.full((n_labeled,), uid, dtype=torch.long))
            per_user_labeled[uid] = (
                ud.labeled_chosen_embs.to(device),
                ud.labeled_rejected_embs.to(device),
            )
        if ud.val_chosen_embs is not None and ud.val_chosen_embs.shape[0] > 0:
            per_user_val[uid] = (
                ud.val_chosen_embs.to(device),
                ud.val_rejected_embs.to(device),
            )

    if labeled_c_list:
        labeled_chosen = torch.cat(labeled_c_list, dim=0).to(device)
        labeled_rejected = torch.cat(labeled_r_list, dim=0).to(device)
        labeled_uids = torch.cat(labeled_u_list, dim=0).to(device)
        labeled_dataset = TensorDataset(labeled_chosen, labeled_rejected, labeled_uids)
    else:
        labeled_chosen = torch.empty(0, dataset.emb_dim, device=device)
        labeled_rejected = torch.empty(0, dataset.emb_dim, device=device)
        labeled_uids = torch.empty(0, dtype=torch.long, device=device)
        labeled_dataset = None

    test_c_list, test_r_list, test_u_list = [], [], []
    for uid in dataset.user_ids:
        ud = dataset.user_data[uid]
        n_test = ud.test_chosen_embs.shape[0]
        if n_test > 0:
            test_c_list.append(ud.test_chosen_embs.to(device))
            test_r_list.append(ud.test_rejected_embs.to(device))
            test_u_list.append(torch.full((n_test,), uid, dtype=torch.long, device=device))
    if test_c_list:
        test_chosen = torch.cat(test_c_list, dim=0)
        test_rejected = torch.cat(test_r_list, dim=0)
        test_uids = torch.cat(test_u_list, dim=0)
    else:
        test_chosen = test_rejected = test_uids = None

    # Fall back to labeled for val if no dedicated val split exists.
    val_src = per_user_val if per_user_val else per_user_labeled
    val_c_list, val_r_list, val_u_list = [], [], []
    for uid in dataset.user_ids:
        if uid not in val_src:
            continue
        c, r = val_src[uid]
        val_c_list.append(c)
        val_r_list.append(r)
        val_u_list.append(torch.full((c.shape[0],), uid, dtype=torch.long, device=device))
    if val_c_list:
        val_chosen = torch.cat(val_c_list, dim=0)
        val_rejected = torch.cat(val_r_list, dim=0)
        val_uids = torch.cat(val_u_list, dim=0)
    else:
        val_chosen = val_rejected = val_uids = None

    return TrainingTensors(
        labeled_chosen=labeled_chosen,
        labeled_rejected=labeled_rejected,
        labeled_uids=labeled_uids,
        labeled_dataset=labeled_dataset,
        per_user_labeled=per_user_labeled,
        per_user_val=per_user_val,
        val_chosen=val_chosen,
        val_rejected=val_rejected,
        val_uids=val_uids,
        test_chosen=test_chosen,
        test_rejected=test_rejected,
        test_uids=test_uids,
    )


# ============================================================================
# Per-user DataLoader wrapper
# ============================================================================


class PerUserDataset(Dataset):
    """One item = one user's full labeled preference data.

    The DataLoader iterates users (batch_size=1 with ``per_user_collate``);
    the trainer's inner loop samples mini-batches inside each user.
    """

    def __init__(self, embedding_dataset: EmbeddingDataset, device: torch.device):
        self._items: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for uid in embedding_dataset.user_ids:
            ud = embedding_dataset.user_data[uid]
            if ud.labeled_chosen_embs.shape[0] == 0:
                continue
            self._items.append((
                ud.labeled_chosen_embs.to(device),
                ud.labeled_rejected_embs.to(device),
                int(uid),
            ))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self._items[idx]


def per_user_collate(batch):
    """Identity collate: keep per-user items as a list (no stacking)."""
    return batch
