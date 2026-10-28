"""Iter-based self-training trainers for PerSPL.

Algorithm (per outer iteration):

    Step 1 (train):
        - Re-initialise V (unless frozen) and W; rebuild optimiser + LR
          scheduler.
        - For ``num_epochs`` epochs over ``PerUserDataset`` (one user per
          loader step, ``tasks_per_batch`` users accumulated per optimiser
          step):
            * sample a labeled mini-batch (``mini_batch_size`` pairs)
            * iter > 0 only: sample an unlabeled mini-batch from this user's
              frozen pseudo-labeled set D̃_u (size ``unlabeled_mini_batch_size``,
              with replacement)
            * loss = α · bt_lab.mean() + (1-α) · bt_unl.mean()
                                       + λ · ((r_c + r_r)/2)²
              where bt_unl = -logσ(sign · (r_u - threshold[u])), sign = 2y-1.
              (iter 0: total = bt_lab.mean() + reg.mean(); α not applied.)
            * end of epoch: refit per-user isotonic on labeled rewards
              → fresh per-user threshold (used by the next epoch's bt_unl).

    Step 2 (re-fit IR): same call as the last end-of-epoch refresh.

    Step 3 (re-pseudo-label):
        - For each user: uniformly sample ``n_unlabeled_per_user`` candidates
          (without replacement) from the unlabeled pool. (Falls back to
          ``dataset.shared_unlabeled_embs`` if the user has no dedicated
          pool; errors if the available pool is smaller than
          ``n_unlabeled_per_user``.) Then score the sampled candidates with
          the just-trained head, run through that user's IR → posterior,
          keep ``post > τ`` (label=1) or ``post < 1-τ`` (label=0), drop
          the middle. The kept (embs, labels) is D̃_u for the next iter.

Iter 0 has no D̃_u, so its loss is purely labeled BT + reg (a plain
warmup that produces the first IR / threshold / pseudo set).

Per-iter best epoch + acc are recorded; the globally-best checkpoint
across all iters is restored at the end. ``train()`` also returns the
last-epoch state (before the best-state restore) for downstream phase-2
fine-tuning.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader

from config import PerSPLPhaseConfig
from data import (
    EmbeddingDataset,
    PerUserDataset,
    build_training_tensors,
    per_user_collate,
)
from model import PerSPLRewardHead
from utils import build_lr_scheduler, compute_threshold


class PerSPLIterTrainer:
    """Outer iter loop + inner epoch loop for PerSPL self-training."""

    def __init__(
        self,
        head: PerSPLRewardHead,
        dataset: EmbeddingDataset,
        config: PerSPLPhaseConfig,
        device: str = "cuda",
        *,
        freeze_V: bool = False,
    ):
        self.config = config
        self.device = torch.device(device)
        self.head = head.to(self.device)
        self.dataset = dataset
        self.num_users = len(dataset.user_ids)
        self.freeze_V = freeze_V

        if freeze_V:
            self.head.V.requires_grad_(False)
            self._stage1_V = self.head.V.detach().clone()
        else:
            self._stage1_V = None

        # Flat labeled / test tensors (re-used every iter for IR fit + eval).
        t = build_training_tensors(dataset, self.device)
        self.labeled_chosen = t.labeled_chosen
        self.labeled_rejected = t.labeled_rejected
        self.labeled_uids = t.labeled_uids
        self.per_user_labeled = t.per_user_labeled
        self._test_chosen = t.test_chosen
        self._test_rejected = t.test_rejected
        self._test_uids = t.test_uids

        # Per-user labeled slices into the flat tensors.
        self._labeled_uid_slices: dict[int, slice] = {}
        off = 0
        for uid in dataset.user_ids:
            if uid in self.per_user_labeled:
                n = self.per_user_labeled[uid][0].shape[0]
                self._labeled_uid_slices[uid] = slice(off, off + n)
                off += n

        # Per-user unlabeled candidate pools kept on CPU; sampling moves a
        # small batch to GPU for scoring. (Some shared pools are >10 GiB,
        # so pinning the whole pool to GPU is impractical.)
        self.per_user_unlabeled: dict[int, torch.Tensor] = {}
        for uid in dataset.user_ids:
            ud = dataset.user_data[uid]
            if ud.unlabeled_embs is not None and ud.unlabeled_embs.shape[0] > 0:
                self.per_user_unlabeled[int(uid)] = ud.unlabeled_embs
        if (dataset.shared_unlabeled_embs is not None
                and dataset.shared_unlabeled_embs.shape[0] > 0):
            self.shared_unlabeled = dataset.shared_unlabeled_embs
        else:
            self.shared_unlabeled = None

        # Per-user calibration (re-fit each epoch end during training).
        self.per_user_threshold = torch.zeros(self.num_users, device=self.device)
        self.per_user_iso: dict[int, IsotonicRegression] = {}

        # Frozen pseudo-label set for the current iter. Empty for iter 0;
        # populated at end of each iter by ``_regenerate_pseudo_set``.
        self.per_user_pseudo: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # Validate every user has a pool of at least n_unlabeled_per_user.
        n_req = self.config.n_unlabeled_per_user
        shared_n = self.shared_unlabeled.shape[0] if self.shared_unlabeled is not None else 0
        for uid in dataset.user_ids:
            uid_int = int(uid)
            pool = self.per_user_unlabeled.get(uid_int)
            n_pool = pool.shape[0] if pool is not None else shared_n
            if n_pool < n_req:
                raise ValueError(
                    f"User {uid_int}: unlabeled pool size {n_pool} < "
                    f"n_unlabeled_per_user {n_req}."
                )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self) -> dict:
        cfg = self.config
        print(f"{'=' * 60}")
        print(
            f"{self.method_name()}: {cfg.n_iter} iters × {cfg.num_epochs} epochs, "
            f"{self.num_users} users, mini_batch={cfg.mini_batch_size}, "
            f"u_mini_batch={cfg.unlabeled_mini_batch_size}, "
            f"tasks_per_batch={cfg.tasks_per_batch}, α={cfg.labeled_loss_weight}, "
            f"τ={cfg.iso_filter_threshold}"
        )
        print(f"{'=' * 60}")

        per_iter: list[dict] = []
        global_best_score = float("-inf")
        global_best_state: Optional[dict] = None
        global_best_iter = -1
        global_best_epoch = -1

        for it in range(cfg.n_iter):
            print(f"\n[{self.method_name()} iter {it + 1}/{cfg.n_iter}] "
                  f"{'(labeled-only warmup)' if it == 0 else 'with pseudo-labels'}")
            self._reinit_model()
            use_unlabeled = (it > 0) and bool(self.per_user_pseudo)
            (best_score, best_epoch, best_state, last_score,
             best_extras, last_extras) = self._train_one_iter(it, use_unlabeled)
            per_iter.append({
                "iter": it + 1,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "last_score": last_score,
                **{f"best_{k}": v for k, v in best_extras.items()},
                **{f"last_{k}": v for k, v in last_extras.items()},
            })
            print(f"[{self.method_name()} iter {it + 1}] "
                  f"best epoch={best_epoch}  score={best_score:.4f}  "
                  f"last epoch={cfg.num_epochs}  score={last_score:.4f}")
            if best_score > global_best_score:
                global_best_score = best_score
                global_best_state = best_state
                global_best_iter = it + 1
                global_best_epoch = best_epoch

            # Step 3: regenerate pseudo set for the next iter.
            if it < cfg.n_iter - 1:
                self._regenerate_pseudo_set()

        # Last-epoch state of the final iter, captured BEFORE we restore the
        # globally-best checkpoint. Phase 2 uses this for V continuity.
        last_epoch_state = copy.deepcopy(self.head.state_dict())

        if global_best_state is not None:
            self.head.load_state_dict(global_best_state)
            print(
                f"\n[{self.method_name()}] restored global best from "
                f"iter {global_best_iter} epoch {global_best_epoch} "
                f"(score={global_best_score:.4f})"
            )

        print(f"\n{'=' * 60}")
        print(f"[{self.method_name()}] per-iter best summary")
        print(f"{'=' * 60}")
        for row in per_iter:
            print(f"  iter {row['iter']:>2}: "
                  f"best epoch {row['best_epoch']:>3}  "
                  f"score {row['best_score']:.4f}  |  "
                  f"last epoch {cfg.num_epochs:>3}  "
                  f"score {row['last_score']:.4f}")

        return {
            "per_iter": per_iter,
            "global_best_iter": global_best_iter,
            "global_best_epoch": global_best_epoch,
            "global_best_score": global_best_score,
            "last_epoch_state": last_epoch_state,
        }

    def _reinit_model(self) -> None:
        """Re-init V (unless frozen) + W; optimizer state is re-built per iter."""
        with torch.no_grad():
            if not self.freeze_V:
                nn.init.normal_(self.head.V)
            else:
                self.head.V.copy_(self._stage1_V)
            nn.init.normal_(self.head.W)
        if self.freeze_V:
            self.head.V.requires_grad_(False)
        self.per_user_threshold = torch.zeros(self.num_users, device=self.device)
        self.per_user_iso = {}

    def _train_one_iter(
        self, iter_idx: int, use_unlabeled: bool,
    ) -> tuple[float, int, dict, float, dict, dict]:
        cfg = self.config
        optimizer = self._build_optimizer()
        steps_per_epoch = max(self.num_users // max(cfg.tasks_per_batch, 1), 1)
        total_steps = cfg.num_epochs * steps_per_epoch
        scheduler = build_lr_scheduler(
            optimizer, cfg.lr_schedule, total_steps,
            lr=cfg.lr,
            lr_min=cfg.lr_min,
            warmup_frac=cfg.warmup_frac,
            linear_start_factor=cfg.linear_start_factor,
            onecycle_pct_start=cfg.onecycle_pct_start,
            onecycle_div_factor=cfg.onecycle_div_factor,
            onecycle_final_div_factor=cfg.onecycle_final_div_factor,
            onecycle_anneal_strategy=cfg.onecycle_anneal_strategy,
            onecycle_cycle_momentum=(not cfg.onecycle_no_cycle_momentum),
            onecycle_three_phase=cfg.onecycle_three_phase,
        )

        # Seed first-epoch threshold for iter > 0 (iter 0 does not consume it).
        if use_unlabeled:
            self._refresh_threshold_and_ir()

        loader = DataLoader(
            PerUserDataset(self.dataset, self.device),
            batch_size=1, shuffle=True, num_workers=0,
            collate_fn=per_user_collate,
        )

        best_score = float("-inf")
        best_epoch = -1
        best_state: Optional[dict] = None
        last_score = float("nan")
        best_extras: dict = {}
        last_extras: dict = {}

        for epoch in range(cfg.num_epochs):
            agg = {"bt_lab": 0.0, "bt_unl": 0.0, "reg": 0.0, "n": 0}
            user_buf: list[tuple[torch.Tensor, torch.Tensor, int]] = []

            for user_batch in loader:
                chosen_all, rejected_all, uid = user_batch[0]
                mini_c, mini_r = self._sample_mini_batch(
                    chosen_all, rejected_all, cfg.mini_batch_size,
                )
                user_buf.append((mini_c, mini_r, uid))

                if len(user_buf) == cfg.tasks_per_batch:
                    parts = self._compute_batch_loss(user_buf, use_unlabeled)
                    optimizer.zero_grad()
                    parts["total"].backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    agg["bt_lab"] += parts["bt_lab"]
                    agg["bt_unl"] += parts["bt_unl"]
                    agg["reg"] += parts["reg"]
                    agg["n"] += 1
                    user_buf.clear()

            # End-of-epoch threshold/IR refresh:
            #   - iter > 0: needed every epoch (next epoch consumes it)
            #   - iter 0: only the final epoch's refresh matters (Step 3)
            if use_unlabeled or epoch == cfg.num_epochs - 1:
                self._refresh_threshold_and_ir()

            if (epoch + 1) % cfg.log_interval == 0:
                n = max(agg["n"], 1)
                score, suffix, eval_extras = self._eval_summary()
                improved = score > best_score
                marker = " *best" if improved else ""
                extras = (
                    f"  bt_lab={agg['bt_lab'] / n:.4f}  "
                    f"bt_unl={agg['bt_unl'] / n:.4f}  "
                    f"reg={agg['reg'] / n:.4f}"
                )
                print(
                    f"    [iter {iter_idx + 1} epoch {epoch + 1:>3}/{cfg.num_epochs}] "
                    f"steps={agg['n']}{extras}  {suffix}{marker}"
                )
                if improved:
                    best_score = score
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.head.state_dict())
                    best_extras = dict(eval_extras)
                if epoch == cfg.num_epochs - 1:
                    last_score = score
                    last_extras = dict(eval_extras)

        # Fallback if log_interval > num_epochs.
        if best_state is None:
            best_score, _, best_extras = self._eval_summary()
            best_extras = dict(best_extras)
            best_epoch = cfg.num_epochs
            best_state = copy.deepcopy(self.head.state_dict())
        if last_score != last_score:
            last_score, _, le = self._eval_summary()
            last_extras = dict(le)

        return best_score, best_epoch, best_state, last_score, best_extras, last_extras

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #

    def _compute_batch_loss(
        self,
        users: list[tuple[torch.Tensor, torch.Tensor, int]],
        use_unlabeled: bool,
    ) -> dict:
        cfg = self.config
        device = self.device

        # Labeled side (always present).
        mini_c = torch.cat([u[0] for u in users])
        mini_r = torch.cat([u[1] for u in users])
        uids_l = torch.cat([
            torch.full((u[0].shape[0],), u[2], dtype=torch.long, device=device)
            for u in users
        ])
        r_c = self.head.score_single(mini_c, uids_l)
        r_r = self.head.score_single(mini_r, uids_l)

        bt_lab = -F.logsigmoid(r_c - r_r).mean()
        reg = cfg.reward_reg_lambda * (((r_c + r_r) / 2.0) ** 2).mean()

        if not use_unlabeled:
            total = bt_lab + reg
            return {
                "total": total,
                "bt_lab": float(bt_lab.detach()),
                "bt_unl": 0.0,
                "reg": float(reg.detach()),
            }

        # Unlabeled side: sample k_u per user from frozen D̃_u (with replacement).
        k_u = cfg.unlabeled_mini_batch_size
        u_embs_list, u_labels_list, u_uids_list = [], [], []
        for _, _, uid in users:
            ps = self.per_user_pseudo.get(int(uid))
            if ps is None or ps[0].shape[0] == 0:
                continue
            embs_pool, labels_pool = ps
            idx = torch.randint(0, embs_pool.shape[0], (k_u,), device=device)
            u_embs_list.append(embs_pool[idx])
            u_labels_list.append(labels_pool[idx])
            u_uids_list.append(
                torch.full((k_u,), int(uid), dtype=torch.long, device=device)
            )

        if u_embs_list:
            u_embs = torch.cat(u_embs_list)
            u_labels = torch.cat(u_labels_list)
            u_uids = torch.cat(u_uids_list)
            r_u = self.head.score_single(u_embs, u_uids)
            thr_u = self.per_user_threshold[u_uids]
            signs = 2.0 * u_labels - 1.0
            bt_unl = -F.logsigmoid(signs * (r_u - thr_u)).mean()
        else:
            bt_unl = torch.zeros((), device=device)

        alpha = cfg.labeled_loss_weight
        total = alpha * bt_lab + (1.0 - alpha) * bt_unl + reg
        return {
            "total": total,
            "bt_lab": float(bt_lab.detach()),
            "bt_unl": float(bt_unl.detach()),
            "reg": float(reg.detach()),
        }

    # ------------------------------------------------------------------ #
    # Threshold / IR / pseudo-set
    # ------------------------------------------------------------------ #

    def _refresh_threshold_and_ir(self) -> None:
        if self.labeled_chosen.shape[0] == 0:
            return
        method = self.config.threshold_method
        self.head.eval()
        with torch.no_grad():
            r_c_all = self.head.score_single(self.labeled_chosen, self.labeled_uids)
            r_r_all = self.head.score_single(self.labeled_rejected, self.labeled_uids)
            for uid in self.dataset.user_ids:
                sl = self._labeled_uid_slices.get(uid)
                if sl is None:
                    continue
                r_c = r_c_all[sl]
                r_r = r_r_all[sl]
                ir = None
                if r_c.numel() >= 5 and r_r.numel() >= 5:
                    X = torch.cat([r_c, r_r]).cpu().numpy().astype(np.float64)
                    y = np.concatenate([
                        np.ones(r_c.numel(), dtype=np.float64),
                        np.zeros(r_r.numel(), dtype=np.float64),
                    ])
                    ir = IsotonicRegression(
                        out_of_bounds="clip", y_min=0.0, y_max=1.0,
                    )
                    ir.fit(X, y)
                    self.per_user_iso[int(uid)] = ir
                self.per_user_threshold[uid] = compute_threshold(
                    r_c, r_r, method=method, prior=0.5, ir=ir,
                )
        self.head.train()

    def _regenerate_pseudo_set(self) -> None:
        """Step 3: per-user sample, score, filter via IR posterior."""
        cfg = self.config
        self.per_user_pseudo = {}
        self.head.eval()
        kept_total = 0
        sampled_total = 0
        users_with_pseudo = 0
        with torch.no_grad():
            for uid in self.dataset.user_ids:
                uid_int = int(uid)
                pool = self.per_user_unlabeled.get(uid_int)
                if pool is None:
                    pool = self.shared_unlabeled
                if pool is None or pool.shape[0] == 0:
                    continue
                if pool.shape[0] < cfg.n_unlabeled_per_user:
                    raise ValueError(
                        f"User {uid_int}: pool {pool.shape[0]} < "
                        f"n_unlabeled_per_user {cfg.n_unlabeled_per_user}"
                    )
                rand_idx = torch.randperm(
                    pool.shape[0], device=pool.device,
                )[:cfg.n_unlabeled_per_user]
                sampled = pool[rand_idx].to(self.device, non_blocking=True)
                uid_vec = torch.full(
                    (cfg.n_unlabeled_per_user,), uid_int,
                    dtype=torch.long, device=self.device,
                )
                r = self.head.score_single(sampled, uid_vec)
                ir = self.per_user_iso.get(uid_int)
                if ir is None:
                    continue
                post = ir.predict(r.cpu().numpy().astype(np.float64))
                tau = cfg.iso_filter_threshold
                keep_chosen = post > tau
                keep_rejected = post < (1.0 - tau)
                keep = keep_chosen | keep_rejected
                n_keep = int(keep.sum())
                sampled_total += cfg.n_unlabeled_per_user
                kept_total += n_keep
                if n_keep == 0:
                    continue
                keep_t = torch.from_numpy(keep).to(self.device)
                kept_embs = sampled[keep_t]
                kept_labels = torch.from_numpy(
                    keep_chosen[keep].astype(np.float32)
                ).to(self.device)
                self.per_user_pseudo[uid_int] = (kept_embs, kept_labels)
                users_with_pseudo += 1
        self.head.train()
        sampled_total = max(sampled_total, 1)
        print(
            f"    [pseudo-set] users={users_with_pseudo}/{self.num_users}, "
            f"kept={kept_total}/{sampled_total} "
            f"({kept_total / sampled_total:.1%})"
        )

    # ------------------------------------------------------------------ #
    # Optimizer / weight-decay groups
    # ------------------------------------------------------------------ #

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        target = cfg.weight_decay_target
        wd = cfg.weight_decay
        trainable = [p for p in self.head.parameters() if p.requires_grad]
        if target == "all" or wd <= 0:
            return torch.optim.Adam(
                trainable, lr=cfg.lr, weight_decay=wd, foreach=False,
            )
        V, W = self.head.V, self.head.W
        other = [p for p in trainable if (p is not V and p is not W)]
        if target == "user":
            groups = [
                {"params": [W], "weight_decay": wd},
                {"params": ([V] if V.requires_grad else []) + other,
                 "weight_decay": 0.0},
            ]
        elif target == "shared":
            groups = [
                {"params": ([V] if V.requires_grad else []), "weight_decay": wd},
                {"params": [W] + other, "weight_decay": 0.0},
            ]
        else:
            raise ValueError(
                f"weight_decay_target must be all/user/shared, got {target!r}"
            )
        groups = [g for g in groups if g["params"]]
        return torch.optim.Adam(groups, lr=cfg.lr, foreach=False)

    # ------------------------------------------------------------------ #
    # Sampling / eval
    # ------------------------------------------------------------------ #

    def _sample_mini_batch(
        self, chosen_all: torch.Tensor, rejected_all: torch.Tensor, k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = chosen_all.shape[0]
        kk = min(k, n)
        idx = torch.randperm(n, device=self.device)[:kk]
        return chosen_all[idx], rejected_all[idx]

    def _per_user_test_accs(self) -> list[float]:
        if self._test_chosen is None:
            return []
        M = self.num_users
        self.head.eval()
        with torch.no_grad():
            r_c = self.head.score_single(self._test_chosen, self._test_uids)
            r_r = self.head.score_single(self._test_rejected, self._test_uids)
            correct = (r_c > r_r).float()
            per_user_sum = torch.zeros(M, device=self.device)
            per_user_count = torch.zeros(M, device=self.device)
            per_user_sum.scatter_add_(0, self._test_uids, correct)
            per_user_count.scatter_add_(0, self._test_uids, torch.ones_like(correct))
            has = per_user_count > 0
            accs = (per_user_sum[has] / per_user_count[has]).tolist()
        self.head.train()
        return accs

    def _compute_test_accuracy(self) -> float:
        accs = self._per_user_test_accs()
        return float(sum(accs) / len(accs)) if accs else 0.0

    def _eval_summary(self) -> tuple[float, str, dict]:
        acc = self._compute_test_accuracy()
        return acc, f"test_acc={acc:.4f}", {}

    def method_name(self) -> str:
        return "PerSPL"


# ============================================================================
# Unseen-user trainer: V frozen, W trained via the same iter protocol.
# Best-epoch selection drives by combined seen+unseen acc.
# ============================================================================


class PerSPLIterUnseenTrainer(PerSPLIterTrainer):

    def __init__(
        self,
        head: PerSPLRewardHead,
        dataset: EmbeddingDataset,
        seen_per_user_accs: list[float],
        config: PerSPLPhaseConfig,
        device: str = "cuda",
    ):
        super().__init__(
            head=head, dataset=dataset, config=config,
            device=device, freeze_V=True,
        )
        self._seen_accs = list(seen_per_user_accs)
        self._seen_mean = (
            float(sum(self._seen_accs) / len(self._seen_accs))
            if self._seen_accs else 0.0
        )

    def method_name(self) -> str:
        return "PerSPL-unseen"

    def _eval_summary(self) -> tuple[float, str, dict]:
        unseen = self._per_user_test_accs()
        unseen_mean = (
            float(sum(unseen) / len(unseen)) if unseen else 0.0
        )
        combined = self._seen_accs + unseen
        total_mean = (
            float(sum(combined) / len(combined)) if combined else 0.0
        )
        extras = {"unseen_acc": unseen_mean, "seen_acc": self._seen_mean}
        return total_mean, (
            f"seen_acc={self._seen_mean:.4f}  "
            f"unseen_acc={unseen_mean:.4f}  "
            f"combined={total_mean:.4f}"
        ), extras


# ============================================================================
# Phase-2 entry: run_unseen_eval
# ============================================================================


def _seen_per_user_accs(
    head: PerSPLRewardHead,
    dataset: EmbeddingDataset,
    device,
) -> list[float]:
    """Per-user accuracy on the seen-user test split for the given head."""
    kept_uids: list[int] = []
    test_c_list, test_r_list, test_u_list = [], [], []
    for uid in dataset.user_ids:
        ud = dataset.user_data[uid]
        n_test = ud.test_chosen_embs.shape[0]
        if n_test == 0:
            continue
        kept_uids.append(uid)
        test_c_list.append(ud.test_chosen_embs.to(device))
        test_r_list.append(ud.test_rejected_embs.to(device))
        test_u_list.append(
            torch.full((n_test,), uid, dtype=torch.long, device=device)
        )
    if not test_c_list:
        return []
    test_chosen = torch.cat(test_c_list)
    test_rejected = torch.cat(test_r_list)
    test_uids = torch.cat(test_u_list)
    M = max(dataset.user_ids) + 1
    head.eval()
    with torch.no_grad():
        r_c = head.score_single(test_chosen, test_uids)
        r_r = head.score_single(test_rejected, test_uids)
        correct = (r_c > r_r).float()
        per_user_sum = torch.zeros(M, device=device)
        per_user_count = torch.zeros(M, device=device)
        per_user_sum.scatter_add_(0, test_uids, correct)
        per_user_count.scatter_add_(0, test_uids, torch.ones_like(correct))
        kept_idx = torch.tensor(kept_uids, dtype=torch.long, device=device)
        accs = (per_user_sum[kept_idx] / per_user_count[kept_idx]).tolist()
    head.train()
    return accs


def _build_unseen_dataset(dataset: EmbeddingDataset) -> EmbeddingDataset:
    return EmbeddingDataset(
        user_data=dataset.unseen_user_data,
        user_ids=list(dataset.unseen_user_ids),
        unseen_user_data={},
        unseen_user_ids=[],
        emb_dim=dataset.emb_dim,
        V_ref=dataset.V_ref,
        shared_unlabeled_embs=dataset.shared_unlabeled_embs,
        metadata=dataset.metadata,
    )


def run_unseen_eval(
    stage1_head: PerSPLRewardHead,
    dataset: EmbeddingDataset,
    config: PerSPLPhaseConfig,
    device,
) -> tuple[Optional[PerSPLRewardHead], dict]:
    """Phase 2: freeze stage-1 V, train unseen-user W via the iter protocol.

    Returns ``(unseen_head, accuracy_dict)``. The accuracy dict uses
    explicit (subset × epoch) keys — no ambiguous ``mean`` fields.
    """
    n_unseen = len(dataset.unseen_user_ids)
    if n_unseen == 0:
        return None, {}

    seen_accs = _seen_per_user_accs(stage1_head, dataset, device)

    unseen_head = PerSPLRewardHead(
        emb_dim=stage1_head.emb_dim,
        num_users=n_unseen,
        rank=stage1_head.rank,
        temperature=stage1_head.temperature,
    ).to(device)
    with torch.no_grad():
        unseen_head.V.copy_(stage1_head.V)
        unseen_head.V_ref.copy_(stage1_head.V_ref)
    unseen_head.V.requires_grad_(False)

    unseen_ds = _build_unseen_dataset(dataset)
    trainer = PerSPLIterUnseenTrainer(
        head=unseen_head,
        dataset=unseen_ds,
        seen_per_user_accs=seen_accs,
        config=config,
        device=device,
    )

    _, init_suffix, _ = trainer._eval_summary()
    print(f"    [PerSPL-unseen init (random W, V from stage 1)] {init_suffix}")

    train_summary = trainer.train()
    last_epoch_state = train_summary.pop("last_epoch_state", None)

    # Best-epoch unseen acc (head was restored to globally-best by train()).
    unseen_best_accs = trainer._per_user_test_accs()
    unseen_best_mean = (
        float(sum(unseen_best_accs) / len(unseen_best_accs))
        if unseen_best_accs else 0.0
    )

    # Last-epoch unseen acc. V is frozen → seen acc doesn't change.
    if last_epoch_state is not None:
        best_state = copy.deepcopy(unseen_head.state_dict())
        unseen_head.load_state_dict(last_epoch_state)
        unseen_last_accs = trainer._per_user_test_accs()
        unseen_last_mean = (
            float(sum(unseen_last_accs) / len(unseen_last_accs))
            if unseen_last_accs else 0.0
        )
        # Restore best so the returned head is canonical (best-epoch).
        unseen_head.load_state_dict(best_state)
    else:
        unseen_last_mean = unseen_best_mean

    seen_mean = (
        float(sum(seen_accs) / len(seen_accs)) if seen_accs else 0.0
    )

    accuracy = {
        "seen_best_epoch_acc": seen_mean,
        "seen_last_epoch_acc": seen_mean,  # V frozen → identical to best
        "unseen_best_epoch_acc": unseen_best_mean,
        "unseen_last_epoch_acc": unseen_last_mean,
        "seen_per_user_accs": seen_accs,
        "unseen_per_user_accs": unseen_best_accs,
    }
    return unseen_head, {"accuracy": accuracy, "phase2_summary": train_summary}
