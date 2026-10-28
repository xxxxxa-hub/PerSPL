"""Dataset loaders: reddit_tldr, personalllm, helpsteer2.

Each loader reads pre-extracted embedding caches from
``./cache/<dataset>/`` and produces an ``EmbeddingDataset``. Embedding
extraction is **not** included in this release — see README.md for how
to obtain the per-dataset cache files.

Cache file conventions (auto-discovered by each loader):

  reddit_tldr/  →  reddit.pt + reddit.json
  personalllm/  →  train_embeddings.pt + test_embeddings.pt
                   + reward_tensor.pt + reward_tensor_test.pt
                   + V_ref.pt
  helpsteer2/   →  train_embeddings.pt + validation_embeddings.pt
                   + train_scores.pt + validation_scores.pt
                   + train_prompt_groups.json + validation_prompt_groups.json
                   + V_ref.pt
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data import EmbeddingDataset, LazyIndexedEmbedding, UserData


# ============================================================================
# Shared embedding tensor cache (avoid loading the same .pt twice)
# ============================================================================


@lru_cache(maxsize=4)
def _shared_emb_tensor(path_str: str) -> torch.Tensor:
    """Cache the heavy .pt embedding tensor by absolute path.

    Treat the returned tensor as read-only — downstream code clones
    per-pair to avoid in-place mutation of the shared store.
    """
    return torch.load(path_str, weights_only=True).float()


# ============================================================================
# Reddit TL;DR (MRM-format cache)
# ============================================================================


def _build_reddit_user_data(meta: dict, emb: torch.Tensor, *,
                            val_ratio: float, seed: int,
                            seen_train_limit: int, unseen_train_limit: int):
    """MRM-style build: returns (seen_user_data, unseen_user_data, remainder).

    Splits users 50/50 seen/unseen (deterministic given seed), then per
    user splits pairs into train/test, caps train to ``*_train_limit``,
    and dumps the sliced-off pairs into a remainder pool that becomes the
    shared unlabeled set.
    """
    user_ids = sorted(meta.keys())
    random.Random(seed).shuffle(user_ids)
    n = len(user_ids)
    seen_user_ids = user_ids[: n // 2]
    unseen_user_ids = user_ids[n // 2:]

    remainder_embs: list[torch.Tensor] = []

    def make_user_data(uids, train_limit):
        out = {}
        for uid in uids:
            entry_list = meta[uid]
            train_examples, test_examples = [], []
            pair_idx = 0
            for e in entry_list:
                conv_id = f"conv_{pair_idx}"
                ex = {
                    "conv_id": conv_id,
                    "chosen_emb": emb[e["chosen_idx"]].clone(),
                    "rejected_emb": emb[e["rejected_idx"]].clone(),
                }
                pair_idx += 1
                if e.get("split") == "train":
                    train_examples.append(ex)
                elif e.get("split") == "val":
                    test_examples.append(ex)
                else:
                    if random.random() < 0.5:
                        train_examples.append(ex)
                    else:
                        test_examples.append(ex)
            rnd = random.Random(
                seed + int(hashlib.md5(uid.encode()).hexdigest(), 16) % (2**32)
            )
            rnd.shuffle(train_examples)
            if train_limit > 0:
                for ex in train_examples[train_limit:]:
                    remainder_embs.append(ex["chosen_emb"])
                    remainder_embs.append(ex["rejected_emb"])
                train_examples = train_examples[:train_limit]
            out[uid] = {"train": train_examples, "test": test_examples}
        return out

    seen_ud = make_user_data(seen_user_ids, seen_train_limit)
    unseen_ud = make_user_data(unseen_user_ids, unseen_train_limit)

    if remainder_embs:
        remainder_tensor = torch.stack(remainder_embs)
    else:
        remainder_tensor = torch.empty(0, emb.shape[-1])

    return seen_ud, unseen_ud, remainder_tensor


def _mrm_user_data_to_user_data(mrm_user_data: dict, uid_to_int: dict,
                                emb_dim: int) -> dict[int, UserData]:
    out: dict[int, UserData] = {}
    for uid_str, data in mrm_user_data.items():
        uid_int = uid_to_int[uid_str]
        train_c = torch.stack([ex["chosen_emb"] for ex in data["train"]])
        train_r = torch.stack([ex["rejected_emb"] for ex in data["train"]])
        test_c = torch.stack([ex["chosen_emb"] for ex in data["test"]])
        test_r = torch.stack([ex["rejected_emb"] for ex in data["test"]])
        out[uid_int] = UserData(
            labeled_chosen_embs=train_c,
            labeled_rejected_embs=train_r,
            unlabeled_embs=torch.empty(0, emb_dim),
            test_chosen_embs=test_c,
            test_rejected_embs=test_r,
        )
    return out


def _cap_unlabeled(pool: torch.Tensor, max_unlabeled: int, seed: int) -> torch.Tensor:
    """Subsample the shared unlabeled pool to ``max_unlabeled`` rows. <=0 = no cap."""
    if pool.shape[0] == 0:
        return pool
    if max_unlabeled <= 0 or pool.shape[0] <= max_unlabeled:
        return pool
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(pool.shape[0], generator=g)[:max_unlabeled]
    return pool[idx]


def _collect_unselected_embs(user_data: dict, selected_uids: set,
                             emb_dim: int) -> torch.Tensor:
    """Flatten all train+test embs of users outside the selected subset."""
    embs: list[torch.Tensor] = []
    for uid, ud in user_data.items():
        if uid in selected_uids:
            continue
        for ex in ud.get("train", []):
            embs.append(ex["chosen_emb"])
            embs.append(ex["rejected_emb"])
        for ex in ud.get("test", []):
            embs.append(ex["chosen_emb"])
            embs.append(ex["rejected_emb"])
    if not embs:
        return torch.empty(0, emb_dim)
    return torch.stack(embs)


def load_reddit_tldr(
    cache_dir: str = "./cache/reddit_tldr",
    n_seen: int = 20,
    n_unseen: int = 20,
    seen_labeled_per_user: int = 100,
    unseen_labeled_per_user: int = 50,
    max_unlabeled: int = -1,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> EmbeddingDataset:
    """Load reddit_tldr from cache files ``reddit.pt`` + ``reddit.json``.

    Sliced-off pairs (beyond ``*_labeled_per_user``) + all pairs from
    users outside the requested counts become the shared unlabeled pool
    (optionally capped by ``max_unlabeled``; <=0 = no cap).
    """
    cache = Path(cache_dir)
    pt_path = cache / "reddit.pt"
    json_path = cache / "reddit.json"
    if not pt_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"reddit_tldr cache missing at {cache}: expected reddit.pt + reddit.json. "
            f"See README.md for cache provisioning instructions."
        )
    with open(json_path) as f:
        meta = json.load(f)
    emb = _shared_emb_tensor(str(pt_path))
    emb_dim = emb.shape[-1]

    seen_ud, unseen_ud, remainder = _build_reddit_user_data(
        meta, emb,
        val_ratio=val_ratio, seed=seed,
        seen_train_limit=seen_labeled_per_user,
        unseen_train_limit=unseen_labeled_per_user,
    )

    # MRM splits 50/50; pick the first n_seen / n_unseen from each half.
    seen_uids = sorted(seen_ud.keys())[:n_seen] if n_seen > 0 else []
    unseen_uids = sorted(unseen_ud.keys())[:n_unseen] if n_unseen > 0 else []
    # Stable order (we don't pre-shuffle here; build_reddit_user_data already shuffled).
    seen_uids = list(seen_ud.keys())[:n_seen] if n_seen > 0 else []
    unseen_uids = list(unseen_ud.keys())[:n_unseen] if n_unseen > 0 else []

    if n_seen > 0 and len(seen_uids) < n_seen:
        raise ValueError(
            f"requested n_seen={n_seen} but only {len(seen_uids)} seen users available"
        )
    if n_unseen > 0 and len(unseen_uids) < n_unseen:
        raise ValueError(
            f"requested n_unseen={n_unseen} but only {len(unseen_uids)} unseen users available"
        )

    seen_subset = {uid: seen_ud[uid] for uid in seen_uids}
    unseen_subset = {uid: unseen_ud[uid] for uid in unseen_uids}
    seen_map = {uid: i for i, uid in enumerate(seen_uids)}
    unseen_map = {uid: i for i, uid in enumerate(unseen_uids)}

    seen_user_data = _mrm_user_data_to_user_data(seen_subset, seen_map, emb_dim)
    unseen_user_data = _mrm_user_data_to_user_data(unseen_subset, unseen_map, emb_dim)

    selected = set(seen_uids) | set(unseen_uids)
    extra_seen = _collect_unselected_embs(seen_ud, selected, emb_dim)
    extra_unseen = _collect_unselected_embs(unseen_ud, selected, emb_dim)
    pool = torch.cat([remainder, extra_seen, extra_unseen], dim=0)
    shared_unlabeled = _cap_unlabeled(pool, max_unlabeled, seed)

    print(
        f"reddit_tldr: {len(seen_user_data)} seen, {len(unseen_user_data)} unseen, "
        f"labeled/user seen={seen_labeled_per_user} unseen={unseen_labeled_per_user}, "
        f"unlabeled pool={shared_unlabeled.shape[0]}"
    )

    return EmbeddingDataset(
        user_data=seen_user_data,
        user_ids=list(seen_user_data.keys()),
        unseen_user_data=unseen_user_data,
        unseen_user_ids=list(unseen_user_data.keys()),
        emb_dim=emb_dim,
        shared_unlabeled_embs=shared_unlabeled,
        metadata={
            "dataset": "reddit_tldr",
            "seen_labeled_per_user": seen_labeled_per_user,
            "unseen_labeled_per_user": unseen_labeled_per_user,
            "max_unlabeled": max_unlabeled,
            "seed": seed,
        },
    )


# ============================================================================
# PersonalLLM (synthetic Dirichlet users over 10 reward models)
# ============================================================================


def load_personalllm(
    cache_dir: str = "./cache/personalllm",
    n_seen: int = 100,
    n_unseen: int = 0,
    alpha: float = 0.1,
    seen_labeled_per_user: int = 45,
    unseen_labeled_per_user: Optional[int] = 9,
    max_unlabeled: int = 5000,
    val_fraction: float = 0.2,
    label_flip_ratio: float = 0.0,
    seed: int = 42,
) -> EmbeddingDataset:
    """Load PersonalLLM from a pre-extracted embedding cache.

    Expected cache files:
        train_embeddings.pt   (num_train, 8, emb_dim)
        test_embeddings.pt    (num_test, 8, emb_dim)
        reward_tensor.pt      (num_train, 8, 10)
        reward_tensor_test.pt (num_test, 8, 10)
        V_ref.pt              (emb_dim, k)

    Users are synthetic: each user u has a 10-vector weight w_u sampled
    from Dir(alpha * 1_10). For each prompt, chosen = argmax(reward @ w_u)
    and rejected = argmin(reward @ w_u) over the 8 candidate responses.
    """
    cache_path = Path(cache_dir)
    required = [
        "train_embeddings.pt", "test_embeddings.pt",
        "reward_tensor.pt", "reward_tensor_test.pt", "V_ref.pt",
    ]
    missing = [f for f in required if not (cache_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"personalllm cache at {cache_path} missing: {missing}. "
            f"See README.md for cache provisioning instructions."
        )

    train_embs = torch.load(cache_path / "train_embeddings.pt", weights_only=True)
    test_embs = torch.load(cache_path / "test_embeddings.pt", weights_only=True)
    reward_tensor = torch.load(cache_path / "reward_tensor.pt", weights_only=True).float()
    reward_tensor_test = torch.load(cache_path / "reward_tensor_test.pt",
                                    weights_only=True).float()
    V_ref = torch.load(cache_path / "V_ref.pt", weights_only=True)

    num_train = train_embs.shape[0]
    num_test = test_embs.shape[0]
    emb_dim = train_embs.shape[2]

    n_total = n_seen + n_unseen
    unseen_desc = ("all" if unseen_labeled_per_user is None
                   else str(unseen_labeled_per_user))
    print(f"personalllm: {num_train} train, {num_test} test prompts, "
          f"emb_dim={emb_dim}, {n_seen} seen + {n_unseen} unseen users, "
          f"alpha={alpha}, seen_labeled={seen_labeled_per_user}, "
          f"unseen_labeled={unseen_desc}")

    if not 0.0 <= label_flip_ratio <= 0.5:
        raise ValueError(
            f"label_flip_ratio must be in [0, 0.5]; got {label_flip_ratio}."
        )

    rng_np = np.random.RandomState(seed)
    W = torch.from_numpy(rng_np.dirichlet(alpha * np.ones(10), n_total)).float()
    rng_flip = np.random.RandomState(seed + 6363)

    def _flip_pairs(c, r):
        if label_flip_ratio <= 0 or c is None or c.shape[0] == 0:
            return c, r
        flips = rng_flip.rand(c.shape[0]) < label_flip_ratio
        idx = np.where(flips)[0]
        if len(idx) == 0:
            return c, r
        idx_t = torch.from_numpy(idx).long()
        c_new, r_new = c.clone(), r.clone()
        c_new[idx_t], r_new[idx_t] = r[idx_t], c[idx_t]
        return c_new, r_new

    test_embs_f = test_embs.float()
    train_chosen_response = (reward_tensor @ W.T).argmax(dim=1)
    train_rejected_response = (reward_tensor @ W.T).argmin(dim=1)

    # Shared unlabeled pool — sample from the (user, prompt) pairs left
    # over after capping each seen user to seen_labeled_per_user.
    n_remain_per_user = num_train - seen_labeled_per_user
    total_remain = n_seen * n_remain_per_user
    n_sample = total_remain if max_unlabeled <= 0 else min(max_unlabeled, total_remain)
    rng_pool = np.random.RandomState(seed + 9999)
    pool_flat = torch.from_numpy(
        rng_pool.choice(total_remain, n_sample, replace=False)
    )
    pool_uid = (pool_flat // n_remain_per_user).tolist()
    pool_pos = (pool_flat % n_remain_per_user).tolist()
    pool_by_user: dict[int, list[int]] = {}
    for uid_val, pos_val in zip(pool_uid, pool_pos):
        pool_by_user.setdefault(uid_val, []).append(pos_val)

    unlabeled_c_list, unlabeled_r_list = [], []

    def _split_labeled(labeled_idx, n_cap, uid_src):
        n_val = max(1, int(n_cap * val_fraction)) if val_fraction > 0 else 0
        n_train = n_cap - n_val
        lab_perm = torch.randperm(
            n_cap,
            generator=torch.Generator().manual_seed(seed + uid_src + 7777),
        )
        return labeled_idx[lab_perm[:n_train]], labeled_idx[lab_perm[n_train:]]

    user_data: dict[int, UserData] = {}
    for uid in range(n_seen):
        test_scores_u = reward_tensor_test @ W[uid]
        test_chosen_idx = test_scores_u.argmax(dim=1)
        test_rejected_idx = test_scores_u.argmin(dim=1)

        perm = torch.randperm(num_train,
                              generator=torch.Generator().manual_seed(seed + uid))
        labeled_idx = perm[:seen_labeled_per_user]
        train_sub, val_sub = _split_labeled(labeled_idx, seen_labeled_per_user, uid)

        chosen_resp_u = train_chosen_response[:, uid]
        rejected_resp_u = train_rejected_response[:, uid]

        labeled_c = train_embs[train_sub, chosen_resp_u[train_sub]].float()
        labeled_r = train_embs[train_sub, rejected_resp_u[train_sub]].float()
        val_c = (train_embs[val_sub, chosen_resp_u[val_sub]].float()
                 if val_sub.numel() > 0 else None)
        val_r = (train_embs[val_sub, rejected_resp_u[val_sub]].float()
                 if val_sub.numel() > 0 else None)

        labeled_c, labeled_r = _flip_pairs(labeled_c, labeled_r)
        val_c, val_r = _flip_pairs(val_c, val_r)

        user_data[uid] = UserData(
            labeled_chosen_embs=labeled_c,
            labeled_rejected_embs=labeled_r,
            unlabeled_embs=torch.empty(0, emb_dim),
            test_chosen_embs=LazyIndexedEmbedding(test_embs_f, test_chosen_idx),
            test_rejected_embs=LazyIndexedEmbedding(test_embs_f, test_rejected_idx),
            val_chosen_embs=val_c,
            val_rejected_embs=val_r,
        )

        if uid in pool_by_user:
            remain_idx = perm[seen_labeled_per_user:]
            for pos in pool_by_user[uid]:
                pidx = remain_idx[pos]
                unlabeled_c_list.append(train_embs[pidx, chosen_resp_u[pidx]].clone())
                unlabeled_r_list.append(train_embs[pidx, rejected_resp_u[pidx]].clone())

    # Unseen users (re-indexed 0..n_unseen-1).
    unseen_user_data: dict[int, UserData] = {}
    for i in range(n_unseen):
        uid_src = n_seen + i
        test_scores_u = reward_tensor_test @ W[uid_src]
        test_chosen_idx = test_scores_u.argmax(dim=1)
        test_rejected_idx = test_scores_u.argmin(dim=1)

        chosen_resp_u = train_chosen_response[:, uid_src]
        rejected_resp_u = train_rejected_response[:, uid_src]

        if unseen_labeled_per_user is None or unseen_labeled_per_user <= 0:
            train_sub_u = torch.arange(num_train)
            val_sub_u = torch.empty(0, dtype=torch.long)
        else:
            n_cap_u = min(unseen_labeled_per_user, num_train)
            perm_u = torch.randperm(
                num_train,
                generator=torch.Generator().manual_seed(seed + uid_src),
            )
            labeled_idx_u = perm_u[:n_cap_u]
            train_sub_u, val_sub_u = _split_labeled(labeled_idx_u, n_cap_u, uid_src)

        all_c = train_embs[train_sub_u, chosen_resp_u[train_sub_u]].float()
        all_r = train_embs[train_sub_u, rejected_resp_u[train_sub_u]].float()
        val_c_u = (train_embs[val_sub_u, chosen_resp_u[val_sub_u]].float()
                   if val_sub_u.numel() > 0 else None)
        val_r_u = (train_embs[val_sub_u, rejected_resp_u[val_sub_u]].float()
                   if val_sub_u.numel() > 0 else None)

        all_c, all_r = _flip_pairs(all_c, all_r)
        val_c_u, val_r_u = _flip_pairs(val_c_u, val_r_u)

        unseen_user_data[i] = UserData(
            labeled_chosen_embs=all_c,
            labeled_rejected_embs=all_r,
            unlabeled_embs=torch.empty(0, emb_dim),
            test_chosen_embs=LazyIndexedEmbedding(test_embs_f, test_chosen_idx),
            test_rejected_embs=LazyIndexedEmbedding(test_embs_f, test_rejected_idx),
            val_chosen_embs=val_c_u,
            val_rejected_embs=val_r_u,
        )

    if unlabeled_c_list:
        sampled_c = torch.stack(unlabeled_c_list)
        sampled_r = torch.stack(unlabeled_r_list)
        pick_chosen = torch.randint(
            0, 2, (sampled_c.shape[0],),
            generator=torch.Generator().manual_seed(seed + 8888),
        ).bool()
        shared_unlabeled = torch.where(
            pick_chosen.unsqueeze(1), sampled_c, sampled_r,
        ).float()
    else:
        shared_unlabeled = torch.empty(0, emb_dim)

    return EmbeddingDataset(
        user_data=user_data,
        user_ids=list(range(n_seen)),
        unseen_user_data=unseen_user_data,
        unseen_user_ids=list(range(n_unseen)),
        emb_dim=emb_dim,
        V_ref=V_ref,
        shared_unlabeled_embs=shared_unlabeled,
        metadata={
            "dataset": "personalllm",
            "n_seen": n_seen, "n_unseen": n_unseen,
            "alpha": alpha,
            "seen_labeled_per_user": seen_labeled_per_user,
            "unseen_labeled_per_user": unseen_labeled_per_user,
            "max_unlabeled": max_unlabeled,
            "seed": seed,
        },
    )


# ============================================================================
# HelpSteer2 (Dirichlet synthetic users over 5 attribute scores)
# ============================================================================


HELPSTEER2_ATTRIBUTES = (
    "helpfulness", "correctness", "coherence", "complexity", "verbosity",
)


def _load_helpsteer2_cache(cache_path: Path):
    required = [
        "train_embeddings.pt", "train_scores.pt", "train_prompt_groups.json",
        "validation_embeddings.pt", "validation_scores.pt",
        "validation_prompt_groups.json", "V_ref.pt",
    ]
    missing = [f for f in required if not (cache_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"helpsteer2 cache at {cache_path} missing: {missing}. "
            f"See README.md for cache provisioning instructions."
        )
    train_embs = torch.load(cache_path / "train_embeddings.pt", weights_only=True).float()
    val_embs = torch.load(cache_path / "validation_embeddings.pt", weights_only=True).float()
    train_scores = torch.load(cache_path / "train_scores.pt", weights_only=True).float()
    val_scores = torch.load(cache_path / "validation_scores.pt", weights_only=True).float()
    V_ref = torch.load(cache_path / "V_ref.pt", weights_only=True)
    with open(cache_path / "train_prompt_groups.json") as f:
        train_groups = json.load(f)
    with open(cache_path / "validation_prompt_groups.json") as f:
        val_groups = json.load(f)
    return train_embs, val_embs, train_scores, val_scores, V_ref, train_groups, val_groups


def _argmax_argmin_tiebreak(scores: torch.Tensor, rng) -> tuple[int, int]:
    """argmax / argmin with random tie-breaking via ``rng.choice``."""
    s = scores.tolist()
    smax = max(s)
    smin = min(s)
    max_ties = [i for i, v in enumerate(s) if v == smax]
    min_ties = [i for i, v in enumerate(s) if v == smin]
    chosen = rng.choice(max_ties)
    if smax == smin:
        candidates = [i for i in min_ties if i != chosen] or min_ties
        rejected = rng.choice(candidates)
    else:
        rejected = rng.choice(min_ties)
    return chosen, rejected


def load_helpsteer2(
    cache_dir: str = "./cache/helpsteer2",
    n_seen: int = 50,
    n_unseen: int = 50,
    alpha: float = 0.01,
    seen_labeled_per_user: int = 50,
    unseen_labeled_per_user: Optional[int] = 10,
    max_unlabeled: int = 50000,
    val_fraction: float = 0.2,
    score_diff_threshold: float = 0.0,
    label_flip_ratio: float = 0.0,
    seed: int = 42,
) -> EmbeddingDataset:
    """Load HelpSteer2 with PersonalLLM-style synthetic Dirichlet users.

    Each of ``n_seen + n_unseen`` users is a 5-vector w sampled from
    Dir(alpha * 1_5) over the HelpSteer2 attributes (helpfulness,
    correctness, coherence, complexity, verbosity). User-u score for a
    (prompt, response) row is ``w · raw_scores``; per train prompt we
    take argmax (chosen) / argmin (rejected) responses, then keep prompts
    with score-diff >= ``score_diff_threshold`` as confident pairs.

    Test set is the HelpSteer2 validation split (no score-diff filter,
    random tie-break on equal weighted scores).
    """
    cache_path = Path(cache_dir)
    (train_embs, val_embs, train_scores, val_scores,
     V_ref, train_groups, val_groups) = _load_helpsteer2_cache(cache_path)

    emb_dim = train_embs.shape[1]
    K = train_scores.shape[1]
    if K != len(HELPSTEER2_ATTRIBUTES):
        raise ValueError(
            f"helpsteer2 cache has {K} attribute columns; expected "
            f"{len(HELPSTEER2_ATTRIBUTES)} ({HELPSTEER2_ATTRIBUTES})."
        )
    if not 0.0 <= label_flip_ratio <= 0.5:
        raise ValueError(f"label_flip_ratio must be in [0, 0.5]; got {label_flip_ratio}.")

    n_total = n_seen + n_unseen
    rng_np = np.random.RandomState(seed)
    W = torch.from_numpy(rng_np.dirichlet(alpha * np.ones(K), n_total)).float()

    print(f"helpsteer2: {train_embs.shape[0]} train rows, "
          f"{val_embs.shape[0]} test rows, emb_dim={emb_dim}, "
          f"{n_seen} seen + {n_unseen} unseen users, alpha={alpha}, "
          f"seen_labeled={seen_labeled_per_user}, "
          f"unseen_labeled={unseen_labeled_per_user}, "
          f"score_diff_threshold={score_diff_threshold}")

    def _build_user_pairs(prompt_groups, scores_tensor, weight_vec, rng):
        pairs: list[tuple[int, int]] = []
        confident: list[bool] = []
        for row_idxs in prompt_groups:
            if len(row_idxs) < 2:
                continue
            row_idxs_t = torch.tensor(row_idxs, dtype=torch.long)
            scores_p = scores_tensor[row_idxs_t] @ weight_vec
            c_local, r_local = _argmax_argmin_tiebreak(scores_p, rng)
            c_global = row_idxs[c_local]
            r_global = row_idxs[r_local]
            diff = float(scores_p[c_local] - scores_p[r_local])
            pairs.append((c_global, r_global))
            confident.append(diff >= score_diff_threshold)
        return pairs, confident

    per_user_train: list[tuple[list[tuple[int, int]], list[bool]]] = []
    per_user_test: list[list[tuple[int, int]]] = []
    for u in range(n_total):
        rng_u = random.Random(seed * 1_000_003 + u)
        train_pairs, train_conf = _build_user_pairs(
            train_groups, train_scores, W[u], rng_u,
        )
        per_user_train.append((train_pairs, train_conf))
        rng_u_val = random.Random(seed * 1_000_003 + u + 7)
        val_pairs, _ = _build_user_pairs(
            val_groups, val_scores, W[u], rng_u_val,
        )
        per_user_test.append(val_pairs)

    rng_split = np.random.RandomState(seed + 4242)
    rng_flip = np.random.RandomState(seed + 6363)
    pending_unlabeled: list[tuple[int, int, int]] = []   # (user, c_row, r_row)

    def _carve_val(n_lab: int) -> int:
        return max(1, int(n_lab * val_fraction)) if val_fraction > 0 else 0

    user_data: dict[int, UserData] = {}
    for u in range(n_seen):
        train_pairs, train_conf = per_user_train[u]
        confident_pairs = [p for p, ok in zip(train_pairs, train_conf) if ok]
        n_confident = len(confident_pairs)

        if n_confident == 0:
            labeled_pairs: list[tuple[int, int]] = []
        else:
            n_lab_target = (seen_labeled_per_user if seen_labeled_per_user > 0
                            else n_confident)
            n_lab = min(n_lab_target, n_confident)
            perm = rng_split.permutation(n_confident)
            labeled_pairs = [confident_pairs[i] for i in perm[:n_lab]]
            for i in perm[n_lab:]:
                c, r = confident_pairs[i]
                pending_unlabeled.append((u, c, r))

        if label_flip_ratio > 0 and labeled_pairs:
            flips = rng_flip.rand(len(labeled_pairs)) < label_flip_ratio
            labeled_pairs = [
                (r, c) if f else (c, r)
                for (c, r), f in zip(labeled_pairs, flips)
            ]

        n_lab = len(labeled_pairs)
        n_val = _carve_val(n_lab) if n_lab > 0 else 0
        n_train = n_lab - n_val
        train_c = (torch.stack([train_embs[c] for c, _ in labeled_pairs[:n_train]])
                   if n_train > 0 else torch.empty(0, emb_dim))
        train_r = (torch.stack([train_embs[r] for _, r in labeled_pairs[:n_train]])
                   if n_train > 0 else torch.empty(0, emb_dim))
        val_c = (torch.stack([train_embs[c] for c, _ in labeled_pairs[n_train:]])
                 if n_val > 0 else None)
        val_r = (torch.stack([train_embs[r] for _, r in labeled_pairs[n_train:]])
                 if n_val > 0 else None)

        test_pairs = per_user_test[u]
        if test_pairs:
            test_c = torch.stack([val_embs[c] for c, _ in test_pairs])
            test_r = torch.stack([val_embs[r] for _, r in test_pairs])
        else:
            test_c = torch.empty(0, emb_dim)
            test_r = torch.empty(0, emb_dim)

        user_data[u] = UserData(
            labeled_chosen_embs=train_c,
            labeled_rejected_embs=train_r,
            unlabeled_embs=torch.empty(0, emb_dim),
            test_chosen_embs=test_c,
            test_rejected_embs=test_r,
            val_chosen_embs=val_c,
            val_rejected_embs=val_r,
        )

    # Shared unlabeled: sample from pending, keep one side per entry via 50/50 coin.
    n_pending = len(pending_unlabeled)
    n_sample = (n_pending if max_unlabeled <= 0
                else min(max_unlabeled, n_pending))
    if n_sample > 0:
        rng_pool = np.random.RandomState(seed + 9999)
        pick_idx = rng_pool.choice(n_pending, n_sample, replace=False)
        rng_coin = torch.Generator().manual_seed(seed + 8888)
        keep_chosen = torch.randint(0, 2, (n_sample,), generator=rng_coin).bool()
        rows = []
        for k, idx in enumerate(pick_idx):
            _u_src, c, r = pending_unlabeled[idx]
            rows.append(train_embs[c if bool(keep_chosen[k]) else r])
        shared_unlabeled = torch.stack(rows).float()
    else:
        shared_unlabeled = torch.empty(0, emb_dim)

    # Unseen users.
    n_cap_unseen = (unseen_labeled_per_user if unseen_labeled_per_user is not None
                    else seen_labeled_per_user)
    rng_split_u = np.random.RandomState(seed + 5151)
    unseen_user_data: dict[int, UserData] = {}
    for i in range(n_unseen):
        u_src = n_seen + i
        train_pairs, train_conf = per_user_train[u_src]
        confident_pairs = [p for p, ok in zip(train_pairs, train_conf) if ok]
        n_confident = len(confident_pairs)

        if n_cap_unseen is None or n_cap_unseen <= 0:
            labeled_pairs = list(confident_pairs)
        else:
            n_lab = min(n_cap_unseen, n_confident)
            perm = (rng_split_u.permutation(n_confident) if n_confident > 0
                    else np.array([], dtype=int))
            labeled_pairs = [confident_pairs[k] for k in perm[:n_lab]]

        if label_flip_ratio > 0 and labeled_pairs:
            flips = rng_flip.rand(len(labeled_pairs)) < label_flip_ratio
            labeled_pairs = [
                (r, c) if f else (c, r)
                for (c, r), f in zip(labeled_pairs, flips)
            ]

        n_lab = len(labeled_pairs)
        n_val = _carve_val(n_lab) if n_lab > 0 else 0
        n_train = n_lab - n_val
        train_c = (torch.stack([train_embs[c] for c, _ in labeled_pairs[:n_train]])
                   if n_train > 0 else torch.empty(0, emb_dim))
        train_r = (torch.stack([train_embs[r] for _, r in labeled_pairs[:n_train]])
                   if n_train > 0 else torch.empty(0, emb_dim))
        val_c = (torch.stack([train_embs[c] for c, _ in labeled_pairs[n_train:]])
                 if n_val > 0 else None)
        val_r = (torch.stack([train_embs[r] for _, r in labeled_pairs[n_train:]])
                 if n_val > 0 else None)

        test_pairs = per_user_test[u_src]
        if test_pairs:
            test_c = torch.stack([val_embs[c] for c, _ in test_pairs])
            test_r = torch.stack([val_embs[r] for _, r in test_pairs])
        else:
            test_c = torch.empty(0, emb_dim)
            test_r = torch.empty(0, emb_dim)

        unseen_user_data[i] = UserData(
            labeled_chosen_embs=train_c,
            labeled_rejected_embs=train_r,
            unlabeled_embs=torch.empty(0, emb_dim),
            test_chosen_embs=test_c,
            test_rejected_embs=test_r,
            val_chosen_embs=val_c,
            val_rejected_embs=val_r,
        )

    return EmbeddingDataset(
        user_data=user_data,
        user_ids=list(range(n_seen)),
        unseen_user_data=unseen_user_data,
        unseen_user_ids=list(range(n_unseen)),
        emb_dim=emb_dim,
        V_ref=V_ref,
        shared_unlabeled_embs=shared_unlabeled,
        metadata={
            "dataset": "helpsteer2",
            "n_seen": n_seen, "n_unseen": n_unseen,
            "alpha": alpha,
            "seen_labeled_per_user": seen_labeled_per_user,
            "unseen_labeled_per_user": unseen_labeled_per_user,
            "max_unlabeled": max_unlabeled,
            "score_diff_threshold": score_diff_threshold,
            "seed": seed,
        },
    )
