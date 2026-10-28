# PerSPL

**Per**sonalized **S**emi-Supervised **P**reference **L**earning — a
two-phase training method for low-rank personalized reward models with
iter-based self-training on unlabeled data.

Phase 1 trains a shared low-rank basis V and per-user weights W jointly
on seen users (Bradley-Terry loss on labeled pairs + reward regulariser
+ self-training BT loss on iteratively pseudo-labeled unlabeled pairs).
Phase 2 freezes V from phase 1 and trains W for held-out unseen users
under the same protocol.

A single JSON config drives both phases. One command reproduces a run.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.1+, a CUDA GPU (CPU works but slow).

## Quickstart

```bash
cd PerSPL
python train.py --config configs/reddit_tldr_s100.json --seed 42
```

The seed comes from the CLI so one config drives all five seeds:

```bash
for s in 42 43 44 45 46; do
  python train.py --config configs/reddit_tldr_s100.json --seed $s
done
```

Five canonical configs ship in [configs/](configs/), plus one variant:

| Config | Description |
|---|---|
| `reddit_tldr_s100.json` | 20 seen + 20 unseen, 100 labeled/user, rank=2 |
| `reddit_tldr_s100_phase2_k2.json` | same as `reddit_tldr_s100.json`, but phase 2 self-trains on unlabeled data (n_iter=2, 20 epochs) instead of labeled-only (n_iter=1) |
| `reddit_tldr_s150.json` | 20 seen + 20 unseen, 150 labeled/user, rank=2 |
| `helpsteer2_a0.01.json` | 50 seen + 50 unseen, Dirichlet α=0.01, rank=5 |
| `helpsteer2_a0.1.json`  | 50 seen + 50 unseen, Dirichlet α=0.1,  rank=5 |
| `personalllm.json` | 100 seen + 100 unseen, rank=2 (default hp; tune for your setup) |

Output: `outputs/<dataset>/<TS>_seed<S>/results.json`. The file has
explicit accuracy fields:

```json
{
  "dataset": "reddit_tldr",
  "seed": 42,
  "n_seen": 20,
  "n_unseen": 20,
  "accuracy": {
    "seen_best_epoch_acc":   0.7612,
    "seen_last_epoch_acc":   0.7612,
    "unseen_best_epoch_acc": 0.6534,
    "unseen_last_epoch_acc": 0.6481,
    "seen_per_user_accs":    [...],
    "unseen_per_user_accs":  [...]
  },
  "phase1_summary": {...},
  "phase2_summary": {...}
}
```

`seen_*` are computed with the loaded phase-1 V+W (frozen in phase 2,
so `best == last`). `unseen_*` come from phase-2 training.

## Datasets

Three datasets are supported. Each needs a pre-extracted embedding
cache at `./cache/<dataset>/`. Extraction is not included in this
release — see the parent project for extraction scripts, or place
your own caches with the expected files.

### reddit_tldr

```
cache/reddit_tldr/
├── reddit.pt     # (N, emb_dim) embedding tensor
└── reddit.json   # per-user metadata: {user_id: [{chosen_idx, rejected_idx, split}, ...]}
```

### personalllm

```
cache/personalllm/
├── train_embeddings.pt    # (num_train, 8, emb_dim)
├── test_embeddings.pt     # (num_test, 8, emb_dim)
├── reward_tensor.pt       # (num_train, 8, 10)
├── reward_tensor_test.pt  # (num_test, 8, 10)
└── V_ref.pt               # (emb_dim, k) reference projection
```

10 reward models score 8 candidate responses per prompt. Synthetic
users are 10-vectors drawn from `Dir(alpha · 1)` over those reward
models; each user's chosen / rejected response per prompt is the
argmax / argmin of the user's weighted reward.

### helpsteer2

```
cache/helpsteer2/
├── train_embeddings.pt          # (N_train, emb_dim)
├── train_scores.pt              # (N_train, 5)  — 5 attribute scores
├── train_prompt_groups.json     # [[row_idx, ...], ...] groups rows by prompt
├── validation_embeddings.pt
├── validation_scores.pt
├── validation_prompt_groups.json
└── V_ref.pt
```

Synthetic users are 5-vectors drawn from `Dir(alpha · 1)` over the
attributes `(helpfulness, correctness, coherence, complexity,
verbosity)`. Chosen / rejected per prompt = argmax / argmin under
each user's weighted scores.

## Config schema

A single JSON file with shared parameters at the top level and two
phase blocks:

```json
{
  "dataset": "reddit_tldr",

  "n_seen": 20, "n_unseen": 20,
  "seen_labeled_per_user": 100, "unseen_labeled_per_user": 50,
  "max_unlabeled": -1,
  "rank": 2, "temperature": 100.0,

  "phase1": {
    "lr": 0.03, "num_epochs": 500, "n_iter": 2,
    "mini_batch_size": 2, "tasks_per_batch": 2,
    "unlabeled_mini_batch_size": 16, "n_unlabeled_per_user": 1000,
    "labeled_loss_weight": 0.6, "iso_filter_threshold": 0.8,
    "reward_reg_lambda": 0.01, "threshold_method": "isotonic",
    "lr_schedule": "onecycle",
    "onecycle_pct_start": 0.3, "onecycle_div_factor": 25.0,
    "onecycle_final_div_factor": 10000.0, "onecycle_anneal_strategy": "cos",
    "onecycle_no_cycle_momentum": true, "onecycle_three_phase": false
  },
  "phase2": {
    "lr": 0.01, "num_epochs": 80, "n_iter": 2,
    "mini_batch_size": 4, "tasks_per_batch": 4,
    "unlabeled_mini_batch_size": 16, "n_unlabeled_per_user": 1000,
    "labeled_loss_weight": 0.8, "iso_filter_threshold": 0.9,
    "reward_reg_lambda": 0.01, "threshold_method": "isotonic",
    "lr_schedule": "onecycle",
    "onecycle_pct_start": 0.3, "onecycle_div_factor": 25.0,
    "onecycle_final_div_factor": 10000.0, "onecycle_anneal_strategy": "cos",
    "onecycle_no_cycle_momentum": true, "onecycle_three_phase": false
  }
}
```

**Top-level**:
- `dataset` — `reddit_tldr` | `personalllm` | `helpsteer2`
- `seed` — *optional in JSON*; pass via `--seed` to sweep without editing the file
- `n_seen`, `n_unseen`, `seen_labeled_per_user`, `unseen_labeled_per_user`,
  `max_unlabeled` — per-user data sizing
- `rank`, `temperature` — model shape (V is reused across phases)

**Dataset-specific (optional, default to safe values)**:
- `alpha` — Dirichlet concentration (personalllm, helpsteer2)
- `val_fraction` — held-out val carve from labeled pool
- `label_flip_ratio` — Bernoulli label noise on labeled pairs only
- `score_diff_threshold` — confident-pair filter (helpsteer2)

**Per-phase (anything else can differ between phases)**:
- Training schedule: `lr`, `num_epochs`, `n_iter`, `mini_batch_size`,
  `unlabeled_mini_batch_size`, `n_unlabeled_per_user`, `tasks_per_batch`
- Loss: `labeled_loss_weight` (α), `iso_filter_threshold` (τ),
  `reward_reg_lambda` (λ), `threshold_method`
- LR schedule: `lr_schedule`, `lr_min`, `warmup_frac`,
  `linear_start_factor`, `onecycle_*`
- Optimizer: `weight_decay`, `weight_decay_target`
- Logging: `log_interval`

`load_config()` in [config.py](config.py) parses + validates the JSON
into a `TwoPhaseConfig`. Missing required fields fail with an explicit
error naming the field; extra fields are rejected.

## Repo layout

Flat, eight Python files + one config directory:

```
PerSPL/
├── README.md           ← this file
├── requirements.txt
├── train.py            ← entry: `python train.py --config configs/X.json`
├── model.py            ← PerSPLRewardHead (low-rank V + per-user W head)
├── trainer.py          ← PerSPLIterTrainer, PerSPLIterUnseenTrainer,
│                          run_unseen_eval (phase-2 driver)
├── config.py           ← TwoPhaseConfig + PerSPLPhaseConfig + load_config
├── inference.py        ← evaluate_per_user (test-set helper)
├── data.py             ← EmbeddingDataset, UserData, PerUserDataset,
│                          per_user_collate, build_training_tensors
├── loaders.py          ← load_reddit_tldr, load_personalllm, load_helpsteer2
├── utils.py            ← seed_everything, build_lr_scheduler,
│                          compute_threshold, per_user_accuracy
└── configs/            ← one JSON per dataset (seed is a CLI flag)
```

No subpackages. Imports are flat (`from model import PerSPLRewardHead`).
You must `cd PerSPL` before running `python train.py`.
