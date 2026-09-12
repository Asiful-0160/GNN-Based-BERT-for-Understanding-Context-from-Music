#!/usr/bin/env python3

"""
Script 32 — Formal Task-1 BERT model-selection trainer

Pipeline Plan v6 experiments:
    T1-A: Frozen BERT, classifier head only, LR=1e-3
    T1-B: Full BERT fine-tuning, LR=2e-5

Important:
- TRAIN is used for optimization.
- VAL is used for early stopping and checkpoint selection.
- TEST is never loaded for labels or inference.
- No threshold sweep is performed here.
- Winning Task-1 configuration is NOT chosen here.
- Script 33 will compare completed T1-A and T1-B runs.

Selection metric:
    mean per-label validation Average Precision
    over all 20 labels

This operationalizes the project requirement:
    "AUC-PR per tag; report mean AUC-PR over tags."

Scheduler implementation clarification:
    10% linear warmup -> linear decay
    warmup steps = ceil(0.10 * maximum optimizer steps)

Frozen artifacts are verified before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
import yaml

from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
)


# ============================================================
# ROOT / PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs/config.yaml"

VOCAB_PATH = (
    ROOT
    / "data/splits/label_vocab.json"
)

INITIAL_LOCK_PATH = (
    ROOT
    / "data/splits/frozen_hashes.json"
)

PRETRAINING_LOCK_PATH = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

TARGETS_PATH = (
    ROOT
    / "data/processed/labels/musiccaps_targets.parquet"
)

BERT_MANIFEST_PATH = (
    ROOT
    / "data/processed/bert_tokens/bert_token_manifest.parquet"
)

BERT_TOKENS_PATH = (
    ROOT
    / "data/processed/bert_tokens/bert_tokens.npz"
)

RESULTS_CSV_DIR = ROOT / "results/csv"
RUNS_DIR = ROOT / "results/runs/task1"
CANDIDATE_MODEL_DIR = ROOT / "models/task1_candidates"

EPOCH_METRICS_PATH = (
    RESULTS_CSV_DIR
    / "epoch_metrics.csv"
)

TRIALS_PATH = (
    RESULTS_CSV_DIR
    / "hyperparameter_trials.csv"
)


# ============================================================
# FROZEN EXPERIMENT VALUES
# ============================================================

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_DATASET_TRACKS = 5140
EXPECTED_TRAIN_TRACKS = 4112
EXPECTED_VAL_TRACKS = 514

NUM_LABELS = 20
MAX_LENGTH = 112

TARGET_COLUMNS = [
    f"y_{i:02d}"
    for i in range(NUM_LABELS)
]


# ============================================================
# PIPELINE PLAN v6 — TASK 1
# ============================================================

MODEL_NAME = "bert-base-uncased"

SEED = 42

BATCH_SIZE = 16
MAX_EPOCHS = 8

DROPOUT = 0.1

WEIGHT_DECAY = 0.01

WARMUP_FRACTION = 0.10

EARLY_STOP_PATIENCE = 2

POS_WEIGHT_CAP = 10.0

VARIANT_CONFIG = {
    "T1-A": {
        "freeze_bert": True,
        "learning_rate": 1.0e-3,
        "description": "Frozen BERT; classifier head only",
    },
    "T1-B": {
        "freeze_bert": False,
        "learning_rate": 2.0e-5,
        "description": "Full BERT fine-tuning",
    },
}

# Implementation choices not explicitly specified by v6.
SCHEDULER_NAME = "linear_warmup_linear_decay"
WARMUP_ROUNDING = "ceil"

# Metric implementation.
SELECTION_METRIC_NAME = "val_auc_pr_macro_mean_ap"

# Numerical implementation.
USE_AMP = True

# No min-delta was specified by v6.
# A new score must be strictly greater than the old score.
IMPROVEMENT_EPSILON = 1.0e-12


# ============================================================
# INITIAL FROZEN-HASH PATH MAPPING
#
# This mapping is based on the actual schema inspected by
# Script 31. It is no longer guessed.
# ============================================================

INITIAL_HASH_PATHS = {
    "canonical_dataset":
        ROOT / "data/interim/musiccaps_canonical.parquet",

    "train_split":
        ROOT / "data/splits/train.json",

    "val_split":
        ROOT / "data/splits/val.json",

    "test_split":
        ROOT / "data/splits/test.json",

    "aspect_synonyms":
        ROOT / "data/splits/aspect_synonyms.json",

    "label_vocab":
        ROOT / "data/splits/label_vocab.json",

    "targets_parquet":
        ROOT / "data/processed/labels/musiccaps_targets.parquet",

    "targets_csv":
        ROOT / "data/processed/labels/musiccaps_targets.csv",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        commit = result.stdout.strip()

        if commit:
            return commit

    except Exception:
        pass

    return "UNAVAILABLE"


def safe_relative(
    path: Path,
) -> str:

    try:
        return str(
            path.relative_to(ROOT)
        )

    except ValueError:
        return str(path)


# ============================================================
# CSV HELPERS
# ============================================================

def write_dataframe_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def append_rows_to_csv(
    rows: list[dict],
    path: Path,
) -> None:

    new_frame = pd.DataFrame(rows)

    if path.exists():

        existing = pd.read_csv(path)

        combined = pd.concat(
            [
                existing,
                new_frame,
            ],
            ignore_index=True,
        )

    else:
        combined = new_frame

    write_dataframe_atomic(
        combined,
        path,
    )


# ============================================================
# ACCIDENTAL DUPLICATE FORMAL RUN PROTECTION
# ============================================================

def assert_no_completed_duplicate(
    variant: str,
) -> None:

    if not TRIALS_PATH.exists():
        return

    trials = pd.read_csv(
        TRIALS_PATH
    )

    required = {
        "task",
        "variant",
        "seed",
        "status",
    }

    if not required.issubset(
        set(trials.columns)
    ):
        return

    duplicate = trials.loc[
        (trials["task"] == "task1")
        & (trials["variant"] == variant)
        & (trials["seed"] == SEED)
        & (trials["status"] == "COMPLETE")
    ]

    require(
        len(duplicate) == 0,
        (
            f"A completed formal {variant} "
            f"seed-{SEED} model-selection run "
            "already exists in "
            "hyperparameter_trials.csv. "
            "Refusing an accidental duplicate."
        ),
    )


# ============================================================
# FROZEN ARTIFACT VERIFICATION
# ============================================================

def verify_frozen_artifacts() -> dict:

    section(
        "1. FROZEN ARTIFACT VERIFICATION"
    )

    actual_lock_sha = sha256_file(
        PRETRAINING_LOCK_PATH
    )

    print(
        "Expected pretraining lock SHA256:",
        EXPECTED_PRETRAINING_LOCK_SHA256,
    )

    print(
        "Actual pretraining lock SHA256:  ",
        actual_lock_sha,
    )

    require(
        actual_lock_sha
        == EXPECTED_PRETRAINING_LOCK_SHA256,
        (
            "pretraining_frozen_hashes.json "
            "changed — ABORT."
        ),
    )

    with PRETRAINING_LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        pretraining_lock = json.load(f)

    require(
        pretraining_lock.get("lock_type")
        == "pretraining_artifact_lock",
        "Unexpected pretraining lock type.",
    )

    require(
        pretraining_lock.get("dataset_tracks")
        == EXPECTED_DATASET_TRACKS,
        "Unexpected frozen dataset size.",
    )

    critical_files = pretraining_lock.get(
        "critical_files"
    )

    require(
        isinstance(critical_files, dict),
        (
            "pretraining lock critical_files "
            "is not a dictionary."
        ),
    )

    print()
    print(
        "Verifying pretraining critical files..."
    )

    for key, record in critical_files.items():

        require(
            isinstance(record, dict),
            (
                f"Malformed critical-file "
                f"record: {key}"
            ),
        )

        relative_path = record.get(
            "path"
        )

        expected_sha = record.get(
            "sha256"
        )

        require(
            isinstance(relative_path, str),
            (
                f"Missing path for critical "
                f"file {key}"
            ),
        )

        require(
            isinstance(expected_sha, str),
            (
                f"Missing SHA256 for critical "
                f"file {key}"
            ),
        )

        path = ROOT / relative_path

        require(
            path.is_file(),
            (
                f"Critical artifact missing: "
                f"{relative_path}"
            ),
        )

        actual_sha = sha256_file(
            path
        )

        require(
            actual_sha == expected_sha,
            (
                f"Critical artifact changed: "
                f"{relative_path}"
            ),
        )

        print(
            f"PASS  {key:<24} "
            f"{relative_path}"
        )

    print()
    print(
        "Verifying original dataset/split lock..."
    )

    with INITIAL_LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        initial_lock = json.load(f)

    require(
        initial_lock.get("frozen") is True,
        "Initial dataset lock is not marked frozen.",
    )

    require(
        initial_lock.get("dataset_tracks")
        == EXPECTED_DATASET_TRACKS,
        "Initial dataset lock size mismatch.",
    )

    require(
        initial_lock.get("label_count")
        == NUM_LABELS,
        "Initial label_count mismatch.",
    )

    expected_counts = {
        "train": EXPECTED_TRAIN_TRACKS,
        "val": EXPECTED_VAL_TRACKS,
        "test": 514,
    }

    require(
        initial_lock.get("split_counts")
        == expected_counts,
        "Frozen split counts changed.",
    )

    hashes = initial_lock.get(
        "hashes"
    )

    require(
        isinstance(hashes, dict),
        "Initial lock hashes missing.",
    )

    require(
        set(hashes.keys())
        == set(INITIAL_HASH_PATHS.keys()),
        (
            "Initial frozen hash key set "
            "does not match the inspected "
            "Script-31 schema."
        ),
    )

    for key, path in INITIAL_HASH_PATHS.items():

        require(
            path.is_file(),
            (
                f"Frozen artifact missing: "
                f"{safe_relative(path)}"
            ),
        )

        actual_sha = sha256_file(
            path
        )

        expected_sha = hashes[key]

        require(
            actual_sha == expected_sha,
            (
                f"Frozen artifact changed: "
                f"{safe_relative(path)}"
            ),
        )

        print(
            f"PASS  {key:<24} "
            f"{safe_relative(path)}"
        )

    print()
    print(
        "Frozen-artifact verification: PASS"
    )

    print(
        "Note: test.json was HASHED only."
    )

    print(
        "No TEST labels/examples were loaded "
        "or used for model development."
    )

    return {
        "pretraining_lock_sha256":
            actual_lock_sha,
        "initial_lock_sha256":
            sha256_file(
                INITIAL_LOCK_PATH
            ),
    }


# ============================================================
# CONFIG / VOCAB
# ============================================================

def validate_config_and_vocab():

    section(
        "2. CONFIG + VOCABULARY"
    )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    require(
        isinstance(config, dict),
        "config.yaml did not load as a mapping.",
    )

    text_config = config.get(
        "text"
    )

    require(
        isinstance(text_config, dict),
        "Missing config text section.",
    )

    require(
        text_config.get("model_name")
        == MODEL_NAME,
        (
            "Frozen text.model_name differs "
            "from bert-base-uncased."
        ),
    )

    require(
        int(text_config.get("max_length"))
        == MAX_LENGTH,
        "Frozen BERT max_length changed.",
    )

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        vocab = json.load(f)

    labels = vocab.get(
        "labels"
    )

    label_to_index = vocab.get(
        "label_to_index"
    )

    require(
        isinstance(labels, list)
        and len(labels) == NUM_LABELS,
        "Expected exactly 20 labels.",
    )

    require(
        isinstance(label_to_index, dict),
        "Invalid label_to_index.",
    )

    label_names = []

    frozen_support = []

    for expected_index, item in enumerate(
        labels
    ):

        require(
            int(item["index"])
            == expected_index,
            "Frozen label order changed.",
        )

        label_name = item["label"]

        require(
            int(
                label_to_index[
                    label_name
                ]
            )
            == expected_index,
            (
                "label_to_index mismatch for "
                f"{label_name!r}"
            ),
        )

        label_names.append(
            label_name
        )

        frozen_support.append(
            int(
                item["train_support"]
            )
        )

    print(
        "Model name:",
        MODEL_NAME,
    )

    print(
        "Max length:",
        MAX_LENGTH,
    )

    print(
        "Labels:",
        NUM_LABELS,
    )

    print(
        "Config/vocabulary validation: PASS"
    )

    return (
        config,
        vocab,
        label_names,
        np.asarray(
            frozen_support,
            dtype=np.int64,
        ),
    )


# ============================================================
# TRAIN + VAL DATA
# ============================================================

def load_development_data(
    frozen_support: np.ndarray,
):

    section(
        "3. TRAIN + VAL DATA"
    )

    # Deliberately load ONLY TRAIN and VAL
    # target rows.
    train_manifest = pd.read_parquet(
        BERT_MANIFEST_PATH,
        filters=[
            ("split", "==", "train")
        ],
    )

    val_manifest = pd.read_parquet(
        BERT_MANIFEST_PATH,
        filters=[
            ("split", "==", "val")
        ],
    )

    train_targets = pd.read_parquet(
        TARGETS_PATH,
        filters=[
            ("split", "==", "train")
        ],
    )

    val_targets = pd.read_parquet(
        TARGETS_PATH,
        filters=[
            ("split", "==", "val")
        ],
    )

    require(
        len(train_manifest)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN manifest count mismatch.",
    )

    require(
        len(val_manifest)
        == EXPECTED_VAL_TRACKS,
        "VAL manifest count mismatch.",
    )

    require(
        len(train_targets)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN target count mismatch.",
    )

    require(
        len(val_targets)
        == EXPECTED_VAL_TRACKS,
        "VAL target count mismatch.",
    )

    train_frame = train_manifest.merge(
        train_targets,
        on="track_id",
        how="inner",
        suffixes=(
            "_manifest",
            "_target",
        ),
        validate="one_to_one",
    )

    val_frame = val_manifest.merge(
        val_targets,
        on="track_id",
        how="inner",
        suffixes=(
            "_manifest",
            "_target",
        ),
        validate="one_to_one",
    )

    require(
        len(train_frame)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN one-to-one merge failed.",
    )

    require(
        len(val_frame)
        == EXPECTED_VAL_TRACKS,
        "VAL one-to-one merge failed.",
    )

    require(
        (
            train_frame["split_manifest"]
            == train_frame["split_target"]
        ).all(),
        "TRAIN split mismatch.",
    )

    require(
        (
            val_frame["split_manifest"]
            == val_frame["split_target"]
        ).all(),
        "VAL split mismatch.",
    )

    train_ids = set(
        train_frame["track_id"]
    )

    val_ids = set(
        val_frame["track_id"]
    )

    require(
        train_ids.isdisjoint(
            val_ids
        ),
        "TRAIN/VAL track overlap detected.",
    )

    train_frame = (
        train_frame
        .sort_values("cache_index")
        .reset_index(drop=True)
    )

    val_frame = (
        val_frame
        .sort_values("cache_index")
        .reset_index(drop=True)
    )

    train_y = train_frame[
        TARGET_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    val_y = val_frame[
        TARGET_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    require(
        set(
            np.unique(
                train_y
            ).tolist()
        ).issubset({0.0, 1.0}),
        "TRAIN targets are not binary.",
    )

    require(
        set(
            np.unique(
                val_y
            ).tolist()
        ).issubset({0.0, 1.0}),
        "VAL targets are not binary.",
    )

    train_support = (
        train_y.sum(axis=0)
        .astype(np.int64)
    )

    require(
        np.array_equal(
            train_support,
            frozen_support,
        ),
        (
            "TRAIN support no longer matches "
            "frozen label vocabulary."
        ),
    )

    val_support = (
        val_y.sum(axis=0)
        .astype(np.int64)
    )

    require(
        np.all(
            val_support > 0
        ),
        (
            "A validation label has "
            "zero positives."
        ),
    )

    require(
        np.all(
            val_support
            < EXPECTED_VAL_TRACKS
        ),
        (
            "A validation label has "
            "zero negatives."
        ),
    )

    negative_count = (
        EXPECTED_TRAIN_TRACKS
        - train_support
    )

    raw_pos_weight = (
        negative_count
        / train_support
    )

    pos_weight = np.minimum(
        raw_pos_weight,
        POS_WEIGHT_CAP,
    ).astype(
        np.float32
    )

    require(
        np.isfinite(
            pos_weight
        ).all(),
        "Non-finite pos_weight.",
    )

    print(
        "TRAIN:",
        len(train_frame),
    )

    print(
        "VAL:  ",
        len(val_frame),
    )

    print(
        "TEST target rows loaded: 0"
    )

    print(
        "TRAIN support == frozen vocab: PASS"
    )

    print(
        "TRAIN-only pos_weight: PASS"
    )

    return (
        train_frame,
        val_frame,
        pos_weight,
    )


# ============================================================
# TOKEN CACHE
# ============================================================

def load_token_cache(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
):

    section(
        "4. TOKEN CACHE"
    )

    with np.load(
        BERT_TOKENS_PATH,
        allow_pickle=False,
    ) as cache:

        require(
            set(cache.files)
            == {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            },
            "Unexpected BERT token-cache keys.",
        )

        input_ids = cache[
            "input_ids"
        ]

        attention_mask = cache[
            "attention_mask"
        ]

        token_type_ids = cache[
            "token_type_ids"
        ]

    expected_shape = (
        EXPECTED_DATASET_TRACKS,
        MAX_LENGTH,
    )

    for name, array in (
        ("input_ids", input_ids),
        (
            "attention_mask",
            attention_mask,
        ),
        (
            "token_type_ids",
            token_type_ids,
        ),
    ):

        require(
            array.shape
            == expected_shape,
            (
                f"{name} shape mismatch: "
                f"{array.shape}"
            ),
        )

    for split_name, frame in (
        ("TRAIN", train_frame),
        ("VAL", val_frame),
    ):

        indices = frame[
            "cache_index"
        ].to_numpy(
            dtype=np.int64
        )

        require(
            len(indices)
            == len(np.unique(indices)),
            (
                f"Duplicate {split_name} "
                "cache index."
            ),
        )

        actual_length = (
            attention_mask[
                indices
            ].sum(axis=1)
        )

        expected_length = frame[
            "cached_token_length"
        ].to_numpy()

        require(
            np.array_equal(
                actual_length,
                expected_length,
            ),
            (
                f"{split_name} token-cache "
                "row alignment failed."
            ),
        )

    print(
        "Token cache shape:",
        expected_shape,
    )

    print(
        "TRAIN cache-index mapping: PASS"
    )

    print(
        "VAL cache-index mapping:   PASS"
    )

    return (
        input_ids,
        attention_mask,
        token_type_ids,
    )


# ============================================================
# DATASET
# ============================================================

class BertCachedDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray,
    ) -> None:

        self.cache_indices = frame[
            "cache_index"
        ].to_numpy(
            dtype=np.int64
        )

        self.targets = frame[
            TARGET_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        self.input_ids = input_ids
        self.attention_mask = (
            attention_mask
        )
        self.token_type_ids = (
            token_type_ids
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.cache_indices
        )

    def __getitem__(
        self,
        index: int,
    ):

        cache_index = int(
            self.cache_indices[
                index
            ]
        )

        return (
            torch.as_tensor(
                self.input_ids[
                    cache_index
                ],
                dtype=torch.long,
            ),
            torch.as_tensor(
                self.attention_mask[
                    cache_index
                ],
                dtype=torch.long,
            ),
            torch.as_tensor(
                self.token_type_ids[
                    cache_index
                ],
                dtype=torch.long,
            ),
            torch.tensor(
                self.targets[
                    index
                ],
                dtype=torch.float32,
            ),
        )


# ============================================================
# MODEL
# ============================================================

class BertTask1Model(nn.Module):

    def __init__(
        self,
        freeze_bert: bool,
    ) -> None:

        super().__init__()

        self.freeze_bert = (
            freeze_bert
        )

        self.bert = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        hidden_size = int(
            self.bert.config.hidden_size
        )

        require(
            hidden_size == 768,
            (
                "Expected BERT hidden size "
                f"768, found {hidden_size}."
            ),
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.classifier = nn.Linear(
            hidden_size,
            NUM_LABELS,
        )

        if freeze_bert:

            for parameter in (
                self.bert.parameters()
            ):
                parameter.requires_grad = (
                    False
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        cls_embedding = (
            outputs
            .last_hidden_state[
                :,
                0,
                :,
            ]
        )

        cls_embedding = (
            self.dropout(
                cls_embedding
            )
        )

        return self.classifier(
            cls_embedding
        )


# ============================================================
# MODEL HASH
# ============================================================

def hash_bert_state(
    bert: nn.Module,
) -> str:

    digest = hashlib.sha256()

    state = bert.state_dict()

    for name in sorted(
        state.keys()
    ):

        tensor = (
            state[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tuple(
                    tensor.shape
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            tensor.numpy()
            .tobytes()
        )

    return digest.hexdigest()


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(
    model: nn.Module,
    learning_rate: float,
):

    decay_parameters = []

    no_decay_parameters = []

    no_decay_terms = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if any(
            term in name
            for term in no_decay_terms
        ):

            no_decay_parameters.append(
                parameter
            )

        else:

            decay_parameters.append(
                parameter
            )

    require(
        (
            len(decay_parameters)
            + len(no_decay_parameters)
        ) > 0,
        "No trainable parameters.",
    )

    groups = [
        {
            "params":
                decay_parameters,
            "weight_decay":
                WEIGHT_DECAY,
        },
        {
            "params":
                no_decay_parameters,
            "weight_decay":
                0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
    )

    return optimizer


# ============================================================
# METRICS
# ============================================================

def compute_auc_pr_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
):

    require(
        y_true.shape
        == y_probability.shape,
        (
            "Metric target/prediction "
            "shape mismatch."
        ),
    )

    require(
        y_true.shape[1]
        == NUM_LABELS,
        "Metric output dimension != 20.",
    )

    per_label_ap = (
        average_precision_score(
            y_true,
            y_probability,
            average=None,
        )
    )

    require(
        len(per_label_ap)
        == NUM_LABELS,
        "Expected 20 per-label AP values.",
    )

    require(
        np.isfinite(
            per_label_ap
        ).all(),
        "Non-finite per-label AP.",
    )

    macro_mean_ap = float(
        np.mean(
            per_label_ap
        )
    )

    micro_ap = float(
        average_precision_score(
            y_true,
            y_probability,
            average="micro",
        )
    )

    require(
        math.isfinite(
            macro_mean_ap
        ),
        "Non-finite macro AP.",
    )

    require(
        math.isfinite(
            micro_ap
        ),
        "Non-finite micro AP.",
    )

    return (
        macro_mean_ap,
        micro_ap,
        per_label_ap.astype(
            np.float64
        ),
    )


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(
    model: BertTask1Model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
):

    model.train()

    # T1-A means the pretrained BERT representation
    # is genuinely frozen. Keep encoder dropout disabled.
    # The explicit Task-1 head Dropout(0.1) remains active.
    if model.freeze_bert:
        model.bert.eval()

    total_loss = 0.0
    examples = 0

    for batch in loader:

        (
            input_ids,
            attention_mask,
            token_type_ids,
            targets,
        ) = batch

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        attention_mask = (
            attention_mask.to(
                device,
                non_blocking=True,
            )
        )

        token_type_ids = (
            token_type_ids.to(
                device,
                non_blocking=True,
            )
        )

        targets = targets.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            require(
                logits.shape
                == (
                    targets.shape[0],
                    NUM_LABELS,
                ),
                (
                    "Unexpected training "
                    "logit shape."
                ),
            )

            require(
                bool(
                    torch.isfinite(
                        logits
                    ).all()
                ),
                (
                    "Non-finite training "
                    "logits."
                ),
            )

            loss = criterion(
                logits,
                targets,
            )

        require(
            bool(
                torch.isfinite(
                    loss
                )
            ),
            "Non-finite training loss.",
        )

        scaler.scale(
            loss
        ).backward()

        scale_before = float(
            scaler.get_scale()
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        scale_after = float(
            scaler.get_scale()
        )

        require(
            not (
                scale_after
                < scale_before
            ),
            (
                "AMP detected an overflow "
                "and skipped an optimizer "
                "step. Formal run aborted "
                "rather than silently "
                "changing optimization."
            ),
        )

        scheduler.step()

        batch_size = int(
            targets.shape[0]
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
            )
            * batch_size
        )

        examples += (
            batch_size
        )

    require(
        examples
        == EXPECTED_TRAIN_TRACKS,
        (
            "TRAIN epoch did not process "
            "exactly 4112 examples."
        ),
    )

    return (
        total_loss
        / examples
    )


@torch.no_grad()
def validate(
    model: BertTask1Model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):

    model.eval()

    total_loss = 0.0
    examples = 0

    all_targets = []
    all_probabilities = []

    for batch in loader:

        (
            input_ids,
            attention_mask,
            token_type_ids,
            targets,
        ) = batch

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        attention_mask = (
            attention_mask.to(
                device,
                non_blocking=True,
            )
        )

        token_type_ids = (
            token_type_ids.to(
                device,
                non_blocking=True,
            )
        )

        targets = targets.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = criterion(
                logits,
                targets,
            )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            (
                "Non-finite validation "
                "logits."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    loss
                )
            ),
            (
                "Non-finite validation "
                "loss."
            ),
        )

        probabilities = torch.sigmoid(
            logits.float()
        )

        require(
            bool(
                torch.isfinite(
                    probabilities
                ).all()
            ),
            (
                "Non-finite validation "
                "probabilities."
            ),
        )

        batch_size = int(
            targets.shape[0]
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
            )
            * batch_size
        )

        examples += (
            batch_size
        )

        all_targets.append(
            targets.detach()
            .cpu()
            .numpy()
        )

        all_probabilities.append(
            probabilities.detach()
            .cpu()
            .numpy()
        )

    require(
        examples
        == EXPECTED_VAL_TRACKS,
        (
            "Validation did not process "
            "exactly 514 examples."
        ),
    )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_probability = np.concatenate(
        all_probabilities,
        axis=0,
    )

    (
        macro_mean_ap,
        micro_ap,
        per_label_ap,
    ) = compute_auc_pr_metrics(
        y_true,
        y_probability,
    )

    return {
        "loss":
            total_loss / examples,

        "macro_mean_ap":
            macro_mean_ap,

        "micro_ap":
            micro_ap,

        "per_label_ap":
            per_label_ap,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model: BertTask1Model,
    variant: str,
    epoch: int,
    val_macro_mean_ap: float,
    val_loss: float,
    pos_weight: np.ndarray,
    run_id: str,
    artifact_hashes: dict,
    initial_bert_sha256: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dict = {
        key:
            value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }

    checkpoint = {
        "task": "task1",
        "variant": variant,
        "run_id": run_id,
        "seed": SEED,
        "epoch": epoch,

        "architecture": (
            "bert-base-uncased -> CLS(768) "
            "-> Dropout(0.1) -> Linear(768,20)"
        ),

        "num_labels": NUM_LABELS,

        "selection_metric":
            SELECTION_METRIC_NAME,

        "val_auc_pr_macro_mean_ap":
            val_macro_mean_ap,

        "val_loss":
            val_loss,

        "pos_weight":
            pos_weight.tolist(),

        "pretraining_lock_sha256":
            artifact_hashes[
                "pretraining_lock_sha256"
            ],

        "initial_dataset_lock_sha256":
            artifact_hashes[
                "initial_lock_sha256"
            ],

        "initial_pretrained_bert_sha256":
            initial_bert_sha256,

        "model_state_dict":
            state_dict,
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(
        path
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Formal Task-1 BERT "
            "model-selection trainer."
        )
    )

    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "T1-A",
            "T1-B",
        ],
        help=(
            "T1-A = frozen BERT; "
            "T1-B = full fine-tuning."
        ),
    )

    args = parser.parse_args()

    variant = args.variant

    variant_spec = (
        VARIANT_CONFIG[
            variant
        ]
    )

    freeze_bert = bool(
        variant_spec[
            "freeze_bert"
        ]
    )

    learning_rate = float(
        variant_spec[
            "learning_rate"
        ]
    )

    print()
    print("=" * 78)
    print(
        "SCRIPT 32 — FORMAL TASK-1 "
        "BERT MODEL-SELECTION TRAINER"
    )
    print("=" * 78)

    print()
    print(
        "Variant:",
        variant,
    )

    print(
        "Description:",
        variant_spec[
            "description"
        ],
    )

    print(
        "Learning rate:",
        learning_rate,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "TRAIN used for optimization."
    )

    print(
        "VAL used for early stopping "
        "and checkpoint selection."
    )

    print(
        "TEST labels/inference: NOT USED."
    )

    print(
        "Threshold sweep: NOT RUN."
    )

    assert_no_completed_duplicate(
        variant
    )

    # --------------------------------------------------------
    # Frozen input enforcement
    # --------------------------------------------------------

    artifact_hashes = (
        verify_frozen_artifacts()
    )

    (
        config,
        vocab,
        label_names,
        frozen_support,
    ) = validate_config_and_vocab()

    (
        train_frame,
        val_frame,
        pos_weight_np,
    ) = load_development_data(
        frozen_support
    )

    (
        input_ids,
        attention_mask,
        token_type_ids,
    ) = load_token_cache(
        train_frame,
        val_frame,
    )

    # --------------------------------------------------------
    # CUDA / reproducibility
    # --------------------------------------------------------

    section(
        "5. ENVIRONMENT + REPRODUCIBILITY"
    )

    require(
        torch.cuda.is_available(),
        "CUDA is required for formal Task-1 training.",
    )

    device = torch.device(
        "cuda"
    )

    set_seed(
        SEED
    )

    gpu_name = (
        torch.cuda
        .get_device_name(0)
    )

    git_commit = (
        get_git_commit()
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "Transformers:",
        transformers.__version__,
    )

    print(
        "scikit-learn:",
        sklearn.__version__,
    )

    print(
        "GPU:",
        gpu_name,
    )

    print(
        "Git commit:",
        git_commit,
    )

    print(
        "cudnn.deterministic:",
        torch.backends
        .cudnn.deterministic,
    )

    print(
        "cudnn.benchmark:",
        torch.backends
        .cudnn.benchmark,
    )

    print(
        "AMP:",
        USE_AMP,
    )

    if git_commit == "UNAVAILABLE":
        print(
            "WARNING: Git commit metadata "
            "is unavailable. The run will "
            "record this explicitly."
        )

    # --------------------------------------------------------
    # Run directories
    # --------------------------------------------------------

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    run_id = (
        f"task1_{variant}_"
        f"seed{SEED}_{timestamp}"
    )

    run_dir = (
        RUNS_DIR
        / run_id
    )

    checkpoint_path = (
        CANDIDATE_MODEL_DIR
        / f"{run_id}_best.pt"
    )

    require(
        not run_dir.exists(),
        (
            "Run directory already exists: "
            f"{safe_relative(run_dir)}"
        ),
    )

    require(
        not checkpoint_path.exists(),
        (
            "Candidate checkpoint already exists: "
            f"{safe_relative(checkpoint_path)}"
        ),
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    CANDIDATE_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset/loaders
    # --------------------------------------------------------

    section(
        "6. DATALOADERS"
    )

    train_dataset = BertCachedDataset(
        train_frame,
        input_ids,
        attention_mask,
        token_type_ids,
    )

    val_dataset = BertCachedDataset(
        val_frame,
        input_ids,
        attention_mask,
        token_type_ids,
    )

    shuffle_generator = (
        torch.Generator()
    )

    shuffle_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    steps_per_epoch = len(
        train_loader
    )

    require(
        steps_per_epoch == 257,
        (
            "Expected exactly 257 "
            "optimizer steps per epoch."
        ),
    )

    maximum_optimizer_steps = (
        steps_per_epoch
        * MAX_EPOCHS
    )

    require(
        maximum_optimizer_steps
        == 2056,
        (
            "Expected 2056 maximum "
            "optimizer steps."
        ),
    )

    warmup_steps = math.ceil(
        WARMUP_FRACTION
        * maximum_optimizer_steps
    )

    require(
        warmup_steps == 206,
        "Expected 206 warmup steps.",
    )

    print(
        "TRAIN batches/epoch:",
        steps_per_epoch,
    )

    print(
        "VAL batches:",
        len(val_loader),
    )

    print(
        "Maximum optimizer steps:",
        maximum_optimizer_steps,
    )

    print(
        "Warmup steps:",
        warmup_steps,
    )

    print(
        "Warmup rounding:",
        WARMUP_ROUNDING,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    section(
        "7. MODEL"
    )

    model = BertTask1Model(
        freeze_bert=freeze_bert
    )

    print(
        "Hashing initial pretrained "
        "BERT weights..."
    )

    initial_bert_sha256 = (
        hash_bert_state(
            model.bert
        )
    )

    print(
        "Initial BERT SHA256:",
        initial_bert_sha256,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    if freeze_bert:

        bert_trainable = sum(
            parameter.numel()
            for parameter
            in model.bert.parameters()
            if parameter.requires_grad
        )

        require(
            bert_trainable == 0,
            (
                "T1-A requires zero "
                "trainable BERT parameters."
            ),
        )

        expected_head_parameters = (
            768 * NUM_LABELS
            + NUM_LABELS
        )

        require(
            trainable_parameters
            == expected_head_parameters,
            (
                "Unexpected T1-A trainable "
                "parameter count."
            ),
        )

    else:

        bert_trainable = sum(
            parameter.numel()
            for parameter
            in model.bert.parameters()
            if parameter.requires_grad
        )

        require(
            bert_trainable > 0,
            (
                "T1-B requires trainable "
                "BERT parameters."
            ),
        )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Trainable BERT parameters:",
        f"{bert_trainable:,}",
    )

    print(
        "Head dropout:",
        DROPOUT,
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Loss / optimizer / scheduler
    # --------------------------------------------------------

    section(
        "8. OPTIMIZATION CONFIG"
    )

    pos_weight = torch.tensor(
        pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = build_optimizer(
        model,
        learning_rate,
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(
                maximum_optimizer_steps
            ),
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    print(
        "Loss:",
        "BCEWithLogitsLoss("
        "train-only pos_weight)",
    )

    print(
        "Optimizer:",
        "torch.optim.AdamW",
    )

    print(
        "Peak LR:",
        learning_rate,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print(
        "Bias/LayerNorm decay:",
        0.0,
    )

    print(
        "Scheduler:",
        SCHEDULER_NAME,
    )

    print(
        "Warmup fraction requested:",
        WARMUP_FRACTION,
    )

    print(
        "Warmup steps:",
        warmup_steps,
    )

    print(
        "Early-stop patience:",
        EARLY_STOP_PATIENCE,
    )

    print(
        "Selection metric:",
        SELECTION_METRIC_NAME,
    )

    # --------------------------------------------------------
    # Save run specification BEFORE training
    # --------------------------------------------------------

    run_spec = {
        "run_id": run_id,
        "task": "task1",
        "variant": variant,
        "description":
            variant_spec[
                "description"
            ],
        "formal_model_selection_run":
            True,
        "seed": SEED,
        "test_split_used":
            False,
        "threshold_tuning_used":
            False,

        "architecture": {
            "backbone":
                MODEL_NAME,
            "representation":
                "last_hidden_state[:,0,:]",
            "hidden_size":
                768,
            "dropout":
                DROPOUT,
            "classifier":
                "Linear(768,20)",
            "num_labels":
                NUM_LABELS,
        },

        "training": {
            "freeze_bert":
                freeze_bert,
            "learning_rate":
                learning_rate,
            "batch_size":
                BATCH_SIZE,
            "max_epochs":
                MAX_EPOCHS,
            "optimizer":
                "torch.optim.AdamW",
            "weight_decay":
                WEIGHT_DECAY,
            "no_decay_for":
                [
                    "bias",
                    "LayerNorm.weight",
                    "layer_norm.weight",
                ],
            "warmup_fraction":
                WARMUP_FRACTION,
            "warmup_steps":
                warmup_steps,
            "warmup_rounding":
                WARMUP_ROUNDING,
            "scheduler":
                SCHEDULER_NAME,
            "early_stop_patience":
                EARLY_STOP_PATIENCE,
            "amp":
                USE_AMP,
        },

        "loss": {
            "name":
                "BCEWithLogitsLoss",
            "pos_weight_source":
                "TRAIN only",
            "formula":
                "min(negative/positive,10)",
            "pos_weight":
                pos_weight_np.tolist(),
        },

        "selection": {
            "metric":
                SELECTION_METRIC_NAME,
            "aggregation":
                "mean over 20 labels",
            "implementation":
                (
                    "sklearn.metrics."
                    "average_precision_score("
                    "average=None), then mean"
                ),
            "threshold_free":
                True,
        },

        "dataset": {
            "train_tracks":
                EXPECTED_TRAIN_TRACKS,
            "val_tracks":
                EXPECTED_VAL_TRACKS,
            "test_tracks_used":
                0,
            "max_length":
                MAX_LENGTH,
            "cache_index_mapping":
                "bert_token_manifest.cache_index",
        },

        "frozen_artifacts": {
            **artifact_hashes,
            "initial_pretrained_bert_sha256":
                initial_bert_sha256,
        },

        "environment": {
            "python_note":
                "project environment Python 3.12.3",
            "torch":
                torch.__version__,
            "transformers":
                transformers.__version__,
            "sklearn":
                sklearn.__version__,
            "gpu":
                gpu_name,
            "git_commit":
                git_commit,
        },
    }

    with (
        run_dir
        / "run_spec.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_spec,
            f,
            indent=2,
        )

    shutil.copy2(
        CONFIG_PATH,
        run_dir
        / "frozen_config_snapshot.yaml",
    )

    # --------------------------------------------------------
    # Formal training
    # --------------------------------------------------------

    section(
        "9. FORMAL TRAINING"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(
        device
    )

    overall_start = time.perf_counter()

    best_metric = -math.inf
    best_epoch = None
    best_val_loss = None
    best_per_label_ap = None

    epochs_without_improvement = 0
    stopped_early = False

    history_rows = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        validation = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = float(
            validation[
                "loss"
            ]
        )

        val_macro_ap = float(
            validation[
                "macro_mean_ap"
            ]
        )

        val_micro_ap = float(
            validation[
                "micro_ap"
            ]
        )

        current_lr = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        improved = (
            val_macro_ap
            > (
                best_metric
                + IMPROVEMENT_EPSILON
            )
        )

        if improved:

            best_metric = (
                val_macro_ap
            )

            best_epoch = epoch

            best_val_loss = (
                val_loss
            )

            best_per_label_ap = (
                validation[
                    "per_label_ap"
                ].copy()
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                variant=variant,
                epoch=epoch,
                val_macro_mean_ap=(
                    val_macro_ap
                ),
                val_loss=val_loss,
                pos_weight=pos_weight_np,
                run_id=run_id,
                artifact_hashes=(
                    artifact_hashes
                ),
                initial_bert_sha256=(
                    initial_bert_sha256
                ),
            )

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        row = {
            "run_id": run_id,
            "task": "task1",
            "variant": variant,
            "seed": SEED,
            "epoch": epoch,
            "train_loss":
                train_loss,
            "val_loss":
                val_loss,
            "val_auc_pr_macro":
                val_macro_ap,
            "val_auc_pr_micro_diagnostic":
                val_micro_ap,
            "learning_rate_end":
                current_lr,
            "improved":
                improved,
            "best_val_auc_pr_so_far":
                best_metric,
            "epochs_without_improvement":
                epochs_without_improvement,
            "epoch_seconds":
                epoch_seconds,
            "test_split_used":
                False,
        }

        history_rows.append(
            row
        )

        write_dataframe_atomic(
            pd.DataFrame(
                history_rows
            ),
            run_dir
            / "history.csv",
        )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss="
            f"{train_loss:.6f} | "
            f"val_loss="
            f"{val_loss:.6f} | "
            f"val_macro_AP="
            f"{val_macro_ap:.6f} | "
            f"val_micro_AP="
            f"{val_micro_ap:.6f} | "
            f"lr="
            f"{current_lr:.8g} | "
            f"time="
            f"{epoch_seconds:.1f}s"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            stopped_early = True

            print()
            print(
                "Early stopping triggered:"
            )

            print(
                f"{EARLY_STOP_PATIENCE} "
                "consecutive epochs "
                "without validation "
                "macro-AP improvement."
            )

            break

    total_runtime_seconds = (
        time.perf_counter()
        - overall_start
    )

    require(
        best_epoch is not None,
        "No best epoch was selected.",
    )

    require(
        checkpoint_path.is_file(),
        "Best checkpoint was not written.",
    )

    # --------------------------------------------------------
    # Reload + independently validate saved best checkpoint
    # --------------------------------------------------------

    section(
        "10. BEST CHECKPOINT REVALIDATION"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    require(
        checkpoint["variant"]
        == variant,
        "Checkpoint variant mismatch.",
    )

    require(
        checkpoint["seed"]
        == SEED,
        "Checkpoint seed mismatch.",
    )

    require(
        checkpoint[
            "pretraining_lock_sha256"
        ]
        == artifact_hashes[
            "pretraining_lock_sha256"
        ],
        (
            "Checkpoint frozen-lock "
            "identity mismatch."
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    revalidated = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    revalidated_metric = float(
        revalidated[
            "macro_mean_ap"
        ]
    )

    require(
        abs(
            revalidated_metric
            - best_metric
        )
        <= 1.0e-8,
        (
            "Reloaded checkpoint does not "
            "reproduce its recorded best "
            "validation metric."
        ),
    )

    require(
        abs(
            float(
                checkpoint[
                    "val_auc_pr_macro_mean_ap"
                ]
            )
            - best_metric
        )
        <= 1.0e-12,
        (
            "Checkpoint metadata metric "
            "does not equal selected metric."
        ),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Recorded best macro AP:",
        f"{best_metric:.8f}",
    )

    print(
        "Reloaded macro AP:",
        f"{revalidated_metric:.8f}",
    )

    print(
        "Checkpoint revalidation: PASS"
    )

    # --------------------------------------------------------
    # Per-label validation AP for BEST checkpoint
    # --------------------------------------------------------

    best_per_label_frame = (
        pd.DataFrame(
            {
                "index":
                    np.arange(
                        NUM_LABELS
                    ),
                "label":
                    label_names,
                "val_average_precision":
                    revalidated[
                        "per_label_ap"
                    ],
            }
        )
    )

    write_dataframe_atomic(
        best_per_label_frame,
        run_dir
        / "best_val_per_label_ap.csv",
    )

    # --------------------------------------------------------
    # Final run metadata
    # --------------------------------------------------------

    peak_vram_bytes = (
        torch.cuda
        .max_memory_allocated(
            device
        )
    )

    peak_vram_gb = (
        peak_vram_bytes
        / (1024 ** 3)
    )

    epochs_completed = len(
        history_rows
    )

    final_result = {
        "status": "COMPLETE",
        "run_id": run_id,
        "task": "task1",
        "variant": variant,
        "seed": SEED,

        "formal_model_selection_run":
            True,

        "best_epoch":
            best_epoch,

        "best_val_auc_pr_macro":
            best_metric,

        "best_val_loss":
            best_val_loss,

        "epochs_completed":
            epochs_completed,

        "stopped_early":
            stopped_early,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "checkpoint_path":
            safe_relative(
                checkpoint_path
            ),

        "runtime_seconds":
            total_runtime_seconds,

        "peak_vram_gb":
            peak_vram_gb,

        "initial_pretrained_bert_sha256":
            initial_bert_sha256,

        "pretraining_lock_sha256":
            artifact_hashes[
                "pretraining_lock_sha256"
            ],
    }

    with (
        run_dir
        / "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_result,
            f,
            indent=2,
        )

    # Global epoch CSV is updated only once the run
    # has completed successfully.
    append_rows_to_csv(
        history_rows,
        EPOCH_METRICS_PATH,
    )

    trial_row = {
        "run_id":
            run_id,

        "status":
            "COMPLETE",

        "task":
            "task1",

        "variant":
            variant,

        "strategy":
            variant_spec[
                "description"
            ],

        "seed":
            SEED,

        "learning_rate":
            learning_rate,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "epochs_completed":
            epochs_completed,

        "dropout":
            DROPOUT,

        "weight_decay":
            WEIGHT_DECAY,

        "warmup_fraction":
            WARMUP_FRACTION,

        "warmup_steps":
            warmup_steps,

        "warmup_rounding":
            WARMUP_ROUNDING,

        "scheduler":
            SCHEDULER_NAME,

        "early_stop_patience":
            EARLY_STOP_PATIENCE,

        "selection_metric":
            SELECTION_METRIC_NAME,

        "best_epoch":
            best_epoch,

        "best_val_auc_pr_macro":
            best_metric,

        "best_val_loss":
            best_val_loss,

        "stopped_early":
            stopped_early,

        "trainable_parameters":
            trainable_parameters,

        "total_parameters":
            total_parameters,

        "runtime_seconds":
            total_runtime_seconds,

        "peak_vram_gb":
            peak_vram_gb,

        "checkpoint_path":
            safe_relative(
                checkpoint_path
            ),

        "pretraining_lock_sha256":
            artifact_hashes[
                "pretraining_lock_sha256"
            ],

        "initial_pretrained_bert_sha256":
            initial_bert_sha256,

        "git_commit":
            git_commit,

        "test_split_used":
            False,
    }

    append_rows_to_csv(
        [trial_row],
        TRIALS_PATH,
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    section(
        "FORMAL TASK-1 RUN COMPLETE"
    )

    print(
        "Variant:",
        variant,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation macro mean AP:",
        f"{best_metric:.8f}",
    )

    print(
        "Epochs completed:",
        epochs_completed,
    )

    print(
        "Stopped early:",
        stopped_early,
    )

    print(
        "Runtime:",
        f"{total_runtime_seconds / 60:.2f} min",
    )

    print(
        "Peak allocated VRAM:",
        f"{peak_vram_gb:.3f} GB",
    )

    print(
        "Candidate checkpoint:",
        safe_relative(
            checkpoint_path
        ),
    )

    print(
        "Run directory:",
        safe_relative(
            run_dir
        ),
    )

    print(
        "Epoch log:",
        safe_relative(
            EPOCH_METRICS_PATH
        ),
    )

    print(
        "Trial log:",
        safe_relative(
            TRIALS_PATH
        ),
    )

    print()
    print(
        "TEST examples used for labels/"
        "inference: 0"
    )

    print(
        "Threshold tuning performed: NO"
    )

    print()
    print(
        "DO NOT run the other Task-1 "
        "variant or select the winner "
        "until this run output has been "
        "reviewed."
    )


if __name__ == "__main__":
    main()