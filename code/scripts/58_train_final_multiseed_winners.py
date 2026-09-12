#!/usr/bin/env python3
"""
Script 58 — Final Three-Seed Winner Training
=============================================

Pipeline Plan v6 final winning configurations:

Task 1:
    T1-B — fully fine-tuned BERT

Task 2:
    A1 — CNN

Task 3:
    M3 — Early Fusion, Phase A -> Phase B

Final seeds:
    42, 1337, 2026

Important Task-3 multiseed definition
-------------------------------------
The selected Task-3 encoder artifacts remain fixed:

    T1-B seed-42 selected checkpoint
    A4 GAT/G2 seed-42 selected checkpoint

These exact encoders were part of the frozen Task-3 design and model selection.

For each final seed:
    - M3 Phase A is retrained with that seed using the frozen
      TRAIN+VAL encoder representations.
    - M3 Phase B starts from the same selected T1-B/A4 encoder
      artifacts plus that seed's Phase-A fusion state.
    - BERT layers 10-11 + complete selected GNN encoder + fusion
      are trainable.

Task 1 and Task 2 are independently retrained from scratch at
each final seed.

No TEST:
    - no TEST labels loaded
    - no TEST inference
    - no TEST metrics
    - no threshold re-tuning

The frozen thresholds from Script 57 are not used for training.
They will be applied only during final TEST evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool

from transformers import (
    AutoModel,
    BertConfig,
    BertModel,
    get_linear_schedule_with_warmup,
)


# =============================================================================
# 0. ROOT / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRETRAINING_LOCK = (
    ROOT / "data/splits/pretraining_frozen_hashes.json"
)

TARGETS = (
    ROOT / "data/processed/labels/musiccaps_targets.parquet"
)

TOKEN_CACHE = (
    ROOT / "data/processed/bert_tokens/bert_tokens.npz"
)

TOKEN_MANIFEST = (
    ROOT / "data/processed/bert_tokens/bert_token_manifest.parquet"
)

AUDIO_MANIFEST = (
    ROOT / "data/processed/audio_feature_manifest.parquet"
)

GRAPH_MANIFEST = (
    ROOT / "data/processed/graph_manifest.parquet"
)


# Frozen Task-3 Phase-A encoder representations.
BERT_SEQUENCE_CACHE = (
    ROOT
    / "data/processed/bert_seq_frozen/"
    "bert_sequence_trainval.npy"
)

BERT_SEQUENCE_MANIFEST = (
    ROOT
    / "data/processed/bert_seq_frozen/"
    "bert_seq_manifest.parquet"
)

GAT_EMBEDDING_CACHE = (
    ROOT
    / "data/processed/gat_g2_frozen/"
    "gat_g2_embedding_trainval.npy"
)

GAT_EMBEDDING_MANIFEST = (
    ROOT
    / "data/processed/gat_g2_frozen/"
    "gat_g2_manifest.parquet"
)


# Selected source encoders used by Task 3.
SELECTED_T1 = (
    ROOT
    / "models/task1_candidates/"
    "task1_T1-B_seed42_20260828T182131Z_best.pt"
)

SELECTED_A4 = (
    ROOT
    / "models/task2_candidates/"
    "task2_A4_GAT_G2_seed42_20260828T193949Z_best.pt"
)


# Accepted seed-42 Task-2 winner, used only as reference.
SELECTED_A1 = (
    ROOT
    / "models/task2_candidates/"
    "task2_A1_CNN_seed42_20260828T184558Z_best.pt"
)


# Frozen Task-3 selection.
TASK3_SELECTION = (
    ROOT
    / "results/runs/task3/"
    "task3_model_selection.json"
)


# Frozen thresholds.
THRESHOLD_LOCK = (
    ROOT
    / "results/runs/final_eval/"
    "winner_thresholds_validation.json"
)


MODEL_ROOT = (
    ROOT / "models/final_multiseed"
)

RUN_ROOT = (
    ROOT
    / "results/runs/final_eval/"
    "multiseed"
)

FINAL_MANIFEST = (
    RUN_ROOT
    / "final_multiseed_training_manifest.json"
)


# =============================================================================
# 1. FROZEN HASH IDENTITIES
# =============================================================================

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_TARGETS_SHA256 = (
    "a6f80bb42b0b48b2c47dac5f27ad2fd66f8893d83bd5d4e552d356cee3547056"
)

EXPECTED_TOKEN_CACHE_SHA256 = (
    "0df45ab266884abbc2f3f7f96a5ba789bbf92bf8da49ac4cb57a1e4eb476188f"
)

EXPECTED_TOKEN_MANIFEST_SHA256 = (
    "4be874ab26375ce63bb0f4b55706d178bfac8a951fee96fd75d2d49b28a390b8"
)

EXPECTED_AUDIO_MANIFEST_SHA256 = (
    "e0479ac47a6d41ca633d6a47bbee16962607cf5a1749cca2e56b349c310aaa49"
)

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6ad0492bf90eccc929cadc1aad1da9e158"
)

EXPECTED_BERT_SEQUENCE_SHA256 = (
    "a1c723ec939392b8b0d9bafadb8327d8a0967ecb9bd8e89ffcd559306633e133"
)

EXPECTED_BERT_SEQUENCE_MANIFEST_SHA256 = (
    "ebd86077489b716abc8768247f550890ca8e5a26b8f0967c32882c22c4e85d2f"
)

EXPECTED_GAT_EMBEDDING_SHA256 = (
    "3d39a343d1acc678576f9dd3d88b088ca18a81017190246b0b32841292b92852"
)

EXPECTED_GAT_EMBEDDING_MANIFEST_SHA256 = (
    "a029c12ec6a98922c98f373bbc263b6c529825b56b1446a14f659f6854ed371a"
)

EXPECTED_SELECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_SELECTED_A1_SHA256 = (
    "4bb2369ce7f93693d67b1579cc1c7b4ac766076a8a0406ae7194ab8b0cbc1b35"
)

EXPECTED_SELECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e"
)

EXPECTED_TASK3_SELECTION_SHA256 = (
    "1d02aafbc0aef9702194874ba02de7c0cbf7483dfac4b15d191fbfdeaac31e5e"
)

EXPECTED_THRESHOLD_LOCK_SHA256 = (
    "441f3bf1229afd4a578a5485111d1a5ad9ab049bd4b163d7bd21c44f64040403"
)


# =============================================================================
# 2. DATASET / METRIC CONTRACT
# =============================================================================

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

N_TOTAL = 5140
N_TRAIN = 4112
N_VAL = 514
N_DEV = 4626

NUM_LABELS = 20
MAX_LENGTH = 112

MODEL_NAME = "bert-base-uncased"

IMPROVEMENT_EPSILON = 1.0e-12


# =============================================================================
# 3. TASK-1 WINNER CONTRACT
# =============================================================================

T1_BATCH_SIZE = 16
T1_MAX_EPOCHS = 8

T1_LR = 2.0e-5
T1_DROPOUT = 0.1
T1_WEIGHT_DECAY = 0.01

T1_WARMUP_FRACTION = 0.10
T1_PATIENCE = 2

T1_USE_AMP = True


# =============================================================================
# 4. TASK-2 A1 WINNER CONTRACT
# =============================================================================

A1_BATCH_SIZE = 32
A1_MAX_EPOCHS = 15

A1_LR = 1.0e-3
A1_WEIGHT_DECAY = 1.0e-4
A1_PATIENCE = 3


# =============================================================================
# 5. TASK-3 M3 CONTRACT
# =============================================================================

# Phase A
M3A_BATCH_SIZE = 32
M3A_MAX_EPOCHS = 20
M3A_PATIENCE = 3

M3A_LR = 1.0e-3
M3A_WEIGHT_DECAY = 1.0e-4


# Phase B
M3B_BATCH_SIZE = 16
M3B_EPOCHS = 5

M3B_LR = 5.0e-6

M3B_BERT_WEIGHT_DECAY = 1.0e-2
M3B_GNN_FUSION_WEIGHT_DECAY = 1.0e-4


# =============================================================================
# 6. ACCEPTED SEED-42 RESULTS — PARITY REFERENCE ONLY
# =============================================================================

SEED42_EXPECTED = {
    "task1_val_macro_ap":
        0.8405750022914198,

    "task2_val_macro_ap":
        0.22887393178444362,

    "task3_phaseA_val_macro_ap":
        0.830729168310,

    "task3_phaseB_val_macro_ap":
        0.839664696610,
}

SEED42_PARITY_ATOL = 1.0e-8


# =============================================================================
# 7. REPORTING / HELPERS
# =============================================================================

def banner(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:

        raise RuntimeError(
            message
        )

    print(
        f"PASS  {message}"
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                chunk_size
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        value = json.load(f)

    if not isinstance(
        value,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not a JSON object"
        )

    return value


def load_checkpoint(
    path: Path,
) -> dict[str, Any]:

    try:

        value = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        value = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        value,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not a checkpoint dictionary"
        )

    return value


def cpu_state_dict(
    module: nn.Module,
) -> dict[str, torch.Tensor]:

    return {
        key:
            value
            .detach()
            .cpu()
            .clone()

        for key, value
        in module.state_dict().items()
    }


def atomic_torch_save(
    value: Any,
    path: Path,
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if path.exists():

        raise RuntimeError(
            f"output already exists: {path}"
        )

    if temp.exists():

        raise RuntimeError(
            f"temporary output exists: {temp}"
        )

    torch.save(
        value,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def atomic_json(
    value: dict[str, Any],
    path: Path,
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if path.exists():

        raise RuntimeError(
            f"output already exists: {path}"
        )

    if temp.exists():

        raise RuntimeError(
            f"temporary output exists: {temp}"
        )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            value,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        f.write("\n")

    os.replace(
        temp,
        path,
    )


def atomic_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if path.exists():

        raise RuntimeError(
            f"output already exists: {path}"
        )

    if temp.exists():

        raise RuntimeError(
            f"temporary output exists: {temp}"
        )

    pd.DataFrame(
        rows
    ).to_csv(
        temp,
        index=False,
    )

    os.replace(
        temp,
        path,
    )


def seed_everything(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    # Required by Pipeline v6.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_ap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    float,
    float,
    np.ndarray,
]:

    per_label = average_precision_score(
        y_true,
        probabilities,
        average=None,
    )

    macro_ap = float(
        np.mean(
            per_label
        )
    )

    micro_ap = float(
        average_precision_score(
            y_true,
            probabilities,
            average="micro",
        )
    )

    return (
        macro_ap,
        micro_ap,
        per_label.astype(
            np.float64
        ),
    )


# =============================================================================
# 8. FROZEN INPUT VALIDATION
# =============================================================================

def verify_global_dependencies() -> None:

    banner(
        "1. FROZEN FINAL-EVALUATION DEPENDENCIES"
    )

    expected = {
        PRETRAINING_LOCK:
            EXPECTED_PRETRAINING_LOCK_SHA256,

        TARGETS:
            EXPECTED_TARGETS_SHA256,

        TOKEN_CACHE:
            EXPECTED_TOKEN_CACHE_SHA256,

        TOKEN_MANIFEST:
            EXPECTED_TOKEN_MANIFEST_SHA256,

        AUDIO_MANIFEST:
            EXPECTED_AUDIO_MANIFEST_SHA256,

        GRAPH_MANIFEST:
            EXPECTED_GRAPH_MANIFEST_SHA256,

        BERT_SEQUENCE_CACHE:
            EXPECTED_BERT_SEQUENCE_SHA256,

        BERT_SEQUENCE_MANIFEST:
            EXPECTED_BERT_SEQUENCE_MANIFEST_SHA256,

        GAT_EMBEDDING_CACHE:
            EXPECTED_GAT_EMBEDDING_SHA256,

        GAT_EMBEDDING_MANIFEST:
            EXPECTED_GAT_EMBEDDING_MANIFEST_SHA256,

        SELECTED_T1:
            EXPECTED_SELECTED_T1_SHA256,

        SELECTED_A1:
            EXPECTED_SELECTED_A1_SHA256,

        SELECTED_A4:
            EXPECTED_SELECTED_A4_SHA256,

        TASK3_SELECTION:
            EXPECTED_TASK3_SELECTION_SHA256,

        THRESHOLD_LOCK:
            EXPECTED_THRESHOLD_LOCK_SHA256,
    }

    for path, expected_sha in (
        expected.items()
    ):

        require(
            path.is_file(),
            (
                "artifact exists: "
                f"{path.relative_to(ROOT)}"
            ),
        )

        require(
            sha256_file(path)
            == expected_sha,
            (
                f"{path.name} SHA256 "
                "matches frozen identity"
            ),
        )

    threshold_lock = load_json(
        THRESHOLD_LOCK
    )

    require(
        threshold_lock.get(
            "artifact_type"
        )
        ==
        "final_winner_validation_thresholds",
        (
            "Script-57 threshold lock "
            "has correct artifact type"
        ),
    )

    require(
        float(
            threshold_lock[
                "task1"
            ][
                "selected_threshold"
            ]
        )
        == 0.70,
        "T1-B threshold remains 0.70",
    )

    require(
        float(
            threshold_lock[
                "task2"
            ][
                "selected_threshold"
            ]
        )
        == 0.50,
        "A1 threshold remains 0.50",
    )

    require(
        float(
            threshold_lock[
                "task3"
            ][
                "selected_threshold"
            ]
        )
        == 0.70,
        "M3 threshold remains 0.70",
    )

    task3_selection = load_json(
        TASK3_SELECTION
    )

    require(
        task3_selection.get(
            "selected_task3_fusion"
        )
        == "M3_phaseB",
        (
            "Task-3 selected configuration "
            "remains M3 Phase B"
        ),
    )


# =============================================================================
# 9. COMMON TRAIN+VAL DATA
# =============================================================================

def load_development_data():

    banner(
        "2. LOAD FROZEN TRAIN+VAL DATA"
    )

    manifest = pd.read_parquet(
        TOKEN_MANIFEST
    )

    require(
        len(manifest)
        == N_TOTAL,
        (
            "token manifest contains "
            "5140 rows"
        ),
    )

    manifest = manifest.copy()

    manifest["track_id"] = (
        manifest[
            "track_id"
        ].astype(str)
    )

    manifest["split"] = (
        manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    counts = (
        manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    require(
        counts
        == {
            "train": 4112,
            "val": 514,
            "test": 514,
        },
        (
            "frozen split metadata "
            "= 4112/514/514"
        ),
    )

    dev_manifest = (
        manifest.loc[
            manifest[
                "split"
            ].isin(
                [
                    "train",
                    "val",
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    require(
        len(dev_manifest)
        == N_DEV,
        (
            "development manifest "
            "= 4626 rows"
        ),
    )

    require(
        "test"
        not in set(
            dev_manifest[
                "split"
            ]
        ),
        (
            "development manifest "
            "contains zero TEST rows"
        ),
    )

    dev_ids = (
        dev_manifest[
            "track_id"
        ].tolist()
    )

    y_columns = [
        f"y_{index:02d}"
        for index in range(
            NUM_LABELS
        )
    ]

    dataset = pads.dataset(
        str(TARGETS),
        format="parquet",
    )

    table = dataset.to_table(
        columns=[
            "track_id",
            "split",
            *y_columns,
        ],
        filter=(
            (pads.field("split") == "train")
            |
            (pads.field("split") == "val")
        ),
    )

    target_frame = table.to_pandas()

    target_frame[
        "track_id"
    ] = (
        target_frame[
            "track_id"
        ].astype(str)
    )

    target_frame[
        "split"
    ] = (
        target_frame[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    require(
        len(target_frame)
        == N_DEV,
        (
            "exactly 4626 TRAIN+VAL "
            "target rows loaded"
        ),
    )

    require(
        "test"
        not in set(
            target_frame[
                "split"
            ]
        ),
        (
            "TEST target rows loaded = 0"
        ),
    )

    target_lookup = (
        target_frame.set_index(
            "track_id",
            drop=False,
        )
    )

    require(
        set(dev_ids)
        ==
        set(
            target_frame[
                "track_id"
            ]
        ),
        (
            "development tokens and "
            "targets contain identical IDs"
        ),
    )

    aligned_targets = (
        target_lookup.loc[
            dev_ids
        ]
        .reset_index(
            drop=True
        )
    )

    require(
        aligned_targets[
            "split"
        ].tolist()
        ==
        dev_manifest[
            "split"
        ].tolist(),
        (
            "target splits align "
            "row-for-row"
        ),
    )

    y = (
        aligned_targets[
            y_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    cache_indices = (
        dev_manifest[
            "cache_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    with np.load(
        TOKEN_CACHE,
        allow_pickle=False,
    ) as cache:

        input_ids = np.asarray(
            cache[
                "input_ids"
            ][
                cache_indices
            ],
            dtype=np.int32,
        )

        attention_mask = np.asarray(
            cache[
                "attention_mask"
            ][
                cache_indices
            ],
            dtype=np.uint8,
        )

        token_type_ids = np.asarray(
            cache[
                "token_type_ids"
            ][
                cache_indices
            ],
            dtype=np.uint8,
        )

    require(
        input_ids.shape
        == (
            N_DEV,
            MAX_LENGTH,
        ),
        (
            "development token matrix "
            "= (4626,112)"
        ),
    )

    split_array = (
        dev_manifest[
            "split"
        ].to_numpy()
    )

    train_indices = np.flatnonzero(
        split_array == "train"
    )

    val_indices = np.flatnonzero(
        split_array == "val"
    )

    require(
        len(train_indices)
        == N_TRAIN,
        "TRAIN indices = 4112",
    )

    require(
        len(val_indices)
        == N_VAL,
        "VAL indices = 514",
    )

    train_targets = (
        y[
            train_indices
        ].astype(
            np.float64
        )
    )

    positive = train_targets.sum(
        axis=0
    )

    negative = (
        N_TRAIN
        - positive
    )

    pos_weight = np.minimum(
        negative / positive,
        10.0,
    ).astype(
        np.float32
    )

    print()
    print(
        "TEST target rows loaded: 0"
    )

    return {
        "manifest":
            dev_manifest,

        "track_ids":
            dev_ids,

        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "token_type_ids":
            token_type_ids,

        "targets":
            y,

        "train_indices":
            train_indices,

        "val_indices":
            val_indices,

        "pos_weight":
            pos_weight,
    }


# =============================================================================
# 10. GENERIC INDEXED TOKEN DATASET
# =============================================================================

class TokenDataset(Dataset):

    def __init__(
        self,
        data,
        indices,
    ):

        self.data = data

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):

        index = int(
            self.indices[
                item
            ]
        )

        return (
            torch.from_numpy(
                self.data[
                    "input_ids"
                ][
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "attention_mask"
                ][
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "token_type_ids"
                ][
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "targets"
                ][
                    index
                ].astype(
                    np.float32,
                    copy=True,
                )
            ),
        )


# =============================================================================
# 11. TASK-1 T1-B
# =============================================================================

class T1BModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        require(
            int(
                self.bert.config.hidden_size
            )
            == 768,
            (
                "T1-B BERT hidden size "
                "= 768"
            ),
        )

        self.dropout = nn.Dropout(
            T1_DROPOUT
        )

        self.classifier = nn.Linear(
            768,
            NUM_LABELS,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        cls = output.last_hidden_state[
            :,
            0,
            :
        ]

        cls = self.dropout(
            cls
        )

        return self.classifier(
            cls
        )


def build_t1_optimizer(
    model,
):

    no_decay = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )

    decay_parameters = []

    no_decay_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if any(
            token in name
            for token in no_decay
        ):

            no_decay_parameters.append(
                parameter
            )

        else:

            decay_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    decay_parameters,

                "weight_decay":
                    T1_WEIGHT_DECAY,
            },

            {
                "params":
                    no_decay_parameters,

                "weight_decay":
                    0.0,
            },
        ],
        lr=T1_LR,
    )

    return optimizer


def evaluate_t1(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_rows = 0

    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            targets,
        ) in loader:

            input_ids = input_ids.to(
                device,
                non_blocking=True,
            )

            attention_mask = attention_mask.to(
                device,
                non_blocking=True,
            )

            token_type_ids = token_type_ids.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            # Exact Script-32 validation numerical path.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=T1_USE_AMP,
            ):

                logits = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                loss = criterion(
                    logits,
                    targets,
                )

            probabilities = torch.sigmoid(
                logits.float()
            )

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

            all_targets.append(
                targets
                .detach()
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

    require(
        total_rows
        == N_VAL,
        (
            "T1-B validation processed "
            "514 rows"
        ),
    )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    probability = np.concatenate(
        all_probabilities,
        axis=0,
    )

    macro_ap, micro_ap, _ = (
        compute_ap(
            y_true,
            probability,
        )
    )

    return {
        "loss":
            total_loss / total_rows,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,
    }


def train_task1(
    seed,
    data,
    device,
    model_dir,
    run_dir,
):

    banner(
        f"TASK 1 — T1-B — SEED {seed}"
    )

    seed_everything(
        seed
    )

    model = T1BModel().to(
        device
    )

    train_dataset = TokenDataset(
        data,
        data[
            "train_indices"
        ],
    )

    val_dataset = TokenDataset(
        data,
        data[
            "val_indices"
        ],
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=T1_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=T1_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    require(
        len(train_loader)
        == 257,
        (
            "T1-B has exactly "
            "257 optimizer steps/epoch"
        ),
    )

    maximum_steps = (
        len(train_loader)
        * T1_MAX_EPOCHS
    )

    require(
        maximum_steps
        == 2056,
        (
            "T1-B maximum optimizer "
            "steps = 2056"
        ),
    )

    warmup_steps = math.ceil(
        T1_WARMUP_FRACTION
        * maximum_steps
    )

    require(
        warmup_steps
        == 206,
        (
            "T1-B warmup steps = 206"
        ),
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            data[
                "pos_weight"
            ],
            device=device,
            dtype=torch.float32,
        )
    )

    optimizer = build_t1_optimizer(
        model
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                maximum_steps,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=T1_USE_AMP,
    )

    best_score = -float("inf")

    best_epoch = None
    best_state = None
    best_val = None

    bad_epochs = 0

    history = []

    # Standard AMP may occasionally skip an unsafe optimizer
    # update. Record such events for reproducibility.
    amp_overflow_skips = 0

    start = time.perf_counter()

    for epoch in range(
        1,
        T1_MAX_EPOCHS + 1,
    ):

        model.train()

        train_loss_total = 0.0
        train_rows = 0

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            targets,
        ) in train_loader:

            input_ids = input_ids.to(
                device,
                non_blocking=True,
            )

            attention_mask = attention_mask.to(
                device,
                non_blocking=True,
            )

            token_type_ids = token_type_ids.to(
                device,
                non_blocking=True,
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
                enabled=T1_USE_AMP,
            ):

                logits = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
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
                (
                    "T1-B TRAIN loss "
                    "is finite"
                ),
            )

            scaler.scale(
                loss
            ).backward()

            scale_before = (
                scaler.get_scale()
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scale_after = (
                scaler.get_scale()
            )

            optimizer_step_skipped = (
                scale_after
                < scale_before
            )

            if optimizer_step_skipped:

                amp_overflow_skips += 1

                print(
                    "WARNING: T1-B AMP overflow; "
                    "optimizer step skipped safely. "
                    f"Total skips={amp_overflow_skips}"
                )

            else:

                # Advance scheduler only when an optimizer
                # update actually occurred.
                scheduler.step()

            rows = int(
                targets.shape[0]
            )

            train_rows += rows

            train_loss_total += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

        require(
            train_rows
            == N_TRAIN,
            (
                "T1-B TRAIN epoch "
                "processed 4112 rows"
            ),
        )

        train_loss = (
            train_loss_total
            / train_rows
        )

        val = evaluate_t1(
            model,
            val_loader,
            criterion,
            device,
        )

        improved = (
            val[
                "macro_ap"
            ]
            >
            best_score
            + IMPROVEMENT_EPSILON
        )

        if improved:

            best_score = val[
                "macro_ap"
            ]

            best_epoch = epoch

            best_val = dict(
                val
            )

            best_state = (
                cpu_state_dict(
                    model
                )
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val[
                        "loss"
                    ],

                "val_macro_ap":
                    val[
                        "macro_ap"
                    ],

                "val_micro_ap":
                    val[
                        "micro_ap"
                    ],

                "improved":
                    improved,
            }
        )

        marker = (
            " BEST"
            if improved
            else ""
        )

        print(
            f"T1-B epoch {epoch:02d}/8  "
            f"train_loss={train_loss:.6f}  "
            f"val_macro_AP="
            f"{val['macro_ap']:.8f}"
            f"{marker}"
        )

        if (
            bad_epochs
            >= T1_PATIENCE
        ):

            print(
                "T1-B early stopping."
            )

            break

    if (
        best_state is None
        or best_epoch is None
        or best_val is None
    ):

        raise RuntimeError(
            "T1-B did not select "
            "a valid checkpoint"
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    restored = evaluate_t1(
        model,
        val_loader,
        criterion,
        device,
    )

    require(
        np.isclose(
            restored[
                "macro_ap"
            ],
            best_val[
                "macro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "T1-B restored best state "
            "reproduces VAL macro AP"
        ),
    )

    checkpoint_path = (
        model_dir
        / "task1_T1B_best.pt"
    )

    epoch_path = (
        run_dir
        / "task1_T1B_epochs.csv"
    )

    checkpoint = {
        "task":
            "task1",

        "variant":
            "T1-B",

        "final_multiseed":
            True,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "val_loss":
            restored[
                "loss"
            ],

        "threshold":
            0.70,

        "model_state_dict":
            best_state,

        "training": {
            "learning_rate":
                T1_LR,

            "batch_size":
                T1_BATCH_SIZE,

            "max_epochs":
                T1_MAX_EPOCHS,

            "weight_decay":
                T1_WEIGHT_DECAY,

            "warmup_fraction":
                T1_WARMUP_FRACTION,

            "warmup_steps":
                warmup_steps,

            "early_stop_patience":
                T1_PATIENCE,

            "amp":
                True,

            "amp_overflow_skipped_optimizer_steps":
                amp_overflow_skips,
        },

        "test_examples_evaluated":
            0,

        "test_target_rows_loaded":
            0,
    }

    atomic_torch_save(
        checkpoint,
        checkpoint_path,
    )

    atomic_csv(
        history,
        epoch_path,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    result = {
        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "checkpoint":
            str(
                checkpoint_path
                .relative_to(
                    ROOT
                )
            ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),

        "epoch_log":
            str(
                epoch_path
                .relative_to(
                    ROOT
                )
            ),

        "runtime_seconds":
            runtime,

        "amp_overflow_skipped_optimizer_steps":
            amp_overflow_skips,
    }

    if seed == 42:
        print(
            "Seed-42 reference comparison — "
            "diagnostic only, not an acceptance gate:"
        )
        print(
            f"  original selection AP = "
            f"{SEED42_EXPECTED['task1_val_macro_ap']:.12f}"
        )
        print(
            f"  final rerun AP         = "
            f"{result['val_macro_ap']:.12f}"
        )
        print(
            f"  delta                  = "
            f"{result['val_macro_ap'] - SEED42_EXPECTED['task1_val_macro_ap']:+.12f}"
        )

    return (
        result,
        model,
    )


# =============================================================================
# 12. A1 CNN DATA
# =============================================================================

def resolve_feature_path(
    value: str,
) -> Path:

    raw = Path(
        value
    )

    candidates = []

    if raw.is_absolute():

        candidates.append(
            raw
        )

    else:

        candidates.extend(
            [
                ROOT / raw,

                AUDIO_MANIFEST.parent
                / raw,

                ROOT
                / "data/processed/mel"
                / raw.name,
            ]
        )

    matches = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = (
                candidate.resolve()
            )

            matches[
                str(
                    resolved
                )
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            "Could not uniquely resolve "
            f"mel path: {value}"
        )

    return next(
        iter(
            matches.values()
        )
    )


def load_development_mels(
    dev_ids,
):

    banner(
        "3. LOAD TRAIN+VAL MEL FEATURES"
    )

    manifest = pd.read_parquet(
        AUDIO_MANIFEST
    )

    manifest = manifest.copy()

    manifest[
        "track_id"
    ] = (
        manifest[
            "track_id"
        ].astype(str)
    )

    require(
        len(manifest)
        == N_TOTAL,
        (
            "audio manifest contains "
            "5140 rows"
        ),
    )

    lookup = manifest.set_index(
        "track_id",
        drop=False,
    )

    require(
        set(dev_ids).issubset(
            set(
                manifest[
                    "track_id"
                ]
            )
        ),
        (
            "all TRAIN+VAL tracks "
            "have mel features"
        ),
    )

    values = np.empty(
        (
            N_DEV,
            1,
            128,
            431,
        ),
        dtype=np.float32,
    )

    for index, track_id in enumerate(
        dev_ids
    ):

        row = lookup.loc[
            track_id
        ]

        path = resolve_feature_path(
            str(
                row[
                    "mel_path"
                ]
            )
        )

        mel = np.load(
            path,
            allow_pickle=False,
        )

        if (
            mel.shape
            != (
                128,
                431,
            )
            or
            mel.dtype
            != np.float32
            or
            not np.isfinite(
                mel
            ).all()
        ):

            raise RuntimeError(
                f"Invalid mel file: "
                f"{track_id}"
            )

        values[
            index,
            0,
            :,
            :
        ] = mel

        number = index + 1

        if (
            number == 1
            or
            number % 500 == 0
            or
            number == N_DEV
        ):

            print(
                f"  loaded "
                f"{number}/{N_DEV}"
            )

    print(
        "TEST mel files loaded: 0"
    )

    return values


class MelDataset(Dataset):

    def __init__(
        self,
        mels,
        targets,
        indices,
    ):

        self.mels = mels
        self.targets = targets

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):

        index = int(
            self.indices[
                item
            ]
        )

        return (
            torch.from_numpy(
                self.mels[
                    index
                ]
            ),

            torch.from_numpy(
                self.targets[
                    index
                ].astype(
                    np.float32,
                    copy=True,
                )
            ),
        )


class A1CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 32, 3, 1, 1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32, 64, 3, 1, 1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                64, 128, 3, 1, 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                128, 256, 3, 1, 1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.adaptive_pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Linear(
            256,
            NUM_LABELS,
        )

    def forward(
        self,
        x,
    ):

        x = self.features(
            x
        )

        x = self.adaptive_pool(
            x
        )

        x = torch.flatten(
            x,
            1,
        )

        return self.classifier(
            x
        )


def evaluate_a1(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_rows = 0
    total_loss = 0.0

    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for (
            mels,
            targets,
        ) in loader:

            mels = mels.to(
                device
            )

            targets = targets.to(
                device
            )

            # Formal A1 validation parity from Script 57:
            # FP32, not autocast.
            logits = model(
                mels
            )

            loss = criterion(
                logits,
                targets
            )

            probabilities = (
                torch.sigmoid(
                    logits.float()
                )
            )

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

            all_targets.append(
                targets
                .detach()
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

    require(
        total_rows
        == N_VAL,
        (
            "A1 validation "
            "processed 514 rows"
        ),
    )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    probability = np.concatenate(
        all_probabilities,
        axis=0,
    )

    macro_ap, micro_ap, _ = (
        compute_ap(
            y_true,
            probability,
        )
    )

    return {
        "loss":
            total_loss
            / total_rows,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,
    }


def train_a1(
    seed,
    data,
    mels,
    device,
    model_dir,
    run_dir,
):

    banner(
        f"TASK 2 — A1 CNN — SEED {seed}"
    )

    seed_everything(
        seed
    )

    model = A1CNN().to(
        device
    )

    train_dataset = MelDataset(
        mels,
        data[
            "targets"
        ],
        data[
            "train_indices"
        ],
    )

    val_dataset = MelDataset(
        mels,
        data[
            "targets"
        ],
        data[
            "val_indices"
        ],
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=A1_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=A1_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            data[
                "pos_weight"
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=A1_LR,
        weight_decay=
            A1_WEIGHT_DECAY,
    )

    best_score = -float("inf")

    best_epoch = None
    best_state = None
    best_val = None

    bad_epochs = 0

    history = []

    start = time.perf_counter()

    for epoch in range(
        1,
        A1_MAX_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_rows = 0

        for (
            batch_mels,
            targets,
        ) in train_loader:

            batch_mels = (
                batch_mels.to(
                    device
                )
            )

            targets = targets.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_mels
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
                (
                    "A1 TRAIN loss "
                    "is finite"
                ),
            )

            loss.backward()

            optimizer.step()

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

        train_loss = (
            total_loss
            / total_rows
        )

        val = evaluate_a1(
            model,
            val_loader,
            criterion,
            device,
        )

        improved = (
            val[
                "macro_ap"
            ]
            >
            best_score
            + IMPROVEMENT_EPSILON
        )

        if improved:

            best_score = (
                val[
                    "macro_ap"
                ]
            )

            best_epoch = epoch

            best_val = dict(
                val
            )

            best_state = (
                cpu_state_dict(
                    model
                )
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val[
                        "loss"
                    ],

                "val_macro_ap":
                    val[
                        "macro_ap"
                    ],

                "val_micro_ap":
                    val[
                        "micro_ap"
                    ],

                "improved":
                    improved,
            }
        )

        marker = (
            " BEST"
            if improved
            else ""
        )

        print(
            f"A1 epoch {epoch:02d}/15  "
            f"train_loss={train_loss:.6f}  "
            f"val_macro_AP="
            f"{val['macro_ap']:.8f}"
            f"{marker}"
        )

        if (
            bad_epochs
            >= A1_PATIENCE
        ):

            print(
                "A1 early stopping."
            )

            break

    if (
        best_state is None
        or
        best_epoch is None
        or
        best_val is None
    ):

        raise RuntimeError(
            "A1 did not select "
            "a checkpoint"
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    restored = evaluate_a1(
        model,
        val_loader,
        criterion,
        device,
    )

    require(
        np.isclose(
            restored[
                "macro_ap"
            ],
            best_val[
                "macro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "A1 restored best state "
            "reproduces VAL macro AP"
        ),
    )

    checkpoint_path = (
        model_dir
        / "task2_A1_CNN_best.pt"
    )

    epoch_path = (
        run_dir
        / "task2_A1_CNN_epochs.csv"
    )

    checkpoint = {
        "task":
            "task2",

        "variant":
            "A1",

        "model":
            "CNN",

        "final_multiseed":
            True,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "val_loss":
            restored[
                "loss"
            ],

        "threshold":
            0.50,

        "model_state_dict":
            best_state,

        "test_examples_evaluated":
            0,

        "test_target_rows_loaded":
            0,
    }

    atomic_torch_save(
        checkpoint,
        checkpoint_path,
    )

    atomic_csv(
        history,
        epoch_path,
    )

    result = {
        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "checkpoint":
            str(
                checkpoint_path
                .relative_to(
                    ROOT
                )
            ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),

        "epoch_log":
            str(
                epoch_path
                .relative_to(
                    ROOT
                )
            ),

        "runtime_seconds":
            (
                time.perf_counter()
                - start
            ),
    }

    if seed == 42:
        print(
            "Seed-42 reference comparison — "
            "diagnostic only, not an acceptance gate:"
        )
        print(
            f"  original selection AP = "
            f"{SEED42_EXPECTED['task2_val_macro_ap']:.12f}"
        )
        print(
            f"  final rerun AP         = "
            f"{result['val_macro_ap']:.12f}"
        )
        print(
            f"  delta                  = "
            f"{result['val_macro_ap'] - SEED42_EXPECTED['task2_val_macro_ap']:+.12f}"
        )

    return result


# =============================================================================
# 13. SELECTED A4 GAT USED INSIDE TASK 3
# =============================================================================

class A4GAT(nn.Module):

    def __init__(self):

        super().__init__()

        self.node_projection = nn.Linear(
            140,
            64,
        )

        self.gat1 = GATConv(
            64,
            32,
            heads=4,
            concat=True,
            dropout=0.2,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.gat2 = GATConv(
            128,
            32,
            heads=4,
            concat=True,
            dropout=0.2,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.gat3 = GATConv(
            128,
            128,
            heads=4,
            concat=False,
            dropout=0.2,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.classifier = nn.Linear(
            128,
            20,
        )

    def encode_graph(
        self,
        data,
    ):

        x = self.node_projection(
            data.x
        )

        x = F.relu(
            self.gat1(
                x,
                data.edge_index,
            )
        )

        x = F.relu(
            self.gat2(
                x,
                data.edge_index,
            )
        )

        x = self.gat3(
            x,
            data.edge_index,
        )

        return global_mean_pool(
            x,
            data.batch,
        )


# =============================================================================
# 14. M3 FUSION
# =============================================================================

class M3Fusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_projection = nn.Linear(
            768,
            256,
        )

        self.graph_projection = nn.Linear(
            128,
            256,
        )

        self.fusion = nn.Linear(
            512,
            256,
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.classifier = nn.Linear(
            256,
            20,
        )

    def forward(
        self,
        text_embedding,
        graph_embedding,
    ):

        text_prime = (
            self.text_projection(
                text_embedding
            )
        )

        graph_prime = (
            self.graph_projection(
                graph_embedding
            )
        )

        fused = torch.cat(
            [
                text_prime,
                graph_prime,
            ],
            dim=-1,
        )

        fused = self.fusion(
            fused
        )

        fused = F.relu(
            fused
        )

        fused = self.dropout(
            fused
        )

        return self.classifier(
            fused
        )


# =============================================================================
# 15. M3 PHASE-A CACHE
# =============================================================================

def load_m3_phaseA_cache(
    dev_ids,
    targets,
):

    banner(
        "4. LOAD FROZEN M3 PHASE-A REPRESENTATIONS"
    )

    bert_manifest = pd.read_parquet(
        BERT_SEQUENCE_MANIFEST
    )

    gat_manifest = pd.read_parquet(
        GAT_EMBEDDING_MANIFEST
    )

    bert_manifest[
        "track_id"
    ] = (
        bert_manifest[
            "track_id"
        ].astype(str)
    )

    gat_manifest[
        "track_id"
    ] = (
        gat_manifest[
            "track_id"
        ].astype(str)
    )

    require(
        len(bert_manifest)
        == N_DEV,
        (
            "BERT Phase-A manifest "
            "= 4626 rows"
        ),
    )

    require(
        len(gat_manifest)
        == N_DEV,
        (
            "GAT Phase-A manifest "
            "= 4626 rows"
        ),
    )

    require(
        bert_manifest[
            "track_id"
        ].tolist()
        == dev_ids,
        (
            "BERT Phase-A cache "
            "matches canonical dev order"
        ),
    )

    require(
        gat_manifest[
            "track_id"
        ].tolist()
        == dev_ids,
        (
            "GAT Phase-A cache "
            "matches canonical dev order"
        ),
    )

    bert_cache = np.load(
        BERT_SEQUENCE_CACHE,
        mmap_mode="r",
    )

    gat_cache = np.load(
        GAT_EMBEDDING_CACHE,
        mmap_mode="r",
    )

    require(
        bert_cache.shape
        == (
            N_DEV,
            112,
            768,
        ),
        (
            "BERT sequence cache shape "
            "= (4626,112,768)"
        ),
    )

    require(
        gat_cache.shape
        == (
            N_DEV,
            128,
        ),
        (
            "GAT embedding cache shape "
            "= (4626,128)"
        ),
    )

    # M3 uses CLS only.
    text_cls = np.asarray(
        bert_cache[
            :,
            0,
            :
        ],
        dtype=np.float32,
    ).copy()

    graph_embedding = np.asarray(
        gat_cache,
        dtype=np.float32,
    ).copy()

    return {
        "text":
            text_cls,

        "graph":
            graph_embedding,

        "targets":
            targets,
    }


class M3CacheDataset(Dataset):

    def __init__(
        self,
        cache,
        indices,
    ):

        self.cache = cache

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):

        index = int(
            self.indices[
                item
            ]
        )

        return (
            torch.from_numpy(
                self.cache[
                    "text"
                ][
                    index
                ].copy()
            ),

            torch.from_numpy(
                self.cache[
                    "graph"
                ][
                    index
                ].copy()
            ),

            torch.from_numpy(
                self.cache[
                    "targets"
                ][
                    index
                ].astype(
                    np.float32,
                    copy=True,
                )
            ),
        )


def evaluate_m3a(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_rows = 0
    total_loss = 0.0

    y_all = []
    p_all = []

    with torch.no_grad():

        for (
            text,
            graph,
            targets,
        ) in loader:

            text = text.to(
                device
            )

            graph = graph.to(
                device
            )

            targets = targets.to(
                device
            )

            logits = model(
                text,
                graph,
            )

            loss = criterion(
                logits,
                targets,
            )

            probabilities = torch.sigmoid(
                logits
            )

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

            y_all.append(
                targets
                .detach()
                .cpu()
                .numpy()
            )

            p_all.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

    require(
        total_rows
        == N_VAL,
        (
            "M3 Phase-A validation "
            "processed 514 rows"
        ),
    )

    y_true = np.concatenate(
        y_all,
        axis=0,
    )

    probability = np.concatenate(
        p_all,
        axis=0,
    )

    macro_ap, micro_ap, _ = compute_ap(
        y_true,
        probability,
    )

    return {
        "loss":
            total_loss
            / total_rows,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,
    }


def train_m3_phaseA(
    seed,
    cache,
    data,
    device,
    model_dir,
    run_dir,
):

    banner(
        f"TASK 3 — M3 PHASE A — SEED {seed}"
    )

    seed_everything(
        seed
    )

    model = M3Fusion().to(
        device
    )

    train_dataset = M3CacheDataset(
        cache,
        data[
            "train_indices"
        ],
    )

    val_dataset = M3CacheDataset(
        cache,
        data[
            "val_indices"
        ],
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=M3A_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=M3A_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            data[
                "pos_weight"
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=M3A_LR,
        weight_decay=
            M3A_WEIGHT_DECAY,
    )

    best_score = -float("inf")

    best_epoch = None
    best_state = None
    best_val = None

    bad_epochs = 0

    history = []

    start = time.perf_counter()

    for epoch in range(
        1,
        M3A_MAX_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_rows = 0

        for (
            text,
            graph,
            targets,
        ) in train_loader:

            text = text.to(
                device
            )

            graph = graph.to(
                device
            )

            targets = targets.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                text,
                graph,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

        train_loss = (
            total_loss
            / total_rows
        )

        val = evaluate_m3a(
            model,
            val_loader,
            criterion,
            device,
        )

        improved = (
            val[
                "macro_ap"
            ]
            >
            best_score
            + IMPROVEMENT_EPSILON
        )

        if improved:

            best_score = (
                val[
                    "macro_ap"
                ]
            )

            best_epoch = epoch

            best_val = dict(
                val
            )

            best_state = (
                cpu_state_dict(
                    model
                )
            )

            bad_epochs = 0

        else:

            bad_epochs += 1

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val[
                        "loss"
                    ],

                "val_macro_ap":
                    val[
                        "macro_ap"
                    ],

                "val_micro_ap":
                    val[
                        "micro_ap"
                    ],

                "improved":
                    improved,
            }
        )

        marker = (
            " BEST"
            if improved
            else ""
        )

        print(
            f"M3A epoch "
            f"{epoch:02d}/20  "
            f"val_macro_AP="
            f"{val['macro_ap']:.8f}"
            f"{marker}"
        )

        if (
            bad_epochs
            >= M3A_PATIENCE
        ):

            print(
                "M3 Phase-A "
                "early stopping."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "M3 Phase-A "
            "selected no checkpoint"
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    restored = evaluate_m3a(
        model,
        val_loader,
        criterion,
        device,
    )

    require(
        np.isclose(
            restored[
                "macro_ap"
            ],
            best_val[
                "macro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M3 Phase-A restored state "
            "reproduces VAL macro AP"
        ),
    )

    checkpoint_path = (
        model_dir
        / "task3_M3_phaseA_best.pt"
    )

    epoch_path = (
        run_dir
        / "task3_M3_phaseA_epochs.csv"
    )

    atomic_torch_save(
        {
            "task":
                "task3",

            "variant":
                "M3",

            "phase":
                "A",

            "final_multiseed":
                True,

            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "val_macro_ap":
                restored[
                    "macro_ap"
                ],

            "val_micro_ap":
                restored[
                    "micro_ap"
                ],

            "model_state_dict":
                best_state,

            "source_encoder_policy":
                (
                    "fixed selected seed42 "
                    "T1-B + A4 representations"
                ),

            "test_examples_evaluated":
                0,

            "test_target_rows_loaded":
                0,
        },
        checkpoint_path,
    )

    atomic_csv(
        history,
        epoch_path,
    )

    result = {
        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "checkpoint":
            str(
                checkpoint_path
                .relative_to(
                    ROOT
                )
            ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),

        "runtime_seconds":
            (
                time.perf_counter()
                - start
            ),
    }

    if seed == 42:
        print(
            "Seed-42 reference comparison — "
            "diagnostic only, not an acceptance gate:"
        )
        print(
            f"  original selection AP = "
            f"{SEED42_EXPECTED['task3_phaseA_val_macro_ap']:.12f}"
        )
        print(
            f"  final rerun AP         = "
            f"{result['val_macro_ap']:.12f}"
        )
        print(
            f"  delta                  = "
            f"{result['val_macro_ap'] - SEED42_EXPECTED['task3_phaseA_val_macro_ap']:+.12f}"
        )

    return (
        result,
        best_state,
    )


# =============================================================================
# 16. G2 GRAPH LOADING FOR PHASE B
# =============================================================================

EXPECTED_GRAPH_KEYS = {
    "x",
    "edge_index_g1",
    "edge_index_g2",
    "similarity_threshold",
    "nonlocal_only",
    "extra_degree_cap",
    "rule_version",
}


def resolve_graph_path(
    value,
):

    raw = Path(
        value
    )

    candidates = []

    if raw.is_absolute():

        candidates.append(
            raw
        )

    else:

        candidates.extend(
            [
                ROOT / raw,

                GRAPH_MANIFEST.parent
                / raw,

                ROOT
                / "data/processed/graphs"
                / raw.name,
            ]
        )

    matches = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = (
                candidate.resolve()
            )

            matches[
                str(
                    resolved
                )
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            f"Cannot uniquely resolve "
            f"graph path: {value}"
        )

    return next(
        iter(
            matches.values()
        )
    )


def load_graph(
    path,
):

    with np.load(
        path,
        allow_pickle=False,
    ) as graph:

        if set(
            graph.files
        ) != EXPECTED_GRAPH_KEYS:

            raise RuntimeError(
                "Unexpected graph schema: "
                f"{path.name}"
            )

        x = np.asarray(
            graph[
                "x"
            ]
        )

        edge_index = np.asarray(
            graph[
                "edge_index_g2"
            ]
        )

    if (
        x.shape
        != (
            9,
            140,
        )
        or
        x.dtype
        != np.float32
    ):

        raise RuntimeError(
            f"Invalid graph node matrix: "
            f"{path.name}"
        )

    if (
        edge_index.ndim
        != 2
        or
        edge_index.shape[0]
        != 2
        or
        edge_index.dtype
        != np.int64
    ):

        raise RuntimeError(
            f"Invalid G2 edge index: "
            f"{path.name}"
        )

    return Data(
        x=torch.from_numpy(
            x.copy()
        ),

        edge_index=torch.from_numpy(
            edge_index.copy()
        ),
    )


def load_development_graphs(
    dev_ids,
):

    banner(
        "5. LOAD TRAIN+VAL G2 GRAPHS"
    )

    manifest = pd.read_parquet(
        GRAPH_MANIFEST
    )

    manifest = manifest.copy()

    manifest[
        "track_id"
    ] = (
        manifest[
            "track_id"
        ].astype(str)
    )

    manifest[
        "split"
    ] = (
        manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    lookup = manifest.set_index(
        "track_id",
        drop=False,
    )

    graphs = []

    for index, track_id in enumerate(
        dev_ids
    ):

        row = lookup.loc[
            track_id
        ]

        if str(
            row[
                "split"
            ]
        ).lower() == "test":

            raise RuntimeError(
                "Forbidden TEST graph "
                "encountered"
            )

        path = resolve_graph_path(
            str(
                row[
                    "graph_path"
                ]
            )
        )

        graphs.append(
            load_graph(
                path
            )
        )

        number = index + 1

        if (
            number == 1
            or
            number % 500 == 0
            or
            number == N_DEV
        ):

            print(
                f"  loaded "
                f"{number}/{N_DEV}"
            )

    require(
        len(graphs)
        == N_DEV,
        (
            "exactly 4626 TRAIN+VAL "
            "G2 graphs loaded"
        ),
    )

    print(
        "TEST graph files opened: 0"
    )

    return graphs


# =============================================================================
# 17. M3 PHASE-B MODEL
# =============================================================================

class M3PhaseBModel(nn.Module):

    def __init__(
        self,
        bert,
        gnn,
        fusion,
    ):

        super().__init__()

        self.bert = bert
        self.gnn = gnn
        self.fusion = fusion

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        graph_batch,
    ):

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        text_cls = (
            output
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

        graph_embedding = (
            self.gnn.encode_graph(
                graph_batch
            )
        )

        return self.fusion(
            text_cls,
            graph_embedding,
        )


class PhaseBDataset(Dataset):

    def __init__(
        self,
        data,
        graphs,
        indices,
    ):

        self.data = data
        self.graphs = graphs

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):

        index = int(
            self.indices[
                item
            ]
        )

        return (
            torch.from_numpy(
                self.data[
                    "input_ids"
                ][
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "attention_mask"
                ][
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "token_type_ids"
                ][
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            self.graphs[
                index
            ],

            torch.from_numpy(
                self.data[
                    "targets"
                ][
                    index
                ].astype(
                    np.float32,
                    copy=True,
                )
            ),
        )


def collate_phase_b(
    samples,
):

    return (
        torch.stack(
            [
                sample[0]
                for sample
                in samples
            ]
        ),

        torch.stack(
            [
                sample[1]
                for sample
                in samples
            ]
        ),

        torch.stack(
            [
                sample[2]
                for sample
                in samples
            ]
        ),

        Batch.from_data_list(
            [
                sample[3]
                for sample
                in samples
            ]
        ),

        torch.stack(
            [
                sample[4]
                for sample
                in samples
            ]
        ),
    )


def reconstruct_m3_phase_b(
    phase_a_state,
    device,
):

    # Exact selected BERT source.
    config = BertConfig.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    selected_t1 = load_checkpoint(
        SELECTED_T1
    )

    source_state = selected_t1[
        "model_state_dict"
    ]

    bert_state = {
        key[len("bert."):]:
            value

        for key, value
        in source_state.items()

        if key.startswith(
            "bert."
        )
    }

    bert.load_state_dict(
        bert_state,
        strict=True,
    )

    # Exact selected A4 source.
    gnn = A4GAT()

    selected_a4 = load_checkpoint(
        SELECTED_A4
    )

    gnn.load_state_dict(
        selected_a4[
            "model_state_dict"
        ],
        strict=True,
    )

    fusion = M3Fusion()

    fusion.load_state_dict(
        phase_a_state,
        strict=True,
    )

    # Freeze all encoder parameters.
    for parameter in bert.parameters():

        parameter.requires_grad_(
            False
        )

    for parameter in gnn.parameters():

        parameter.requires_grad_(
            False
        )

    # Unfreeze last two BERT layers.
    for layer_index in (
        10,
        11,
    ):

        for parameter in (
            bert.encoder.layer[
                layer_index
            ].parameters()
        ):

            parameter.requires_grad_(
                True
            )

    # Unfreeze complete selected GNN encoder.
    for module in (
        gnn.node_projection,
        gnn.gat1,
        gnn.gat2,
        gnn.gat3,
    ):

        for parameter in module.parameters():

            parameter.requires_grad_(
                True
            )

    # Original A4 classifier stays unused/frozen.
    for parameter in (
        gnn.classifier.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    for parameter in fusion.parameters():

        parameter.requires_grad_(
            True
        )

    model = M3PhaseBModel(
        bert,
        gnn,
        fusion,
    )

    return model.to(
        device
    )


def evaluate_m3b(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_rows = 0
    total_loss = 0.0

    y_all = []
    p_all = []

    with torch.no_grad():

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
            targets,
        ) in loader:

            input_ids = input_ids.to(
                device
            )

            attention_mask = (
                attention_mask.to(
                    device
                )
            )

            token_type_ids = (
                token_type_ids.to(
                    device
                )
            )

            graph_batch = (
                graph_batch.to(
                    device
                )
            )

            targets = targets.to(
                device
            )

            logits = model(
                input_ids,
                attention_mask,
                token_type_ids,
                graph_batch,
            )

            loss = criterion(
                logits,
                targets,
            )

            probability = (
                torch.sigmoid(
                    logits
                )
            )

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

            y_all.append(
                targets
                .detach()
                .cpu()
                .numpy()
            )

            p_all.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    require(
        total_rows
        == N_VAL,
        (
            "M3 Phase-B validation "
            "processed 514 rows"
        ),
    )

    y_true = np.concatenate(
        y_all,
        axis=0,
    )

    probability = np.concatenate(
        p_all,
        axis=0,
    )

    macro_ap, micro_ap, _ = compute_ap(
        y_true,
        probability,
    )

    return {
        "loss":
            total_loss
            / total_rows,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,
    }


def train_m3_phase_b(
    seed,
    phase_a_state,
    phase_a_result,
    data,
    graphs,
    device,
    model_dir,
    run_dir,
):

    banner(
        f"TASK 3 — M3 PHASE B — SEED {seed}"
    )

    seed_everything(
        seed
    )

    model = reconstruct_m3_phase_b(
        phase_a_state,
        device,
    )

    train_dataset = PhaseBDataset(
        data,
        graphs,
        data[
            "train_indices"
        ],
    )

    val_dataset = PhaseBDataset(
        data,
        graphs,
        data[
            "val_indices"
        ],
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=M3B_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        collate_fn=
            collate_phase_b,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=M3B_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=
            collate_phase_b,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            data[
                "pos_weight"
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    # The live reconstruction must reproduce
    # the just-completed Phase-A state.
    initial_val = evaluate_m3b(
        model,
        val_loader,
        criterion,
        device,
    )

    print(
        "Phase-B live starting parity:"
    )

    print(
        f"  Phase-A stored = "
        f"{phase_a_result['val_macro_ap']:.12f}"
    )

    print(
        f"  live model     = "
        f"{initial_val['macro_ap']:.12f}"
    )

    require(
        np.isclose(
            initial_val[
                "macro_ap"
            ],
            phase_a_result[
                "val_macro_ap"
            ],
            rtol=0.0,
            atol=1e-8,
        ),
        (
            "M3 Phase-B live model "
            "reproduces its Phase-A start"
        ),
    )

    bert_parameters = [
        parameter

        for parameter
        in model.bert.parameters()

        if parameter.requires_grad
    ]

    gnn_parameters = [
        parameter

        for parameter
        in model.gnn.parameters()

        if parameter.requires_grad
    ]

    fusion_parameters = [
        parameter

        for parameter
        in model.fusion.parameters()

        if parameter.requires_grad
    ]

    require(
        sum(
            parameter.numel()
            for parameter
            in bert_parameters
        )
        == 14_175_744,
        (
            "M3 Phase-B BERT trainable "
            "parameters = 14,175,744"
        ),
    )

    require(
        sum(
            parameter.numel()
            for parameter
            in gnn_parameters
        )
        == 101_056,
        (
            "M3 Phase-B GNN trainable "
            "parameters = 101,056"
        ),
    )

    require(
        sum(
            parameter.numel()
            for parameter
            in fusion_parameters
        )
        == 366_356,
        (
            "M3 Phase-B fusion trainable "
            "parameters = 366,356"
        ),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    bert_parameters,

                "lr":
                    M3B_LR,

                "weight_decay":
                    M3B_BERT_WEIGHT_DECAY,
            },

            {
                "params":
                    (
                        gnn_parameters
                        +
                        fusion_parameters
                    ),

                "lr":
                    M3B_LR,

                "weight_decay":
                    M3B_GNN_FUSION_WEIGHT_DECAY,
            },
        ]
    )

    best_score = -float("inf")

    best_epoch = None
    best_state = None
    best_val = None

    history = []

    start = time.perf_counter()

    for epoch in range(
        1,
        M3B_EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_rows = 0

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
            targets,
        ) in train_loader:

            input_ids = input_ids.to(
                device
            )

            attention_mask = (
                attention_mask.to(
                    device
                )
            )

            token_type_ids = (
                token_type_ids.to(
                    device
                )
            )

            graph_batch = (
                graph_batch.to(
                    device
                )
            )

            targets = targets.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                input_ids,
                attention_mask,
                token_type_ids,
                graph_batch,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            rows = int(
                targets.shape[0]
            )

            total_rows += rows

            total_loss += (
                float(
                    loss
                    .detach()
                    .cpu()
                )
                * rows
            )

        train_loss = (
            total_loss
            / total_rows
        )

        val = evaluate_m3b(
            model,
            val_loader,
            criterion,
            device,
        )

        improved = (
            val[
                "macro_ap"
            ]
            >
            best_score
            + IMPROVEMENT_EPSILON
        )

        if improved:

            best_score = (
                val[
                    "macro_ap"
                ]
            )

            best_epoch = epoch

            best_val = dict(
                val
            )

            best_state = {
                "bert_layer_10":
                    cpu_state_dict(
                        model.bert
                        .encoder
                        .layer[10]
                    ),

                "bert_layer_11":
                    cpu_state_dict(
                        model.bert
                        .encoder
                        .layer[11]
                    ),

                "gnn":
                    cpu_state_dict(
                        model.gnn
                    ),

                "fusion":
                    cpu_state_dict(
                        model.fusion
                    ),
            }

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val[
                        "loss"
                    ],

                "val_macro_ap":
                    val[
                        "macro_ap"
                    ],

                "val_micro_ap":
                    val[
                        "micro_ap"
                    ],

                "improved":
                    improved,
            }
        )

        marker = (
            " BEST"
            if improved
            else ""
        )

        print(
            f"M3B epoch "
            f"{epoch:02d}/5  "
            f"val_macro_AP="
            f"{val['macro_ap']:.8f}"
            f"{marker}"
        )

    if best_state is None:

        raise RuntimeError(
            "M3 Phase-B selected "
            "no checkpoint"
        )

    model.bert.encoder.layer[
        10
    ].load_state_dict(
        best_state[
            "bert_layer_10"
        ],
        strict=True,
    )

    model.bert.encoder.layer[
        11
    ].load_state_dict(
        best_state[
            "bert_layer_11"
        ],
        strict=True,
    )

    model.gnn.load_state_dict(
        best_state[
            "gnn"
        ],
        strict=True,
    )

    model.fusion.load_state_dict(
        best_state[
            "fusion"
        ],
        strict=True,
    )

    restored = evaluate_m3b(
        model,
        val_loader,
        criterion,
        device,
    )

    require(
        np.isclose(
            restored[
                "macro_ap"
            ],
            best_val[
                "macro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M3 Phase-B restored best "
            "state reproduces VAL AP"
        ),
    )

    checkpoint_path = (
        model_dir
        / "task3_M3_phaseB_best.pt"
    )

    epoch_path = (
        run_dir
        / "task3_M3_phaseB_epochs.csv"
    )

    checkpoint = {
        "task":
            "task3",

        "variant":
            "M3",

        "phase":
            "B",

        "final_multiseed":
            True,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "threshold":
            0.70,

        "selected_T1_source_sha256":
            EXPECTED_SELECTED_T1_SHA256,

        "selected_A4_source_sha256":
            EXPECTED_SELECTED_A4_SHA256,

        "bert_layer_10_state_dict":
            best_state[
                "bert_layer_10"
            ],

        "bert_layer_11_state_dict":
            best_state[
                "bert_layer_11"
            ],

        "gnn_state_dict":
            best_state[
                "gnn"
            ],

        "fusion_state_dict":
            best_state[
                "fusion"
            ],

        "test_examples_evaluated":
            0,

        "test_target_rows_loaded":
            0,
    }

    atomic_torch_save(
        checkpoint,
        checkpoint_path,
    )

    atomic_csv(
        history,
        epoch_path,
    )

    result = {
        "best_epoch":
            best_epoch,

        "val_macro_ap":
            restored[
                "macro_ap"
            ],

        "val_micro_ap":
            restored[
                "micro_ap"
            ],

        "checkpoint":
            str(
                checkpoint_path
                .relative_to(
                    ROOT
                )
            ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),

        "runtime_seconds":
            (
                time.perf_counter()
                - start
            ),
    }

    if seed == 42:
        print(
            "Seed-42 reference comparison — "
            "diagnostic only, not an acceptance gate:"
        )
        print(
            f"  original selection AP = "
            f"{SEED42_EXPECTED['task3_phaseB_val_macro_ap']:.12f}"
        )
        print(
            f"  final rerun AP         = "
            f"{result['val_macro_ap']:.12f}"
        )
        print(
            f"  delta                  = "
            f"{result['val_macro_ap'] - SEED42_EXPECTED['task3_phaseB_val_macro_ap']:+.12f}"
        )

    return result


# =============================================================================
# 18. RUN ONE COMPLETE FINAL SEED
# =============================================================================

def run_seed(
    seed,
    data,
    mels,
    m3_cache,
    graphs,
    device,
):

    banner(
        f"FINAL MULTISEED REPLICATE — SEED {seed}"
    )

    model_final = (
        MODEL_ROOT
        / f"seed{seed}"
    )

    model_temp = (
        MODEL_ROOT
        / f"seed{seed}.__tmp__"
    )

    run_final = (
        RUN_ROOT
        / f"seed{seed}"
    )

    run_temp = (
        RUN_ROOT
        / f"seed{seed}.__tmp__"
    )

    for path in (
        model_final,
        model_temp,
        run_final,
        run_temp,
    ):

        require(
            not path.exists(),
            (
                f"output path unused: "
                f"{path.relative_to(ROOT)}"
            ),
        )

    model_temp.mkdir(
        parents=True,
        exist_ok=False,
    )

    run_temp.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:

        task1, task1_model = train_task1(
            seed,
            data,
            device,
            model_temp,
            run_temp,
        )

        # Task-1 model is not Task-3's source;
        # Task 3 intentionally retains its frozen selected
        # seed-42 encoder artifact.
        del task1_model

        torch.cuda.empty_cache()

        task2 = train_a1(
            seed,
            data,
            mels,
            device,
            model_temp,
            run_temp,
        )

        torch.cuda.empty_cache()

        (
            task3_phase_a,
            phase_a_state,
        ) = train_m3_phaseA(
            seed,
            m3_cache,
            data,
            device,
            model_temp,
            run_temp,
        )

        torch.cuda.empty_cache()

        task3_phase_b = (
            train_m3_phase_b(
                seed,
                phase_a_state,
                task3_phase_a,
                data,
                graphs,
                device,
                model_temp,
                run_temp,
            )
        )

        summary = {
            "artifact_type":
                "final_multiseed_seed_run",

            "seed":
                seed,

            "task1":
                task1,

            "task2":
                task2,

            "task3_phaseA":
                task3_phase_a,

            "task3":
                task3_phase_b,

            "thresholds": {
                "task1":
                    0.70,

                "task2":
                    0.50,

                "task3":
                    0.70,
            },

            "task3_encoder_policy":
                (
                    "fixed selected T1-B seed42 "
                    "and A4 GAT/G2 seed42 artifacts "
                    "across all fusion seeds"
                ),

            "test_target_rows_loaded":
                0,

            "test_examples_evaluated":
                0,

            "threshold_retuning":
                False,
        }

        summary_path = (
            run_temp
            / "seed_summary.json"
        )

        atomic_json(
            summary,
            summary_path,
        )

        # Commit complete seed only after every stage passes.
        os.replace(
            model_temp,
            model_final,
        )

        os.replace(
            run_temp,
            run_final,
        )

    except Exception:

        print()
        print(
            "Seed run failed."
        )

        print(
            "Temporary outputs were retained "
            "for inspection:"
        )

        if model_temp.exists():
            print(
                f"  {model_temp}"
            )

        if run_temp.exists():
            print(
                f"  {run_temp}"
            )

        raise

    banner(
        f"SEED {seed} COMPLETE"
    )

    print(
        f"T1-B VAL macro AP: "
        f"{task1['val_macro_ap']:.8f}"
    )

    print(
        f"A1   VAL macro AP: "
        f"{task2['val_macro_ap']:.8f}"
    )

    print(
        f"M3A  VAL macro AP: "
        f"{task3_phase_a['val_macro_ap']:.8f}"
    )

    print(
        f"M3B  VAL macro AP: "
        f"{task3_phase_b['val_macro_ap']:.8f}"
    )

    print()

    print(
        "TEST target rows loaded: 0"
    )

    print(
        "TEST inference examples: 0"
    )


# =============================================================================
# 19. FINAL TRAINING MANIFEST
# =============================================================================

def create_final_manifest() -> None:

    if FINAL_MANIFEST.exists():

        raise RuntimeError(
            "Final multi-seed training "
            "manifest already exists"
        )

    seed_summaries = {}

    for seed in FINAL_SEEDS:

        path = (
            RUN_ROOT
            / f"seed{seed}"
            / "seed_summary.json"
        )

        require(
            path.is_file(),
            (
                f"seed {seed} summary exists"
            ),
        )

        seed_summaries[
            str(seed)
        ] = load_json(
            path
        )

    artifact = {
        "artifact_type":
            "final_multiseed_training_manifest",

        "version":
            1,

        "seeds":
            list(
                FINAL_SEEDS
            ),

        "winning_configurations": {
            "task1":
                "T1-B BERT",

            "task2":
                "A1 CNN",

            "task3":
                "M3 Early Fusion Phase B",
        },

        "frozen_thresholds": {
            "task1":
                0.70,

            "task2":
                0.50,

            "task3":
                0.70,
        },

        "threshold_lock_sha256":
            EXPECTED_THRESHOLD_LOCK_SHA256,

        "task3_encoder_policy":
            (
                "selected T1-B seed42 and "
                "selected A4 GAT/G2 seed42 "
                "remain fixed encoder sources "
                "for every Task-3 final seed"
            ),

        "runs":
            seed_summaries,

        "test_target_rows_loaded":
            0,

        "test_examples_evaluated":
            0,

        "ready_for_final_test":
            True,
    }

    atomic_json(
        artifact,
        FINAL_MANIFEST,
    )

    banner(
        "FINAL THREE-SEED TRAINING COMPLETE"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Completed final seeds:"
    )

    print(
        "  42"
    )

    print(
        "  1337"
    )

    print(
        "  2026"
    )

    print()

    print(
        "TEST inference performed: 0"
    )

    print(
        "TEST target rows loaded:  0"
    )

    print()

    print(
        "Final training manifest:"
    )

    print(
        f"  "
        f"{FINAL_MANIFEST.relative_to(ROOT)}"
    )

    print()

    print(
        "Manifest SHA256:"
    )

    print(
        f"  "
        f"{sha256_file(FINAL_MANIFEST)}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  Final TEST evaluation and benchmarks."
    )

    print()

    print(
        "STOP HERE."
    )


# =============================================================================
# 20. MAIN
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--seed",
        type=int,
        choices=list(
            FINAL_SEEDS
        ),
    )

    group.add_argument(
        "--all-seeds",
        action="store_true",
    )

    args = parser.parse_args()

    banner(
        "SCRIPT 58 — FINAL THREE-SEED WINNER TRAINING"
    )

    print(
        "Winning configurations are frozen."
    )

    print(
        "No architecture/hyperparameter "
        "selection occurs here."
    )

    print()

    print(
        "TEST labels:       NO"
    )

    print(
        "TEST inference:    NO"
    )

    print(
        "Threshold tuning:  NO"
    )

    verify_global_dependencies()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        "CUDA:"
    )

    print(
        f"  "
        f"{torch.cuda.get_device_name(device)}"
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load development inputs once.
    data = load_development_data()

    # Load reusable Task-2 data once.
    mels = load_development_mels(
        data[
            "track_ids"
        ]
    )

    # Load frozen Phase-A representations once.
    m3_cache = load_m3_phaseA_cache(
        data[
            "track_ids"
        ],
        data[
            "targets"
        ],
    )

    # Load raw G2 graphs once for M3 Phase B.
    graphs = load_development_graphs(
        data[
            "track_ids"
        ]
    )

    if args.all_seeds:

        seeds = FINAL_SEEDS

    else:

        seeds = (
            args.seed,
        )

    try:

        for seed in seeds:

            completed_model_dir = (
                MODEL_ROOT
                / f"seed{seed}"
            )

            completed_summary = (
                RUN_ROOT
                / f"seed{seed}"
                / "seed_summary.json"
            )

            model_complete = (
                completed_model_dir.is_dir()
            )

            summary_complete = (
                completed_summary.is_file()
            )

            if (
                model_complete
                and
                summary_complete
            ):

                banner(
                    f"SEED {seed} ALREADY COMPLETE"
                )

                print(
                    "Existing completed seed retained; "
                    "training will not be repeated."
                )

                continue

            if (
                model_complete
                !=
                summary_complete
            ):

                raise RuntimeError(
                    f"seed {seed} has inconsistent "
                    "partial final outputs; inspect "
                    "before continuing"
                )

            run_seed(
                seed,
                data,
                mels,
                m3_cache,
                graphs,
                device,
            )

        # Only create the final manifest when all
        # three complete seed runs exist.
        all_complete = all(
            (
                RUN_ROOT
                / f"seed{seed}"
                / "seed_summary.json"
            ).is_file()

            for seed in FINAL_SEEDS
        )

        if all_complete:

            create_final_manifest()

        else:

            banner(
                "SCRIPT 58 PARTIAL COMPLETION"
            )

            print(
                "Requested seed completed."
            )

            print(
                "Final TEST is NOT yet allowed."
            )

            print(
                "Complete all seeds "
                "{42,1337,2026} first."
            )

    except Exception as exc:

        banner(
            "SCRIPT 58 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "Do NOT begin final TEST."
        )

        return 1

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )