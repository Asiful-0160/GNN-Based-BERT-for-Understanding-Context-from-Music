#!/usr/bin/env python3
"""
Script 50 — Formal Task-3 M3 Early-Fusion Phase-A Training
==========================================================

Frozen design
-------------
Task-3 Phase-A design SHA256:

    0a9d50413b65aeed58d02d6a2dff70d99363ae25fee115d30bc9749f54e1c9ef

M3:
    BERT CLS 768 -> Linear(768,256)
    GAT      128 -> Linear(128,256)

    concat -> 512
           -> Linear(512,256)
           -> ReLU
           -> Dropout(0.2)
           -> Linear(256,20)

Phase A:
    frozen BERT representation
    frozen GAT representation

    AdamW
    lr           = 1e-3
    weight_decay = 1e-4
    batch_size   = 32
    max_epochs   = 20
    patience     = 3
    seed         = 42

Loss:
    BCEWithLogitsLoss(TRAIN-only pos_weight)

Primary selection metric:
    validation macro mean per-label Average Precision

STRICT TEST POLICY
------------------
TEST representations:  NOT PRESENT in Task-3 caches
TEST labels:           NOT LOADED
TEST inference:        NOT RUN
TEST metrics:          NOT RUN
Threshold tuning:      NOT RUN

This is a FORMAL development run.

The script refuses to start if a previous formal M3 Phase-A seed-42
run already exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 0. PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Frozen Task-3 design
# -----------------------------------------------------------------------------

DESIGN_PATH = (
    ROOT
    / "results/runs/task3/task3_phaseA_design.json"
)


# -----------------------------------------------------------------------------
# Frozen original artifacts
# -----------------------------------------------------------------------------

PRETRAINING_LOCK = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

TARGETS = (
    ROOT
    / "data/processed/labels/musiccaps_targets.parquet"
)

T1_CHECKPOINT = (
    ROOT
    / "models/task1_candidates/"
    "task1_T1-B_seed42_20260828T182131Z_best.pt"
)

A4_CHECKPOINT = (
    ROOT
    / "models/task2_candidates/"
    "task2_A4_GAT_G2_seed42_20260828T193949Z_best.pt"
)


# -----------------------------------------------------------------------------
# Frozen Script-47 BERT cache
# -----------------------------------------------------------------------------

BERT_DIR = (
    ROOT
    / "data/processed/bert_seq_frozen"
)

BERT_SEQUENCE = (
    BERT_DIR
    / "bert_sequence_trainval.npy"
)

BERT_MASK = (
    BERT_DIR
    / "attention_mask_trainval.npy"
)

BERT_MANIFEST = (
    BERT_DIR
    / "bert_seq_manifest.parquet"
)

BERT_LOCK = (
    BERT_DIR
    / "bert_seq_frozen_hashes.json"
)


# -----------------------------------------------------------------------------
# Frozen Script-48 GAT cache
# -----------------------------------------------------------------------------

GAT_DIR = (
    ROOT
    / "data/processed/gat_g2_frozen"
)

GAT_EMBEDDING = (
    GAT_DIR
    / "gat_g2_embedding_trainval.npy"
)

GAT_MANIFEST = (
    GAT_DIR
    / "gat_g2_manifest.parquet"
)

GAT_LOCK = (
    GAT_DIR
    / "gat_g2_frozen_hashes.json"
)


# -----------------------------------------------------------------------------
# Formal Task-3 output locations
# -----------------------------------------------------------------------------

MODEL_DIR = (
    ROOT
    / "models/task3_candidates"
)

RUN_DIR = (
    ROOT
    / "results/runs/task3"
)


# =============================================================================
# 1. IMMUTABLE HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "0a9d50413b65aeed58d02d6a2dff70d99363ae25fee115d30bc9749f54e1c9ef"
)

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_TARGETS_SHA256 = (
    "a6f80bb42b0b48b2c47dac5f27ad2fd66f8893d83bd5d4e552d356cee3547056"
)

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e"
)


# Script 47
EXPECTED_BERT_SEQUENCE_SHA256 = (
    "a1c723ec939392b8b0d9bafadb8327d8a0967ecb9bd8e89ffcd559306633e133"
)

EXPECTED_BERT_MASK_SHA256 = (
    "633b11d5ca9146000da84b282f9081c6ff33c1279486c2e3112e4c89137d197b"
)

EXPECTED_BERT_MANIFEST_SHA256 = (
    "ebd86077489b716abc8768247f550890ca8e5a26b8f0967c32882c22c4e85d2f"
)

EXPECTED_BERT_LOCK_SHA256 = (
    "c57cc66666d4531c02c6821964585b5511e516872f1a922cf12e01487d464228"
)


# Script 48
EXPECTED_GAT_EMBEDDING_SHA256 = (
    "3d39a343d1acc678576f9dd3d88b088ca18a81017190246b0b32841292b92852"
)

EXPECTED_GAT_MANIFEST_SHA256 = (
    "a029c12ec6a98922c98f373bbc263b6c529825b56b1446a14f659f6854ed371a"
)

EXPECTED_GAT_LOCK_SHA256 = (
    "3ce775a3bb89532554883c11ac8f0746e2f3f4a7f983a2e92d30ca18df4eb5b0"
)


# =============================================================================
# 2. FROZEN EXPERIMENT CONTRACT
# =============================================================================

N_TRAIN = 4112
N_VAL = 514
N_DEV = 4626

NUM_LABELS = 20

BERT_LENGTH = 112
BERT_DIM = 768

GAT_DIM = 128

SHARED_DIM = 256

SEED = 42

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 32

MAX_EPOCHS = 20
PATIENCE = 3

DROPOUT = 0.2


# Frozen Phase-A baselines
M1_VAL_MACRO_AP = (
    0.8405750022914198
)

M2_VAL_MACRO_AP = (
    0.18752693895679443
)

PHASE_B_REFERENCE = max(
    M1_VAL_MACRO_AP,
    M2_VAL_MACRO_AP,
)


# =============================================================================
# 3. REPORTING
# =============================================================================

FAILURES: list[str] = []


def banner(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(message: str) -> None:

    print(
        f"PASS  {message}"
    )


def failed(message: str) -> None:

    print(
        f"FAIL  {message}"
    )

    FAILURES.append(
        message
    )


def check(
    condition: bool,
    message: str,
) -> bool:

    if condition:

        passed(
            message
        )

        return True

    failed(
        message
    )

    return False


# =============================================================================
# 4. BASIC HELPERS
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def write_json_atomic(
    path: Path,
    obj: Any,
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if temp.exists():

        raise RuntimeError(
            f"Temporary JSON already exists: {temp}"
        )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        f.write(
            "\n"
        )

    os.replace(
        temp,
        path,
    )


def save_checkpoint_atomic(
    checkpoint: dict[str, Any],
    path: Path,
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if temp.exists():

        temp.unlink()

    torch.save(
        checkpoint,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def load_checkpoint(
    path: Path,
) -> dict[str, Any]:

    try:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        obj = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        obj,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not a checkpoint dictionary"
        )

    return obj


def cpu_state_dict(
    model: nn.Module,
) -> dict[str, torch.Tensor]:

    return {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }


def to_numpy_1d(
    value: Any,
) -> np.ndarray:

    if torch.is_tensor(
        value
    ):

        value = (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value,
        dtype=np.float64,
    ).reshape(-1)


# =============================================================================
# 5. REPRODUCIBILITY
# =============================================================================

def configure_reproducibility() -> None:

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

        if hasattr(
            torch.backends.cuda.matmul,
            "allow_tf32",
        ):

            torch.backends.cuda.matmul.allow_tf32 = False

        if hasattr(
            torch.backends.cudnn,
            "allow_tf32",
        ):

            torch.backends.cudnn.allow_tf32 = False

        torch.backends.cudnn.benchmark = False

    try:

        torch.set_float32_matmul_precision(
            "highest"
        )

    except Exception:

        pass


# =============================================================================
# 6. EXACT FROZEN M3 ARCHITECTURE
# =============================================================================

class M3EarlyFusion(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.text_projection = (
            nn.Linear(
                BERT_DIM,
                SHARED_DIM,
            )
        )

        self.graph_projection = (
            nn.Linear(
                GAT_DIM,
                SHARED_DIM,
            )
        )

        self.fusion = (
            nn.Linear(
                SHARED_DIM * 2,
                SHARED_DIM,
            )
        )

        self.dropout = (
            nn.Dropout(
                DROPOUT
            )
        )

        self.classifier = (
            nn.Linear(
                SHARED_DIM,
                NUM_LABELS,
            )
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:

        text_prime = (
            self.text_projection(
                text_cls
            )
        )

        graph_prime = (
            self.graph_projection(
                graph_embedding
            )
        )

        z = torch.cat(
            [
                text_prime,
                graph_prime,
            ],
            dim=-1,
        )

        z = self.fusion(
            z
        )

        z = F.relu(
            z
        )

        z = self.dropout(
            z
        )

        return self.classifier(
            z
        )


# =============================================================================
# 7. OUTPUT PROTECTION
# =============================================================================

def verify_no_prior_formal_run() -> None:

    banner(
        "1. FORMAL-RUN OUTPUT PROTECTION"
    )

    existing: list[Path] = []

    patterns = (
        "task3_M3_phaseA_seed42_*",
        "task3_m3_phaseA_seed42_*",
    )

    for base in (
        MODEL_DIR,
        RUN_DIR,
    ):

        if not base.exists():

            continue

        for pattern in patterns:

            existing.extend(
                base.glob(
                    pattern
                )
            )

    unique_existing = sorted(
        {
            p.resolve()
            for p in existing
        }
    )

    if unique_existing:

        print(
            "Existing M3 Phase-A seed-42 "
            "formal artifacts:"
        )

        for path in unique_existing:

            print(
                f"  {path}"
            )

    check(
        len(
            unique_existing
        )
        == 0,
        (
            "no previous formal M3 "
            "Phase-A seed-42 run exists"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Formal-run collision detected. "
            "Refusing to create another M3 "
            "Phase-A seed-42 result."
        )


# =============================================================================
# 8. VERIFY FROZEN DESIGN
# =============================================================================

def verify_design() -> dict[str, Any]:

    banner(
        "2. FROZEN TASK-3 DESIGN"
    )

    check(
        DESIGN_PATH.is_file(),
        (
            "Task-3 Phase-A design "
            "artifact exists"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Task-3 design is missing"
        )

    observed = sha256_file(
        DESIGN_PATH
    )

    print(
        f"Observed design SHA256:"
    )

    print(
        f"  {observed}"
    )

    print(
        f"Expected design SHA256:"
    )

    print(
        f"  {EXPECTED_DESIGN_SHA256}"
    )

    check(
        observed
        == EXPECTED_DESIGN_SHA256,
        (
            "Task-3 Phase-A design "
            "SHA256 matches frozen design"
        ),
    )

    design = load_json(
        DESIGN_PATH
    )

    check(
        design.get(
            "artifact_type"
        )
        ==
        "task3_phaseA_design_lock",
        (
            "design artifact type "
            "is correct"
        ),
    )

    check(
        design.get(
            "design_frozen_before_any_fusion_result"
        )
        is True,
        (
            "design was frozen before "
            "fusion results"
        ),
    )

    check(
        design.get(
            "test_examples_used"
        )
        == 0,
        (
            "design records zero TEST "
            "examples used"
        ),
    )

    check(
        design.get(
            "test_target_rows_loaded"
        )
        == 0,
        (
            "design records zero TEST "
            "target rows loaded"
        ),
    )

    check(
        design.get(
            "threshold_tuning_used"
        )
        is False,
        (
            "design records no "
            "threshold tuning"
        ),
    )

    m3 = design.get(
        "m3",
        {},
    )

    check(
        m3.get(
            "text_projection"
        )
        == "Linear(768,256)",
        "M3 text projection is frozen",
    )

    check(
        m3.get(
            "graph_projection"
        )
        == "Linear(128,256)",
        "M3 graph projection is frozen",
    )

    check(
        m3.get(
            "fusion"
        )
        == "Linear(512,256)",
        "M3 fusion layer is frozen",
    )

    check(
        float(
            m3.get(
                "dropout",
                -1,
            )
        )
        == DROPOUT,
        (
            "M3 dropout is frozen "
            "at 0.2"
        ),
    )

    check(
        m3.get(
            "classifier"
        )
        == "Linear(256,20)",
        "M3 classifier is frozen",
    )

    check(
        int(
            m3.get(
                "trainable_parameters",
                -1,
            )
        )
        == 366_356,
        (
            "M3 trainable parameter "
            "count is frozen at 366,356"
        ),
    )

    training = design.get(
        "phase_a_training",
        {},
    )

    expected_training = {
        "seed":
            SEED,

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "early_stop_patience":
            PATIENCE,

        "scheduler":
            None,
    }

    for key, expected in (
        expected_training.items()
    ):

        check(
            training.get(
                key
            )
            == expected,
            (
                f"Phase-A '{key}' "
                f"matches frozen value "
                f"{expected!r}"
            ),
        )

    check(
        training.get(
            "encoders"
        )
        == "frozen",
        (
            "Phase-A encoders are "
            "frozen"
        ),
    )

    check(
        training.get(
            "threshold_tuning_during_training"
        )
        is False,
        (
            "threshold tuning during "
            "training is disabled"
        ),
    )

    phase_b = design.get(
        "phase_b_gate",
        {},
    )

    check(
        phase_b.get(
            "application"
        )
        == "per_variant",
        (
            "Phase-B gate is frozen "
            "per fusion variant"
        ),
    )

    check(
        bool(
            phase_b.get(
                "strict"
            )
        ),
        (
            "Phase-B comparison is "
            "strictly greater-than"
        ),
    )

    check(
        np.isclose(
            float(
                phase_b.get(
                    "effective_reference_to_beat"
                )
            ),
            PHASE_B_REFERENCE,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "Phase-B reference remains "
            "the frozen M1 score"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen Task-3 design validation failed"
        )

    return design


# =============================================================================
# 9. VERIFY FROZEN DEPENDENCIES
# =============================================================================

def verify_frozen_dependencies() -> None:

    banner(
        "3. FROZEN DEPENDENCY IDENTITIES"
    )

    expected = {
        PRETRAINING_LOCK:
            EXPECTED_PRETRAINING_LOCK_SHA256,

        TARGETS:
            EXPECTED_TARGETS_SHA256,

        T1_CHECKPOINT:
            EXPECTED_T1_SHA256,

        A4_CHECKPOINT:
            EXPECTED_A4_SHA256,

        BERT_SEQUENCE:
            EXPECTED_BERT_SEQUENCE_SHA256,

        BERT_MASK:
            EXPECTED_BERT_MASK_SHA256,

        BERT_MANIFEST:
            EXPECTED_BERT_MANIFEST_SHA256,

        BERT_LOCK:
            EXPECTED_BERT_LOCK_SHA256,

        GAT_EMBEDDING:
            EXPECTED_GAT_EMBEDDING_SHA256,

        GAT_MANIFEST:
            EXPECTED_GAT_MANIFEST_SHA256,

        GAT_LOCK:
            EXPECTED_GAT_LOCK_SHA256,
    }

    for path in expected:

        check(
            path.is_file(),
            (
                "artifact exists: "
                f"{path.relative_to(ROOT)}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Required frozen dependency missing"
        )

    for path, expected_sha in (
        expected.items()
    ):

        observed = sha256_file(
            path
        )

        print(
            f"{path.name}:"
        )

        print(
            f"  {observed}"
        )

        check(
            observed
            == expected_sha,
            (
                f"{path.name} SHA256 "
                "matches frozen identity"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen dependency hash validation failed"
        )


# =============================================================================
# 10. LOAD / ALIGN DEVELOPMENT TARGETS
# =============================================================================

def load_aligned_development_data():

    banner(
        "4. TRAIN+VAL DATA ALIGNMENT"
    )

    bert_manifest = (
        pd.read_parquet(
            BERT_MANIFEST
        )
    )

    gat_manifest = (
        pd.read_parquet(
            GAT_MANIFEST
        )
    )

    check(
        len(
            bert_manifest
        )
        == N_DEV,
        (
            "BERT manifest contains "
            "exactly 4626 rows"
        ),
    )

    check(
        len(
            gat_manifest
        )
        == N_DEV,
        (
            "GAT manifest contains "
            "exactly 4626 rows"
        ),
    )

    bert_ids = (
        bert_manifest[
            "track_id"
        ]
        .astype(str)
        .tolist()
    )

    gat_ids = (
        gat_manifest[
            "track_id"
        ]
        .astype(str)
        .tolist()
    )

    check(
        bert_ids
        == gat_ids,
        (
            "BERT and GAT track IDs "
            "match row-for-row"
        ),
    )

    bert_splits = (
        bert_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
        .tolist()
    )

    gat_splits = (
        gat_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
        .tolist()
    )

    check(
        bert_splits
        == gat_splits,
        (
            "BERT and GAT split values "
            "match row-for-row"
        ),
    )

    counts = (
        pd.Series(
            bert_splits
        )
        .value_counts()
        .to_dict()
    )

    print(
        f"Development split counts: "
        f"{counts}"
    )

    check(
        counts
        == {
            "train": N_TRAIN,
            "val": N_VAL,
        },
        (
            "development cache contains "
            "4112 TRAIN + 514 VAL"
        ),
    )

    check(
        "test"
        not in set(
            bert_splits
        ),
        (
            "development representation "
            "manifests contain zero TEST rows"
        ),
    )

    # -------------------------------------------------------------------------
    # Read actual labels for TRAIN + VAL only.
    # -------------------------------------------------------------------------

    y_columns = [
        f"y_{i:02d}"
        for i in range(
            NUM_LABELS
        )
    ]

    target_dataset = (
        pads.dataset(
            str(
                TARGETS
            ),
            format="parquet",
        )
    )

    dev_filter = (
        (pads.field("split") == "train")
        |
        (pads.field("split") == "val")
    )

    table = (
        target_dataset.to_table(
            columns=[
                "track_id",
                "split",
                *y_columns,
            ],
            filter=dev_filter,
        )
    )

    targets = (
        table.to_pandas()
    )

    targets["track_id"] = (
        targets[
            "track_id"
        ].astype(str)
    )

    targets["split"] = (
        targets[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    target_counts = (
        targets[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"Target rows loaded: "
        f"{target_counts}"
    )

    check(
        len(
            targets
        )
        == N_DEV,
        (
            "exactly 4626 TRAIN+VAL "
            "target rows loaded"
        ),
    )

    check(
        target_counts
        == {
            "train": N_TRAIN,
            "val": N_VAL,
        },
        (
            "loaded target values contain "
            "TRAIN+VAL only"
        ),
    )

    check(
        "test"
        not in set(
            targets[
                "split"
            ]
        ),
        (
            "TEST target rows loaded = 0"
        ),
    )

    lookup = (
        targets
        .set_index(
            "track_id",
            drop=False,
        )
    )

    check(
        set(
            bert_ids
        )
        ==
        set(
            targets[
                "track_id"
            ]
        ),
        (
            "representation and target "
            "track-ID sets are identical"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Development target identity mismatch"
        )

    aligned_targets = (
        lookup.loc[
            bert_ids
        ]
        .reset_index(
            drop=True
        )
    )

    check(
        aligned_targets[
            "track_id"
        ].tolist()
        == bert_ids,
        (
            "target row i matches "
            "representation row i"
        ),
    )

    check(
        aligned_targets[
            "split"
        ].tolist()
        == bert_splits,
        (
            "target split row i matches "
            "representation split row i"
        ),
    )

    target_matrix = (
        aligned_targets[
            y_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    check(
        target_matrix.shape
        == (
            N_DEV,
            NUM_LABELS,
        ),
        (
            "aligned target matrix is "
            "exactly (4626,20)"
        ),
    )

    unique_targets = set(
        np.unique(
            target_matrix
        ).tolist()
    )

    check(
        unique_targets.issubset(
            {
                0.0,
                1.0,
            }
        ),
        (
            "development targets are binary"
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen representations.
    # -------------------------------------------------------------------------

    bert_sequence = (
        np.load(
            BERT_SEQUENCE,
            mmap_mode="r",
            allow_pickle=False,
        )
    )

    gat_embedding = (
        np.load(
            GAT_EMBEDDING,
            mmap_mode="r",
            allow_pickle=False,
        )
    )

    check(
        bert_sequence.shape
        == (
            N_DEV,
            BERT_LENGTH,
            BERT_DIM,
        ),
        (
            "BERT cache shape is "
            "(4626,112,768)"
        ),
    )

    check(
        gat_embedding.shape
        == (
            N_DEV,
            GAT_DIM,
        ),
        (
            "GAT cache shape is "
            "(4626,128)"
        ),
    )

    check(
        bert_sequence.dtype
        == np.float32,
        "BERT cache dtype is float32",
    )

    check(
        gat_embedding.dtype
        == np.float32,
        "GAT cache dtype is float32",
    )

    split_array = np.asarray(
        bert_splits
    )

    train_indices = np.flatnonzero(
        split_array == "train"
    )

    val_indices = np.flatnonzero(
        split_array == "val"
    )

    check(
        len(
            train_indices
        )
        == N_TRAIN,
        (
            "TRAIN indices = 4112"
        ),
    )

    check(
        len(
            val_indices
        )
        == N_VAL,
        (
            "VAL indices = 514"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Development data validation failed"
        )

    return (
        bert_sequence,
        gat_embedding,
        target_matrix,
        train_indices,
        val_indices,
    )


# =============================================================================
# 11. TRAIN-ONLY POS_WEIGHT
# =============================================================================

def obtain_pos_weight(
    targets: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:

    banner(
        "5. TRAIN-ONLY POS_WEIGHT"
    )

    train_targets = (
        targets[
            train_indices
        ].astype(
            np.float64
        )
    )

    positive = (
        train_targets.sum(
            axis=0
        )
    )

    negative = (
        N_TRAIN
        - positive
    )

    check(
        bool(
            (
                positive > 0
            ).all()
        ),
        (
            "all labels have positive "
            "TRAIN support"
        ),
    )

    weight = np.minimum(
        negative / positive,
        10.0,
    )

    t1 = load_checkpoint(
        T1_CHECKPOINT
    )

    a4 = load_checkpoint(
        A4_CHECKPOINT
    )

    t1_weight = to_numpy_1d(
        t1[
            "pos_weight"
        ]
    )

    a4_weight = to_numpy_1d(
        a4[
            "pos_weight"
        ]
    )

    check(
        np.allclose(
            weight,
            t1_weight,
            rtol=1e-6,
            atol=1e-6,
        ),
        (
            "TRAIN-only pos_weight "
            "matches T1-B"
        ),
    )

    check(
        np.allclose(
            weight,
            a4_weight,
            rtol=1e-6,
            atol=1e-6,
        ),
        (
            "TRAIN-only pos_weight "
            "matches A4"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Class-weight validation failed"
        )

    print(
        "PASS  shared frozen TRAIN-only "
        "pos_weight confirmed"
    )

    return weight.astype(
        np.float32
    )


# =============================================================================
# 12. DATASET
# =============================================================================

class M3Dataset(Dataset):

    def __init__(
        self,
        bert_sequence,
        gat_embedding,
        targets: np.ndarray,
        indices: np.ndarray,
    ) -> None:

        self.bert_sequence = (
            bert_sequence
        )

        self.gat_embedding = (
            gat_embedding
        )

        self.targets = (
            targets
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self) -> int:

        return len(
            self.indices
        )

    def __getitem__(
        self,
        item: int,
    ):

        index = int(
            self.indices[
                item
            ]
        )

        # M3 requires only the final-layer CLS vector.
        text_cls = np.array(
            self.bert_sequence[
                index,
                0,
                :
            ],
            dtype=np.float32,
            copy=True,
        )

        graph = np.array(
            self.gat_embedding[
                index
            ],
            dtype=np.float32,
            copy=True,
        )

        target = np.array(
            self.targets[
                index
            ],
            dtype=np.float32,
            copy=True,
        )

        return (
            torch.from_numpy(
                text_cls
            ),
            torch.from_numpy(
                graph
            ),
            torch.from_numpy(
                target
            ),
        )


# =============================================================================
# 13. DATALOADERS
# =============================================================================

def build_loaders(
    bert_sequence,
    gat_embedding,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
):

    banner(
        "6. FORMAL TRAIN / VAL DATALOADERS"
    )

    train_dataset = (
        M3Dataset(
            bert_sequence,
            gat_embedding,
            targets,
            train_indices,
        )
    )

    val_dataset = (
        M3Dataset(
            bert_sequence,
            gat_embedding,
            targets,
            val_indices,
        )
    )

    check(
        len(
            train_dataset
        )
        == N_TRAIN,
        "TRAIN dataset contains 4112 rows",
    )

    check(
        len(
            val_dataset
        )
        == N_VAL,
        "VAL dataset contains 514 rows",
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    )

    print(
        f"TRAIN batches: "
        f"{len(train_loader)}"
    )

    print(
        f"VAL batches:   "
        f"{len(val_loader)}"
    )

    print(
        "TEST loader:    NOT CREATED"
    )

    if FAILURES:

        raise RuntimeError(
            "DataLoader construction failed"
        )

    return (
        train_loader,
        val_loader,
    )


# =============================================================================
# 14. MODEL VALIDATION
# =============================================================================

def build_model(
    device: torch.device,
) -> M3EarlyFusion:

    banner(
        "7. FORMAL M3 MODEL"
    )

    configure_reproducibility()

    model = (
        M3EarlyFusion()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    check(
        trainable_parameters
        == 366_356,
        (
            "formal M3 contains exactly "
            "366,356 trainable parameters"
        ),
    )

    # There are no BERT or GAT modules in this model.
    module_names = [
        name.lower()
        for name, _
        in model.named_modules()
    ]

    check(
        not any(
            "bert"
            in name
            for name
            in module_names
        ),
        (
            "formal M3 model contains "
            "no BERT encoder module"
        ),
    )

    check(
        not any(
            "gat"
            in name
            for name
            in module_names
        ),
        (
            "formal M3 model contains "
            "no GAT encoder module"
        ),
    )

    model.to(
        device
    )

    if FAILURES:

        raise RuntimeError(
            "Formal M3 architecture validation failed"
        )

    return model


# =============================================================================
# 15. TRAINING / VALIDATION
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:

    model.train()

    total_loss = 0.0
    total_rows = 0

    for (
        text_cls,
        graph,
        target,
    ) in loader:

        text_cls = text_cls.to(
            device
        )

        graph = graph.to(
            device
        )

        target = target.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            text_cls,
            graph,
        )

        loss = criterion(
            logits,
            target,
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite TRAIN loss detected"
            )

        loss.backward()

        optimizer.step()

        batch_rows = int(
            target.shape[0]
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_rows
        )

        total_rows += (
            batch_rows
        )

    if total_rows != N_TRAIN:

        raise RuntimeError(
            "TRAIN epoch did not consume "
            f"exactly {N_TRAIN} rows; "
            f"got {total_rows}"
        )

    return (
        total_loss
        / total_rows
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:

    model.eval()

    total_loss = 0.0
    total_rows = 0

    all_targets = []
    all_probabilities = []

    with torch.inference_mode():

        for (
            text_cls,
            graph,
            target,
        ) in loader:

            text_cls = text_cls.to(
                device
            )

            graph = graph.to(
                device
            )

            target = target.to(
                device
            )

            logits = model(
                text_cls,
                graph,
            )

            if not torch.isfinite(
                logits
            ).all():

                raise RuntimeError(
                    "Non-finite VAL logits detected"
                )

            loss = criterion(
                logits,
                target,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite VAL loss detected"
                )

            probability = (
                torch.sigmoid(
                    logits
                )
            )

            batch_rows = int(
                target.shape[0]
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_rows
            )

            total_rows += (
                batch_rows
            )

            all_targets.append(
                target.detach()
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probability.detach()
                .cpu()
                .numpy()
            )

    if total_rows != N_VAL:

        raise RuntimeError(
            "VAL evaluation did not consume "
            f"exactly {N_VAL} rows; "
            f"got {total_rows}"
        )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_probability = np.concatenate(
        all_probabilities,
        axis=0,
    )

    if y_true.shape != (
        N_VAL,
        NUM_LABELS,
    ):

        raise RuntimeError(
            "Unexpected VAL target shape: "
            f"{y_true.shape}"
        )

    if y_probability.shape != (
        N_VAL,
        NUM_LABELS,
    ):

        raise RuntimeError(
            "Unexpected VAL probability shape: "
            f"{y_probability.shape}"
        )

    if not np.isfinite(
        y_probability
    ).all():

        raise RuntimeError(
            "Non-finite VAL probabilities detected"
        )

    per_label_ap = (
        average_precision_score(
            y_true,
            y_probability,
            average=None,
        )
    )

    macro_ap = float(
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

    return {
        "loss":
            total_loss / total_rows,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,

        "per_label_ap":
            [
                float(x)
                for x in per_label_ap
            ],
    }


# =============================================================================
# 16. FORMAL TRAINING LOOP
# =============================================================================

def run_formal_training(
    model: M3EarlyFusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: np.ndarray,
    device: torch.device,
    checkpoint_path: Path,
):

    banner(
        "8. FORMAL M3 PHASE-A TRAINING"
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Maximum epochs: "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Early-stop patience: "
        f"{PATIENCE}"
    )

    print(
        "Selection: highest VAL macro AP"
    )

    print(
        "Improvement rule: strictly greater"
    )

    print(
        "Threshold tuning: NO"
    )

    print(
        "TEST inference: NO"
    )

    best_macro_ap = (
        -float("inf")
    )

    best_epoch = None
    best_val = None

    epochs_without_improvement = 0

    history: list[
        dict[str, Any]
    ] = []

    start_time = (
        time.perf_counter()
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        val = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        improved = (
            val[
                "macro_ap"
            ]
            >
            best_macro_ap
        )

        if improved:

            best_macro_ap = (
                val[
                    "macro_ap"
                ]
            )

            best_epoch = (
                epoch
            )

            best_val = dict(
                val
            )

            epochs_without_improvement = 0

            checkpoint = {
                "task":
                    "task3",

                "phase":
                    "A",

                "variant":
                    "M3",

                "model":
                    "EarlyFusion",

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "architecture": {
                    "text_source":
                        (
                            "T1-B final-layer "
                            "CLS from frozen cache"
                        ),

                    "text_projection":
                        "Linear(768,256)",

                    "graph_source":
                        (
                            "A4 GAT/G2 "
                            "post-pool embedding"
                        ),

                    "graph_projection":
                        "Linear(128,256)",

                    "concat_dimension":
                        512,

                    "fusion":
                        "Linear(512,256)",

                    "activation":
                        "ReLU",

                    "dropout":
                        DROPOUT,

                    "classifier":
                        "Linear(256,20)",

                    "trainable_parameters":
                        366_356,
                },

                "training": {
                    "optimizer":
                        "AdamW",

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "batch_size":
                        BATCH_SIZE,

                    "max_epochs":
                        MAX_EPOCHS,

                    "early_stop_patience":
                        PATIENCE,

                    "scheduler":
                        None,

                    "loss":
                        (
                            "BCEWithLogitsLoss("
                            "TRAIN-only pos_weight)"
                        ),

                    "selection_metric":
                        (
                            "VAL macro mean "
                            "per-label Average Precision"
                        ),

                    "threshold_tuning":
                        False,
                },

                "val_loss":
                    float(
                        val[
                            "loss"
                        ]
                    ),

                "val_auc_pr_macro_mean_ap":
                    float(
                        val[
                            "macro_ap"
                        ]
                    ),

                "val_micro_ap":
                    float(
                        val[
                            "micro_ap"
                        ]
                    ),

                "val_per_label_ap":
                    val[
                        "per_label_ap"
                    ],

                "pos_weight":
                    [
                        float(x)
                        for x
                        in pos_weight
                    ],

                "task3_phaseA_design_path":
                    str(
                        DESIGN_PATH.relative_to(
                            ROOT
                        )
                    ),

                "task3_phaseA_design_sha256":
                    EXPECTED_DESIGN_SHA256,

                "pretraining_lock_sha256":
                    EXPECTED_PRETRAINING_LOCK_SHA256,

                "targets_sha256":
                    EXPECTED_TARGETS_SHA256,

                "source_t1_checkpoint_sha256":
                    EXPECTED_T1_SHA256,

                "source_a4_checkpoint_sha256":
                    EXPECTED_A4_SHA256,

                "bert_sequence_sha256":
                    EXPECTED_BERT_SEQUENCE_SHA256,

                "gat_embedding_sha256":
                    EXPECTED_GAT_EMBEDDING_SHA256,

                "test_split_used":
                    False,

                "test_examples_evaluated":
                    0,

                "test_labels_accessed":
                    False,

                "threshold_tuning_used":
                    False,

                "model_state_dict":
                    cpu_state_dict(
                        model
                    ),
            }

            save_checkpoint_atomic(
                checkpoint,
                checkpoint_path,
            )

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    train_loss
                ),

            "val_loss":
                float(
                    val[
                        "loss"
                    ]
                ),

            "val_macro_ap":
                float(
                    val[
                        "macro_ap"
                    ]
                ),

            "val_micro_ap":
                float(
                    val[
                        "micro_ap"
                    ]
                ),

            "improved":
                bool(
                    improved
                ),

            "epochs_without_improvement":
                int(
                    epochs_without_improvement
                ),

            "epoch_seconds":
                float(
                    epoch_seconds
                ),
        }

        history.append(
            row
        )

        marker = (
            "  BEST"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS}  "
            f"train_loss="
            f"{train_loss:.6f}  "
            f"val_loss="
            f"{val['loss']:.6f}  "
            f"val_macro_AP="
            f"{val['macro_ap']:.8f}  "
            f"val_micro_AP="
            f"{val['micro_ap']:.8f}"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered: "
                f"{PATIENCE} consecutive "
                "non-improving epochs"
            )

            break

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    peak_vram_bytes = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    if (
        best_epoch is None
        or best_val is None
        or not checkpoint_path.is_file()
    ):

        raise RuntimeError(
            "No valid best checkpoint was produced"
        )

    return {
        "history":
            history,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val":
            best_val,

        "runtime_seconds":
            float(
                runtime_seconds
            ),

        "peak_vram_bytes":
            int(
                peak_vram_bytes
            ),
    }


# =============================================================================
# 17. BEST-CHECKPOINT REVALIDATION
# =============================================================================

def revalidate_best_checkpoint(
    checkpoint_path: Path,
    val_loader: DataLoader,
    pos_weight: np.ndarray,
    device: torch.device,
    expected_best: dict[str, Any],
) -> dict[str, Any]:

    banner(
        "9. BEST-CHECKPOINT REVALIDATION"
    )

    checkpoint = (
        load_checkpoint(
            checkpoint_path
        )
    )

    check(
        checkpoint.get(
            "task"
        )
        == "task3",
        "checkpoint task = task3",
    )

    check(
        checkpoint.get(
            "phase"
        )
        == "A",
        "checkpoint phase = A",
    )

    check(
        checkpoint.get(
            "variant"
        )
        == "M3",
        "checkpoint variant = M3",
    )

    check(
        checkpoint.get(
            "task3_phaseA_design_sha256"
        )
        ==
        EXPECTED_DESIGN_SHA256,
        (
            "checkpoint records frozen "
            "Script-49 design SHA256"
        ),
    )

    check(
        checkpoint.get(
            "test_split_used"
        )
        is False,
        (
            "checkpoint records TEST "
            "split unused"
        ),
    )

    model = (
        M3EarlyFusion()
    )

    incompatible = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    check(
        len(
            incompatible.missing_keys
        )
        == 0,
        (
            "best checkpoint strict load "
            "has zero missing keys"
        ),
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "best checkpoint strict load "
            "has zero unexpected keys"
        ),
    )

    model.to(
        device
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    observed = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(
        "Stored best VAL macro AP:"
    )

    print(
        f"  "
        f"{expected_best['macro_ap']:.12f}"
    )

    print(
        "Reloaded VAL macro AP:"
    )

    print(
        f"  "
        f"{observed['macro_ap']:.12f}"
    )

    check(
        np.isclose(
            observed[
                "macro_ap"
            ],
            expected_best[
                "macro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "reloaded best checkpoint "
            "reproduces best VAL macro AP"
        ),
    )

    check(
        np.isclose(
            observed[
                "micro_ap"
            ],
            expected_best[
                "micro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "reloaded best checkpoint "
            "reproduces VAL micro AP"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Best-checkpoint revalidation failed"
        )

    return observed


# =============================================================================
# 18. WRITE RUN ARTIFACTS
# =============================================================================

def write_run_artifacts(
    run_id: str,
    training_result: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha: str,
) -> tuple[Path, Path]:

    banner(
        "10. WRITE FORMAL RUN ARTIFACTS"
    )

    epoch_path = (
        RUN_DIR
        / f"{run_id}_epochs.csv"
    )

    summary_path = (
        RUN_DIR
        / f"{run_id}_summary.json"
    )

    check(
        not epoch_path.exists(),
        "epoch-log path is unused",
    )

    check(
        not summary_path.exists(),
        "run-summary path is unused",
    )

    if FAILURES:

        raise RuntimeError(
            "Run-output collision detected"
        )

    history_df = (
        pd.DataFrame(
            training_result[
                "history"
            ]
        )
    )

    epoch_temp = Path(
        str(
            epoch_path
        )
        + ".__tmp__"
    )

    history_df.to_csv(
        epoch_temp,
        index=False,
    )

    os.replace(
        epoch_temp,
        epoch_path,
    )

    best_val = (
        training_result[
            "best_val"
        ]
    )

    phase_b_required = (
        best_val[
            "macro_ap"
        ]
        <=
        PHASE_B_REFERENCE
    )

    summary = {
        "task":
            "task3",

        "phase":
            "A",

        "variant":
            "M3",

        "model":
            "EarlyFusion",

        "run_id":
            run_id,

        "seed":
            SEED,

        "formal_run":
            True,

        "task3_phaseA_design_sha256":
            EXPECTED_DESIGN_SHA256,

        "best_epoch":
            training_result[
                "best_epoch"
            ],

        "best_val_loss":
            float(
                best_val[
                    "loss"
                ]
            ),

        "best_val_auc_pr_macro_mean_ap":
            float(
                best_val[
                    "macro_ap"
                ]
            ),

        "best_val_micro_ap":
            float(
                best_val[
                    "micro_ap"
                ]
            ),

        "best_val_per_label_ap":
            best_val[
                "per_label_ap"
            ],

        "checkpoint":
            str(
                checkpoint_path.relative_to(
                    ROOT
                )
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "epoch_log":
            str(
                epoch_path.relative_to(
                    ROOT
                )
            ),

        "runtime_seconds":
            training_result[
                "runtime_seconds"
            ],

        "runtime_minutes":
            (
                training_result[
                    "runtime_seconds"
                ]
                / 60.0
            ),

        "peak_vram_bytes":
            training_result[
                "peak_vram_bytes"
            ],

        "peak_vram_gb":
            (
                training_result[
                    "peak_vram_bytes"
                ]
                / (1024 ** 3)
            ),

        "baseline_references": {
            "M1_T1B_val_macro_ap":
                M1_VAL_MACRO_AP,

            "M2_A4_GAT_G2_val_macro_ap":
                M2_VAL_MACRO_AP,
        },

        "phase_b_gate": {
            "application":
                "M3 independently",

            "strict_requirement_to_skip":
                (
                    "M3 Phase-A VAL macro AP "
                    "> M1 AND > M2"
                ),

            "effective_reference_to_beat":
                PHASE_B_REFERENCE,

            "m3_phase_a_val_macro_ap":
                float(
                    best_val[
                        "macro_ap"
                    ]
                ),

            "phase_b_required_for_m3":
                bool(
                    phase_b_required
                ),

            "action_now":
                (
                    "Do not run Phase B yet; "
                    "complete M4 Phase A first."
                ),
        },

        "training": {
            "optimizer":
                "AdamW",

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "patience":
                PATIENCE,

            "scheduler":
                None,

            "seed":
                SEED,
        },

        "test_split_used":
            False,

        "test_target_rows_loaded":
            0,

        "test_examples_evaluated":
            0,

        "threshold_tuning_used":
            False,

        "environment": {
            "python":
                sys.version.split()[0],

            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,

            "torch":
                torch.__version__,

            "sklearn":
                sklearn.__version__,

            "cuda_device":
                torch.cuda.get_device_name(
                    0
                ),
        },
    }

    write_json_atomic(
        summary_path,
        summary,
    )

    print(
        f"Epoch log:"
    )

    print(
        f"  "
        f"{epoch_path.relative_to(ROOT)}"
    )

    print(
        f"Summary:"
    )

    print(
        f"  "
        f"{summary_path.relative_to(ROOT)}"
    )

    return (
        epoch_path,
        summary_path,
    )


# =============================================================================
# 19. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 50 — FORMAL TASK-3 M3 EARLY-FUSION PHASE-A"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()

    print(
        "Formal experiment:       YES"
    )

    print(
        "Variant:                 M3 Early Fusion"
    )

    print(
        "Phase:                   A"
    )

    print(
        "Seed:                    42"
    )

    print()

    print(
        "BERT encoder training:   NO"
    )

    print(
        "GAT encoder training:    NO"
    )

    print(
        "Fusion-head training:    YES"
    )

    print()

    print(
        "TRAIN examples:          4112"
    )

    print(
        "VAL examples:             514"
    )

    print(
        "TEST examples:              0"
    )

    print()

    print(
        "Threshold tuning:         NO"
    )

    configure_reproducibility()

    try:

        # ---------------------------------------------------------------------
        # Output protection first.
        # ---------------------------------------------------------------------

        verify_no_prior_formal_run()

        # ---------------------------------------------------------------------
        # Frozen experiment design.
        # ---------------------------------------------------------------------

        verify_design()

        # ---------------------------------------------------------------------
        # Immutable dependencies.
        # ---------------------------------------------------------------------

        verify_frozen_dependencies()

        # ---------------------------------------------------------------------
        # TRAIN+VAL aligned representations / targets only.
        # ---------------------------------------------------------------------

        (
            bert_sequence,
            gat_embedding,
            targets,
            train_indices,
            val_indices,
        ) = load_aligned_development_data()

        # ---------------------------------------------------------------------
        # TRAIN-only class weights.
        # ---------------------------------------------------------------------

        pos_weight = obtain_pos_weight(
            targets=targets,
            train_indices=train_indices,
        )

        # ---------------------------------------------------------------------
        # CUDA.
        # ---------------------------------------------------------------------

        banner(
            "5A. TRAINING DEVICE"
        )

        check(
            torch.cuda.is_available(),
            "CUDA is available",
        )

        if FAILURES:

            raise RuntimeError(
                "Expected CUDA environment unavailable"
            )

        device = torch.device(
            "cuda"
        )

        print(
            "CUDA device:"
        )

        print(
            f"  "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"PyTorch: "
            f"{torch.__version__}"
        )

        print(
            f"scikit-learn: "
            f"{sklearn.__version__}"
        )

        # ---------------------------------------------------------------------
        # Dataloaders.
        # ---------------------------------------------------------------------

        (
            train_loader,
            val_loader,
        ) = build_loaders(
            bert_sequence=bert_sequence,
            gat_embedding=gat_embedding,
            targets=targets,
            train_indices=train_indices,
            val_indices=val_indices,
        )

        # ---------------------------------------------------------------------
        # Formal M3 model.
        # ---------------------------------------------------------------------

        model = build_model(
            device
        )

        # ---------------------------------------------------------------------
        # Formal run ID.
        # ---------------------------------------------------------------------

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

        run_id = (
            "task3_M3_phaseA_"
            f"seed{SEED}_"
            f"{timestamp}"
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        RUN_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            MODEL_DIR
            / f"{run_id}_best.pt"
        )

        check(
            not checkpoint_path.exists(),
            (
                "formal best-checkpoint "
                "path is unused"
            ),
        )

        if FAILURES:

            raise RuntimeError(
                "Checkpoint collision detected"
            )

        print()
        print(
            f"Run ID:"
        )

        print(
            f"  {run_id}"
        )

        print(
            f"Best checkpoint:"
        )

        print(
            f"  "
            f"{checkpoint_path.relative_to(ROOT)}"
        )

        # ---------------------------------------------------------------------
        # Formal training.
        # ---------------------------------------------------------------------

        training_result = (
            run_formal_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                pos_weight=pos_weight,
                device=device,
                checkpoint_path=checkpoint_path,
            )
        )

        # ---------------------------------------------------------------------
        # Re-load and independently revalidate selected checkpoint.
        # ---------------------------------------------------------------------

        revalidated = (
            revalidate_best_checkpoint(
                checkpoint_path=checkpoint_path,
                val_loader=val_loader,
                pos_weight=pos_weight,
                device=device,
                expected_best=training_result[
                    "best_val"
                ],
            )
        )

        # Replace with independently revalidated metrics.
        training_result[
            "best_val"
        ] = revalidated

        # ---------------------------------------------------------------------
        # Freeze checkpoint identity.
        # ---------------------------------------------------------------------

        checkpoint_sha = sha256_file(
            checkpoint_path
        )

        print()
        print(
            "Selected M3 checkpoint SHA256:"
        )

        print(
            f"  {checkpoint_sha}"
        )

        # ---------------------------------------------------------------------
        # Run-local formal logs.
        #
        # We deliberately do NOT alter the older global experiment CSV schemas
        # in this script because Script 49 did not revalidate their schemas.
        # The formal M3 run remains fully self-contained and auditable here.
        # ---------------------------------------------------------------------

        (
            epoch_path,
            summary_path,
        ) = write_run_artifacts(
            run_id=run_id,
            training_result=training_result,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
        )

        epoch_sha = sha256_file(
            epoch_path
        )

        summary_sha = sha256_file(
            summary_path
        )

    except Exception as exc:

        banner(
            "SCRIPT 50 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        if FAILURES:

            print()

            print(
                f"Failed checks: "
                f"{len(FAILURES)}"
            )

            for message in FAILURES:

                print(
                    f"  - {message}"
                )

        print()

        print(
            "Do NOT proceed to M4 Phase-A training."
        )

        print(
            "Inspect any partial Script-50 "
            "artifact before rerunning."
        )

        return 1

    # =========================================================================
    # Final summary
    # =========================================================================

    best_val = (
        training_result[
            "best_val"
        ]
    )

    phase_b_required = (
        best_val[
            "macro_ap"
        ]
        <=
        PHASE_B_REFERENCE
    )

    banner(
        "SCRIPT 50 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Formal model:"
    )

    print(
        "  Task 3 / M3 / Early Fusion / Phase A"
    )

    print(
        "  seed = 42"
    )

    print()

    print(
        "Best epoch:"
    )

    print(
        f"  "
        f"{training_result['best_epoch']}"
    )

    print()

    print(
        "Best validation metrics:"
    )

    print(
        f"  VAL macro AP = "
        f"{best_val['macro_ap']:.8f}"
    )

    print(
        f"  VAL micro AP = "
        f"{best_val['micro_ap']:.8f}"
    )

    print(
        f"  VAL loss     = "
        f"{best_val['loss']:.8f}"
    )

    print()

    print(
        "Frozen references:"
    )

    print(
        f"  M1 BERT-only = "
        f"{M1_VAL_MACRO_AP:.8f}"
    )

    print(
        f"  M2 GNN-only  = "
        f"{M2_VAL_MACRO_AP:.8f}"
    )

    print()

    print(
        "M3 Phase-B gate:"
    )

    print(
        f"  effective score to beat = "
        f"{PHASE_B_REFERENCE:.8f}"
    )

    print(
        f"  M3 Phase B required = "
        f"{phase_b_required}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  Do NOT run M3 Phase B yet."
    )

    print(
        "  M4 Phase A must be completed first."
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        f"  "
        f"{checkpoint_path.relative_to(ROOT)}"
    )

    print(
        "Checkpoint SHA256:"
    )

    print(
        f"  {checkpoint_sha}"
    )

    print()

    print(
        "Epoch log SHA256:"
    )

    print(
        f"  {epoch_sha}"
    )

    print()

    print(
        "Run summary SHA256:"
    )

    print(
        f"  {summary_sha}"
    )

    print()

    print(
        f"Runtime: "
        f"{training_result['runtime_seconds'] / 60.0:.2f} min"
    )

    print(
        f"Peak VRAM: "
        f"{training_result['peak_vram_bytes'] / (1024 ** 3):.3f} GB"
    )

    print()

    print(
        "BERT/GAT encoder parameters trained: 0"
    )

    print(
        "M3 trainable parameters:       366,356"
    )

    print(
        "TEST target rows loaded:             0"
    )

    print(
        "TEST inference examples:              0"
    )

    print(
        "Threshold tuning:                     0"
    )

    print()

    print(
        "STOP HERE."
    )

    print(
        "Review this formal M3 result before "
        "constructing Script 51 for M4 Phase A."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )