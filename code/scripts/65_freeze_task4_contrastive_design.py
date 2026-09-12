#!/usr/bin/env python3
"""
SCRIPT 65 — FREEZE TASK-4 CONTRASTIVE RETRIEVAL DESIGN

Purpose
-------
Freeze the complete Task-4 experimental protocol BEFORE any Task-4
training or retrieval result is observed.

This script:

1. Re-verifies the critical frozen Task-1/Task-2/Task-3-development inputs.
2. Confirms TRAIN+VAL paired representation alignment.
3. Freezes:
   - T4-A frozen-encoder contrastive model
   - T4-B partial-fine-tuning contrastive model
   - projection dimensions
   - symmetric InfoNCE
   - temperature
   - optimizer/training settings
   - VAL retrieval metrics
   - model-selection rule
   - multi-seed policy
   - final TEST policy
   - zero-shot evaluation protocol
   - qualitative retrieval policy
4. Writes a versioned design JSON and a SHA256 lock.

IMPORTANT
---------
NO model construction
NO forward pass
NO training
NO optimizer execution
NO InfoNCE computation
NO retrieval computation
NO TEST examples
NO TEST labels
NO TEST inference
NO TEST retrieval

This script changes only Task-4 protocol metadata.
It does NOT modify any frozen Tasks 1–3 artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# ROOT / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SCRIPT64_PATH = (
    ROOT / "scripts/64_task4_contrastive_preflight.py"
)

PRETRAINING_LOCK_PATH = (
    ROOT / "data/splits/pretraining_frozen_hashes.json"
)

TASK1_SELECTION_PATH = (
    ROOT / "results/runs/task1/task1_model_selection.json"
)

TASK2_GRAPH_SELECTION_PATH = (
    ROOT / "results/runs/task2/task2_graph_selection.json"
)

TASK2_MODEL_SELECTION_PATH = (
    ROOT / "results/runs/task2/task2_model_selection.json"
)

T1B_CHECKPOINT_PATH = (
    ROOT
    / "models/task1_candidates/"
      "task1_T1-B_seed42_20260828T182131Z_best.pt"
)

A4_CHECKPOINT_PATH = (
    ROOT
    / "models/task2_candidates/"
      "task2_A4_GAT_G2_seed42_20260828T193949Z_best.pt"
)

BERT_CACHE_DIR = (
    ROOT / "data/processed/bert_seq_frozen"
)

BERT_SEQUENCE_PATH = (
    BERT_CACHE_DIR / "bert_sequence_trainval.npy"
)

BERT_MASK_PATH = (
    BERT_CACHE_DIR / "attention_mask_trainval.npy"
)

BERT_MANIFEST_PATH = (
    BERT_CACHE_DIR / "bert_seq_manifest.parquet"
)

BERT_METADATA_PATH = (
    BERT_CACHE_DIR / "bert_seq_cache_metadata.json"
)

BERT_LOCK_PATH = (
    BERT_CACHE_DIR / "bert_seq_frozen_hashes.json"
)

GAT_CACHE_DIR = (
    ROOT / "data/processed/gat_g2_frozen"
)

GAT_EMBEDDING_PATH = (
    GAT_CACHE_DIR / "gat_g2_embedding_trainval.npy"
)

GAT_MANIFEST_PATH = (
    GAT_CACHE_DIR / "gat_g2_manifest.parquet"
)

GAT_METADATA_PATH = (
    GAT_CACHE_DIR / "gat_g2_cache_metadata.json"
)

GAT_LOCK_PATH = (
    GAT_CACHE_DIR / "gat_g2_frozen_hashes.json"
)

OUTPUT_DIR = (
    ROOT / "results/runs/task4"
)

DESIGN_PATH = (
    OUTPUT_DIR / "task4_contrastive_design.json"
)

DESIGN_LOCK_PATH = (
    OUTPUT_DIR / "task4_contrastive_design_lock.json"
)


# =============================================================================
# ACCEPTED FROZEN IDENTITIES
# =============================================================================

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28"
    "f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_T1B_SHA256 = (
    "3b675acbacb7191d514e3cc513669442"
    "71e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311"
    "daf74ea0a24f6b02e78e5513078fae1e"
)

EXPECTED_BERT_SEQUENCE_SHA256 = (
    "a1c723ec939392b8b0d9bafadb8327d"
    "8a0967ecb9bd8e89ffcd559306633e133"
)

EXPECTED_BERT_MASK_SHA256 = (
    "633b11d5ca9146000da84b282f9081c6"
    "ff33c1279486c2e3112e4c89137d197b"
)

EXPECTED_BERT_MANIFEST_SHA256 = (
    "ebd86077489b716abc8768247f550890"
    "ca8e5a26b8f0967c32882c22c4e85d2f"
)

EXPECTED_BERT_METADATA_SHA256 = (
    "d50d25fbea4cd0ea83a946ae57e3c867"
    "3bd53303efd58a2c2e9bf3998e792c73"
)

EXPECTED_BERT_LOCK_SHA256 = (
    "c57cc66666d4531c02c6821964585b551"
    "1e516872f1a922cf12e01487d464228"
)

EXPECTED_GAT_EMBEDDING_SHA256 = (
    "3d39a343d1acc678576f9dd3d88b088"
    "ca18a81017190246b0b32841292b92852"
)

EXPECTED_GAT_MANIFEST_SHA256 = (
    "a029c12ec6a98922c98f373bbc263b6"
    "c529825b56b1446a14f659f6854ed371a"
)

EXPECTED_GAT_METADATA_SHA256 = (
    "d2ea876aa58cd0b32ae4cf06c849937"
    "a88d4a47c1e4b69d7f481ece0c273467c"
)

EXPECTED_GAT_LOCK_SHA256 = (
    "3ce775a3bb89532554883c11ac8f0746"
    "e2f3f4a7f983a2e92d30ca18df4eb5b0"
)


# =============================================================================
# FROZEN DATA CONTRACT
# =============================================================================

EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_DEV = 4626
EXPECTED_TEST = 514
EXPECTED_TOTAL = 5140

EXPECTED_BERT_SEQUENCE_SHAPE = (
    EXPECTED_DEV,
    112,
    768,
)

EXPECTED_GAT_SHAPE = (
    EXPECTED_DEV,
    128,
)


# =============================================================================
# TASK-4 DESIGN — FROZEN HERE
# =============================================================================

DESIGN_VERSION = 1

PRIMARY_SELECTION_SEED = 42

FINAL_SEEDS = [
    42,
    1337,
    2026,
]

SHARED_EMBEDDING_DIM = 256

TEMPERATURE = 0.07

TRAIN_BATCH_SIZE = 32

GRADIENT_CLIP_NORM = 1.0


# T4-A
T4A_LR = 1.0e-3
T4A_WEIGHT_DECAY = 1.0e-4
T4A_MAX_EPOCHS = 30
T4A_PATIENCE = 5


# T4-B
T4B_PROJECTION_LR = 1.0e-4
T4B_BERT_LR = 2.0e-5
T4B_GAT_LR = 1.0e-4
T4B_WEIGHT_DECAY = 1.0e-4
T4B_MAX_EPOCHS = 20
T4B_PATIENCE = 4


# =============================================================================
# STATE
# =============================================================================

FAILURES: list[str] = []
WARNINGS: list[str] = []


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(message: str) -> None:

    print(f"PASS  {message}")


def failed(message: str) -> None:

    FAILURES.append(message)
    print(f"FAIL  {message}")


def warning(message: str) -> None:

    WARNINGS.append(message)
    print(f"WARN  {message}")


def require(
    condition: bool,
    message: str,
) -> None:

    if condition:
        passed(message)
    else:
        failed(message)


def safe_relative(path: Path) -> str:

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def verify_hash(
    path: Path,
    expected: str,
    label: str,
) -> None:

    if not path.exists():

        failed(
            f"{label} exists"
        )
        return

    observed = sha256_file(path)

    print(
        f"{label}:\n"
        f"  observed = {observed}\n"
        f"  expected = {expected}"
    )

    require(
        observed == expected,
        f"{label} SHA256 matches accepted identity",
    )


def load_json(path: Path) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def json_blob(obj: Any) -> str:

    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
    ).lower()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    temp_path = Path(
        str(path) + ".tmp"
    )

    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())

    os.replace(
        temp_path,
        path,
    )


def split_counts(
    manifest: pd.DataFrame,
) -> dict[str, int]:

    counts = (
        manifest["split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )

    return {
        str(k): int(v)
        for k, v in counts.items()
    }


# =============================================================================
# HEADER
# =============================================================================

section(
    "SCRIPT 65 — FREEZE TASK-4 CONTRASTIVE RETRIEVAL DESIGN"
)

print(f"Project root: {ROOT}")
print("Mode: DESIGN FREEZE")
print()
print("Model construction:       NO")
print("Forward/inference:        NO")
print("Training/backprop:        NO")
print("InfoNCE calculation:      NO")
print("Retrieval calculation:    NO")
print("TEST examples:            NO")
print("TEST labels:              NO")
print("TEST inference:           NO")
print("Task 1–3 modification:    NO")
print()
print(
    "Only new output:"
)
print(
    f"  {safe_relative(DESIGN_PATH)}"
)
print(
    f"  {safe_relative(DESIGN_LOCK_PATH)}"
)


# =============================================================================
# 1. OUTPUT PROTECTION
# =============================================================================

section(
    "1. OUTPUT PROTECTION"
)

require(
    not DESIGN_PATH.exists(),
    "Task-4 design artifact does not already exist",
)

require(
    not DESIGN_LOCK_PATH.exists(),
    "Task-4 design lock does not already exist",
)

if FAILURES:

    print()
    print(
        "Existing Task-4 design detected."
    )
    print(
        "Refusing to overwrite a frozen experimental design."
    )

    sys.exit(1)


# =============================================================================
# 2. VERIFY SCRIPT-64 SOURCE EXISTS
# =============================================================================

section(
    "2. SCRIPT-64 PREFLIGHT PROVENANCE"
)

require(
    SCRIPT64_PATH.exists(),
    "Script 64 source exists",
)

script64_sha256 = None

if SCRIPT64_PATH.exists():

    script64_sha256 = sha256_file(
        SCRIPT64_PATH
    )

    print(
        "Script 64 SHA256:"
    )
    print(
        f"  {script64_sha256}"
    )

    passed(
        "Script 64 source identity recorded for Task-4 provenance"
    )


# =============================================================================
# 3. VERIFY FROZEN UPSTREAM IDENTITIES
# =============================================================================

section(
    "3. FROZEN UPSTREAM ARTIFACT IDENTITIES"
)

verify_hash(
    PRETRAINING_LOCK_PATH,
    EXPECTED_PRETRAINING_LOCK_SHA256,
    "pretraining lock",
)

verify_hash(
    T1B_CHECKPOINT_PATH,
    EXPECTED_T1B_SHA256,
    "selected T1-B checkpoint",
)

verify_hash(
    A4_CHECKPOINT_PATH,
    EXPECTED_A4_SHA256,
    "selected A4 GAT/G2 checkpoint",
)

verify_hash(
    BERT_SEQUENCE_PATH,
    EXPECTED_BERT_SEQUENCE_SHA256,
    "Script-47 BERT sequence cache",
)

verify_hash(
    BERT_MASK_PATH,
    EXPECTED_BERT_MASK_SHA256,
    "Script-47 BERT attention-mask cache",
)

verify_hash(
    BERT_MANIFEST_PATH,
    EXPECTED_BERT_MANIFEST_SHA256,
    "Script-47 BERT manifest",
)

verify_hash(
    BERT_METADATA_PATH,
    EXPECTED_BERT_METADATA_SHA256,
    "Script-47 BERT metadata",
)

verify_hash(
    BERT_LOCK_PATH,
    EXPECTED_BERT_LOCK_SHA256,
    "Script-47 BERT cache lock",
)

verify_hash(
    GAT_EMBEDDING_PATH,
    EXPECTED_GAT_EMBEDDING_SHA256,
    "Script-48 GAT embedding cache",
)

verify_hash(
    GAT_MANIFEST_PATH,
    EXPECTED_GAT_MANIFEST_SHA256,
    "Script-48 GAT manifest",
)

verify_hash(
    GAT_METADATA_PATH,
    EXPECTED_GAT_METADATA_SHA256,
    "Script-48 GAT metadata",
)

verify_hash(
    GAT_LOCK_PATH,
    EXPECTED_GAT_LOCK_SHA256,
    "Script-48 GAT cache lock",
)


# =============================================================================
# 4. VERIFY FROZEN SELECTION ARTIFACTS
# =============================================================================

section(
    "4. FROZEN MODEL-SELECTION REFERENCES"
)

selection_checks = [
    (
        TASK1_SELECTION_PATH,
        [
            "t1-b",
            EXPECTED_T1B_SHA256,
        ],
        "Task-1 selection",
    ),
    (
        TASK2_GRAPH_SELECTION_PATH,
        [
            "g2",
        ],
        "Task-2 graph selection",
    ),
    (
        TASK2_MODEL_SELECTION_PATH,
        [
            "a4",
            "gat",
            "g2",
            EXPECTED_A4_SHA256,
        ],
        "Task-2 model selection",
    ),
]

for path, required_tokens, label in selection_checks:

    if not path.exists():

        failed(
            f"{label} exists"
        )
        continue

    passed(
        f"{label} exists"
    )

    obj = load_json(path)
    blob = json_blob(obj)

    for token in required_tokens:

        require(
            token.lower() in blob,
            f"{label} contains {token!r}",
        )


# =============================================================================
# 5. VERIFY TRAIN+VAL PAIRED DEVELOPMENT POPULATION
# =============================================================================

section(
    "5. TASK-4 DEVELOPMENT POPULATION"
)

bert_manifest = pd.read_parquet(
    BERT_MANIFEST_PATH
)

gat_manifest = pd.read_parquet(
    GAT_MANIFEST_PATH
)

required_columns = {
    "cache_index",
    "track_id",
    "split",
}

require(
    required_columns.issubset(
        bert_manifest.columns
    ),
    "BERT manifest contains cache_index/track_id/split",
)

require(
    required_columns.issubset(
        gat_manifest.columns
    ),
    "GAT manifest contains cache_index/track_id/split",
)

require(
    len(bert_manifest)
    == EXPECTED_DEV,
    "BERT Task-4 development population = 4626",
)

require(
    len(gat_manifest)
    == EXPECTED_DEV,
    "GAT Task-4 development population = 4626",
)

bert_counts = split_counts(
    bert_manifest
)

gat_counts = split_counts(
    gat_manifest
)

print(
    "BERT split counts:",
    bert_counts,
)

print(
    "GAT split counts:",
    gat_counts,
)

expected_counts = {
    "train": EXPECTED_TRAIN,
    "val": EXPECTED_VAL,
}

require(
    bert_counts == expected_counts,
    "BERT development cache is exactly "
    "4112 TRAIN + 514 VAL",
)

require(
    gat_counts == expected_counts,
    "GAT development cache is exactly "
    "4112 TRAIN + 514 VAL",
)

require(
    bert_counts.get(
        "test",
        0,
    )
    == 0,
    "BERT development cache contains zero TEST rows",
)

require(
    gat_counts.get(
        "test",
        0,
    )
    == 0,
    "GAT development cache contains zero TEST rows",
)

bert_ids = (
    bert_manifest["track_id"]
    .astype(str)
    .tolist()
)

gat_ids = (
    gat_manifest["track_id"]
    .astype(str)
    .tolist()
)

require(
    bert_ids == gat_ids,
    "BERT row i and GAT row i have identical track_id",
)

require(
    len(set(bert_ids))
    == EXPECTED_DEV,
    "all Task-4 development track IDs are unique",
)

bert_splits = (
    bert_manifest["split"]
    .astype(str)
    .str.lower()
    .tolist()
)

gat_splits = (
    gat_manifest["split"]
    .astype(str)
    .str.lower()
    .tolist()
)

require(
    bert_splits == gat_splits,
    "BERT row i and GAT row i have identical split",
)


# =============================================================================
# 6. VERIFY REPRESENTATION CONTRACT
# =============================================================================

section(
    "6. FROZEN REPRESENTATION CONTRACT"
)

bert_sequence = np.load(
    BERT_SEQUENCE_PATH,
    mmap_mode="r",
)

gat_embedding = np.load(
    GAT_EMBEDDING_PATH,
    mmap_mode="r",
)

print(
    "BERT sequence shape:",
    bert_sequence.shape,
)

print(
    "GAT embedding shape:",
    gat_embedding.shape,
)

require(
    tuple(bert_sequence.shape)
    == EXPECTED_BERT_SEQUENCE_SHAPE,
    "BERT cache shape is exactly (4626,112,768)",
)

require(
    tuple(gat_embedding.shape)
    == EXPECTED_GAT_SHAPE,
    "GAT cache shape is exactly (4626,128)",
)

require(
    bert_sequence.dtype
    == np.dtype("float32"),
    "BERT sequence cache dtype is float32",
)

require(
    gat_embedding.dtype
    == np.dtype("float32"),
    "GAT embedding cache dtype is float32",
)

print()
print(
    "Frozen Task-4 encoder contract:"
)
print(
    "  text representation  = final T1-B CLS, 768-D"
)
print(
    "  graph representation = A4/G2 pooled GAT, 128-D"
)


# =============================================================================
# 7. ABORT BEFORE FREEZE IF ANY INPUT CHECK FAILED
# =============================================================================

section(
    "7. PRE-FREEZE VALIDATION"
)

if FAILURES:

    print(
        f"FAILURES = {len(FAILURES)}"
    )

    for message in FAILURES:
        print(
            f"  - {message}"
        )

    print()
    print(
        "RESULT: FAIL"
    )

    print(
        "Task-4 design has NOT been written."
    )

    sys.exit(1)


passed(
    "all upstream inputs match the accepted frozen Task-1/2/3 lineage"
)


# =============================================================================
# 8. FREEZE TASK-4 SCIENTIFIC DESIGN
# =============================================================================

section(
    "8. TASK-4 DESIGN TO FREEZE"
)

created_utc = datetime.now(
    timezone.utc
).isoformat()

design: dict[str, Any] = {

    "artifact_type":
        "task4_contrastive_experimental_design",

    "version":
        DESIGN_VERSION,

    "created_utc":
        created_utc,

    "project":
        "GNN-BERT Music Context Understanding",

    "task":
        "Task 4 — Cross-Modal MusicCaps Alignment",

    "status":
        "frozen_before_any_formal_task4_result",

    "research_question":
        (
            "Can frozen or partially fine-tuned graph-audio and "
            "caption encoders learn a shared embedding space in "
            "which corresponding MusicCaps graph-caption pairs "
            "retrieve each other?"
        ),

    "dataset": {

        "name":
            "MusicCaps",

        "total_frozen_tracks":
            EXPECTED_TOTAL,

        "train_tracks":
            EXPECTED_TRAIN,

        "val_tracks":
            EXPECTED_VAL,

        "test_tracks":
            EXPECTED_TEST,

        "development_tracks":
            EXPECTED_DEV,

        "pair_key":
            "track_id",

        "positive_pair":
            "caption and audio graph from the identical frozen track_id",

        "development_scope":
            [
                "train",
                "val",
            ],

        "test_during_development":
            False,
    },

    "upstream_encoders": {

        "text": {

            "model_id":
                "T1-B",

            "model_type":
                "bert-base-uncased fine-tuned Task-1 encoder",

            "checkpoint":
                safe_relative(
                    T1B_CHECKPOINT_PATH
                ),

            "checkpoint_sha256":
                EXPECTED_T1B_SHA256,

            "representation":
                "final-layer CLS hidden state",

            "dimension":
                768,

            "task4_a_state":
                "frozen",

            "task4_b_state":
                (
                    "only final 2 BERT encoder blocks trainable; "
                    "all earlier BERT blocks and embeddings frozen"
                ),
        },

        "graph": {

            "model_id":
                "A4",

            "architecture":
                "GAT",

            "graph_type":
                "G2",

            "checkpoint":
                safe_relative(
                    A4_CHECKPOINT_PATH
                ),

            "checkpoint_sha256":
                EXPECTED_A4_SHA256,

            "representation":
                (
                    "128-D graph representation after final "
                    "GAT encoder block + global mean pooling "
                    "and before Task-2 classifier"
                ),

            "dimension":
                128,

            "task4_a_state":
                "frozen",

            "task4_b_state":
                (
                    "only final GAT message-passing encoder block "
                    "trainable; earlier GAT encoder blocks frozen"
                ),
        },
    },

    "development_caches": {

        "bert_sequence":
            safe_relative(
                BERT_SEQUENCE_PATH
            ),

        "bert_sequence_sha256":
            EXPECTED_BERT_SEQUENCE_SHA256,

        "gat_embedding":
            safe_relative(
                GAT_EMBEDDING_PATH
            ),

        "gat_embedding_sha256":
            EXPECTED_GAT_EMBEDDING_SHA256,

        "row_alignment":
            "exact rowwise equality by track_id",

        "test_rows":
            0,
    },

    "shared_embedding_space": {

        "dimension":
            SHARED_EMBEDDING_DIM,

        "text_projection":
            "Linear(768, 256, bias=False)",

        "graph_projection":
            "Linear(128, 256, bias=False)",

        "projection_activation":
            None,

        "projection_dropout":
            0.0,

        "projection_initialization":
            "Xavier uniform",

        "normalization":
            "L2 normalize along embedding dimension",

        "similarity":
            "cosine similarity via normalized dot product",
    },

    "contrastive_objective": {

        "name":
            "symmetric InfoNCE",

        "temperature":
            TEMPERATURE,

        "temperature_trainable":
            False,

        "similarity_logits":
            "z_audio @ z_text.T / temperature",

        "positive_index":
            "diagonal i == i",

        "negative_policy":
            (
                "all non-matching examples within the current "
                "TRAIN mini-batch are negatives"
            ),

        "audio_to_text_loss":
            "CrossEntropy(similarity_matrix, arange(batch_size))",

        "text_to_audio_loss":
            (
                "CrossEntropy(similarity_matrix.T, "
                "arange(batch_size))"
            ),

        "total_loss":
            (
                "0.5 * audio_to_text_loss + "
                "0.5 * text_to_audio_loss"
            ),
    },

    "t4_a": {

        "name":
            "T4-A Frozen Encoders",

        "purpose":
            (
                "Test whether the already selected Task-1 BERT "
                "and Task-2 GAT representations can be aligned "
                "using projection heads alone."
            ),

        "encoder_training":
            False,

        "uses_frozen_script47_48_caches":
            True,

        "trainable_components":
            [
                "text Linear(768,256,bias=False)",
                "graph Linear(128,256,bias=False)",
            ],

        "optimizer":
            "AdamW",

        "learning_rate":
            T4A_LR,

        "weight_decay":
            T4A_WEIGHT_DECAY,

        "batch_size":
            TRAIN_BATCH_SIZE,

        "shuffle_train":
            True,

        "drop_last":
            False,

        "max_epochs":
            T4A_MAX_EPOCHS,

        "early_stopping_patience":
            T4A_PATIENCE,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "checkpoint_metric":
            "VAL mean retrieval recall (mR)",

        "checkpoint_mode":
            "maximize",

        "checkpoint_tie_rule":
            "earliest epoch",

        "formal_selection_seed":
            PRIMARY_SELECTION_SEED,
    },

    "t4_b": {

        "name":
            "T4-B Partial Dual-Encoder Fine-Tuning",

        "purpose":
            (
                "Allow limited cross-modal adaptation while "
                "preserving most of the already selected "
                "Task-1 and Task-2 encoders."
            ),

        "initialization":
            (
                "initialize projection heads from the corresponding "
                "T4-A checkpoint for the same seed; initialize "
                "encoders from frozen T1-B and A4 checkpoints"
            ),

        "uses_frozen_representation_cache_for_training":
            False,

        "text_trainable_scope":
            "final 2 BERT encoder blocks only",

        "graph_trainable_scope":
            "final GAT message-passing encoder block only",

        "projection_heads_trainable":
            True,

        "optimizer":
            "AdamW with parameter groups",

        "learning_rates": {

            "projection_heads":
                T4B_PROJECTION_LR,

            "bert_final_2_blocks":
                T4B_BERT_LR,

            "gat_final_block":
                T4B_GAT_LR,
        },

        "weight_decay":
            T4B_WEIGHT_DECAY,

        "batch_size":
            TRAIN_BATCH_SIZE,

        "shuffle_train":
            True,

        "drop_last":
            False,

        "max_epochs":
            T4B_MAX_EPOCHS,

        "early_stopping_patience":
            T4B_PATIENCE,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "checkpoint_metric":
            "VAL mean retrieval recall (mR)",

        "checkpoint_mode":
            "maximize",

        "checkpoint_tie_rule":
            "earliest epoch",

        "formal_selection_seed":
            PRIMARY_SELECTION_SEED,

        "mixed_precision":
            (
                "AMP permitted for GPU training, but the accepted "
                "numerical pathway must be fixed by Script 66 "
                "before any formal T4-B result"
            ),
    },

    "validation_retrieval": {

        "candidate_pool":
            "all 514 validation examples",

        "batch_local_retrieval":
            False,

        "similarity_matrix_shape":
            [
                EXPECTED_VAL,
                EXPECTED_VAL,
            ],

        "directions":
            [
                "caption_to_audio",
                "audio_to_caption",
            ],

        "metrics":
            [
                "R@1",
                "R@5",
                "R@10",
            ],

        "correct_match":
            "same track_id / diagonal pair",

        "rank_definition":
            (
                "rank = 1 + number of candidate similarities "
                "strictly greater than the similarity of the "
                "correct paired example"
            ),

        "mean_retrieval_recall_mR":
            (
                "arithmetic mean of C2A R@1,R@5,R@10 "
                "and A2C R@1,R@5,R@10"
            ),

        "primary_model_selection_metric":
            "mR",

        "model_selection":
            (
                "select T4-A or T4-B using highest seed-42 "
                "VAL mR only"
            ),

        "exact_model_tie_rule":
            (
                "if VAL mR is exactly tied, choose simpler T4-A"
            ),

        "test_used_for_selection":
            False,
    },

    "randomness": {

        "formal_model_selection_seed":
            PRIMARY_SELECTION_SEED,

        "final_selected_configuration_seeds":
            FINAL_SEEDS,

        "seed_sources":
            [
                "python random",
                "numpy",
                "torch CPU",
                "torch CUDA",
            ],

        "cudnn_deterministic":
            True,

        "cudnn_benchmark":
            False,

        "tf32":
            False,
    },

    "final_multiseed_policy": {

        "selection_basis":
            "seed-42 validation mR",

        "seeds":
            FINAL_SEEDS,

        "if_t4_a_wins":
            (
                "train T4-A independently under each final seed"
            ),

        "if_t4_b_wins":
            (
                "for each final seed, independently run the full "
                "same-seed T4-A initialization stage followed by "
                "same-seed T4-B partial fine-tuning"
            ),

        "reported_statistics":
            (
                "mean ± standard deviation over seeds for all "
                "six retrieval recalls and mR"
            ),

        "test_access_during_multiseed_training":
            False,
    },

    "final_test_protocol": {

        "allowed_only_after":
            [
                "T4-A formal run complete",
                "T4-B formal run complete",
                "VAL winner frozen",
                "final multi-seed checkpoints complete",
                "pre-TEST Task-4 lock complete",
            ],

        "candidate_pool":
            "all 514 frozen TEST examples",

        "similarity_matrix_shape":
            [
                EXPECTED_TEST,
                EXPECTED_TEST,
            ],

        "metrics":
            [
                "caption_to_audio_R@1",
                "caption_to_audio_R@5",
                "caption_to_audio_R@10",
                "audio_to_caption_R@1",
                "audio_to_caption_R@5",
                "audio_to_caption_R@10",
                "mR",
            ],

        "random_baseline_expected_recall": {

            "R@1":
                1.0 / EXPECTED_TEST,

            "R@5":
                5.0 / EXPECTED_TEST,

            "R@10":
                10.0 / EXPECTED_TEST,
        },

        "no_post_test_model_tuning":
            True,
    },

    "zero_shot_tag_protocol": {

        "status":
            "predeclared_secondary_task4_analysis",

        "label_vocabulary":
            (
                "reuse the already frozen Task-1/2/3 "
                "20-label vocabulary unchanged"
            ),

        "label_text":
            (
                "use each exact canonical label phrase as text; "
                "no prompt engineering after seeing TEST results"
            ),

        "required_mode":
            (
                "caption embedding versus canonical label-text "
                "embeddings"
            ),

        "additional_cross_modal_mode":
            (
                "audio embedding versus canonical label-text "
                "embeddings"
            ),

        "score":
            "cosine similarity in frozen Task-4 embedding space",

        "primary_threshold_free_metric":
            "Macro Average Precision",

        "f1_if_reported":
            (
                "one global threshold selected using VAL only "
                "and then frozen before TEST"
            ),

        "task4_contrastive_training_uses_labels":
            False,
    },

    "qualitative_retrieval_protocol": {

        "required_examples":
            10,

        "query_direction":
            "caption_to_audio",

        "retrieved_items_per_query":
            3,

        "selection_rule":
            (
                "deterministically sample 10 TEST query indices "
                "using seed 42; do not hand-pick based on retrieval "
                "success or failure"
            ),

        "report_fields":
            [
                "query caption",
                "top-1 retrieved track",
                "top-2 retrieved track",
                "top-3 retrieved track",
                "similarities",
                "ground-truth track rank",
            ],
    },

    "human_evaluation_protocol": {

        "planned":
            True,

        "minimum_listeners":
            5,

        "queries":
            10,

        "queries_source":
            (
                "same deterministic 10-query qualitative sample"
            ),

        "evaluated_item":
            "top-1 retrieved audio for each caption",

        "rating_scale":
            [
                1,
                2,
                3,
                4,
                5,
            ],

        "scale_meaning":
            {
                "1": "very poor match",
                "2": "poor match",
                "3": "moderate match",
                "4": "good match",
                "5": "very good match",
            },

        "report":
            "mean and standard deviation of listener ratings",
    },

    "hard_prohibitions": [

        "do not modify the 5140-track canonical dataset",
        "do not regenerate train/val/test splits",
        "do not change the frozen 20-label vocabulary",
        "do not rebuild G2 based on Task-4 results",
        "do not replace T1-B based on Task-4 results",
        "do not replace A4/G2 based on Task-4 results",
        "do not use TEST retrieval to choose architecture",
        "do not use TEST retrieval to tune temperature",
        "do not use TEST retrieval to tune batch size",
        "do not use TEST retrieval to tune projection dimension",
        "do not use 20-label supervision for Task-4 contrastive training",
        "do not alter Tasks 1–3 after Task-4 results",
    ],

    "script_sequence": {

        "64":
            "Task-4 paired-data preflight",

        "65":
            "freeze Task-4 contrastive design",

        "66":
            "contrastive smoke test",

        "67":
            "formal T4-A training",

        "68":
            "formal T4-B partial fine-tuning",

        "69":
            "VAL comparison and Task-4 winner freeze",

        "70":
            "final winner multi-seed reruns",

        "71":
            "final Task-4 pre-TEST lock",

        "72":
            "one final TEST retrieval stage",

        "73":
            "zero-shot tag evaluation",

        "74":
            "10 qualitative retrieval examples",

        "75":
            "human evaluation assets/results",

        "76":
            "Task-4 figures/tables/final artifact lock",
    },

    "provenance": {

        "script64_source":
            safe_relative(
                SCRIPT64_PATH
            ),

        "script64_sha256":
            script64_sha256,

        "script65_source":
            safe_relative(
                Path(__file__).resolve()
            ),

        "script65_sha256":
            sha256_file(
                Path(__file__).resolve()
            ),

        "pretraining_lock_sha256":
            EXPECTED_PRETRAINING_LOCK_SHA256,

        "t1b_checkpoint_sha256":
            EXPECTED_T1B_SHA256,

        "a4_checkpoint_sha256":
            EXPECTED_A4_SHA256,

        "bert_cache_lock_sha256":
            EXPECTED_BERT_LOCK_SHA256,

        "gat_cache_lock_sha256":
            EXPECTED_GAT_LOCK_SHA256,
    },
}


# =============================================================================
# 9. DISPLAY FROZEN DESIGN
# =============================================================================

section(
    "9. FROZEN TASK-4 DESIGN SUMMARY"
)

print(
    f"Shared embedding dimension : {SHARED_EMBEDDING_DIM}"
)

print(
    f"Temperature τ              : {TEMPERATURE}"
)

print(
    "Loss                       : symmetric InfoNCE"
)

print(
    "Similarity                 : cosine / normalized dot product"
)

print(
    f"Batch size                 : {TRAIN_BATCH_SIZE}"
)

print()
print(
    "T4-A:"
)

print(
    "  frozen T1-B + frozen A4/G2"
)

print(
    "  train projection heads only"
)

print(
    f"  LR={T4A_LR}"
)

print(
    f"  max_epochs={T4A_MAX_EPOCHS}"
)

print(
    f"  patience={T4A_PATIENCE}"
)

print()
print(
    "T4-B:"
)

print(
    "  final 2 BERT blocks trainable"
)

print(
    "  final GAT encoder block trainable"
)

print(
    "  projection heads trainable"
)

print(
    f"  projection LR={T4B_PROJECTION_LR}"
)

print(
    f"  BERT LR={T4B_BERT_LR}"
)

print(
    f"  GAT LR={T4B_GAT_LR}"
)

print(
    f"  max_epochs={T4B_MAX_EPOCHS}"
)

print(
    f"  patience={T4B_PATIENCE}"
)

print()
print(
    "VAL model selection:"
)

print(
    "  Caption→Audio R@1/R@5/R@10"
)

print(
    "  Audio→Caption R@1/R@5/R@10"
)

print(
    "  primary = mean of all six recalls (mR)"
)

print()
print(
    "Final seeds:"
)

print(
    f"  {FINAL_SEEDS}"
)

print()
print(
    "TEST during Scripts 65–71:"
)

print(
    "  FORBIDDEN"
)


# =============================================================================
# 10. WRITE FROZEN DESIGN
# =============================================================================

section(
    "10. WRITE TASK-4 DESIGN ARTIFACT"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

atomic_write_json(
    DESIGN_PATH,
    design,
)

require(
    DESIGN_PATH.exists(),
    "Task-4 design JSON written",
)

design_sha256 = sha256_file(
    DESIGN_PATH
)

print(
    "Task-4 design SHA256:"
)

print(
    f"  {design_sha256}"
)


# =============================================================================
# 11. WRITE DESIGN LOCK
# =============================================================================

section(
    "11. WRITE TASK-4 DESIGN LOCK"
)

design_lock = {

    "lock_type":
        "task4_contrastive_design_lock",

    "version":
        1,

    "created_utc":
        created_utc,

    "design_path":
        safe_relative(
            DESIGN_PATH
        ),

    "design_sha256":
        design_sha256,

    "script65_path":
        safe_relative(
            Path(__file__).resolve()
        ),

    "script65_sha256":
        sha256_file(
            Path(__file__).resolve()
        ),

    "parent_artifacts": {

        "pretraining_lock_sha256":
            EXPECTED_PRETRAINING_LOCK_SHA256,

        "t1b_checkpoint_sha256":
            EXPECTED_T1B_SHA256,

        "a4_checkpoint_sha256":
            EXPECTED_A4_SHA256,

        "bert_sequence_sha256":
            EXPECTED_BERT_SEQUENCE_SHA256,

        "gat_embedding_sha256":
            EXPECTED_GAT_EMBEDDING_SHA256,
    },

    "development_population": {

        "train":
            EXPECTED_TRAIN,

        "val":
            EXPECTED_VAL,

        "test":
            0,

        "total":
            EXPECTED_DEV,
    },

    "formal_task4_result_observed_before_lock":
        False,

    "test_retrieval_observed_before_lock":
        False,
}

atomic_write_json(
    DESIGN_LOCK_PATH,
    design_lock,
)

require(
    DESIGN_LOCK_PATH.exists(),
    "Task-4 design lock written",
)

design_lock_sha256 = sha256_file(
    DESIGN_LOCK_PATH
)

print(
    "Task-4 design lock SHA256:"
)

print(
    f"  {design_lock_sha256}"
)


# =============================================================================
# 12. POST-WRITE VALIDATION
# =============================================================================

section(
    "12. POST-WRITE VALIDATION"
)

written_design = load_json(
    DESIGN_PATH
)

written_lock = load_json(
    DESIGN_LOCK_PATH
)

require(
    written_design.get(
        "status"
    )
    == "frozen_before_any_formal_task4_result",
    "design records pre-result freeze status",
)

require(
    written_design[
        "shared_embedding_space"
    ][
        "dimension"
    ]
    == SHARED_EMBEDDING_DIM,
    "shared embedding dimension frozen at 256",
)

require(
    written_design[
        "contrastive_objective"
    ][
        "temperature"
    ]
    == TEMPERATURE,
    "temperature frozen at 0.07",
)

require(
    written_design[
        "t4_a"
    ][
        "batch_size"
    ]
    == TRAIN_BATCH_SIZE,
    "T4-A batch size frozen at 32",
)

require(
    written_design[
        "t4_b"
    ][
        "batch_size"
    ]
    == TRAIN_BATCH_SIZE,
    "T4-B batch size frozen at 32",
)

require(
    written_design[
        "validation_retrieval"
    ][
        "primary_model_selection_metric"
    ]
    == "mR",
    "VAL mR frozen as primary model-selection metric",
)

require(
    written_design[
        "dataset"
    ][
        "test_during_development"
    ]
    is False,
    "TEST remains forbidden during Task-4 development",
)

require(
    written_lock[
        "design_sha256"
    ]
    == sha256_file(
        DESIGN_PATH
    ),
    "design lock matches design artifact",
)


# =============================================================================
# 13. EXECUTION BOUNDARY
# =============================================================================

section(
    "13. EXECUTION-BOUNDARY CONFIRMATION"
)

print(
    "Script 65 did not instantiate BERT."
)

print(
    "Script 65 did not instantiate GAT."
)

print(
    "Script 65 did not perform model inference."
)

print(
    "Script 65 did not perform training."
)

print(
    "Script 65 did not compute InfoNCE."
)

print(
    "Script 65 did not compute retrieval metrics."
)

print(
    "Script 65 did not access TEST examples."
)

print(
    "Script 65 did not access TEST labels."
)

print(
    "Script 65 did not perform TEST retrieval."
)

print(
    "Script 65 did not modify Tasks 1–3."
)

passed(
    "Task-4 design was frozen before any formal Task-4 result"
)


# =============================================================================
# SUMMARY
# =============================================================================

section(
    "SCRIPT 65 SUMMARY"
)

print(
    f"Failures: {len(FAILURES)}"
)

for message in FAILURES:
    print(
        f"  - {message}"
    )

print()
print(
    f"Warnings: {len(WARNINGS)}"
)

for message in WARNINGS:
    print(
        f"  - {message}"
    )

print()

if FAILURES:

    print(
        "RESULT: FAIL"
    )

    print()
    print(
        "Do NOT proceed to Script 66."
    )

    sys.exit(1)


print(
    "RESULT: PASS"
)

print()
print(
    "TASK-4 EXPERIMENTAL DESIGN IS NOW FROZEN."
)

print()
print(
    f"Design:"
)

print(
    f"  {safe_relative(DESIGN_PATH)}"
)

print(
    f"  SHA256 = {design_sha256}"
)

print()
print(
    f"Lock:"
)

print(
    f"  {safe_relative(DESIGN_LOCK_PATH)}"
)

print(
    f"  SHA256 = {design_lock_sha256}"
)

print()
print(
    "No formal Task-4 model has been trained."
)

print(
    "No Task-4 retrieval result has been observed."
)

print(
    "No TEST retrieval has occurred."
)

print()
print(
    "STOP HERE."
)

print(
    "Next: Script 66 — contrastive dual-encoder smoke test."
)

print(
    "Any hardware-mandated design change discovered by Script 66 "
    "must be documented BEFORE Scripts 67/68 formal results."
)