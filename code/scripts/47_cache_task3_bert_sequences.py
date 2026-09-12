#!/usr/bin/env python3
"""
Script 47 — Cache Frozen Task-3 BERT Sequence Representations
=============================================================

Purpose
-------
Generate final-layer BERT sequence representations from the already-selected
Task-1 T1-B checkpoint for Task-3 Phase A.

STRICT DEVELOPMENT BOUNDARY
---------------------------
Included:
    TRAIN
    VAL

Excluded:
    TEST

This script performs inference only with the frozen selected T1-B encoder.

It performs NO:
    - model training
    - gradient updates
    - optimizer steps
    - threshold tuning
    - TEST inference
    - TEST metric calculation
    - TEST label access
    - modification of frozen preprocessing artifacts

Output
------
data/processed/bert_seq_frozen/

    bert_sequence_trainval.npy
        shape = (4626, 112, 768)
        dtype = float32

    attention_mask_trainval.npy
        shape = (4626, 112)
        dtype = uint8

    bert_seq_manifest.parquet

    bert_seq_cache_metadata.json

    bert_seq_frozen_hashes.json

The full sequence is cached because:

    M3 Early Fusion:
        uses sequence[:, 0, :]  -> CLS representation, 768-D

    M4 Cross-Attention:
        uses the full final-layer sequence, 112 x 768

Thus the two Task-3 variants use the same frozen BERT representation source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import BertConfig, BertModel


# =============================================================================
# 0. PROJECT / FROZEN CONTRACT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETRAINING_LOCK = (
    PROJECT_ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

TASK1_SELECTION = (
    PROJECT_ROOT
    / "results/runs/task1/task1_model_selection.json"
)

T1_CHECKPOINT = (
    PROJECT_ROOT
    / "models/task1_candidates/"
    "task1_T1-B_seed42_20260828T182131Z_best.pt"
)

TOKEN_CACHE = (
    PROJECT_ROOT
    / "data/processed/bert_tokens/bert_tokens.npz"
)

TOKEN_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/bert_tokens/bert_token_manifest.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/bert_seq_frozen"
)

TEMP_DIR = (
    PROJECT_ROOT
    / "data/processed/bert_seq_frozen.__tmp__"
)


# -----------------------------------------------------------------------------
# Frozen hashes established before Script 47
# -----------------------------------------------------------------------------

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_T1_CHECKPOINT_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)


# -----------------------------------------------------------------------------
# Frozen model/data dimensions
# -----------------------------------------------------------------------------

EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_TEST = 514
EXPECTED_TOTAL = 5140

EXPECTED_DEVELOPMENT_ROWS = (
    EXPECTED_TRAIN + EXPECTED_VAL
)

EXPECTED_MAX_LENGTH = 112
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_BERT_LAYERS = 12
EXPECTED_LABELS = 20

EXPECTED_VOCAB_SIZE = 30522
EXPECTED_POSITION_EMBEDDINGS = 512
EXPECTED_TYPE_VOCAB_SIZE = 2

MODEL_NAME = "bert-base-uncased"


# -----------------------------------------------------------------------------
# Cache-generation settings
#
# Batch size changes only inference batching, not the frozen model or scientific
# model-selection configuration. We use the existing project batch size of 32
# conservatively.
# -----------------------------------------------------------------------------

BATCH_SIZE = 32

CACHE_DTYPE = np.float32

SEED = 42


# =============================================================================
# 1. REPORTING
# =============================================================================

FAILURES: list[str] = []


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(message: str) -> None:
    print(f"PASS  {message}")


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    FAILURES.append(message)


def check(condition: bool, message: str) -> bool:
    if condition:
        passed(message)
        return True

    fail(message)
    return False


# =============================================================================
# 2. HELPERS
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """
    Calculate SHA256 without loading the entire file into memory.
    """

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    path: Path,
    obj: Any,
) -> None:
    with path.open(
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

        f.write("\n")


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    """
    Load our own trusted local project checkpoint onto CPU.
    """

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "T1-B checkpoint is not a dictionary"
        )

    return checkpoint


def relative(path: Path) -> str:
    return str(
        path.relative_to(PROJECT_ROOT)
    )


# =============================================================================
# 3. REPRODUCIBILITY / NUMERIC POLICY
# =============================================================================

def configure_reproducibility() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # We deliberately preserve full FP32 inference.
    #
    # No autocast.
    # No float16 inference.
    # No TF32 acceleration.
    #
    # This avoids introducing an unnecessary precision change into the frozen
    # representation cache.

    if torch.cuda.is_available():

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
# 4. VERIFY FROZEN INPUT ARTIFACTS
# =============================================================================

def verify_frozen_inputs() -> None:

    banner(
        "1. FROZEN INPUT ARTIFACTS"
    )

    required_paths = (
        PRETRAINING_LOCK,
        TASK1_SELECTION,
        T1_CHECKPOINT,
        TOKEN_CACHE,
        TOKEN_MANIFEST,
    )

    for path in required_paths:
        check(
            path.is_file(),
            f"required artifact exists: {relative(path)}",
        )

    if FAILURES:
        raise RuntimeError(
            "Required frozen artifacts are missing"
        )

    # -------------------------------------------------------------------------
    # Frozen preprocessing lock
    # -------------------------------------------------------------------------

    lock_sha = sha256_file(
        PRETRAINING_LOCK
    )

    print(
        f"Pretraining lock SHA256: {lock_sha}"
    )

    check(
        lock_sha
        == EXPECTED_PRETRAINING_LOCK_SHA256,
        "pretraining lock remains byte-for-byte unchanged",
    )

    # -------------------------------------------------------------------------
    # Selected checkpoint
    # -------------------------------------------------------------------------

    checkpoint_sha = sha256_file(
        T1_CHECKPOINT
    )

    print(
        f"T1-B checkpoint SHA256: {checkpoint_sha}"
    )

    check(
        checkpoint_sha
        == EXPECTED_T1_CHECKPOINT_SHA256,
        "selected T1-B checkpoint SHA256 is unchanged",
    )

    # -------------------------------------------------------------------------
    # Frozen Task-1 selection
    # -------------------------------------------------------------------------

    selection = load_json(
        TASK1_SELECTION
    )

    selection_text = json.dumps(
        selection,
        sort_keys=True,
    ).lower()

    check(
        "t1-b" in selection_text,
        "Task-1 selection still identifies T1-B",
    )

    check(
        T1_CHECKPOINT.name.lower()
        in selection_text,
        (
            "Task-1 selection still references "
            "the exact T1-B checkpoint"
        ),
    )

    check(
        EXPECTED_T1_CHECKPOINT_SHA256
        in selection_text,
        (
            "Task-1 selection still records "
            "the expected checkpoint SHA256"
        ),
    )

    if isinstance(selection, dict):

        if "test_split_used" in selection:
            check(
                selection["test_split_used"] is False,
                (
                    "Task-1 frozen selection records "
                    "TEST split unused"
                ),
            )

    if FAILURES:
        raise RuntimeError(
            "Frozen input validation failed"
        )


# =============================================================================
# 5. OUTPUT COLLISION PROTECTION
# =============================================================================

def verify_clean_output_location() -> None:

    banner(
        "2. OUTPUT COLLISION PROTECTION"
    )

    check(
        not OUTPUT_DIR.exists(),
        (
            "final bert_seq_frozen output directory "
            "does not already exist"
        ),
    )

    check(
        not TEMP_DIR.exists(),
        (
            "temporary bert_seq_frozen.__tmp__ "
            "directory does not already exist"
        ),
    )

    if FAILURES:
        raise RuntimeError(
            "Output collision detected; refusing to overwrite"
        )


# =============================================================================
# 6. TOKEN MANIFEST VALIDATION
# =============================================================================

def load_and_validate_manifest() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    banner(
        "3. TOKEN MANIFEST / DEVELOPMENT-SPLIT VALIDATION"
    )

    manifest = pd.read_parquet(
        TOKEN_MANIFEST
    )

    expected_columns = [
        "cache_index",
        "track_id",
        "split",
        "cached_token_length",
    ]

    print(
        f"Manifest shape: {manifest.shape}"
    )

    print(
        f"Manifest columns: {list(manifest.columns)}"
    )

    check(
        list(manifest.columns)
        == expected_columns,
        "token manifest has the exact frozen schema",
    )

    check(
        len(manifest)
        == EXPECTED_TOTAL,
        (
            "token manifest contains exactly "
            f"{EXPECTED_TOTAL} tracks"
        ),
    )

    check(
        manifest["track_id"].nunique()
        == EXPECTED_TOTAL,
        "token-manifest track IDs are unique",
    )

    cache_indices = pd.to_numeric(
        manifest["cache_index"],
        errors="raise",
    ).astype(np.int64)

    check(
        np.array_equal(
            cache_indices.to_numpy(),
            np.arange(
                EXPECTED_TOTAL,
                dtype=np.int64,
            ),
        ),
        "cache_index is exactly contiguous 0..5139",
    )

    normalized_split = (
        manifest["split"]
        .astype(str)
        .str.lower()
    )

    split_counts = (
        normalized_split
        .value_counts()
        .to_dict()
    )

    print(
        f"Frozen split counts: {split_counts}"
    )

    check(
        split_counts
        == {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },
        "frozen split counts remain exactly 4112/514/514",
    )

    # -------------------------------------------------------------------------
    # Development rows only.
    #
    # Preserve original frozen token-manifest ordering.
    # -------------------------------------------------------------------------

    development_mask = (
        normalized_split.isin(
            ["train", "val"]
        )
    )

    development = (
        manifest.loc[
            development_mask,
            expected_columns,
        ]
        .copy()
        .reset_index(drop=True)
    )

    development["split"] = (
        development["split"]
        .astype(str)
        .str.lower()
    )

    check(
        len(development)
        == EXPECTED_DEVELOPMENT_ROWS,
        (
            "TRAIN+VAL development subset contains "
            f"exactly {EXPECTED_DEVELOPMENT_ROWS} rows"
        ),
    )

    development_counts = (
        development["split"]
        .value_counts()
        .to_dict()
    )

    print(
        "Development split counts: "
        f"{development_counts}"
    )

    check(
        development_counts
        == {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
        },
        "development cache contains TRAIN and VAL only",
    )

    check(
        not (
            development["split"]
            == "test"
        ).any(),
        "zero TEST rows are present in the development cache plan",
    )

    source_indices = (
        development["cache_index"]
        .astype(np.int64)
        .to_numpy()
    )

    # Prove selected indices correspond only to TRAIN or VAL.
    selected_split_values = (
        normalized_split.iloc[
            source_indices
        ]
        .to_numpy()
    )

    check(
        np.all(
            np.isin(
                selected_split_values,
                ["train", "val"],
            )
        ),
        (
            "every source token-cache index selected "
            "for inference belongs to TRAIN or VAL"
        ),
    )

    if FAILURES:
        raise RuntimeError(
            "Development-split validation failed"
        )

    return manifest, development


# =============================================================================
# 7. TOKEN CACHE VALIDATION
# =============================================================================

def load_and_validate_token_cache() -> dict[str, np.ndarray]:

    banner(
        "4. FROZEN TOKEN CACHE VALIDATION"
    )

    cache_npz = np.load(
        TOKEN_CACHE,
        allow_pickle=False,
    )

    required = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }

    check(
        required.issubset(
            set(cache_npz.files)
        ),
        "token cache contains all three required arrays",
    )

    if FAILURES:
        cache_npz.close()

        raise RuntimeError(
            "Token cache schema validation failed"
        )

    arrays = {
        "input_ids": cache_npz["input_ids"],
        "attention_mask": cache_npz["attention_mask"],
        "token_type_ids": cache_npz["token_type_ids"],
    }

    cache_npz.close()

    expected_shape = (
        EXPECTED_TOTAL,
        EXPECTED_MAX_LENGTH,
    )

    for name, array in arrays.items():

        print(
            f"{name}: "
            f"shape={array.shape}, "
            f"dtype={array.dtype}"
        )

        check(
            array.shape == expected_shape,
            f"{name} has exact shape {expected_shape}",
        )

    check(
        arrays["input_ids"].dtype
        == np.int32,
        "input_ids dtype is int32",
    )

    check(
        arrays["attention_mask"].dtype
        == np.uint8,
        "attention_mask dtype is uint8",
    )

    check(
        arrays["token_type_ids"].dtype
        == np.uint8,
        "token_type_ids dtype is uint8",
    )

    # Global structural checks only.
    # We do not print or inspect individual TEST examples.

    check(
        np.min(arrays["input_ids"]) >= 0,
        "token IDs are non-negative",
    )

    check(
        np.max(arrays["input_ids"])
        < EXPECTED_VOCAB_SIZE,
        (
            "all token IDs are within "
            "bert-base-uncased vocabulary range"
        ),
    )

    unique_mask_values = np.unique(
        arrays["attention_mask"]
    )

    check(
        set(
            unique_mask_values.tolist()
        ).issubset({0, 1}),
        "attention mask is binary",
    )

    unique_type_values = np.unique(
        arrays["token_type_ids"]
    )

    check(
        set(
            unique_type_values.tolist()
        ).issubset({0, 1}),
        "token_type_ids are within the expected BERT range",
    )

    if FAILURES:
        raise RuntimeError(
            "Frozen token-cache validation failed"
        )

    return arrays


# =============================================================================
# 8. LOAD EXACT SELECTED T1-B ENCODER
# =============================================================================

def build_frozen_t1b_encoder(
    device: torch.device,
) -> tuple[
    BertModel,
    dict[str, Any],
    BertConfig,
]:

    banner(
        "5. LOAD SELECTED T1-B BERT ENCODER"
    )

    checkpoint = torch_load_checkpoint(
        T1_CHECKPOINT
    )

    required_checkpoint_fields = (
        "task",
        "variant",
        "seed",
        "epoch",
        "architecture",
        "num_labels",
        "model_state_dict",
    )

    for field in required_checkpoint_fields:
        check(
            field in checkpoint,
            f"T1-B checkpoint contains field '{field}'",
        )

    if FAILURES:
        raise RuntimeError(
            "T1-B checkpoint metadata incomplete"
        )

    check(
        checkpoint["task"] == "task1",
        "checkpoint task is exactly task1",
    )

    check(
        checkpoint["variant"] == "T1-B",
        "checkpoint variant is exactly T1-B",
    )

    check(
        int(checkpoint["seed"]) == 42,
        "checkpoint seed is exactly 42",
    )

    check(
        int(checkpoint["epoch"]) == 8,
        "selected checkpoint epoch is exactly 8",
    )

    check(
        int(checkpoint["num_labels"])
        == EXPECTED_LABELS,
        "checkpoint contains exactly 20 output labels",
    )

    architecture = str(
        checkpoint["architecture"]
    )

    print(
        f"Checkpoint architecture: {architecture}"
    )

    check(
        "CLS(768)" in architecture,
        "checkpoint architecture explicitly uses CLS(768)",
    )

    check(
        "Linear(768,20)" in architecture,
        (
            "checkpoint architecture explicitly "
            "uses Linear(768,20)"
        ),
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "model_state_dict is not a dictionary"
        )

    # -------------------------------------------------------------------------
    # Load the locally cached bert-base-uncased CONFIG only.
    #
    # We deliberately do not download anything and do not trust an inferred
    # architecture. The project already used bert-base-uncased during Task 1,
    # so its config should be locally available.
    # -------------------------------------------------------------------------

    try:
        config = BertConfig.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not load the locally cached "
            "bert-base-uncased configuration. "
            "Refusing to guess the configuration."
        ) from exc

    print(
        "Loaded local BERT config:"
    )

    print(
        f"  model_type           = {config.model_type}"
    )

    print(
        f"  vocab_size           = {config.vocab_size}"
    )

    print(
        f"  hidden_size          = {config.hidden_size}"
    )

    print(
        f"  num_hidden_layers    = {config.num_hidden_layers}"
    )

    print(
        f"  num_attention_heads  = {config.num_attention_heads}"
    )

    print(
        f"  intermediate_size    = {config.intermediate_size}"
    )

    print(
        f"  max_position_embeddings = "
        f"{config.max_position_embeddings}"
    )

    print(
        f"  type_vocab_size      = {config.type_vocab_size}"
    )

    check(
        config.model_type == "bert",
        "local configuration is a BERT configuration",
    )

    check(
        config.vocab_size
        == EXPECTED_VOCAB_SIZE,
        "BERT vocabulary size is exactly 30522",
    )

    check(
        config.hidden_size
        == EXPECTED_HIDDEN_SIZE,
        "BERT hidden size is exactly 768",
    )

    check(
        config.num_hidden_layers
        == EXPECTED_BERT_LAYERS,
        "BERT has exactly 12 encoder layers",
    )

    check(
        config.num_attention_heads == 12,
        "BERT has exactly 12 attention heads",
    )

    check(
        config.intermediate_size == 3072,
        "BERT intermediate size is exactly 3072",
    )

    check(
        config.max_position_embeddings
        == EXPECTED_POSITION_EMBEDDINGS,
        "BERT position-embedding capacity is exactly 512",
    )

    check(
        config.type_vocab_size
        == EXPECTED_TYPE_VOCAB_SIZE,
        "BERT type vocabulary size is exactly 2",
    )

    check(
        EXPECTED_MAX_LENGTH
        <= config.max_position_embeddings,
        (
            "frozen sequence length 112 fits "
            "inside BERT positional capacity"
        ),
    )

    if FAILURES:
        raise RuntimeError(
            "Local BERT configuration does not match "
            "the frozen Task-1 architecture"
        )

    # -------------------------------------------------------------------------
    # Extract the BERT submodule weights.
    #
    # Script 46 proved checkpoint keys use:
    #
    #     bert....
    #     classifier....
    #
    # For Task-3 cache generation we use only the selected BERT encoder.
    # -------------------------------------------------------------------------

    bert_state_dict = {}

    non_bert_tensor_keys = []

    for key, value in state_dict.items():

        if not torch.is_tensor(value):
            continue

        if key.startswith("bert."):
            stripped_key = key[
                len("bert.") :
            ]

            bert_state_dict[
                stripped_key
            ] = value

        else:
            non_bert_tensor_keys.append(
                key
            )

    print(
        f"Full checkpoint tensor count: "
        f"{sum(torch.is_tensor(v) for v in state_dict.values())}"
    )

    print(
        f"BERT tensor count extracted: "
        f"{len(bert_state_dict)}"
    )

    print(
        "Non-BERT tensor keys excluded "
        "from Task-3 encoder:"
    )

    for key in non_bert_tensor_keys:
        print(f"  {key}")

    check(
        set(non_bert_tensor_keys)
        == {
            "classifier.weight",
            "classifier.bias",
        },
        (
            "the only excluded T1-B parameters are "
            "the original 20-label classifier"
        ),
    )

    # Instantiate architecture from verified local config.
    #
    # No pretrained download occurs here.
    # The selected T1-B weights are loaded immediately afterward.

    model = BertModel(
        config,
        add_pooling_layer=True,
    )

    try:
        incompatible = model.load_state_dict(
            bert_state_dict,
            strict=True,
        )

    except RuntimeError as exc:
        raise RuntimeError(
            "Selected T1-B BERT weights do not strictly match "
            "the verified bert-base-uncased architecture"
        ) from exc

    check(
        len(incompatible.missing_keys) == 0,
        "strict T1-B BERT load has zero missing keys",
    )

    check(
        len(incompatible.unexpected_keys) == 0,
        "strict T1-B BERT load has zero unexpected keys",
    )

    # Freeze encoder completely.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()

    check(
        not model.training,
        "T1-B BERT is in eval mode",
    )

    check(
        all(
            not parameter.requires_grad
            for parameter in model.parameters()
        ),
        "all T1-B BERT parameters are frozen",
    )

    model.to(device)

    if FAILURES:
        raise RuntimeError(
            "T1-B encoder loading validation failed"
        )

    return model, checkpoint, config


# =============================================================================
# 9. PREPARE OUTPUT MANIFEST
# =============================================================================

def build_output_manifest(
    development: pd.DataFrame,
) -> pd.DataFrame:

    manifest = pd.DataFrame(
        {
            "cache_index": np.arange(
                len(development),
                dtype=np.int32,
            ),
            "track_id": (
                development["track_id"]
                .astype(str)
                .to_numpy()
            ),
            "split": (
                development["split"]
                .astype(str)
                .str.lower()
                .to_numpy()
            ),
            "source_token_cache_index": (
                development["cache_index"]
                .astype(np.int32)
                .to_numpy()
            ),
            "cached_token_length": (
                development[
                    "cached_token_length"
                ]
                .astype(np.int16)
                .to_numpy()
            ),
        }
    )

    check(
        len(manifest)
        == EXPECTED_DEVELOPMENT_ROWS,
        (
            "output manifest contains exactly "
            f"{EXPECTED_DEVELOPMENT_ROWS} rows"
        ),
    )

    check(
        manifest["track_id"].nunique()
        == EXPECTED_DEVELOPMENT_ROWS,
        "output manifest track IDs are unique",
    )

    check(
        np.array_equal(
            manifest["cache_index"]
            .to_numpy(),
            np.arange(
                EXPECTED_DEVELOPMENT_ROWS,
                dtype=np.int32,
            ),
        ),
        (
            "output cache indices are contiguous "
            "0..4625"
        ),
    )

    split_counts = (
        manifest["split"]
        .value_counts()
        .to_dict()
    )

    check(
        split_counts
        == {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
        },
        (
            "output manifest contains exactly "
            "4112 TRAIN and 514 VAL rows"
        ),
    )

    check(
        "test"
        not in set(
            manifest["split"]
        ),
        "output manifest contains zero TEST rows",
    )

    return manifest


# =============================================================================
# 10. CACHE GENERATION
# =============================================================================

def generate_cache(
    model: BertModel,
    device: torch.device,
    token_arrays: dict[str, np.ndarray],
    output_manifest: pd.DataFrame,
) -> dict[str, Any]:

    banner(
        "6. GENERATE TRAIN+VAL FROZEN BERT SEQUENCE CACHE"
    )

    TEMP_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    seq_path = (
        TEMP_DIR
        / "bert_sequence_trainval.npy"
    )

    mask_path = (
        TEMP_DIR
        / "attention_mask_trainval.npy"
    )

    sequence_shape = (
        EXPECTED_DEVELOPMENT_ROWS,
        EXPECTED_MAX_LENGTH,
        EXPECTED_HIDDEN_SIZE,
    )

    mask_shape = (
        EXPECTED_DEVELOPMENT_ROWS,
        EXPECTED_MAX_LENGTH,
    )

    print(
        f"Sequence cache shape: {sequence_shape}"
    )

    print(
        f"Sequence cache dtype: {CACHE_DTYPE}"
    )

    print(
        f"Attention-mask shape: {mask_shape}"
    )

    # Use open_memmap so the full ~1.6 GB cache is never held in RAM.
    sequence_cache = np.lib.format.open_memmap(
        seq_path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=sequence_shape,
    )

    attention_cache = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=mask_shape,
    )

    source_indices = (
        output_manifest[
            "source_token_cache_index"
        ]
        .astype(np.int64)
        .to_numpy()
    )

    total_batches = math.ceil(
        EXPECTED_DEVELOPMENT_ROWS
        / BATCH_SIZE
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Total inference batches: {total_batches}"
    )

    print(
        "Inference splits: TRAIN + VAL only"
    )

    print(
        "TEST inference examples: 0"
    )

    rows_written = 0
    finite_rows = 0

    global_abs_max = 0.0

    with torch.inference_mode():

        for batch_number, start in enumerate(
            range(
                0,
                EXPECTED_DEVELOPMENT_ROWS,
                BATCH_SIZE,
            ),
            start=1,
        ):

            end = min(
                start + BATCH_SIZE,
                EXPECTED_DEVELOPMENT_ROWS,
            )

            batch_source_indices = (
                source_indices[start:end]
            )

            # -------------------------------------------------------------
            # Safety proof:
            # these source indices came from the TRAIN+VAL-only manifest.
            # TEST indices never enter the model.
            # -------------------------------------------------------------

            batch_splits = (
                output_manifest[
                    "split"
                ]
                .iloc[start:end]
                .to_numpy()
            )

            if np.any(
                batch_splits == "test"
            ):
                raise RuntimeError(
                    "TEST row detected in inference batch. "
                    "Aborting immediately."
                )

            # -------------------------------------------------------------
            # Build only the selected TRAIN/VAL tensors for the current batch.
            # -------------------------------------------------------------

            input_ids_np = (
                token_arrays["input_ids"][
                    batch_source_indices
                ]
            )

            attention_mask_np = (
                token_arrays[
                    "attention_mask"
                ][
                    batch_source_indices
                ]
            )

            token_type_ids_np = (
                token_arrays[
                    "token_type_ids"
                ][
                    batch_source_indices
                ]
            )

            input_ids = torch.from_numpy(
                input_ids_np.astype(
                    np.int64,
                    copy=False,
                )
            ).to(
                device=device,
                dtype=torch.long,
                non_blocking=False,
            )

            attention_mask = torch.from_numpy(
                attention_mask_np.astype(
                    np.int64,
                    copy=False,
                )
            ).to(
                device=device,
                dtype=torch.long,
                non_blocking=False,
            )

            token_type_ids = torch.from_numpy(
                token_type_ids_np.astype(
                    np.int64,
                    copy=False,
                )
            ).to(
                device=device,
                dtype=torch.long,
                non_blocking=False,
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )

            hidden = (
                outputs.last_hidden_state
            )

            expected_batch_shape = (
                end - start,
                EXPECTED_MAX_LENGTH,
                EXPECTED_HIDDEN_SIZE,
            )

            if tuple(hidden.shape) != expected_batch_shape:
                raise RuntimeError(
                    "Unexpected T1-B sequence output shape. "
                    f"Expected {expected_batch_shape}, "
                    f"got {tuple(hidden.shape)}."
                )

            if hidden.dtype != torch.float32:
                raise RuntimeError(
                    "Unexpected BERT output dtype. "
                    "Script 47 requires full float32 inference; "
                    f"got {hidden.dtype}."
                )

            if not torch.isfinite(hidden).all():
                raise RuntimeError(
                    f"Non-finite BERT representation detected "
                    f"in batch {batch_number}."
                )

            hidden_np = (
                hidden
                .detach()
                .cpu()
                .numpy()
                .astype(
                    CACHE_DTYPE,
                    copy=False,
                )
            )

            sequence_cache[
                start:end
            ] = hidden_np

            attention_cache[
                start:end
            ] = attention_mask_np

            rows_written += (
                end - start
            )

            finite_rows += (
                end - start
            )

            batch_abs_max = float(
                np.max(
                    np.abs(
                        hidden_np
                    )
                )
            )

            global_abs_max = max(
                global_abs_max,
                batch_abs_max,
            )

            if (
                batch_number == 1
                or batch_number
                % 10 == 0
                or batch_number
                == total_batches
            ):
                print(
                    f"Batch "
                    f"{batch_number:3d}/"
                    f"{total_batches}: "
                    f"rows {start:4d}..{end - 1:4d} "
                    f"cached"
                )

            # Explicitly release GPU batch tensors.
            del input_ids
            del attention_mask
            del token_type_ids
            del outputs
            del hidden

    sequence_cache.flush()
    attention_cache.flush()

    del sequence_cache
    del attention_cache

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print()
    print(
        f"Rows written: {rows_written}"
    )

    print(
        f"Rows verified finite during inference: "
        f"{finite_rows}"
    )

    print(
        f"Maximum absolute hidden value observed: "
        f"{global_abs_max:.6f}"
    )

    check(
        rows_written
        == EXPECTED_DEVELOPMENT_ROWS,
        (
            "exactly 4626 TRAIN+VAL sequence rows "
            "were written"
        ),
    )

    check(
        finite_rows
        == EXPECTED_DEVELOPMENT_ROWS,
        "all generated sequence rows were finite",
    )

    return {
        "sequence_path": seq_path,
        "mask_path": mask_path,
        "rows_written": rows_written,
        "global_abs_max": global_abs_max,
    }


# =============================================================================
# 11. WRITE MANIFEST / METADATA
# =============================================================================

def write_cache_supporting_artifacts(
    output_manifest: pd.DataFrame,
    checkpoint: dict[str, Any],
    config: BertConfig,
    generation_info: dict[str, Any],
    device: torch.device,
) -> dict[str, Path]:

    banner(
        "7. WRITE CACHE MANIFEST + PROVENANCE"
    )

    manifest_path = (
        TEMP_DIR
        / "bert_seq_manifest.parquet"
    )

    metadata_path = (
        TEMP_DIR
        / "bert_seq_cache_metadata.json"
    )

    output_manifest.to_parquet(
        manifest_path,
        index=False,
    )

    print(
        f"Wrote: {relative(manifest_path)}"
    )

    checkpoint_sha = sha256_file(
        T1_CHECKPOINT
    )

    token_cache_sha = sha256_file(
        TOKEN_CACHE
    )

    token_manifest_sha = sha256_file(
        TOKEN_MANIFEST
    )

    pretraining_lock_sha = sha256_file(
        PRETRAINING_LOCK
    )

    cuda_name = None

    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(
            device
        )

    metadata = {
        "artifact_type": (
            "task3_frozen_bert_sequence_cache"
        ),
        "version": 1,
        "construction_scope": (
            "train_val_only"
        ),
        "test_examples_inferred": 0,
        "test_labels_accessed": False,
        "threshold_tuning_used": False,
        "training_performed": False,
        "encoder_frozen": True,
        "source_task": "task1",
        "source_variant": "T1-B",
        "source_model": MODEL_NAME,
        "source_checkpoint": relative(
            T1_CHECKPOINT
        ),
        "source_checkpoint_sha256": (
            checkpoint_sha
        ),
        "source_checkpoint_seed": int(
            checkpoint["seed"]
        ),
        "source_checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "source_checkpoint_val_auc_pr_macro_mean_ap": (
            float(
                checkpoint[
                    "val_auc_pr_macro_mean_ap"
                ]
            )
        ),
        "source_architecture": str(
            checkpoint["architecture"]
        ),
        "pretraining_lock": relative(
            PRETRAINING_LOCK
        ),
        "pretraining_lock_sha256": (
            pretraining_lock_sha
        ),
        "source_token_cache": relative(
            TOKEN_CACHE
        ),
        "source_token_cache_sha256": (
            token_cache_sha
        ),
        "source_token_manifest": relative(
            TOKEN_MANIFEST
        ),
        "source_token_manifest_sha256": (
            token_manifest_sha
        ),
        "split_counts": {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": 0,
            "total_cached": (
                EXPECTED_DEVELOPMENT_ROWS
            ),
        },
        "sequence_representation": {
            "layer": "final_hidden_state",
            "shape_per_track": [
                EXPECTED_MAX_LENGTH,
                EXPECTED_HIDDEN_SIZE,
            ],
            "dtype": "float32",
            "sequence_length": (
                EXPECTED_MAX_LENGTH
            ),
            "hidden_size": (
                EXPECTED_HIDDEN_SIZE
            ),
            "early_fusion_text_vector": (
                "sequence[:, 0, :]"
            ),
            "early_fusion_text_dimension": (
                EXPECTED_HIDDEN_SIZE
            ),
            "cross_attention_source": (
                "full final-layer token sequence"
            ),
        },
        "attention_mask": {
            "shape_per_track": [
                EXPECTED_MAX_LENGTH
            ],
            "dtype": "uint8",
        },
        "bert_config": {
            "model_type": (
                config.model_type
            ),
            "vocab_size": int(
                config.vocab_size
            ),
            "hidden_size": int(
                config.hidden_size
            ),
            "num_hidden_layers": int(
                config.num_hidden_layers
            ),
            "num_attention_heads": int(
                config.num_attention_heads
            ),
            "intermediate_size": int(
                config.intermediate_size
            ),
            "max_position_embeddings": int(
                config.max_position_embeddings
            ),
            "type_vocab_size": int(
                config.type_vocab_size
            ),
            "hidden_dropout_prob": float(
                config.hidden_dropout_prob
            ),
            "attention_probs_dropout_prob": float(
                config.attention_probs_dropout_prob
            ),
        },
        "cache_generation": {
            "batch_size": BATCH_SIZE,
            "torch_inference_mode": True,
            "model_eval_mode": True,
            "autocast": False,
            "tf32_allowed": False,
            "output_dtype": "float32",
            "rows_written": int(
                generation_info[
                    "rows_written"
                ]
            ),
            "maximum_absolute_hidden_value": (
                float(
                    generation_info[
                        "global_abs_max"
                    ]
                )
            ),
        },
        "environment": {
            "python": (
                sys.version.split()[0]
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "transformers": (
                transformers.__version__
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "device": str(device),
            "cuda_device_name": cuda_name,
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    print(
        f"Wrote: {relative(metadata_path)}"
    )

    return {
        "manifest": manifest_path,
        "metadata": metadata_path,
    }


# =============================================================================
# 12. VALIDATE COMPLETED TEMP CACHE
# =============================================================================

def validate_completed_cache(
    generation_info: dict[str, Any],
    supporting: dict[str, Path],
) -> None:

    banner(
        "8. POST-GENERATION CACHE VALIDATION"
    )

    sequence_path = generation_info[
        "sequence_path"
    ]

    mask_path = generation_info[
        "mask_path"
    ]

    manifest_path = supporting[
        "manifest"
    ]

    metadata_path = supporting[
        "metadata"
    ]

    for path in (
        sequence_path,
        mask_path,
        manifest_path,
        metadata_path,
    ):
        check(
            path.is_file(),
            f"generated artifact exists: {path.name}",
        )

    sequence = np.load(
        sequence_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    masks = np.load(
        mask_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    check(
        sequence.shape
        == (
            EXPECTED_DEVELOPMENT_ROWS,
            EXPECTED_MAX_LENGTH,
            EXPECTED_HIDDEN_SIZE,
        ),
        (
            "saved BERT sequence cache shape is exactly "
            "(4626, 112, 768)"
        ),
    )

    check(
        sequence.dtype
        == np.float32,
        "saved BERT sequence cache dtype is float32",
    )

    check(
        masks.shape
        == (
            EXPECTED_DEVELOPMENT_ROWS,
            EXPECTED_MAX_LENGTH,
        ),
        (
            "saved attention-mask cache shape is exactly "
            "(4626, 112)"
        ),
    )

    check(
        masks.dtype
        == np.uint8,
        "saved attention-mask cache dtype is uint8",
    )

    # Read only selected TRAIN/VAL cache samples.
    #
    # These are already the development-only output cache, so no TEST sample
    # can be accessed here.

    validation_rows = [
        0,
        EXPECTED_TRAIN - 1,
        EXPECTED_TRAIN,
        EXPECTED_DEVELOPMENT_ROWS - 1,
    ]

    for row in validation_rows:

        check(
            np.isfinite(
                sequence[row]
            ).all(),
            (
                f"saved sequence row {row} "
                "contains only finite values"
            ),
        )

        check(
            set(
                np.unique(
                    masks[row]
                ).tolist()
            ).issubset(
                {0, 1}
            ),
            (
                f"saved attention mask row {row} "
                "is binary"
            ),
        )

    saved_manifest = pd.read_parquet(
        manifest_path
    )

    check(
        len(saved_manifest)
        == EXPECTED_DEVELOPMENT_ROWS,
        "saved sequence manifest contains exactly 4626 rows",
    )

    split_counts = (
        saved_manifest["split"]
        .value_counts()
        .to_dict()
    )

    check(
        split_counts
        == {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
        },
        (
            "saved sequence manifest contains "
            "TRAIN+VAL only"
        ),
    )

    check(
        "test"
        not in set(
            saved_manifest["split"]
        ),
        "saved sequence manifest contains zero TEST rows",
    )

    metadata = load_json(
        metadata_path
    )

    check(
        metadata[
            "test_examples_inferred"
        ] == 0,
        "metadata records zero TEST examples inferred",
    )

    check(
        metadata[
            "test_labels_accessed"
        ] is False,
        "metadata records zero TEST-label access",
    )

    check(
        metadata[
            "training_performed"
        ] is False,
        "metadata records no model training",
    )

    check(
        metadata[
            "threshold_tuning_used"
        ] is False,
        "metadata records no threshold tuning",
    )

    del sequence
    del masks

    if FAILURES:
        raise RuntimeError(
            "Post-generation validation failed"
        )


# =============================================================================
# 13. FREEZE OUTPUT HASHES
# =============================================================================

def write_output_hashes(
    generation_info: dict[str, Any],
    supporting: dict[str, Path],
) -> Path:

    banner(
        "9. HASH GENERATED TASK-3 BERT CACHE"
    )

    files_to_hash = {
        "bert_sequence_trainval": (
            generation_info[
                "sequence_path"
            ]
        ),
        "attention_mask_trainval": (
            generation_info[
                "mask_path"
            ]
        ),
        "bert_seq_manifest": (
            supporting[
                "manifest"
            ]
        ),
        "bert_seq_cache_metadata": (
            supporting[
                "metadata"
            ]
        ),
    }

    hashes: dict[str, Any] = {
        "lock_type": (
            "task3_bert_sequence_cache_lock"
        ),
        "version": 1,
        "construction_scope": (
            "train_val_only"
        ),
        "test_examples_inferred": 0,
        "files": {},
    }

    for name, path in files_to_hash.items():

        print(
            f"Hashing {path.name}..."
        )

        digest = sha256_file(
            path
        )

        hashes[
            "files"
        ][name] = {
            "path": (
                f"data/processed/"
                f"bert_seq_frozen/"
                f"{path.name}"
            ),
            "sha256": digest,
            "size_bytes": int(
                path.stat().st_size
            ),
        }

        print(
            f"  SHA256: {digest}"
        )

    hashes_path = (
        TEMP_DIR
        / "bert_seq_frozen_hashes.json"
    )

    write_json(
        hashes_path,
        hashes,
    )

    print(
        f"Wrote: {relative(hashes_path)}"
    )

    return hashes_path


# =============================================================================
# 14. FINAL ATOMIC COMMIT
# =============================================================================

def commit_cache(
    hashes_path: Path,
) -> None:

    banner(
        "10. FINAL ATOMIC COMMIT"
    )

    check(
        hashes_path.is_file(),
        "cache lock/hash artifact exists",
    )

    check(
        not OUTPUT_DIR.exists(),
        (
            "final output location remains absent "
            "before commit"
        ),
    )

    if FAILURES:
        raise RuntimeError(
            "Final commit safety validation failed"
        )

    # Same parent/filesystem:
    # rename changes the temp directory into the final directory only after all
    # generation, validation and hashing have succeeded.

    TEMP_DIR.rename(
        OUTPUT_DIR
    )

    check(
        OUTPUT_DIR.is_dir(),
        "bert_seq_frozen cache committed successfully",
    )

    check(
        not TEMP_DIR.exists(),
        "temporary output directory no longer exists",
    )

    if FAILURES:
        raise RuntimeError(
            "Final cache commit failed"
        )


# =============================================================================
# 15. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 47 — CACHE FROZEN TASK-3 BERT SEQUENCES"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print()
    print(
        "Source encoder:"
    )
    print(
        "  Task-1 T1-B fully fine-tuned BERT"
    )

    print()
    print(
        "Inference scope:"
    )
    print(
        "  TRAIN = 4112"
    )
    print(
        "  VAL   = 514"
    )
    print(
        "  TEST  = 0"
    )

    print()
    print(
        "Output representation:"
    )
    print(
        "  final BERT hidden sequence"
    )
    print(
        "  shape per track = (112, 768)"
    )
    print(
        "  dtype = float32"
    )

    print()
    print(
        "Training:             NO"
    )
    print(
        "Gradient calculation: NO"
    )
    print(
        "Threshold tuning:     NO"
    )
    print(
        "TEST inference:       NO"
    )
    print(
        "TEST label access:    NO"
    )

    configure_reproducibility()

    try:

        # ---------------------------------------------------------------------
        # Frozen inputs
        # ---------------------------------------------------------------------

        verify_frozen_inputs()

        # ---------------------------------------------------------------------
        # Protect against overwriting anything
        # ---------------------------------------------------------------------

        verify_clean_output_location()

        # ---------------------------------------------------------------------
        # Development split
        # ---------------------------------------------------------------------

        _, development = (
            load_and_validate_manifest()
        )

        # ---------------------------------------------------------------------
        # Frozen token cache
        # ---------------------------------------------------------------------

        token_arrays = (
            load_and_validate_token_cache()
        )

        # ---------------------------------------------------------------------
        # Hardware
        # ---------------------------------------------------------------------

        banner(
            "5A. INFERENCE DEVICE"
        )

        check(
            torch.cuda.is_available(),
            "CUDA is available",
        )

        if FAILURES:
            raise RuntimeError(
                "Expected project CUDA environment is unavailable"
            )

        device = torch.device(
            "cuda"
        )

        print(
            f"CUDA device: "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"PyTorch: {torch.__version__}"
        )

        print(
            f"Transformers: "
            f"{transformers.__version__}"
        )

        # ---------------------------------------------------------------------
        # Exact selected encoder
        # ---------------------------------------------------------------------

        model, checkpoint, config = (
            build_frozen_t1b_encoder(
                device
            )
        )

        # ---------------------------------------------------------------------
        # Output manifest
        # ---------------------------------------------------------------------

        output_manifest = (
            build_output_manifest(
                development
            )
        )

        # ---------------------------------------------------------------------
        # Actual TRAIN+VAL-only inference/cache generation
        # ---------------------------------------------------------------------

        generation_info = (
            generate_cache(
                model=model,
                device=device,
                token_arrays=token_arrays,
                output_manifest=output_manifest,
            )
        )

        # Encoder is no longer needed.
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Manifest + provenance
        # ---------------------------------------------------------------------

        supporting = (
            write_cache_supporting_artifacts(
                output_manifest=output_manifest,
                checkpoint=checkpoint,
                config=config,
                generation_info=generation_info,
                device=device,
            )
        )

        # ---------------------------------------------------------------------
        # Validate saved files before freezing them
        # ---------------------------------------------------------------------

        validate_completed_cache(
            generation_info,
            supporting,
        )

        # ---------------------------------------------------------------------
        # Hash cache artifacts
        # ---------------------------------------------------------------------

        hashes_path = (
            write_output_hashes(
                generation_info,
                supporting,
            )
        )

        # ---------------------------------------------------------------------
        # Atomic final commit
        # ---------------------------------------------------------------------

        commit_cache(
            hashes_path
        )

    except Exception as exc:

        banner(
            "SCRIPT 47 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        if FAILURES:

            print(
                f"Failed checks: {len(FAILURES)}"
            )

            for message in FAILURES:
                print(
                    f"  - {message}"
                )

        print()

        if TEMP_DIR.exists():

            print(
                "IMPORTANT:"
            )

            print(
                "A temporary cache directory exists:"
            )

            print(
                f"  {TEMP_DIR}"
            )

            print()

            print(
                "It has NOT been committed as the final cache."
            )

            print(
                "Do not reuse or delete it until its contents "
                "have been inspected."
            )

        print()

        print(
            "Final bert_seq_frozen cache was NOT "
            "successfully frozen."
        )

        print(
            "Do not proceed to the next Task-3 script."
        )

        return 1

    # =========================================================================
    # Final report
    # =========================================================================

    banner(
        "SCRIPT 47 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Frozen encoder:"
    )

    print(
        "  T1-B fully fine-tuned BERT"
    )

    print(
        f"  checkpoint SHA256:"
    )

    print(
        f"  {EXPECTED_T1_CHECKPOINT_SHA256}"
    )

    print()

    print(
        "Cached examples:"
    )

    print(
        f"  TRAIN = {EXPECTED_TRAIN}"
    )

    print(
        f"  VAL   = {EXPECTED_VAL}"
    )

    print(
        "  TEST  = 0"
    )

    print(
        f"  TOTAL = {EXPECTED_DEVELOPMENT_ROWS}"
    )

    print()

    print(
        "Frozen sequence cache:"
    )

    print(
        "  bert_sequence_trainval.npy"
    )

    print(
        "  shape = (4626, 112, 768)"
    )

    print(
        "  dtype = float32"
    )

    print()

    print(
        "Frozen attention mask:"
    )

    print(
        "  attention_mask_trainval.npy"
    )

    print(
        "  shape = (4626, 112)"
    )

    print(
        "  dtype = uint8"
    )

    print()

    print(
        "Task-3 usage contract:"
    )

    print(
        "  M3 Early Fusion:"
    )

    print(
        "    text vector = sequence[:, 0, :]"
    )

    print(
        "    dimension   = 768"
    )

    print()

    print(
        "  M4 Cross-Attention:"
    )

    print(
        "    text input = full sequence"
    )

    print(
        "    shape      = 112 x 768"
    )

    print()

    print(
        "Model training performed: 0"
    )

    print(
        "TEST examples inferred:    0"
    )

    print(
        "TEST labels accessed:      0"
    )

    print(
        "Threshold tuning:          0"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()

    print(
        "STOP HERE."
    )

    print(
        "Review Script 47 output before designing "
        "the next Task-3 stage."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )