#!/usr/bin/env python3
"""
Script 57 — Freeze Validation Thresholds for Final Winning Configurations
=========================================================================

Final selected configurations:

Task 1:
    T1-B — fully fine-tuned BERT

Task 2:
    A1 — CNN

Task 3:
    M3 Phase B — Early Fusion joint fine-tuning

Purpose
-------
1. Reconstruct T1-B and reproduce its accepted validation macro AP.
2. Reconstruct A1 CNN and reproduce its accepted validation macro AP.
3. Tune thresholds for T1-B and A1 on VALIDATION only.
4. Import the already-frozen M3 Phase-B threshold from Script 56.
5. Freeze one combined winner-threshold artifact.

Threshold candidate set:
    0.30, 0.35, ..., 0.70

Threshold selection rule was fixed before T1/A1 threshold results:
    1. highest validation Macro-F1
    2. higher validation Micro-F1
    3. threshold closest to 0.50
    4. lower threshold

Important:
    - model selection remains based on threshold-free validation macro AP
    - no training occurs here
    - no TEST labels are loaded
    - no TEST inference occurs
    - no TEST threshold tuning occurs
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    f1_score,
)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from transformers import AutoModel


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRETRAINING_LOCK = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

TARGETS = (
    ROOT
    / "data/processed/labels/musiccaps_targets.parquet"
)

TOKEN_CACHE = (
    ROOT
    / "data/processed/bert_tokens/bert_tokens.npz"
)

TOKEN_MANIFEST = (
    ROOT
    / "data/processed/bert_tokens/bert_token_manifest.parquet"
)

AUDIO_MANIFEST = (
    ROOT
    / "data/processed/audio_feature_manifest.parquet"
)


# -----------------------------------------------------------------------------
# Selected Task-1 winner
# -----------------------------------------------------------------------------

T1_CHECKPOINT = (
    ROOT
    / "models/task1_candidates/"
    "task1_T1-B_seed42_20260828T182131Z_best.pt"
)


# -----------------------------------------------------------------------------
# Selected Task-2 overall audio winner
# -----------------------------------------------------------------------------

A1_CHECKPOINT = (
    ROOT
    / "models/task2_candidates/"
    "task2_A1_CNN_seed42_20260828T184558Z_best.pt"
)


# -----------------------------------------------------------------------------
# Already-completed Task-3 threshold selection
# -----------------------------------------------------------------------------

M3_THRESHOLD = (
    ROOT
    / "results/runs/task3/"
    "task3_M3_phaseB_seed42_threshold.json"
)


# -----------------------------------------------------------------------------
# Final combined threshold lock
# -----------------------------------------------------------------------------

OUTPUT = (
    ROOT
    / "results/runs/final_eval/"
    "winner_thresholds_validation.json"
)

OUTPUT_TEMP = Path(
    str(OUTPUT)
    + ".__tmp__"
)


# =============================================================================
# 1. FROZEN HASHES
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

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A1_SHA256 = (
    "4bb2369ce7f93693d67b1579cc1c7b4ac766076a8a0406ae7194ab8b0cbc1b35"
)

EXPECTED_M3_THRESHOLD_SHA256 = (
    "50a2f4e89985405ee4951aa48783135246be09cc7b0566293264b4e48c351b57"
)


# =============================================================================
# 2. ACCEPTED MODEL RESULTS
# =============================================================================

N_TOTAL = 5140
N_VAL = 514

NUM_LABELS = 20
MAX_LENGTH = 112

MODEL_NAME = "bert-base-uncased"

EXPECTED_T1_VAL_AP = (
    0.8405750022914198
)

EXPECTED_A1_VAL_AP = (
    0.22887393178444362
)

EXPECTED_M3_VAL_AP = (
    0.839664696610
)

EXPECTED_M3_THRESHOLD = (
    0.70
)

THRESHOLDS = np.arange(
    0.30,
    0.7001,
    0.05,
).round(2)

AP_PARITY_ATOL = 1e-8


# =============================================================================
# 3. HELPERS
# =============================================================================

def banner(
    title: str,
) -> None:

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

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        value = json.load(
            f
        )

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


def write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_TEMP.exists():

        raise RuntimeError(
            "Temporary threshold artifact "
            f"already exists: {OUTPUT_TEMP}"
        )

    with OUTPUT_TEMP.open(
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
        OUTPUT_TEMP,
        path,
    )


def seed_everything() -> None:

    random.seed(42)
    np.random.seed(42)

    torch.manual_seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            42
        )

    # Matches formal reproducibility policy.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 4. TASK-1 T1-B MODEL
# =============================================================================

class T1BModel(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        # Match Script 32 construction path.
        self.bert = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        hidden_size = int(
            self.bert.config.hidden_size
        )

        if hidden_size != 768:

            raise RuntimeError(
                "Unexpected BERT hidden size: "
                f"{hidden_size}"
            )

        self.dropout = nn.Dropout(
            0.1
        )

        self.classifier = nn.Linear(
            768,
            NUM_LABELS,
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

        # Exact Script-32 representation.
        cls_embedding = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
        )

        cls_embedding = self.dropout(
            cls_embedding
        )

        return self.classifier(
            cls_embedding
        )


# =============================================================================
# 5. TASK-2 A1 CNN MODEL
# =============================================================================

class A1CNN(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
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
        x: torch.Tensor,
    ) -> torch.Tensor:

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


# =============================================================================
# 6. VERIFY FINAL WINNER IDENTITIES
# =============================================================================

def verify_dependencies() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    banner(
        "1. FROZEN WINNER IDENTITIES"
    )

    require(
        not OUTPUT.exists(),
        (
            "combined winner-threshold "
            "artifact does not already exist"
        ),
    )

    require(
        not OUTPUT_TEMP.exists(),
        (
            "temporary winner-threshold "
            "artifact does not exist"
        ),
    )

    expected_files = {
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

        T1_CHECKPOINT:
            EXPECTED_T1_SHA256,

        A1_CHECKPOINT:
            EXPECTED_A1_SHA256,

        M3_THRESHOLD:
            EXPECTED_M3_THRESHOLD_SHA256,
    }

    for path, expected_sha in (
        expected_files.items()
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

    t1 = load_checkpoint(
        T1_CHECKPOINT
    )

    a1 = load_checkpoint(
        A1_CHECKPOINT
    )

    require(
        t1.get(
            "variant"
        )
        == "T1-B",
        (
            "Task-1 selected checkpoint "
            "is T1-B"
        ),
    )

    require(
        int(
            t1.get(
                "seed",
                -1,
            )
        )
        == 42,
        "T1-B selection seed = 42",
    )

    require(
        np.isclose(
            float(
                t1[
                    "val_auc_pr_macro_mean_ap"
                ]
            ),
            EXPECTED_T1_VAL_AP,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "T1-B stored VAL macro AP "
            "matches accepted result"
        ),
    )

    require(
        a1.get(
            "variant"
        )
        == "A1",
        (
            "Task-2 selected checkpoint "
            "is A1"
        ),
    )

    require(
        str(
            a1.get(
                "model",
                ""
            )
        ).upper()
        == "CNN",
        "A1 model = CNN",
    )

    require(
        int(
            a1.get(
                "seed",
                -1,
            )
        )
        == 42,
        "A1 selection seed = 42",
    )

    require(
        np.isclose(
            float(
                a1[
                    "val_auc_pr_macro_mean_ap"
                ]
            ),
            EXPECTED_A1_VAL_AP,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "A1 stored VAL macro AP "
            "matches accepted result"
        ),
    )

    architecture = a1.get(
        "architecture",
        {}
    )

    require(
        architecture.get(
            "input"
        )
        == [
            1,
            128,
            431,
        ],
        (
            "A1 frozen input shape "
            "= [1,128,431]"
        ),
    )

    require(
        architecture.get(
            "channels"
        )
        == [
            1,
            32,
            64,
            128,
            256,
        ],
        (
            "A1 frozen channels "
            "= 1→32→64→128→256"
        ),
    )

    return (
        t1,
        a1,
    )


# =============================================================================
# 7. COMMON VALIDATION IDENTITIES + TARGETS
# =============================================================================

def load_validation_identity() -> tuple[
    pd.DataFrame,
    list[str],
    np.ndarray,
]:

    banner(
        "2. LOAD VALIDATION IDENTITY + TARGETS"
    )

    manifest = pd.read_parquet(
        TOKEN_MANIFEST
    )

    require(
        list(
            manifest.columns
        )
        == [
            "cache_index",
            "track_id",
            "split",
            "cached_token_length",
        ],
        (
            "BERT token manifest schema "
            "is unchanged"
        ),
    )

    require(
        len(manifest)
        == N_TOTAL,
        (
            "BERT token manifest "
            "contains 5140 rows"
        ),
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

    split_counts = (
        manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    require(
        split_counts
        == {
            "train": 4112,
            "val": 514,
            "test": 514,
        },
        (
            "frozen split metadata remains "
            "4112/514/514"
        ),
    )

    val_manifest = (
        manifest.loc[
            manifest[
                "split"
            ]
            == "val"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    require(
        len(val_manifest)
        == N_VAL,
        (
            "VAL manifest contains "
            "exactly 514 rows"
        ),
    )

    val_ids = (
        val_manifest[
            "track_id"
        ].tolist()
    )

    require(
        len(
            set(
                val_ids
            )
        )
        == N_VAL,
        (
            "VAL track IDs "
            "are unique"
        ),
    )

    # -------------------------------------------------------------------------
    # VALIDATION LABELS ONLY
    # -------------------------------------------------------------------------

    y_columns = [
        f"y_{i:02d}"
        for i in range(
            NUM_LABELS
        )
    ]

    dataset = pads.dataset(
        str(
            TARGETS
        ),
        format="parquet",
    )

    table = dataset.to_table(
        columns=[
            "track_id",
            "split",
            *y_columns,
        ],
        filter=(
            pads.field(
                "split"
            )
            == "val"
        ),
    )

    target_frame = (
        table.to_pandas()
    )

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
        len(
            target_frame
        )
        == N_VAL,
        (
            "VAL target rows loaded = 514"
        ),
    )

    require(
        set(
            target_frame[
                "split"
            ]
        )
        == {
            "val"
        },
        (
            "target-value load "
            "contains VAL only"
        ),
    )

    require(
        set(
            val_ids
        )
        ==
        set(
            target_frame[
                "track_id"
            ]
        ),
        (
            "VAL token and target "
            "track IDs match"
        ),
    )

    target_lookup = (
        target_frame.set_index(
            "track_id",
            drop=False,
        )
    )

    aligned_targets = (
        target_lookup.loc[
            val_ids
        ]
        .reset_index(
            drop=True
        )
    )

    y_true = (
        aligned_targets[
            y_columns
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    require(
        y_true.shape
        == (
            N_VAL,
            NUM_LABELS,
        ),
        (
            "VAL target matrix "
            "= (514,20)"
        ),
    )

    require(
        set(
            np.unique(
                y_true
            ).tolist()
        )
        <= {
            0,
            1,
        },
        (
            "VAL targets are binary"
        ),
    )

    print()
    print(
        "TEST target rows loaded: 0"
    )

    return (
        val_manifest,
        val_ids,
        y_true,
    )


# =============================================================================
# 8. T1-B VALIDATION INFERENCE — EXACT SCRIPT-32 NUMERICAL PATH
# =============================================================================

def evaluate_t1(
    checkpoint: dict[str, Any],
    val_manifest: pd.DataFrame,
    y_true: np.ndarray,
    device: torch.device,
) -> tuple[
    np.ndarray,
    float,
]:

    banner(
        "3. T1-B VALIDATION INFERENCE"
    )

    model = T1BModel()

    incompatible = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    require(
        len(
            incompatible.missing_keys
        )
        == 0,
        (
            "T1-B strict load has "
            "zero missing keys"
        ),
    )

    require(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "T1-B strict load has "
            "zero unexpected keys"
        ),
    )

    model = model.to(
        device
    )

    model.eval()

    cache_indices = (
        val_manifest[
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

        require(
            set(
                cache.files
            )
            == {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            },
            (
                "BERT token-cache arrays "
                "are unchanged"
            ),
        )

        input_ids = np.asarray(
            cache[
                "input_ids"
            ][
                cache_indices
            ],
            dtype=np.int64,
        )

        attention_mask = np.asarray(
            cache[
                "attention_mask"
            ][
                cache_indices
            ],
            dtype=np.int64,
        )

        token_type_ids = np.asarray(
            cache[
                "token_type_ids"
            ][
                cache_indices
            ],
            dtype=np.int64,
        )

    require(
        input_ids.shape
        == (
            N_VAL,
            MAX_LENGTH,
        ),
        (
            "T1-B VAL input shape "
            "= (514,112)"
        ),
    )

    dataset = TensorDataset(
        torch.from_numpy(
            input_ids
        ),
        torch.from_numpy(
            attention_mask
        ),
        torch.from_numpy(
            token_type_ids
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    require(
        len(dataset)
        == N_VAL,
        (
            "T1-B validation dataset "
            "contains 514 examples"
        ),
    )

    require(
        len(loader)
        == 33,
        (
            "T1-B validation loader "
            "contains 33 batches"
        ),
    )

    probabilities: list[
        np.ndarray
    ] = []

    examples = 0

    # Script 32 used @torch.no_grad(), not training mode.
    with torch.no_grad():

        for (
            ids,
            mask,
            token_types,
        ) in loader:

            ids = ids.to(
                device,
                non_blocking=True,
            )

            mask = mask.to(
                device,
                non_blocking=True,
            )

            token_types = (
                token_types.to(
                    device,
                    non_blocking=True,
                )
            )

            # -------------------------------------------------------------
            # EXACT numerical validation path from Script 32:
            #
            # torch.autocast(
            #     device_type="cuda",
            #     dtype=torch.float16,
            #     enabled=True,
            # )
            #
            # probabilities =
            #     sigmoid(logits.float())
            # -------------------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                logits = model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=
                        token_types,
                )

            require(
                bool(
                    torch.isfinite(
                        logits
                    ).all()
                ),
                (
                    "T1-B validation "
                    "logits are finite"
                ),
            )

            probability = torch.sigmoid(
                logits.float()
            )

            require(
                bool(
                    torch.isfinite(
                        probability
                    ).all()
                ),
                (
                    "T1-B validation "
                    "probabilities are finite"
                ),
            )

            batch_rows = int(
                probability.shape[0]
            )

            examples += (
                batch_rows
            )

            # IMPORTANT:
            # This append was missing from the
            # failed patched Script 57 path.
            probabilities.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    require(
        examples
        == N_VAL,
        (
            "T1-B inference processed "
            "exactly 514 validation examples"
        ),
    )

    require(
        len(
            probabilities
        )
        == len(
            loader
        ),
        (
            "T1-B produced one probability "
            "array per validation batch"
        ),
    )

    probability_matrix = np.concatenate(
        probabilities,
        axis=0,
    )

    require(
        probability_matrix.shape
        == (
            N_VAL,
            NUM_LABELS,
        ),
        (
            "T1-B VAL probability matrix "
            "= (514,20)"
        ),
    )

    per_label_ap = (
        average_precision_score(
            y_true,
            probability_matrix,
            average=None,
        )
    )

    macro_ap = float(
        np.mean(
            per_label_ap
        )
    )

    print()
    print(
        "Stored T1-B VAL macro AP:"
    )

    print(
        f"  {EXPECTED_T1_VAL_AP:.12f}"
    )

    print(
        "Reconstructed T1-B VAL macro AP:"
    )

    print(
        f"  {macro_ap:.12f}"
    )

    print(
        "Absolute difference:"
    )

    print(
        f"  "
        f"{abs(macro_ap - EXPECTED_T1_VAL_AP):.12e}"
    )

    require(
        np.isclose(
            macro_ap,
            EXPECTED_T1_VAL_AP,
            rtol=0.0,
            atol=AP_PARITY_ATOL,
        ),
        (
            "T1-B reproduces accepted "
            "VAL macro AP using Script-32 "
            "FP16-autocast validation path"
        ),
    )

    return (
        probability_matrix,
        macro_ap,
    )


# =============================================================================
# 9. MEL PATH RESOLUTION
# =============================================================================

def resolve_mel_path(
    value: str,
) -> Path:

    raw = Path(
        value
    )

    candidates: list[
        Path
    ] = []

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

    matches: dict[
        str,
        Path,
    ] = {}

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

    if len(
        matches
    ) != 1:

        raise RuntimeError(
            "Could not uniquely resolve "
            f"mel file '{value}'. "
            f"Matches={len(matches)}"
        )

    return next(
        iter(
            matches.values()
        )
    )


# =============================================================================
# 10. LOAD A1 VALIDATION MEL FEATURES
# =============================================================================

def load_val_mels(
    val_ids: list[str],
) -> np.ndarray:

    banner(
        "4. LOAD A1 VALIDATION MEL FEATURES"
    )

    manifest = pd.read_parquet(
        AUDIO_MANIFEST
    )

    expected_columns = [
        "track_id",
        "mel_path",
        "chroma_path",
        "mel_bins",
        "mel_frames",
        "chroma_bins",
        "chroma_frames",
        "cache_status",
    ]

    require(
        list(
            manifest.columns
        )
        == expected_columns,
        (
            "audio feature manifest "
            "has frozen 8-column schema"
        ),
    )

    require(
        len(
            manifest
        )
        == N_TOTAL,
        (
            "audio feature manifest "
            "contains 5140 rows"
        ),
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
        manifest[
            "track_id"
        ].nunique()
        == N_TOTAL,
        (
            "audio feature manifest "
            "track IDs are unique"
        ),
    )

    require(
        set(
            val_ids
        ).issubset(
            set(
                manifest[
                    "track_id"
                ]
            )
        ),
        (
            "all 514 VAL tracks "
            "exist in audio manifest"
        ),
    )

    lookup = manifest.set_index(
        "track_id",
        drop=False,
    )

    values = np.empty(
        (
            N_VAL,
            1,
            128,
            431,
        ),
        dtype=np.float32,
    )

    print(
        "Loading 514 VAL mel arrays..."
    )

    for index, track_id in enumerate(
        val_ids
    ):

        row = lookup.loc[
            track_id
        ]

        if int(
            row[
                "mel_bins"
            ]
        ) != 128:

            raise RuntimeError(
                f"{track_id}: "
                "mel_bins != 128"
            )

        if int(
            row[
                "mel_frames"
            ]
        ) != 431:

            raise RuntimeError(
                f"{track_id}: "
                "mel_frames != 431"
            )

        path = resolve_mel_path(
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

        if mel.shape != (
            128,
            431,
        ):

            raise RuntimeError(
                f"{track_id}: "
                f"mel shape = {mel.shape}, "
                "expected (128,431)"
            )

        if mel.dtype != np.float32:

            raise RuntimeError(
                f"{track_id}: "
                f"mel dtype = {mel.dtype}, "
                "expected float32"
            )

        if not np.isfinite(
            mel
        ).all():

            raise RuntimeError(
                f"{track_id}: "
                "non-finite mel values"
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
            or number % 100 == 0
            or number == N_VAL
        ):

            print(
                f"  loaded "
                f"{number}/{N_VAL}"
            )

    require(
        values.shape
        == (
            N_VAL,
            1,
            128,
            431,
        ),
        (
            "A1 VAL mel tensor "
            "= (514,1,128,431)"
        ),
    )

    print()
    print(
        "TEST mel files loaded: 0"
    )

    return values


# =============================================================================
# 11. GENERIC A1 INFERENCE
# =============================================================================

def infer_a1(
    model: A1CNN,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> np.ndarray:

    model.eval()

    probabilities: list[
        np.ndarray
    ] = []

    examples = 0

    with torch.no_grad():

        for (
            mel,
        ) in loader:

            mel = mel.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(
                    mel
                )

            if not bool(
                torch.isfinite(
                    logits
                ).all()
            ):

                raise RuntimeError(
                    "A1 produced "
                    "non-finite logits"
                )

            probability = torch.sigmoid(
                logits.float()
            )

            if not bool(
                torch.isfinite(
                    probability
                ).all()
            ):

                raise RuntimeError(
                    "A1 produced "
                    "non-finite probabilities"
                )

            examples += int(
                probability.shape[0]
            )

            probabilities.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    require(
        examples
        == N_VAL,
        (
            "A1 inference processed "
            "exactly 514 validation examples"
        ),
    )

    require(
        len(
            probabilities
        )
        == len(
            loader
        ),
        (
            "A1 produced one probability "
            "array per validation batch"
        ),
    )

    result = np.concatenate(
        probabilities,
        axis=0,
    )

    require(
        result.shape
        == (
            N_VAL,
            NUM_LABELS,
        ),
        (
            "A1 probability matrix "
            "= (514,20)"
        ),
    )

    return result


# =============================================================================
# 12. A1 VALIDATION — RESOLVE ORIGINAL NUMERICAL PATH AUTOMATICALLY
# =============================================================================

def evaluate_a1(
    checkpoint: dict[str, Any],
    mel_values: np.ndarray,
    y_true: np.ndarray,
    device: torch.device,
) -> tuple[
    np.ndarray,
    float,
    str,
]:

    banner(
        "5. A1 CNN VALIDATION INFERENCE"
    )

    model = A1CNN()

    incompatible = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )
    )

    require(
        len(
            incompatible.missing_keys
        )
        == 0,
        (
            "A1 strict load has "
            "zero missing keys"
        ),
    )

    require(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "A1 strict load has "
            "zero unexpected keys"
        ),
    )

    model = model.to(
        device
    )

    dataset = TensorDataset(
        torch.from_numpy(
            mel_values
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    require(
        len(
            dataset
        )
        == N_VAL,
        (
            "A1 validation dataset "
            "contains 514 examples"
        ),
    )

    require(
        len(
            loader
        )
        == 17,
        (
            "A1 validation loader "
            "contains 17 batches"
        ),
    )

    # -------------------------------------------------------------------------
    # We already know the formal A1 checkpoint and its accepted AP.
    #
    # Earlier logs did not preserve Script-36's exact AMP validation detail.
    # Rather than guess or require another manual inspection, reproduce both
    # numerical evaluation paths and accept ONLY the one that reproduces the
    # already-frozen formal A1 AP.
    #
    # This does not alter architecture, parameters, split, or model selection.
    # It resolves only the numerical inference convention.
    # -------------------------------------------------------------------------

    print()
    print(
        "Checking A1 FP32 validation parity..."
    )

    probability_fp32 = infer_a1(
        model=model,
        loader=loader,
        device=device,
        use_amp=False,
    )

    ap_fp32 = float(
        np.mean(
            average_precision_score(
                y_true,
                probability_fp32,
                average=None,
            )
        )
    )

    print(
        f"  FP32 VAL macro AP = "
        f"{ap_fp32:.12f}"
    )

    print(
        "Checking A1 FP16-autocast validation parity..."
    )

    probability_amp = infer_a1(
        model=model,
        loader=loader,
        device=device,
        use_amp=True,
    )

    ap_amp = float(
        np.mean(
            average_precision_score(
                y_true,
                probability_amp,
                average=None,
            )
        )
    )

    print(
        f"  AMP VAL macro AP  = "
        f"{ap_amp:.12f}"
    )

    print()
    print(
        f"Stored A1 VAL macro AP = "
        f"{EXPECTED_A1_VAL_AP:.12f}"
    )

    fp32_error = abs(
        ap_fp32
        - EXPECTED_A1_VAL_AP
    )

    amp_error = abs(
        ap_amp
        - EXPECTED_A1_VAL_AP
    )

    print(
        f"FP32 absolute error = "
        f"{fp32_error:.12e}"
    )

    print(
        f"AMP absolute error  = "
        f"{amp_error:.12e}"
    )

    candidates = [
        (
            fp32_error,
            "FP32",
            ap_fp32,
            probability_fp32,
        ),
        (
            amp_error,
            "FP16_autocast",
            ap_amp,
            probability_amp,
        ),
    ]

    candidates.sort(
        key=lambda x: x[0]
    )

    (
        best_error,
        selected_mode,
        selected_ap,
        selected_probability,
    ) = candidates[0]

    require(
        best_error
        <= AP_PARITY_ATOL,
        (
            "one A1 numerical validation "
            "path reproduces accepted VAL macro AP"
        ),
    )

    print()
    print(
        "Selected A1 numerical validation mode:"
    )

    print(
        f"  {selected_mode}"
    )

    print(
        f"Reproduced A1 VAL macro AP:"
    )

    print(
        f"  {selected_ap:.12f}"
    )

    return (
        selected_probability,
        selected_ap,
        selected_mode,
    )


# =============================================================================
# 13. THRESHOLD SWEEP
# =============================================================================

def sweep_thresholds(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:

    banner(
        f"6. VALIDATION THRESHOLD SWEEP — {model_name}"
    )

    rows: list[
        dict[str, Any]
    ] = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

        macro_f1 = float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        micro_f1 = float(
            f1_score(
                y_true,
                predictions,
                average="micro",
                zero_division=0,
            )
        )

        per_label_f1 = (
            f1_score(
                y_true,
                predictions,
                average=None,
                zero_division=0,
            )
        )

        row = {
            "threshold":
                float(
                    threshold
                ),

            "macro_f1":
                macro_f1,

            "micro_f1":
                micro_f1,

            "per_label_f1":
                [
                    float(
                        value
                    )
                    for value
                    in per_label_f1
                ],
        }

        rows.append(
            row
        )

        print(
            f"threshold={threshold:.2f}  "
            f"macro_F1={macro_f1:.8f}  "
            f"micro_F1={micro_f1:.8f}"
        )

    # Frozen threshold-selection rule.
    best = sorted(
        rows,
        key=lambda row: (
            -row[
                "macro_f1"
            ],
            -row[
                "micro_f1"
            ],
            abs(
                row[
                    "threshold"
                ]
                - 0.50
            ),
            row[
                "threshold"
            ],
        ),
    )[0]

    print()
    print(
        f"Selected threshold for {model_name}:"
    )

    print(
        f"  threshold = "
        f"{best['threshold']:.2f}"
    )

    print(
        f"  VAL Macro-F1 = "
        f"{best['macro_f1']:.8f}"
    )

    print(
        f"  VAL Micro-F1 = "
        f"{best['micro_f1']:.8f}"
    )

    return (
        rows,
        best,
    )


# =============================================================================
# 14. IMPORT SCRIPT-56 M3 THRESHOLD
# =============================================================================

def load_m3_threshold() -> dict[str, Any]:

    banner(
        "7. IMPORT FROZEN M3 PHASE-B THRESHOLD"
    )

    artifact = load_json(
        M3_THRESHOLD
    )

    require(
        artifact.get(
            "artifact_type"
        )
        == "task3_threshold_selection",
        (
            "M3 threshold artifact "
            "type is correct"
        ),
    )

    require(
        artifact.get(
            "variant"
        )
        == "M3",
        (
            "M3 threshold variant "
            "= M3"
        ),
    )

    require(
        artifact.get(
            "phase"
        )
        == "B",
        (
            "M3 threshold phase "
            "= B"
        ),
    )

    require(
        int(
            artifact.get(
                "seed",
                -1,
            )
        )
        == 42,
        (
            "M3 threshold source "
            "seed = 42"
        ),
    )

    stored_ap = float(
        artifact[
            "threshold_independent_validation"
        ][
            "macro_mean_per_label_AP"
        ]
    )

    require(
        np.isclose(
            stored_ap,
            EXPECTED_M3_VAL_AP,
            rtol=0.0,
            atol=1e-8,
        ),
        (
            "M3 threshold artifact "
            "retains accepted VAL macro AP"
        ),
    )

    selected_threshold = float(
        artifact[
            "selected_threshold"
        ]
    )

    require(
        np.isclose(
            selected_threshold,
            EXPECTED_M3_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M3 frozen threshold "
            "= 0.70"
        ),
    )

    require(
        int(
            artifact.get(
                "test_target_rows_loaded",
                -1,
            )
        )
        == 0,
        (
            "M3 threshold tuning loaded "
            "zero TEST target rows"
        ),
    )

    require(
        int(
            artifact.get(
                "test_examples_evaluated",
                -1,
            )
        )
        == 0,
        (
            "M3 threshold tuning evaluated "
            "zero TEST examples"
        ),
    )

    print()
    print(
        f"M3 threshold = "
        f"{selected_threshold:.2f}"
    )

    print(
        f"M3 VAL Macro-F1 = "
        f"{float(artifact['selected_validation_macro_f1']):.8f}"
    )

    print(
        f"M3 VAL Micro-F1 = "
        f"{float(artifact['selected_validation_micro_f1']):.8f}"
    )

    return artifact


# =============================================================================
# 15. WRITE COMBINED THRESHOLD LOCK
# =============================================================================

def freeze_thresholds(
    *,
    t1_ap: float,
    t1_sweep: list[dict[str, Any]],
    t1_best: dict[str, Any],
    a1_ap: float,
    a1_mode: str,
    a1_sweep: list[dict[str, Any]],
    a1_best: dict[str, Any],
    m3: dict[str, Any],
) -> str:

    banner(
        "8. FREEZE ALL WINNER THRESHOLDS"
    )

    artifact = {
        "artifact_type":
            "final_winner_validation_thresholds",

        "version":
            1,

        "threshold_source_seed":
            42,

        "selection_basis":
            "VALIDATION only",

        "model_selection_metric":
            (
                "macro mean per-label "
                "Average Precision"
            ),

        "threshold_candidates":
            [
                float(
                    value
                )
                for value
                in THRESHOLDS
            ],

        "threshold_selection_rule": {
            "primary":
                (
                    "maximum validation "
                    "Macro-F1"
                ),

            "tie_break_1":
                (
                    "higher validation "
                    "Micro-F1"
                ),

            "tie_break_2":
                (
                    "threshold closest "
                    "to 0.50"
                ),

            "tie_break_3":
                "lower threshold",
        },

        "task1": {
            "winner":
                "T1-B",

            "model":
                "BERT",

            "modality":
                "text",

            "selection_seed":
                42,

            "checkpoint":
                str(
                    T1_CHECKPOINT
                    .relative_to(
                        ROOT
                    )
                ),

            "checkpoint_sha256":
                EXPECTED_T1_SHA256,

            "validation_numerical_path":
                (
                    "CUDA FP16 autocast, "
                    "then sigmoid(logits.float()); "
                    "matches Script 32"
                ),

            "val_macro_ap":
                t1_ap,

            "selected_threshold":
                float(
                    t1_best[
                        "threshold"
                    ]
                ),

            "val_macro_f1":
                float(
                    t1_best[
                        "macro_f1"
                    ]
                ),

            "val_micro_f1":
                float(
                    t1_best[
                        "micro_f1"
                    ]
                ),

            "sweep":
                t1_sweep,
        },

        "task2": {
            "winner":
                "A1",

            "model":
                "CNN",

            "modality":
                "audio",

            "selection_seed":
                42,

            "checkpoint":
                str(
                    A1_CHECKPOINT
                    .relative_to(
                        ROOT
                    )
                ),

            "checkpoint_sha256":
                EXPECTED_A1_SHA256,

            "validation_numerical_path":
                a1_mode,

            "val_macro_ap":
                a1_ap,

            "selected_threshold":
                float(
                    a1_best[
                        "threshold"
                    ]
                ),

            "val_macro_f1":
                float(
                    a1_best[
                        "macro_f1"
                    ]
                ),

            "val_micro_f1":
                float(
                    a1_best[
                        "micro_f1"
                    ]
                ),

            "sweep":
                a1_sweep,
        },

        "task3": {
            "winner":
                "M3_phaseB",

            "model":
                "EarlyFusionJoint",

            "modality":
                "text+graph_audio",

            "selection_seed":
                42,

            "source_threshold_artifact":
                str(
                    M3_THRESHOLD
                    .relative_to(
                        ROOT
                    )
                ),

            "source_threshold_artifact_sha256":
                EXPECTED_M3_THRESHOLD_SHA256,

            "val_macro_ap":
                EXPECTED_M3_VAL_AP,

            "selected_threshold":
                float(
                    m3[
                        "selected_threshold"
                    ]
                ),

            "val_macro_f1":
                float(
                    m3[
                        "selected_validation_macro_f1"
                    ]
                ),

            "val_micro_f1":
                float(
                    m3[
                        "selected_validation_micro_f1"
                    ]
                ),
        },

        "test_target_rows_loaded":
            0,

        "test_examples_evaluated":
            0,

        "test_threshold_tuning":
            False,

        "training_performed":
            False,

        "next_stage":
            (
                "three-seed reruns of "
                "frozen winning configurations "
                "with seeds 42, 1337, 2026"
            ),
    }

    write_json_atomic(
        OUTPUT,
        artifact,
    )

    require(
        OUTPUT.is_file(),
        (
            "combined winner-threshold "
            "artifact was created"
        ),
    )

    return sha256_file(
        OUTPUT
    )


# =============================================================================
# 16. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 57 — FREEZE FINAL WINNER VALIDATION THRESHOLDS"
    )

    print(
        "Task 1 winner: T1-B BERT"
    )

    print(
        "Task 2 winner: A1 CNN"
    )

    print(
        "Task 3 winner: M3 Phase B"
    )

    print()

    print(
        "Threshold candidates: "
        "0.30, 0.35, ..., 0.70"
    )

    print(
        "Primary criterion: "
        "VAL Macro-F1"
    )

    print()

    print(
        "Training:       NO"
    )

    print(
        "TEST labels:    NO"
    )

    print(
        "TEST inference: NO"
    )

    seed_everything()

    try:

        # ---------------------------------------------------------------------
        # Frozen identities
        # ---------------------------------------------------------------------

        (
            t1_checkpoint,
            a1_checkpoint,
        ) = verify_dependencies()

        # ---------------------------------------------------------------------
        # Common VAL identity / targets
        # ---------------------------------------------------------------------

        (
            val_manifest,
            val_ids,
            y_true,
        ) = load_validation_identity()

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA unavailable"
            )

        device = torch.device(
            "cuda"
        )

        print()
        print(
            "CUDA device:"
        )

        print(
            f"  "
            f"{torch.cuda.get_device_name(device)}"
        )

        # ---------------------------------------------------------------------
        # Task 1
        # ---------------------------------------------------------------------

        (
            t1_probability,
            t1_ap,
        ) = evaluate_t1(
            checkpoint=
                t1_checkpoint,

            val_manifest=
                val_manifest,

            y_true=
                y_true,

            device=
                device,
        )

        (
            t1_sweep,
            t1_best,
        ) = sweep_thresholds(
            model_name=
                "T1-B BERT",

            y_true=
                y_true,

            probabilities=
                t1_probability,
        )

        del t1_probability

        torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Task 2
        # ---------------------------------------------------------------------

        mel_values = load_val_mels(
            val_ids
        )

        (
            a1_probability,
            a1_ap,
            a1_mode,
        ) = evaluate_a1(
            checkpoint=
                a1_checkpoint,

            mel_values=
                mel_values,

            y_true=
                y_true,

            device=
                device,
        )

        (
            a1_sweep,
            a1_best,
        ) = sweep_thresholds(
            model_name=
                "A1 CNN",

            y_true=
                y_true,

            probabilities=
                a1_probability,
        )

        del a1_probability
        del mel_values

        torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Task 3 — already frozen by Script 56
        # ---------------------------------------------------------------------

        m3 = load_m3_threshold()

        # ---------------------------------------------------------------------
        # Final threshold lock
        # ---------------------------------------------------------------------

        output_sha = freeze_thresholds(
            t1_ap=
                t1_ap,

            t1_sweep=
                t1_sweep,

            t1_best=
                t1_best,

            a1_ap=
                a1_ap,

            a1_mode=
                a1_mode,

            a1_sweep=
                a1_sweep,

            a1_best=
                a1_best,

            m3=
                m3,
        )

    except Exception as exc:

        banner(
            "SCRIPT 57 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "No combined threshold lock "
            "was accepted."
        )

        print(
            "Do NOT start multi-seed training."
        )

        return 1

    banner(
        "SCRIPT 57 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Frozen winner thresholds:"
    )

    print()

    print(
        "Task 1 — T1-B BERT"
    )

    print(
        f"  VAL macro AP = "
        f"{t1_ap:.8f}"
    )

    print(
        f"  threshold    = "
        f"{t1_best['threshold']:.2f}"
    )

    print(
        f"  VAL Macro-F1 = "
        f"{t1_best['macro_f1']:.8f}"
    )

    print(
        f"  VAL Micro-F1 = "
        f"{t1_best['micro_f1']:.8f}"
    )

    print()

    print(
        "Task 2 — A1 CNN"
    )

    print(
        f"  numerical path = "
        f"{a1_mode}"
    )

    print(
        f"  VAL macro AP    = "
        f"{a1_ap:.8f}"
    )

    print(
        f"  threshold       = "
        f"{a1_best['threshold']:.2f}"
    )

    print(
        f"  VAL Macro-F1    = "
        f"{a1_best['macro_f1']:.8f}"
    )

    print(
        f"  VAL Micro-F1    = "
        f"{a1_best['micro_f1']:.8f}"
    )

    print()

    print(
        "Task 3 — M3 Phase B"
    )

    print(
        f"  VAL macro AP = "
        f"{EXPECTED_M3_VAL_AP:.8f}"
    )

    print(
        f"  threshold    = "
        f"{float(m3['selected_threshold']):.2f}"
    )

    print(
        f"  VAL Macro-F1 = "
        f"{float(m3['selected_validation_macro_f1']):.8f}"
    )

    print(
        f"  VAL Micro-F1 = "
        f"{float(m3['selected_validation_micro_f1']):.8f}"
    )

    print()

    print(
        "TEST target rows loaded: 0"
    )

    print(
        "TEST inference examples:  0"
    )

    print(
        "TEST threshold tuning:    0"
    )

    print()

    print(
        "Combined threshold artifact:"
    )

    print(
        f"  "
        f"{OUTPUT.relative_to(ROOT)}"
    )

    print()

    print(
        "Combined threshold SHA256:"
    )

    print(
        f"  {output_sha}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  Multi-seed final winner reruns"
    )

    print(
        "  seeds = {42, 1337, 2026}"
    )

    print()

    print(
        "STOP HERE."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )