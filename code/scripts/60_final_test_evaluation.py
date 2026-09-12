#!/usr/bin/env python3
"""
Script 60 — FINAL TEST Evaluation, Exactly Once
================================================

This is the first target-dependent TEST evaluation.

Frozen before this script:
    - dataset / splits / labels
    - model architectures
    - winning configurations
    - all training
    - three-seed reruns
    - validation thresholds
    - final report thresholds

Final comparison models:
    B1 Majority-class
    B2 A1 CNN
    T1-B BERT
    A3 GraphSAGE/G2
    A4 GAT/G2
    M3 Early Fusion Phase B
    M4 Cross-Attention Phase B

Three-seed final evaluation:
    A1 CNN
    T1-B BERT
    M3 Early Fusion Phase B

Seeds:
    42, 1337, 2026

Metrics:
    Macro-F1
    Micro-F1
    macro mean per-label Average Precision ("AUC-PR" in project)
    Micro Average Precision (diagnostic)
    per-label F1
    per-label AP
    per-label TEST support

IMPORTANT:
    NO threshold tuning.
    NO model selection.
    NO hyperparameter decisions.
    NO retraining.

The script also saves all TEST probability matrices so later
error/statistical analysis can reuse them without new model inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    f1_score,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
)

from torch_geometric.data import (
    Batch,
    Data,
)

from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    global_mean_pool,
)

from transformers import (
    AutoModel,
    BertConfig,
    BertModel,
)


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

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

GRAPH_MANIFEST = (
    ROOT
    / "data/processed/graph_manifest.parquet"
)

PRETRAINING_LOCK = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)


# -----------------------------------------------------------------------------
# Frozen single-seed report models
# -----------------------------------------------------------------------------

A3_CHECKPOINT = (
    ROOT
    / "models/task2_candidates/"
    "task2_A3_GraphSAGE_G2_seed42_20260828T192129Z_best.pt"
)

A4_CHECKPOINT = (
    ROOT
    / "models/task2_candidates/"
    "task2_A4_GAT_G2_seed42_20260828T193949Z_best.pt"
)

SELECTED_T1_CHECKPOINT = (
    ROOT
    / "models/task1_candidates/"
    "task1_T1-B_seed42_20260828T182131Z_best.pt"
)

M4_PHASEA_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M4_phaseA_seed42_20260910T131031Z_best.pt"
)

M4_PHASEB_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M4_phaseB_seed42_20260910T133743Z_best.pt"
)


# -----------------------------------------------------------------------------
# Final evaluation locks
# -----------------------------------------------------------------------------

FINAL_THRESHOLDS = (
    ROOT
    / "results/runs/final_eval/"
    "final_report_thresholds_validation.json"
)

MULTISEED_MANIFEST = (
    ROOT
    / "results/runs/final_eval/multiseed/"
    "final_multiseed_training_manifest.json"
)


# -----------------------------------------------------------------------------
# Output — committed only after complete successful TEST run
# -----------------------------------------------------------------------------

FINAL_DIR = (
    ROOT
    / "results/runs/final_eval/"
    "final_test"
)

TEMP_DIR = (
    ROOT
    / "results/runs/final_eval/"
    "final_test.__tmp__"
)


# =============================================================================
# 1. FROZEN HASHES
# =============================================================================

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

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_FINAL_THRESHOLDS_SHA256 = (
    "20118489eba0b5ee9d0568bfa1d76eb4c4a6a829c50042b83d738b31526b26e9"
)

EXPECTED_MULTISEED_MANIFEST_SHA256 = (
    "ca4955591cb85957d628ac1c7cbcfd5595ad16010fece7362cf79dec8b524446"
)

EXPECTED_A3_SHA256 = (
    "abac9ffed7907177b215332d9758e4cb86482f856b5492f2d73352412d84d8ea"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e"
)

EXPECTED_SELECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_M4_PHASEA_SHA256 = (
    "ea210e30a932c124ee07b6a5fc90d068e0201152301998ed1eeb67559dd2e112"
)

EXPECTED_M4_PHASEB_SHA256 = (
    "7a27a270074ef4e50143aec94b8b5d7ada001de0c0708c18b3aea406d159c165"
)


# =============================================================================
# 2. FIXED DATA CONTRACT
# =============================================================================

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

N_TOTAL = 5140
N_TRAIN = 4112
N_TEST = 514

NUM_LABELS = 20
MAX_LENGTH = 112

MODEL_NAME = "bert-base-uncased"


LABEL_NAMES = [
    "low quality",
    "instrumental",
    "medium tempo",
    "fast tempo",
    "noisy",
    "emotional",
    "energetic",
    "passionate",
    "amateur recording",
    "bass guitar",
    "electric guitar",
    "live performance",
    "slow tempo",
    "male vocal",
    "mono",
    "acoustic drums",
    "groovy",
    "no voices",
    "acoustic guitar",
    "piano",
]


# =============================================================================
# 3. GENERAL HELPERS
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
        raise RuntimeError(message)

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

        obj = json.load(f)

    if not isinstance(
        obj,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not a JSON object"
        )

    return obj


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


def seed_everything(
    seed: int = 42,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_std(
    values: list[float],
) -> float | None:

    if len(values) <= 1:
        return None

    return float(
        np.std(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            ddof=1,
        )
    )


def format_mean_std(
    mean: float,
    std: float | None,
    n: int,
) -> str:

    if (
        n > 1
        and std is not None
    ):

        return (
            f"{mean:.6f} ± "
            f"{std:.6f}"
        )

    return f"{mean:.6f}"


# =============================================================================
# 4. MODEL DEFINITIONS
# =============================================================================

class T1BModel(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        self.dropout = nn.Dropout(
            0.1
        )

        self.classifier = nn.Linear(
            768,
            20,
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

        cls = self.dropout(cls)

        return self.classifier(
            cls
        )


class A1CNN(nn.Module):

    def __init__(self) -> None:

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
            20,
        )

    def forward(
        self,
        x,
    ):

        x = self.features(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(
            x,
            1,
        )

        return self.classifier(x)


class GraphSAGEModel(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.node_projection = nn.Linear(
            140,
            64,
        )

        # Names intentionally match frozen A3 state_dict.
        self.conv1 = SAGEConv(
            64,
            128,
        )

        self.conv2 = SAGEConv(
            128,
            128,
        )

        self.conv3 = SAGEConv(
            128,
            128,
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.classifier = nn.Linear(
            128,
            20,
        )

    def forward(
        self,
        data,
    ):

        x = self.node_projection(
            data.x
        )

        x = self.conv1(
            x,
            data.edge_index,
        )

        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            data.edge_index,
        )

        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(
            x,
            data.edge_index,
        )

        x = global_mean_pool(
            x,
            data.batch,
        )

        return self.classifier(x)


class A4GAT(nn.Module):

    def __init__(self) -> None:

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

    def forward(
        self,
        data,
    ):

        return self.classifier(
            self.encode_graph(data)
        )


class M3Fusion(nn.Module):

    def __init__(self) -> None:

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

        text_prime = self.text_projection(
            text_embedding
        )

        graph_prime = self.graph_projection(
            graph_embedding
        )

        z = torch.cat(
            [
                text_prime,
                graph_prime,
            ],
            dim=-1,
        )

        z = self.fusion(z)

        z = F.relu(z)

        z = self.dropout(z)

        return self.classifier(z)


class M3PhaseBModel(nn.Module):

    def __init__(
        self,
        bert,
        gnn,
        fusion,
    ) -> None:

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

        text_cls = output.last_hidden_state[
            :,
            0,
            :
        ]

        graph_embedding = (
            self.gnn.encode_graph(
                graph_batch
            )
        )

        return self.fusion(
            text_cls,
            graph_embedding,
        )


class M4Fusion(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.text_projection = nn.Linear(
            768,
            256,
        )

        self.graph_projection = nn.Linear(
            128,
            256,
        )

        self.attention = (
            nn.MultiheadAttention(
                embed_dim=256,
                num_heads=4,
                dropout=0.0,
                batch_first=True,
            )
        )

        self.fusion = nn.Linear(
            512,
            256,
        )

        self.classifier = nn.Linear(
            256,
            20,
        )

    def forward(
        self,
        bert_sequence,
        attention_mask,
        graph_embedding,
    ):

        text_sequence = (
            self.text_projection(
                bert_sequence
            )
        )

        graph_prime = (
            self.graph_projection(
                graph_embedding
            )
        )

        query = graph_prime.unsqueeze(1)

        attended, _ = self.attention(
            query,
            text_sequence,
            text_sequence,
            key_padding_mask=
                ~attention_mask.bool(),
            need_weights=False,
        )

        z = torch.cat(
            [
                graph_prime,
                attended.squeeze(1),
            ],
            dim=-1,
        )

        z = F.relu(
            self.fusion(z)
        )

        return self.classifier(z)


class M4Model(nn.Module):

    def __init__(
        self,
        bert,
        gnn,
        fusion,
    ) -> None:

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

        graph_embedding = (
            self.gnn.encode_graph(
                graph_batch
            )
        )

        return self.fusion(
            output.last_hidden_state,
            attention_mask,
            graph_embedding,
        )


# =============================================================================
# 5. VERIFY ALL FINAL DEPENDENCIES BEFORE TEST ACCESS
# =============================================================================

def verify_dependencies():

    banner(
        "1. FINAL FROZEN DEPENDENCIES — BEFORE TEST ACCESS"
    )

    require(
        not FINAL_DIR.exists(),
        (
            "no previous committed "
            "final TEST evaluation exists"
        ),
    )

    require(
        not TEMP_DIR.exists(),
        (
            "no previous/partial TEST-access "
            "directory exists"
        ),
    )

    expected = {
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

        PRETRAINING_LOCK:
            EXPECTED_PRETRAINING_LOCK_SHA256,

        FINAL_THRESHOLDS:
            EXPECTED_FINAL_THRESHOLDS_SHA256,

        MULTISEED_MANIFEST:
            EXPECTED_MULTISEED_MANIFEST_SHA256,

        A3_CHECKPOINT:
            EXPECTED_A3_SHA256,

        A4_CHECKPOINT:
            EXPECTED_A4_SHA256,

        SELECTED_T1_CHECKPOINT:
            EXPECTED_SELECTED_T1_SHA256,

        M4_PHASEA_CHECKPOINT:
            EXPECTED_M4_PHASEA_SHA256,

        M4_PHASEB_CHECKPOINT:
            EXPECTED_M4_PHASEB_SHA256,
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
        FINAL_THRESHOLDS
    )

    require(
        threshold_lock.get(
            "artifact_type"
        )
        ==
        "final_report_validation_thresholds",
        (
            "final report threshold lock "
            "has correct artifact type"
        ),
    )

    require(
        threshold_lock.get(
            "ready_for_final_test"
        )
        is True,
        (
            "threshold lock declares "
            "ready_for_final_test"
        ),
    )

    multiseed = load_json(
        MULTISEED_MANIFEST
    )

    require(
        multiseed.get(
            "artifact_type"
        )
        ==
        "final_multiseed_training_manifest",
        (
            "multiseed manifest "
            "has correct artifact type"
        ),
    )

    require(
        multiseed.get(
            "ready_for_final_test"
        )
        is True,
        (
            "multiseed manifest declares "
            "ready_for_final_test"
        ),
    )

    require(
        tuple(
            multiseed[
                "seeds"
            ]
        )
        == FINAL_SEEDS,
        (
            "final seeds are exactly "
            "{42,1337,2026}"
        ),
    )

    # -------------------------------------------------------------------------
    # Script 58 committed temporary directories by renaming them.
    # Its JSON checkpoint path strings may therefore retain .__tmp__.
    #
    # Do not use those path strings.
    # Resolve canonical committed path by seed and verify against the
    # checkpoint SHA recorded by the frozen manifest.
    # -------------------------------------------------------------------------

    checkpoint_map = {}

    filenames = {
        "task1":
            "task1_T1B_best.pt",

        "task2":
            "task2_A1_CNN_best.pt",

        "task3":
            "task3_M3_phaseB_best.pt",
    }

    for seed in FINAL_SEEDS:

        seed_run = multiseed[
            "runs"
        ][
            str(seed)
        ]

        require(
            int(
                seed_run[
                    "seed"
                ]
            )
            == seed,
            (
                f"multiseed summary "
                f"seed identity = {seed}"
            ),
        )

        checkpoint_map[
            seed
        ] = {}

        for key, filename in (
            filenames.items()
        ):

            path = (
                ROOT
                / "models/final_multiseed"
                / f"seed{seed}"
                / filename
            )

            require(
                path.is_file(),
                (
                    f"final {key} checkpoint "
                    f"exists for seed {seed}"
                ),
            )

            expected_sha = (
                seed_run[
                    key
                ][
                    "checkpoint_sha256"
                ]
            )

            actual_sha = sha256_file(
                path
            )

            require(
                actual_sha
                == expected_sha,
                (
                    f"final {key} seed {seed} "
                    "checkpoint matches "
                    "multiseed manifest SHA"
                ),
            )

            checkpoint_map[
                seed
            ][
                key
            ] = {
                "path":
                    path,

                "sha256":
                    actual_sha,
            }

    # -------------------------------------------------------------------------
    # Thresholds are fixed and read-only from here.
    # -------------------------------------------------------------------------

    models = threshold_lock[
        "models"
    ]

    thresholds = {
        "CNN_A1":
            float(
                models[
                    "B2_A1_CNN"
                ][
                    "threshold"
                ]
            ),

        "BERT_T1B":
            float(
                models[
                    "BERT_T1B"
                ][
                    "threshold"
                ]
            ),

        "GraphSAGE_A3_G2":
            float(
                models[
                    "GraphSAGE_A3_G2"
                ][
                    "threshold"
                ]
            ),

        "GAT_A4_G2":
            float(
                models[
                    "GAT_A4_G2"
                ][
                    "threshold"
                ]
            ),

        "EarlyFusion_M3B":
            float(
                models[
                    "EarlyFusion_M3B"
                ][
                    "threshold"
                ]
            ),

        "CrossAttention_M4B":
            float(
                models[
                    "CrossAttention_M4B"
                ][
                    "threshold"
                ]
            ),
    }

    require(
        thresholds
        == {
            "CNN_A1": 0.50,
            "BERT_T1B": 0.70,
            "GraphSAGE_A3_G2": 0.50,
            "GAT_A4_G2": 0.45,
            "EarlyFusion_M3B": 0.70,
            "CrossAttention_M4B": 0.70,
        },
        (
            "all six model thresholds "
            "match final frozen values"
        ),
    )

    numerical_modes = {
        "GraphSAGE_A3_G2":
            models[
                "GraphSAGE_A3_G2"
            ][
                "numerical_path"
            ],

        "GAT_A4_G2":
            models[
                "GAT_A4_G2"
            ][
                "numerical_path"
            ],

        "CrossAttention_M4B":
            models[
                "CrossAttention_M4B"
            ][
                "numerical_path"
            ],
    }

    for name, mode in (
        numerical_modes.items()
    ):

        require(
            mode
            in {
                "FP32",
                "FP16_autocast",
            },
            (
                f"{name} numerical path "
                "is frozen and recognized"
            ),
        )

    return (
        threshold_lock,
        multiseed,
        checkpoint_map,
        thresholds,
        numerical_modes,
    )


# =============================================================================
# 6. TEST ACCESS MARKER
# =============================================================================

def begin_test_access():

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    marker = (
        TEMP_DIR
        / "TEST_ACCESS_STARTED.txt"
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    marker.write_text(
        (
            "FINAL TEST ACCESS STARTED\n"
            f"UTC: {timestamp}\n"
            "\n"
            "If this run fails after this marker, "
            "do not casually rerun Script 60.\n"
        ),
        encoding="utf-8",
    )

    banner(
        "2. FINAL TEST ACCESS BEGINS NOW"
    )

    print(
        "All model/threshold/training "
        "decisions were frozen beforehand."
    )

    print()
    print(
        "From this point onward:"
    )

    print(
        "  NO model selection"
    )

    print(
        "  NO threshold tuning"
    )

    print(
        "  NO retraining"
    )


# =============================================================================
# 7. LOAD TRAIN MAJORITY STATISTICS + TEST DATA
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


def resolve_mel_path(
    value: str,
) -> Path:

    raw = Path(value)

    candidates = []

    if raw.is_absolute():

        candidates.append(raw)

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

            resolved = candidate.resolve()

            matches[
                str(resolved)
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            "cannot uniquely resolve "
            f"mel path: {value}"
        )

    return next(
        iter(matches.values())
    )


def resolve_graph_path(
    value: str,
) -> Path:

    raw = Path(value)

    candidates = []

    if raw.is_absolute():

        candidates.append(raw)

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

            resolved = candidate.resolve()

            matches[
                str(resolved)
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            "cannot uniquely resolve "
            f"graph path: {value}"
        )

    return next(
        iter(matches.values())
    )


def load_graph(
    path: Path,
) -> Data:

    with np.load(
        path,
        allow_pickle=False,
    ) as graph:

        if set(
            graph.files
        ) != EXPECTED_GRAPH_KEYS:

            raise RuntimeError(
                "unexpected graph schema: "
                f"{path.name}"
            )

        x = np.asarray(
            graph["x"]
        )

        edge = np.asarray(
            graph[
                "edge_index_g2"
            ]
        )

        threshold = float(
            np.asarray(
                graph[
                    "similarity_threshold"
                ]
            ).reshape(-1)[0]
        )

        nonlocal_only = bool(
            np.asarray(
                graph[
                    "nonlocal_only"
                ]
            ).reshape(-1)[0]
        )

        degree_cap = int(
            np.asarray(
                graph[
                    "extra_degree_cap"
                ]
            ).reshape(-1)[0]
        )

        rule_version = int(
            np.asarray(
                graph[
                    "rule_version"
                ]
            ).reshape(-1)[0]
        )

    if x.shape != (
        9,
        140,
    ):

        raise RuntimeError(
            f"{path.name}: "
            f"invalid node shape {x.shape}"
        )

    if x.dtype != np.float32:

        raise RuntimeError(
            f"{path.name}: "
            f"invalid x dtype {x.dtype}"
        )

    if (
        edge.ndim != 2
        or edge.shape[0] != 2
        or edge.dtype != np.int64
    ):

        raise RuntimeError(
            f"{path.name}: "
            "invalid G2 edge_index"
        )

    if not np.isclose(
        threshold,
        0.85,
        rtol=0.0,
        atol=1e-6,
    ):

        raise RuntimeError(
            f"{path.name}: "
            "similarity threshold changed"
        )

    if not nonlocal_only:

        raise RuntimeError(
            f"{path.name}: "
            "nonlocal_only changed"
        )

    if degree_cap != 2:

        raise RuntimeError(
            f"{path.name}: "
            "degree cap changed"
        )

    if rule_version != 2:

        raise RuntimeError(
            f"{path.name}: "
            "rule version changed"
        )

    return Data(
        x=torch.from_numpy(
            x.copy()
        ),

        edge_index=torch.from_numpy(
            edge.copy()
        ),
    )


def load_final_data():

    banner(
        "3. LOAD TRAIN STATISTICS + TEST DATA"
    )

    token_manifest = pd.read_parquet(
        TOKEN_MANIFEST
    ).copy()

    token_manifest[
        "track_id"
    ] = (
        token_manifest[
            "track_id"
        ].astype(str)
    )

    token_manifest[
        "split"
    ] = (
        token_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    counts = (
        token_manifest[
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
            "frozen split counts remain "
            "4112/514/514"
        ),
    )

    test_manifest = (
        token_manifest.loc[
            token_manifest[
                "split"
            ]
            == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    require(
        len(test_manifest)
        == N_TEST,
        "TEST token rows = 514",
    )

    test_ids = (
        test_manifest[
            "track_id"
        ].tolist()
    )

    require(
        len(
            set(test_ids)
        )
        == N_TEST,
        "TEST IDs are unique",
    )

    y_columns = [
        f"y_{i:02d}"
        for i in range(
            NUM_LABELS
        )
    ]

    target_dataset = pads.dataset(
        str(TARGETS),
        format="parquet",
    )

    # -------------------------------------------------------------------------
    # TRAIN labels: B1 is learned only from TRAIN.
    # -------------------------------------------------------------------------

    train_table = (
        target_dataset.to_table(
            columns=[
                "track_id",
                "split",
                *y_columns,
            ],
            filter=(
                pads.field("split")
                == "train"
            ),
        )
    )

    train_frame = (
        train_table.to_pandas()
    )

    require(
        len(train_frame)
        == N_TRAIN,
        (
            "B1 TRAIN target rows "
            "= 4112"
        ),
    )

    train_y = (
        train_frame[
            y_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    train_prevalence = (
        train_y.mean(
            axis=0
        )
    )

    majority_classes = (
        train_prevalence
        >= 0.5
    ).astype(
        np.int64
    )

    print()
    print(
        "B1 majority classes learned "
        "from TRAIN only:"
    )

    print(
        " ",
        majority_classes.tolist(),
    )

    # -------------------------------------------------------------------------
    # TEST TARGETS — first target-dependent TEST load.
    # -------------------------------------------------------------------------

    test_table = (
        target_dataset.to_table(
            columns=[
                "track_id",
                "split",
                *y_columns,
            ],
            filter=(
                pads.field("split")
                == "test"
            ),
        )
    )

    test_frame = (
        test_table.to_pandas()
    )

    test_frame[
        "track_id"
    ] = (
        test_frame[
            "track_id"
        ].astype(str)
    )

    test_frame[
        "split"
    ] = (
        test_frame[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    require(
        len(test_frame)
        == N_TEST,
        (
            "TEST target rows loaded "
            "= 514"
        ),
    )

    require(
        set(
            test_frame[
                "split"
            ]
        )
        == {
            "test"
        },
        (
            "target-value TEST load "
            "contains TEST only"
        ),
    )

    require(
        set(test_ids)
        ==
        set(
            test_frame[
                "track_id"
            ]
        ),
        (
            "TEST token IDs and "
            "target IDs match"
        ),
    )

    test_lookup = (
        test_frame.set_index(
            "track_id",
            drop=False,
        )
    )

    aligned_test = (
        test_lookup.loc[
            test_ids
        ]
        .reset_index(
            drop=True
        )
    )

    y_true = (
        aligned_test[
            y_columns
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    require(
        y_true.shape
        == (
            N_TEST,
            NUM_LABELS,
        ),
        (
            "TEST target matrix "
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
            "TEST targets are binary"
        ),
    )

    support = (
        y_true.sum(
            axis=0
        ).astype(
            np.int64
        )
    )

    # -------------------------------------------------------------------------
    # TEST BERT tokens.
    # -------------------------------------------------------------------------

    cache_indices = (
        test_manifest[
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
            N_TEST,
            MAX_LENGTH,
        ),
        (
            "TEST BERT input shape "
            "= (514,112)"
        ),
    )

    # -------------------------------------------------------------------------
    # TEST mel features.
    # -------------------------------------------------------------------------

    audio_manifest = (
        pd.read_parquet(
            AUDIO_MANIFEST
        )
        .copy()
    )

    audio_manifest[
        "track_id"
    ] = (
        audio_manifest[
            "track_id"
        ].astype(str)
    )

    audio_lookup = (
        audio_manifest.set_index(
            "track_id",
            drop=False,
        )
    )

    require(
        set(test_ids).issubset(
            set(
                audio_manifest[
                    "track_id"
                ]
            )
        ),
        (
            "all TEST IDs have "
            "audio features"
        ),
    )

    mels = np.empty(
        (
            N_TEST,
            1,
            128,
            431,
        ),
        dtype=np.float32,
    )

    print()
    print(
        "Loading 514 TEST mel arrays..."
    )

    for index, track_id in enumerate(
        test_ids
    ):

        row = audio_lookup.loc[
            track_id
        ]

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
                "invalid TEST mel: "
                f"{track_id}"
            )

        mels[
            index,
            0,
            :,
            :
        ] = mel

        number = index + 1

        if (
            number == 1
            or number % 100 == 0
            or number == N_TEST
        ):

            print(
                f"  loaded "
                f"{number}/{N_TEST}"
            )

    require(
        mels.shape
        == (
            N_TEST,
            1,
            128,
            431,
        ),
        (
            "TEST mel tensor "
            "= (514,1,128,431)"
        ),
    )

    # -------------------------------------------------------------------------
    # TEST G2 graphs.
    # -------------------------------------------------------------------------

    graph_manifest = (
        pd.read_parquet(
            GRAPH_MANIFEST
        )
        .copy()
    )

    graph_manifest[
        "track_id"
    ] = (
        graph_manifest[
            "track_id"
        ].astype(str)
    )

    graph_manifest[
        "split"
    ] = (
        graph_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    graph_lookup = (
        graph_manifest.set_index(
            "track_id",
            drop=False,
        )
    )

    graphs = []

    print()
    print(
        "Loading 514 TEST G2 graphs..."
    )

    for index, track_id in enumerate(
        test_ids,
        start=1,
    ):

        row = graph_lookup.loc[
            track_id
        ]

        if str(
            row[
                "split"
            ]
        ).lower() != "test":

            raise RuntimeError(
                "non-TEST graph encountered "
                "during final TEST evaluation"
            )

        graph_path = resolve_graph_path(
            str(
                row[
                    "graph_path"
                ]
            )
        )

        graphs.append(
            load_graph(
                graph_path
            )
        )

        if (
            index == 1
            or index % 100 == 0
            or index == N_TEST
        ):

            print(
                f"  loaded "
                f"{index}/{N_TEST}"
            )

    require(
        len(graphs)
        == N_TEST,
        (
            "exactly 514 TEST "
            "G2 graphs loaded"
        ),
    )

    print()
    print(
        "FINAL TEST DATA LOAD COMPLETE."
    )

    return {
        "track_ids":
            test_ids,

        "targets":
            y_true,

        "support":
            support,

        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "token_type_ids":
            token_type_ids,

        "mels":
            mels,

        "graphs":
            graphs,

        "train_prevalence":
            train_prevalence,

        "majority_classes":
            majority_classes,
    }


# =============================================================================
# 8. DATASETS / LOADERS
# =============================================================================

class FusionTestDataset(Dataset):

    def __init__(
        self,
        data,
    ):

        self.data = data

    def __len__(self):

        return N_TEST

    def __getitem__(
        self,
        index,
    ):

        return (
            torch.from_numpy(
                self.data[
                    "input_ids"
                ][
                    index
                ]
            ),

            torch.from_numpy(
                self.data[
                    "attention_mask"
                ][
                    index
                ]
            ),

            torch.from_numpy(
                self.data[
                    "token_type_ids"
                ][
                    index
                ]
            ),

            self.data[
                "graphs"
            ][
                index
            ],
        )


def collate_fusion(
    samples,
):

    return (
        torch.stack(
            [
                x[0]
                for x in samples
            ]
        ),

        torch.stack(
            [
                x[1]
                for x in samples
            ]
        ),

        torch.stack(
            [
                x[2]
                for x in samples
            ]
        ),

        Batch.from_data_list(
            [
                x[3]
                for x in samples
            ]
        ),
    )


# =============================================================================
# 9. METRICS
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:

    require(
        probabilities.shape
        == y_true.shape,
        (
            "probability and target "
            "shapes match"
        ),
    )

    require(
        np.isfinite(
            probabilities
        ).all(),
        (
            "probabilities are finite"
        ),
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    per_label_f1 = f1_score(
        y_true,
        predictions,
        average=None,
        zero_division=0,
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

    per_label_ap = (
        average_precision_score(
            y_true,
            probabilities,
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
            probabilities,
            average="micro",
        )
    )

    return {
        "macro_f1":
            macro_f1,

        "micro_f1":
            micro_f1,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,

        "per_label_f1":
            per_label_f1.astype(
                np.float64
            ),

        "per_label_ap":
            per_label_ap.astype(
                np.float64
            ),

        "predictions":
            predictions,
    }


def compute_majority_metrics(
    data,
):

    scores = np.tile(
        data[
            "train_prevalence"
        ].reshape(
            1,
            -1,
        ),
        (
            N_TEST,
            1,
        ),
    )

    predictions = np.tile(
        data[
            "majority_classes"
        ].reshape(
            1,
            -1,
        ),
        (
            N_TEST,
            1,
        ),
    )

    y_true = data[
        "targets"
    ]

    per_label_f1 = f1_score(
        y_true,
        predictions,
        average=None,
        zero_division=0,
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

    # For threshold-free PR reporting, use TRAIN prevalence
    # as B1's constant per-label score.
    per_label_ap = (
        average_precision_score(
            y_true,
            scores,
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
            scores,
            average="micro",
        )
    )

    return {
        "macro_f1":
            macro_f1,

        "micro_f1":
            micro_f1,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,

        "per_label_f1":
            per_label_f1.astype(
                np.float64
            ),

        "per_label_ap":
            per_label_ap.astype(
                np.float64
            ),

        "predictions":
            predictions,

        "probabilities":
            scores,
    }


# =============================================================================
# 10. MODEL RECONSTRUCTION HELPERS
# =============================================================================

def build_selected_source_bert():

    config = BertConfig.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    checkpoint = load_checkpoint(
        SELECTED_T1_CHECKPOINT
    )

    state = checkpoint[
        "model_state_dict"
    ]

    bert_state = {
        key[
            len("bert.") :
        ]:
            value

        for key, value in (
            state.items()
        )

        if key.startswith(
            "bert."
        )
    }

    incompatible = (
        bert.load_state_dict(
            bert_state,
            strict=True,
        )
    )

    require(
        len(
            incompatible.missing_keys
        )
        == 0
        and
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "selected T1-B BERT "
            "source strict-loads"
        ),
    )

    return bert


def build_selected_source_a4():

    model = A4GAT()

    checkpoint = load_checkpoint(
        A4_CHECKPOINT
    )

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
        == 0
        and
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "selected A4 GAT "
            "source strict-loads"
        ),
    )

    return model


def reconstruct_final_m3(
    checkpoint_path: Path,
    device,
):

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    require(
        checkpoint.get(
            "variant"
        )
        == "M3",
        (
            "final Task-3 checkpoint "
            "variant = M3"
        ),
    )

    require(
        checkpoint.get(
            "phase"
        )
        == "B",
        (
            "final Task-3 checkpoint "
            "phase = B"
        ),
    )

    bert = build_selected_source_bert()

    gnn = build_selected_source_a4()

    fusion = M3Fusion()

    bert.encoder.layer[
        10
    ].load_state_dict(
        checkpoint[
            "bert_layer_10_state_dict"
        ],
        strict=True,
    )

    bert.encoder.layer[
        11
    ].load_state_dict(
        checkpoint[
            "bert_layer_11_state_dict"
        ],
        strict=True,
    )

    gnn.load_state_dict(
        checkpoint[
            "gnn_state_dict"
        ],
        strict=True,
    )

    fusion.load_state_dict(
        checkpoint[
            "fusion_state_dict"
        ],
        strict=True,
    )

    model = M3PhaseBModel(
        bert,
        gnn,
        fusion,
    )

    model.to(device)
    model.eval()

    return model


def get_phase_b_state(
    checkpoint,
    preferred,
    fallback,
):

    if preferred in checkpoint:

        return checkpoint[
            preferred
        ]

    if fallback in checkpoint:

        return checkpoint[
            fallback
        ]

    raise RuntimeError(
        "missing M4 Phase-B "
        f"state: {preferred}"
    )


def reconstruct_m4(
    device,
):

    bert = build_selected_source_bert()

    gnn = build_selected_source_a4()

    fusion = M4Fusion()

    phase_a = load_checkpoint(
        M4_PHASEA_CHECKPOINT
    )

    fusion.load_state_dict(
        phase_a[
            "model_state_dict"
        ],
        strict=True,
    )

    phase_b = load_checkpoint(
        M4_PHASEB_CHECKPOINT
    )

    bert.encoder.layer[
        10
    ].load_state_dict(
        get_phase_b_state(
            phase_b,
            "bert_layer_10_state_dict",
            "bert_layer_10",
        ),
        strict=True,
    )

    bert.encoder.layer[
        11
    ].load_state_dict(
        get_phase_b_state(
            phase_b,
            "bert_layer_11_state_dict",
            "bert_layer_11",
        ),
        strict=True,
    )

    gnn.load_state_dict(
        get_phase_b_state(
            phase_b,
            "gnn_state_dict",
            "gnn",
        ),
        strict=True,
    )

    fusion.load_state_dict(
        get_phase_b_state(
            phase_b,
            "fusion_state_dict",
            "fusion",
        ),
        strict=True,
    )

    model = M4Model(
        bert,
        gnn,
        fusion,
    )

    model.to(device)
    model.eval()

    return model


# =============================================================================
# 11. INFERENCE FUNCTIONS
# =============================================================================

def infer_t1(
    checkpoint_path,
    data,
    device,
):

    model = T1BModel()

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(device)
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(
            data[
                "input_ids"
            ]
        ),
        torch.from_numpy(
            data[
                "attention_mask"
            ]
        ),
        torch.from_numpy(
            data[
                "token_type_ids"
            ]
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

    all_probability = []

    with torch.no_grad():

        for (
            input_ids,
            attention_mask,
            token_type_ids,
        ) in loader:

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

            # Match formal T1-B numerical evaluation path.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                logits = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

            probability = torch.sigmoid(
                logits.float()
            )

            all_probability.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    result = np.concatenate(
        all_probability,
        axis=0,
    )

    require(
        result.shape
        == (
            N_TEST,
            NUM_LABELS,
        ),
        (
            "T1-B TEST probabilities "
            "= (514,20)"
        ),
    )

    del model

    torch.cuda.empty_cache()

    return result


def infer_a1(
    checkpoint_path,
    data,
    device,
):

    model = A1CNN()

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(device)
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(
            data[
                "mels"
            ]
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

    all_probability = []

    with torch.no_grad():

        for (mels,) in loader:

            mels = mels.to(
                device,
                non_blocking=True,
            )

            # Frozen A1 numerical path = FP32.
            logits = model(
                mels
            )

            probability = torch.sigmoid(
                logits.float()
            )

            all_probability.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    result = np.concatenate(
        all_probability,
        axis=0,
    )

    require(
        result.shape
        == (
            N_TEST,
            NUM_LABELS,
        ),
        (
            "A1 TEST probabilities "
            "= (514,20)"
        ),
    )

    del model

    torch.cuda.empty_cache()

    return result


def mode_uses_amp(
    mode: str,
) -> bool:

    if mode == "FP32":
        return False

    if mode == "FP16_autocast":
        return True

    raise RuntimeError(
        f"unknown numerical mode: {mode}"
    )


def infer_graph_model(
    model,
    graphs,
    device,
    numerical_mode,
):

    model.to(device)
    model.eval()

    use_amp = mode_uses_amp(
        numerical_mode
    )

    all_probability = []

    with torch.no_grad():

        for start in range(
            0,
            N_TEST,
            32,
        ):

            graph_batch = (
                Batch.from_data_list(
                    graphs[
                        start:
                        start + 32
                    ]
                )
                .to(device)
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(
                    graph_batch
                )

            probability = torch.sigmoid(
                logits.float()
            )

            all_probability.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    result = np.concatenate(
        all_probability,
        axis=0,
    )

    require(
        result.shape
        == (
            N_TEST,
            NUM_LABELS,
        ),
        (
            "graph TEST probabilities "
            "= (514,20)"
        ),
    )

    model.cpu()

    torch.cuda.empty_cache()

    return result


def infer_fusion(
    model,
    data,
    device,
    *,
    numerical_mode,
):

    dataset = FusionTestDataset(
        data
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fusion,
    )

    use_amp = mode_uses_amp(
        numerical_mode
    )

    all_probability = []

    model.eval()

    with torch.no_grad():

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
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

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    graph_batch,
                )

            probability = torch.sigmoid(
                logits.float()
            )

            all_probability.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

    result = np.concatenate(
        all_probability,
        axis=0,
    )

    require(
        result.shape
        == (
            N_TEST,
            NUM_LABELS,
        ),
        (
            "fusion TEST probabilities "
            "= (514,20)"
        ),
    )

    return result


# =============================================================================
# 12. RESULT COLLECTION
# =============================================================================

def add_result(
    *,
    model_key,
    display_name,
    modality,
    seed,
    threshold,
    probabilities,
    y_true,
    support,
    checkpoint_sha,
    numerical_mode,
    run_records,
    per_label_records,
    probability_cache,
):

    metrics = compute_metrics(
        y_true,
        probabilities,
        threshold,
    )

    print()
    print(
        f"{display_name}"
        + (
            f" — seed {seed}"
            if seed is not None
            else ""
        )
    )

    print(
        f"  Macro-F1 = "
        f"{metrics['macro_f1']:.8f}"
    )

    print(
        f"  Micro-F1 = "
        f"{metrics['micro_f1']:.8f}"
    )

    print(
        f"  AUC-PR   = "
        f"{metrics['macro_ap']:.8f}"
    )

    run_records.append(
        {
            "model_key":
                model_key,

            "model":
                display_name,

            "modality":
                modality,

            "seed":
                seed,

            "threshold":
                threshold,

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "micro_f1":
                metrics[
                    "micro_f1"
                ],

            "auc_pr_macro_mean_ap":
                metrics[
                    "macro_ap"
                ],

            "micro_average_precision":
                metrics[
                    "micro_ap"
                ],

            "checkpoint_sha256":
                checkpoint_sha,

            "numerical_mode":
                numerical_mode,
        }
    )

    for index in range(
        NUM_LABELS
    ):

        per_label_records.append(
            {
                "model_key":
                    model_key,

                "model":
                    display_name,

                "seed":
                    seed,

                "label_index":
                    index,

                "label":
                    LABEL_NAMES[
                        index
                    ],

                "test_support":
                    int(
                        support[
                            index
                        ]
                    ),

                "f1":
                    float(
                        metrics[
                            "per_label_f1"
                        ][
                            index
                        ]
                    ),

                "average_precision":
                    float(
                        metrics[
                            "per_label_ap"
                        ][
                            index
                        ]
                    ),
            }
        )

    cache_key = (
        f"prob__{model_key}"
        + (
            f"__seed{seed}"
            if seed is not None
            else ""
        )
    )

    probability_cache[
        cache_key
    ] = probabilities.astype(
        np.float32
    )


# =============================================================================
# 13. MAIN FINAL TEST
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 60 — FINAL TEST EVALUATION, EXACTLY ONCE"
    )

    print(
        "No model selection."
    )

    print(
        "No threshold tuning."
    )

    print(
        "No training."
    )

    seed_everything(
        42
    )

    try:

        (
            threshold_lock,
            multiseed_manifest,
            checkpoint_map,
            thresholds,
            numerical_modes,
        ) = verify_dependencies()

    except Exception as exc:

        banner(
            "SCRIPT 60 — ABORTED BEFORE TEST ACCESS"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "TEST labels were NOT loaded."
        )

        return 1

    # =========================================================================
    # From here onward TEST is considered accessed.
    # =========================================================================

    try:

        begin_test_access()

        data = load_final_data()

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
            " ",
            torch.cuda.get_device_name(
                device
            ),
        )

        y_true = data[
            "targets"
        ]

        support = data[
            "support"
        ]

        run_records = []

        per_label_records = []

        probability_cache = {
            "y_true":
                y_true.astype(
                    np.int8
                ),

            "track_ids":
                np.asarray(
                    data[
                        "track_ids"
                    ],
                    dtype=str,
                ),

            "test_support":
                support.astype(
                    np.int64
                ),
        }

        # =====================================================================
        # B1 — Majority-class baseline
        # =====================================================================

        banner(
            "4. B1 MAJORITY-CLASS BASELINE"
        )

        majority = compute_majority_metrics(
            data
        )

        print(
            f"B1 Macro-F1 = "
            f"{majority['macro_f1']:.8f}"
        )

        print(
            f"B1 Micro-F1 = "
            f"{majority['micro_f1']:.8f}"
        )

        print(
            f"B1 AUC-PR   = "
            f"{majority['macro_ap']:.8f}"
        )

        run_records.append(
            {
                "model_key":
                    "B1_Majority",

                "model":
                    "B1 Majority-class",

                "modality":
                    "—",

                "seed":
                    None,

                "threshold":
                    None,

                "macro_f1":
                    majority[
                        "macro_f1"
                    ],

                "micro_f1":
                    majority[
                        "micro_f1"
                    ],

                "auc_pr_macro_mean_ap":
                    majority[
                        "macro_ap"
                    ],

                "micro_average_precision":
                    majority[
                        "micro_ap"
                    ],

                "checkpoint_sha256":
                    None,

                "numerical_mode":
                    (
                        "majority learned per label "
                        "from TRAIN; TRAIN prevalence "
                        "used as constant PR score"
                    ),
            }
        )

        for index in range(
            NUM_LABELS
        ):

            per_label_records.append(
                {
                    "model_key":
                        "B1_Majority",

                    "model":
                        "B1 Majority-class",

                    "seed":
                        None,

                    "label_index":
                        index,

                    "label":
                        LABEL_NAMES[
                            index
                        ],

                    "test_support":
                        int(
                            support[
                                index
                            ]
                        ),

                    "f1":
                        float(
                            majority[
                                "per_label_f1"
                            ][
                                index
                            ]
                        ),

                    "average_precision":
                        float(
                            majority[
                                "per_label_ap"
                            ][
                                index
                            ]
                        ),
                }
            )

        probability_cache[
            "prob__B1_Majority"
        ] = (
            majority[
                "probabilities"
            ].astype(
                np.float32
            )
        )

        probability_cache[
            "pred__B1_Majority"
        ] = (
            majority[
                "predictions"
            ].astype(
                np.int8
            )
        )

        # =====================================================================
        # T1-B BERT — three final seeds
        # =====================================================================

        banner(
            "5. T1-B BERT — FINAL THREE SEEDS"
        )

        for seed in FINAL_SEEDS:

            print()
            print(
                f"Running final TEST inference "
                f"for T1-B seed {seed}..."
            )

            checkpoint_info = (
                checkpoint_map[
                    seed
                ][
                    "task1"
                ]
            )

            probability = infer_t1(
                checkpoint_info[
                    "path"
                ],
                data,
                device,
            )

            add_result(
                model_key=
                    "BERT_T1B",

                display_name=
                    "BERT",

                modality=
                    "Text",

                seed=
                    seed,

                threshold=
                    thresholds[
                        "BERT_T1B"
                    ],

                probabilities=
                    probability,

                y_true=
                    y_true,

                support=
                    support,

                checkpoint_sha=
                    checkpoint_info[
                        "sha256"
                    ],

                numerical_mode=
                    "FP16_autocast",

                run_records=
                    run_records,

                per_label_records=
                    per_label_records,

                probability_cache=
                    probability_cache,
            )

        # =====================================================================
        # A1 CNN — three final seeds
        # =====================================================================

        banner(
            "6. A1 CNN — FINAL THREE SEEDS"
        )

        for seed in FINAL_SEEDS:

            print()
            print(
                f"Running final TEST inference "
                f"for A1 seed {seed}..."
            )

            checkpoint_info = (
                checkpoint_map[
                    seed
                ][
                    "task2"
                ]
            )

            probability = infer_a1(
                checkpoint_info[
                    "path"
                ],
                data,
                device,
            )

            add_result(
                model_key=
                    "CNN_A1",

                display_name=
                    "B2 CNN",

                modality=
                    "Audio",

                seed=
                    seed,

                threshold=
                    thresholds[
                        "CNN_A1"
                    ],

                probabilities=
                    probability,

                y_true=
                    y_true,

                support=
                    support,

                checkpoint_sha=
                    checkpoint_info[
                        "sha256"
                    ],

                numerical_mode=
                    "FP32",

                run_records=
                    run_records,

                per_label_records=
                    per_label_records,

                probability_cache=
                    probability_cache,
            )

        # =====================================================================
        # A3 GraphSAGE / G2 — frozen seed42 report model
        # =====================================================================

        banner(
            "7. A3 GRAPHSAGE/G2"
        )

        a3 = GraphSAGEModel()

        a3_checkpoint = load_checkpoint(
            A3_CHECKPOINT
        )

        a3.load_state_dict(
            a3_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        probability = infer_graph_model(
            a3,
            data[
                "graphs"
            ],
            device,
            numerical_modes[
                "GraphSAGE_A3_G2"
            ],
        )

        add_result(
            model_key=
                "GraphSAGE_A3_G2",

            display_name=
                "GraphSAGE",

            modality=
                "Audio Graph",

            seed=
                42,

            threshold=
                thresholds[
                    "GraphSAGE_A3_G2"
                ],

            probabilities=
                probability,

            y_true=
                y_true,

            support=
                support,

            checkpoint_sha=
                EXPECTED_A3_SHA256,

            numerical_mode=
                numerical_modes[
                    "GraphSAGE_A3_G2"
                ],

            run_records=
                run_records,

            per_label_records=
                per_label_records,

            probability_cache=
                probability_cache,
        )

        del a3

        torch.cuda.empty_cache()

        # =====================================================================
        # A4 GAT / G2 — frozen seed42 report model
        # =====================================================================

        banner(
            "8. A4 GAT/G2"
        )

        a4 = A4GAT()

        a4_checkpoint = load_checkpoint(
            A4_CHECKPOINT
        )

        a4.load_state_dict(
            a4_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        probability = infer_graph_model(
            a4,
            data[
                "graphs"
            ],
            device,
            numerical_modes[
                "GAT_A4_G2"
            ],
        )

        add_result(
            model_key=
                "GAT_A4_G2",

            display_name=
                "GAT",

            modality=
                "Audio Graph",

            seed=
                42,

            threshold=
                thresholds[
                    "GAT_A4_G2"
                ],

            probabilities=
                probability,

            y_true=
                y_true,

            support=
                support,

            checkpoint_sha=
                EXPECTED_A4_SHA256,

            numerical_mode=
                numerical_modes[
                    "GAT_A4_G2"
                ],

            run_records=
                run_records,

            per_label_records=
                per_label_records,

            probability_cache=
                probability_cache,
        )

        del a4

        torch.cuda.empty_cache()

        # =====================================================================
        # M3 Early Fusion — three final seeds
        # =====================================================================

        banner(
            "9. M3 EARLY FUSION — FINAL THREE SEEDS"
        )

        for seed in FINAL_SEEDS:

            print()
            print(
                f"Running final TEST inference "
                f"for M3 seed {seed}..."
            )

            checkpoint_info = (
                checkpoint_map[
                    seed
                ][
                    "task3"
                ]
            )

            model = reconstruct_final_m3(
                checkpoint_info[
                    "path"
                ],
                device,
            )

            probability = infer_fusion(
                model,
                data,
                device,
                numerical_mode=
                    "FP32",
            )

            add_result(
                model_key=
                    "EarlyFusion_M3B",

                display_name=
                    "Early Fusion",

                modality=
                    "Audio+Text",

                seed=
                    seed,

                threshold=
                    thresholds[
                        "EarlyFusion_M3B"
                    ],

                probabilities=
                    probability,

                y_true=
                    y_true,

                support=
                    support,

                checkpoint_sha=
                    checkpoint_info[
                        "sha256"
                    ],

                numerical_mode=
                    "FP32",

                run_records=
                    run_records,

                per_label_records=
                    per_label_records,

                probability_cache=
                    probability_cache,
            )

            del model

            torch.cuda.empty_cache()

        # =====================================================================
        # M4 Cross-Attention — frozen seed42 selected report model
        # =====================================================================

        banner(
            "10. M4 CROSS-ATTENTION PHASE B"
        )

        model = reconstruct_m4(
            device
        )

        probability = infer_fusion(
            model,
            data,
            device,
            numerical_mode=
                numerical_modes[
                    "CrossAttention_M4B"
                ],
        )

        add_result(
            model_key=
                "CrossAttention_M4B",

            display_name=
                "Cross-Attention",

            modality=
                "Audio+Text",

            seed=
                42,

            threshold=
                thresholds[
                    "CrossAttention_M4B"
                ],

            probabilities=
                probability,

            y_true=
                y_true,

            support=
                support,

            checkpoint_sha=
                EXPECTED_M4_PHASEB_SHA256,

            numerical_mode=
                numerical_modes[
                    "CrossAttention_M4B"
                ],

            run_records=
                run_records,

            per_label_records=
                per_label_records,

            probability_cache=
                probability_cache,
        )

        del model

        torch.cuda.empty_cache()

        # =====================================================================
        # 11. SAVE RAW TEST PREDICTIONS — NO MORE TEST INFERENCE NEEDED
        # =====================================================================

        banner(
            "11. SAVE FINAL TEST PREDICTION CACHE"
        )

        prediction_path = (
            TEMP_DIR
            / "final_test_predictions.npz"
        )

        np.savez_compressed(
            prediction_path,
            **probability_cache,
        )

        require(
            prediction_path.is_file(),
            (
                "final TEST probability "
                "cache saved"
            ),
        )

        # =====================================================================
        # 12. PER-RUN TABLE
        # =====================================================================

        run_frame = pd.DataFrame(
            run_records
        )

        per_run_path = (
            TEMP_DIR
            / "final_test_metrics_per_run.csv"
        )

        run_frame.to_csv(
            per_run_path,
            index=False,
        )

        # =====================================================================
        # 13. PER-LABEL RAW TABLE
        # =====================================================================

        per_label_frame = pd.DataFrame(
            per_label_records
        )

        per_label_path = (
            TEMP_DIR
            / "final_test_per_label_per_run.csv"
        )

        per_label_frame.to_csv(
            per_label_path,
            index=False,
        )

        # =====================================================================
        # 14. FINAL COMPARISON — mean ± sample SD for multiseed winners
        # =====================================================================

        banner(
            "12. BUILD FINAL COMPARISON TABLE"
        )

        final_order = [
            (
                "B1_Majority",
                "B1 Majority-class",
                "—",
            ),
            (
                "CNN_A1",
                "B2 CNN",
                "Audio",
            ),
            (
                "BERT_T1B",
                "BERT",
                "Text",
            ),
            (
                "GraphSAGE_A3_G2",
                "GraphSAGE",
                "Audio Graph",
            ),
            (
                "GAT_A4_G2",
                "GAT",
                "Audio Graph",
            ),
            (
                "EarlyFusion_M3B",
                "Early Fusion",
                "Audio+Text",
            ),
            (
                "CrossAttention_M4B",
                "Cross-Attention",
                "Audio+Text",
            ),
        ]

        comparison_rows = []

        display_rows = []

        for (
            model_key,
            display_name,
            modality,
        ) in final_order:

            subset = (
                run_frame.loc[
                    run_frame[
                        "model_key"
                    ]
                    == model_key
                ]
            )

            require(
                len(subset)
                in {
                    1,
                    3,
                },
                (
                    f"{display_name} has "
                    "valid final run count"
                ),
            )

            n_runs = len(
                subset
            )

            macro_values = (
                subset[
                    "macro_f1"
                ]
                .astype(float)
                .tolist()
            )

            micro_values = (
                subset[
                    "micro_f1"
                ]
                .astype(float)
                .tolist()
            )

            ap_values = (
                subset[
                    "auc_pr_macro_mean_ap"
                ]
                .astype(float)
                .tolist()
            )

            macro_mean = float(
                np.mean(
                    macro_values
                )
            )

            micro_mean = float(
                np.mean(
                    micro_values
                )
            )

            ap_mean = float(
                np.mean(
                    ap_values
                )
            )

            macro_std = sample_std(
                macro_values
            )

            micro_std = sample_std(
                micro_values
            )

            ap_std = sample_std(
                ap_values
            )

            comparison_rows.append(
                {
                    "model":
                        display_name,

                    "modality":
                        modality,

                    "n_runs":
                        n_runs,

                    "macro_f1_mean":
                        macro_mean,

                    "macro_f1_sample_std":
                        macro_std,

                    "micro_f1_mean":
                        micro_mean,

                    "micro_f1_sample_std":
                        micro_std,

                    "auc_pr_mean":
                        ap_mean,

                    "auc_pr_sample_std":
                        ap_std,
                }
            )

            display_rows.append(
                {
                    "Model":
                        display_name,

                    "Modality":
                        modality,

                    "Runs":
                        n_runs,

                    "Macro-F1":
                        format_mean_std(
                            macro_mean,
                            macro_std,
                            n_runs,
                        ),

                    "Micro-F1":
                        format_mean_std(
                            micro_mean,
                            micro_std,
                            n_runs,
                        ),

                    "AUC-PR":
                        format_mean_std(
                            ap_mean,
                            ap_std,
                            n_runs,
                        ),
                }
            )

        comparison_path = (
            TEMP_DIR
            / "final_comparison_table.csv"
        )

        pd.DataFrame(
            comparison_rows
        ).to_csv(
            comparison_path,
            index=False,
        )

        display_path = (
            TEMP_DIR
            / "final_comparison_table_display.csv"
        )

        display_frame = pd.DataFrame(
            display_rows
        )

        display_frame.to_csv(
            display_path,
            index=False,
        )

        print()
        print(
            display_frame.to_string(
                index=False
            )
        )

        # =====================================================================
        # 15. PER-LABEL SUMMARY
        # =====================================================================

        banner(
            "13. BUILD PER-LABEL SUMMARY"
        )

        per_label_summary = []

        for (
            model_key,
            display_name,
            _,
        ) in final_order:

            model_frame = (
                per_label_frame.loc[
                    per_label_frame[
                        "model_key"
                    ]
                    == model_key
                ]
            )

            for label_index in range(
                NUM_LABELS
            ):

                label_frame = (
                    model_frame.loc[
                        model_frame[
                            "label_index"
                        ]
                        == label_index
                    ]
                )

                f1_values = (
                    label_frame[
                        "f1"
                    ]
                    .astype(float)
                    .tolist()
                )

                ap_values = (
                    label_frame[
                        "average_precision"
                    ]
                    .astype(float)
                    .tolist()
                )

                n_runs = len(
                    f1_values
                )

                require(
                    n_runs
                    in {
                        1,
                        3,
                    },
                    (
                        f"{display_name} / "
                        f"label {label_index} "
                        "has valid run count"
                    ),
                )

                f1_mean = float(
                    np.mean(
                        f1_values
                    )
                )

                ap_mean = float(
                    np.mean(
                        ap_values
                    )
                )

                per_label_summary.append(
                    {
                        "model":
                            display_name,

                        "label_index":
                            label_index,

                        "label":
                            LABEL_NAMES[
                                label_index
                            ],

                        "test_support":
                            int(
                                support[
                                    label_index
                                ]
                            ),

                        "n_runs":
                            n_runs,

                        "f1_mean":
                            f1_mean,

                        "f1_sample_std":
                            sample_std(
                                f1_values
                            ),

                        "average_precision_mean":
                            ap_mean,

                        "average_precision_sample_std":
                            sample_std(
                                ap_values
                            ),
                    }
                )

        per_label_summary_path = (
            TEMP_DIR
            / "final_test_per_label_summary.csv"
        )

        pd.DataFrame(
            per_label_summary
        ).to_csv(
            per_label_summary_path,
            index=False,
        )

        # =====================================================================
        # 16. RESEARCH-COMPARISON NUMBERS
        # =====================================================================

        comparison_frame = (
            pd.DataFrame(
                comparison_rows
            )
            .set_index(
                "model"
            )
        )

        bert_ap = float(
            comparison_frame.loc[
                "BERT",
                "auc_pr_mean",
            ]
        )

        fusion_ap = float(
            comparison_frame.loc[
                "Early Fusion",
                "auc_pr_mean",
            ]
        )

        gat_ap = float(
            comparison_frame.loc[
                "GAT",
                "auc_pr_mean",
            ]
        )

        cnn_ap = float(
            comparison_frame.loc[
                "B2 CNN",
                "auc_pr_mean",
            ]
        )

        research_comparison = {
            "early_fusion_minus_bert_auc_pr":
                fusion_ap - bert_ap,

            "early_fusion_minus_gat_auc_pr":
                fusion_ap - gat_ap,

            "early_fusion_minus_cnn_auc_pr":
                fusion_ap - cnn_ap,

            "early_fusion_mean_auc_pr":
                fusion_ap,

            "bert_mean_auc_pr":
                bert_ap,

            "gat_seed42_auc_pr":
                gat_ap,

            "cnn_mean_auc_pr":
                cnn_ap,
        }

        # =====================================================================
        # 17. FINAL EVALUATION ARTIFACT
        # =====================================================================

        output_hashes = {
            "final_test_predictions.npz":
                sha256_file(
                    prediction_path
                ),

            "final_test_metrics_per_run.csv":
                sha256_file(
                    per_run_path
                ),

            "final_test_per_label_per_run.csv":
                sha256_file(
                    per_label_path
                ),

            "final_test_per_label_summary.csv":
                sha256_file(
                    per_label_summary_path
                ),

            "final_comparison_table.csv":
                sha256_file(
                    comparison_path
                ),

            "final_comparison_table_display.csv":
                sha256_file(
                    display_path
                ),
        }

        evaluation_artifact = {
            "artifact_type":
                "final_test_evaluation",

            "version":
                1,

            "test_evaluation_status":
                "FINAL",

            "test_access_policy":
                (
                    "first target-dependent TEST evaluation; "
                    "frozen thresholds applied without retuning"
                ),

            "test_examples":
                N_TEST,

            "labels":
                LABEL_NAMES,

            "test_support":
                [
                    int(x)
                    for x in support
                ],

            "final_seeds":
                list(
                    FINAL_SEEDS
                ),

            "multiseed_standard_deviation":
                (
                    "sample standard deviation "
                    "across seeds; ddof=1"
                ),

            "auc_pr_definition":
                (
                    "mean of per-label "
                    "sklearn average_precision_score"
                ),

            "b1_definition":
                (
                    "majority class learned independently "
                    "per label from TRAIN; TRAIN prevalence "
                    "used as constant threshold-free PR score"
                ),

            "frozen_thresholds":
                thresholds,

            "numerical_modes":
                {
                    "BERT_T1B":
                        "FP16_autocast",

                    "CNN_A1":
                        "FP32",

                    "EarlyFusion_M3B":
                        "FP32",

                    **numerical_modes,
                },

            "dependencies": {
                "targets_sha256":
                    EXPECTED_TARGETS_SHA256,

                "token_cache_sha256":
                    EXPECTED_TOKEN_CACHE_SHA256,

                "token_manifest_sha256":
                    EXPECTED_TOKEN_MANIFEST_SHA256,

                "audio_manifest_sha256":
                    EXPECTED_AUDIO_MANIFEST_SHA256,

                "graph_manifest_sha256":
                    EXPECTED_GRAPH_MANIFEST_SHA256,

                "final_threshold_lock_sha256":
                    EXPECTED_FINAL_THRESHOLDS_SHA256,

                "multiseed_manifest_sha256":
                    EXPECTED_MULTISEED_MANIFEST_SHA256,
            },

            "checkpoint_map": {
                str(seed): {
                    key: {
                        "path":
                            str(
                                value[
                                    "path"
                                ].relative_to(
                                    ROOT
                                )
                            ),

                        "sha256":
                            value[
                                "sha256"
                            ],
                    }

                    for key, value
                    in checkpoint_map[
                        seed
                    ].items()
                }

                for seed in FINAL_SEEDS
            },

            "per_run_metrics":
                run_records,

            "comparison_table":
                comparison_rows,

            "research_comparison":
                research_comparison,

            "output_hashes":
                output_hashes,

            "threshold_tuning_on_test":
                False,

            "training_after_test_access":
                False,

            "model_selection_after_test_access":
                False,

            "test_probability_cache_saved":
                True,

            "next_stage":
                (
                    "post-hoc analysis using frozen "
                    "saved TEST outputs; no retuning"
                ),
        }

        evaluation_path = (
            TEMP_DIR
            / "final_test_evaluation.json"
        )

        with evaluation_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                evaluation_artifact,
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )

            f.write("\n")

        evaluation_sha = (
            sha256_file(
                evaluation_path
            )
        )

        # =====================================================================
        # 18. COMMIT COMPLETE TEST DIRECTORY
        # =====================================================================

        require(
            not FINAL_DIR.exists(),
            (
                "final TEST directory "
                "still unused before commit"
            ),
        )

        os.replace(
            TEMP_DIR,
            FINAL_DIR,
        )

    except Exception as exc:

        banner(
            "SCRIPT 60 — FAILED AFTER TEST ACCESS"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "TEST has now been accessed."
        )

        print(
            "Do NOT rerun Script 60 "
            "without inspecting the retained "
            "temporary TEST directory first."
        )

        if TEMP_DIR.exists():

            print()
            print(
                "Retained:"
            )

            print(
                f"  "
                f"{TEMP_DIR.relative_to(ROOT)}"
            )

        return 1

    # =========================================================================
    # SUCCESS SUMMARY
    # =========================================================================

    banner(
        "SCRIPT 60 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "FINAL TEST evaluation complete."
    )

    print(
        "Threshold tuning on TEST: 0"
    )

    print(
        "Training after TEST access: 0"
    )

    print(
        "Model selection after TEST access: 0"
    )

    print()

    print(
        "Final comparison:"
    )

    print()

    print(
        display_frame.to_string(
            index=False
        )
    )

    print()

    print(
        "Primary research comparison "
        "(AUC-PR):"
    )

    print(
        f"  Early Fusion mean = "
        f"{fusion_ap:.8f}"
    )

    print(
        f"  BERT mean         = "
        f"{bert_ap:.8f}"
    )

    print(
        f"  Fusion - BERT     = "
        f"{fusion_ap - bert_ap:+.8f}"
    )

    print()

    print(
        f"  GAT seed42        = "
        f"{gat_ap:.8f}"
    )

    print(
        f"  Fusion - GAT      = "
        f"{fusion_ap - gat_ap:+.8f}"
    )

    print()

    print(
        "Final TEST artifacts:"
    )

    print(
        "  results/runs/final_eval/final_test/"
        "final_test_evaluation.json"
    )

    print(
        "  results/runs/final_eval/final_test/"
        "final_comparison_table.csv"
    )

    print(
        "  results/runs/final_eval/final_test/"
        "final_test_metrics_per_run.csv"
    )

    print(
        "  results/runs/final_eval/final_test/"
        "final_test_per_label_summary.csv"
    )

    print(
        "  results/runs/final_eval/final_test/"
        "final_test_predictions.npz"
    )

    print()

    print(
        "Final evaluation JSON SHA256:"
    )

    print(
        f"  {evaluation_sha}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  Statistical / error / fusion analysis "
        "using these SAVED final outputs only."
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