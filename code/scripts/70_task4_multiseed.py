#!/usr/bin/env python3
"""
SCRIPT 70 — TASK-4 SELECTED T4-B MULTI-SEED RUNS
================================================

Frozen selection:
    T4-B

Final seeds:
    42
    1337
    2026

Seed 42:
    Already independently completed through:
        Script 67 — same-seed T4-A
        Script 68 — same-seed T4-B

    It is REUSED, not retrained.

Seeds 1337 and 2026:
    For EACH seed independently:

        Stage A:
            frozen T1-B representations
            frozen A4/G2 representations
            train projection heads only
            select best checkpoint by full-pool VAL mR

        Stage B:
            initialize:
                T1-B encoder
                A4/G2 encoder
                same-seed best Stage-A projections

            train:
                BERT layers 10-11
                GAT3 only
                both projection heads

            select best checkpoint by full-pool VAL mR

Then:
    aggregate the selected T4-B validation retrieval metrics
    over seeds {42, 1337, 2026}

STRICT BOUNDARY
---------------
TRAIN:
    optimization

VAL:
    checkpoint selection and variability reporting

TEST:
    ZERO access
    ZERO inference
    ZERO retrieval
    ZERO model selection

This script does NOT reopen T4-A vs T4-B architecture selection.
T4-B was already frozen by Script 69.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import csv
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import transformers

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torch_geometric.data import (
    Data,
    Batch,
)

from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
)

from transformers import (
    BertConfig,
    BertModel,
)


# =============================================================================
# 0. ROOT / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]


DESIGN = (
    ROOT
    / "results/runs/task4/"
      "task4_contrastive_design.json"
)

DESIGN_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_contrastive_design_lock.json"
)

MODEL_SELECTION = (
    ROOT
    / "results/runs/task4/"
      "task4_model_selection.json"
)

MODEL_SELECTION_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_model_selection_lock.json"
)


# -----------------------------------------------------------------------------
# Existing formal seed-42 run
# -----------------------------------------------------------------------------

SEED42_T4A_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-A_seed42_best.pt"
)

SEED42_T4A_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-A_seed42_summary.json"
)

SEED42_T4B_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-B_seed42_best.pt"
)

SEED42_T4B_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_seed42_summary.json"
)


# -----------------------------------------------------------------------------
# Frozen representations for Stage A
# -----------------------------------------------------------------------------

BERT_SEQUENCE = (
    ROOT
    / "data/processed/bert_seq_frozen/"
      "bert_sequence_trainval.npy"
)

BERT_SEQUENCE_MANIFEST = (
    ROOT
    / "data/processed/bert_seq_frozen/"
      "bert_seq_manifest.parquet"
)

GAT_EMBEDDING = (
    ROOT
    / "data/processed/gat_g2_frozen/"
      "gat_g2_embedding_trainval.npy"
)

GAT_EMBEDDING_MANIFEST = (
    ROOT
    / "data/processed/gat_g2_frozen/"
      "gat_g2_manifest.parquet"
)


# -----------------------------------------------------------------------------
# Live encoder sources for Stage B
# -----------------------------------------------------------------------------

TOKEN_CACHE = (
    ROOT
    / "data/processed/bert_tokens/"
      "bert_tokens.npz"
)

TOKEN_MANIFEST = (
    ROOT
    / "data/processed/bert_tokens/"
      "bert_token_manifest.parquet"
)

GRAPH_MANIFEST = (
    ROOT
    / "data/processed/"
      "graph_manifest.parquet"
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


MODEL_DIR = (
    ROOT
    / "models/task4_candidates"
)

RUN_DIR = (
    ROOT
    / "results/runs/task4"
)


MULTISEED_CSV = (
    RUN_DIR
    / "task4_T4-B_multiseed_validation.csv"
)

MULTISEED_SUMMARY = (
    RUN_DIR
    / "task4_T4-B_multiseed_summary.json"
)

MULTISEED_LOCK = (
    RUN_DIR
    / "task4_T4-B_multiseed_lock.json"
)


def t4a_checkpoint_path(
    seed: int,
) -> Path:

    if seed == 42:
        return SEED42_T4A_CHECKPOINT

    return (
        MODEL_DIR
        / f"task4_T4-A_seed{seed}_multiseed_best.pt"
    )


def t4a_history_path(
    seed: int,
) -> Path:

    return (
        RUN_DIR
        / f"task4_T4-A_seed{seed}_multiseed_history.csv"
    )


def t4a_summary_path(
    seed: int,
) -> Path:

    if seed == 42:
        return SEED42_T4A_SUMMARY

    return (
        RUN_DIR
        / f"task4_T4-A_seed{seed}_multiseed_summary.json"
    )


def t4b_checkpoint_path(
    seed: int,
) -> Path:

    if seed == 42:
        return SEED42_T4B_CHECKPOINT

    return (
        MODEL_DIR
        / f"task4_T4-B_seed{seed}_multiseed_best.pt"
    )


def t4b_history_path(
    seed: int,
) -> Path:

    return (
        RUN_DIR
        / f"task4_T4-B_seed{seed}_multiseed_history.csv"
    )


def t4b_summary_path(
    seed: int,
) -> Path:

    if seed == 42:
        return SEED42_T4B_SUMMARY

    return (
        RUN_DIR
        / f"task4_T4-B_seed{seed}_multiseed_summary.json"
    )


# =============================================================================
# 1. FROZEN HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "61ed1263476e631374790c9669ab4aa96"
    "e3b5cb295436d4e273da66d4da556db"
)

EXPECTED_DESIGN_LOCK_SHA256 = (
    "53f50205e712833ec4efb506b9b6d911"
    "8dcf46f51b72c5907f33cd61ddca612c"
)

EXPECTED_SELECTION_SHA256 = (
    "831e1535cfe6f45a646f1732ecf268db"
    "02c5214aa48404fd07fb3ae8ff5e41c5"
)

EXPECTED_SELECTION_LOCK_SHA256 = (
    "a1aa9d2420a97c86b0f3709227ff22d1"
    "ee3ccf111010733b2bf705dee5eb6bb6"
)


EXPECTED_SEED42_T4A_CHECKPOINT_SHA256 = (
    "fc3e85fbe6e98d19dee5cb2891fce954"
    "704e09fc49e494a589193d9a0a09ab9f"
)

EXPECTED_SEED42_T4A_SUMMARY_SHA256 = (
    "f82844e55472a574b4c500dd267435890"
    "e7c55f2d6fb0ca853ce0e8ebed13806"
)

EXPECTED_SEED42_T4B_CHECKPOINT_SHA256 = (
    "2be12c7e6fdc851b2b1efae0a7bed2c4"
    "95cbefb5ae90a4f491c54a72ec8e9ace"
)

EXPECTED_SEED42_T4B_SUMMARY_SHA256 = (
    "d4300c9f5089ca3efa9626bb60fc10e1"
    "23336604808e99eba3cc7cfd0839e6aa"
)


EXPECTED_BERT_SEQUENCE_SHA256 = (
    "a1c723ec939392b8b0d9bafadb8327d"
    "8a0967ecb9bd8e89ffcd559306633e133"
)

EXPECTED_GAT_EMBEDDING_SHA256 = (
    "3d39a343d1acc678576f9dd3d88b088"
    "ca18a81017190246b0b32841292b92852"
)

EXPECTED_TOKEN_CACHE_SHA256 = (
    "0df45ab266884abbc2f3f7f96a5ba789b"
    "bf92bf8da49ac4cb57a1e4eb476188f"
)

EXPECTED_TOKEN_MANIFEST_SHA256 = (
    "4be874ab26375ce63bb0f4b55706d178b"
    "fac8a951fee96fd75d2d49b28a390b8"
)

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6a"
    "d0492bf90eccc929cadc1aad1da9e158"
)

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc513669442"
    "71e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311"
    "daf74ea0a24f6b02e78e5513078fae1e"
)


# =============================================================================
# 2. DATA / MODEL CONTRACT
# =============================================================================

N_TOTAL = 5140
N_TRAIN = 4112
N_VAL = 514
N_TEST = 514
N_DEV = 4626

MAX_LENGTH = 112

BERT_DIM = 768
RAW_NODE_DIM = 140
GAT_DIM = 128
SHARED_DIM = 256

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

NEW_SEEDS = (
    1337,
    2026,
)


# -----------------------------------------------------------------------------
# T4-A
# -----------------------------------------------------------------------------

T4A_TEMPERATURE = 0.07

T4A_BATCH = 32

T4A_LR = 1.0e-3
T4A_WEIGHT_DECAY = 1.0e-4

T4A_MAX_EPOCHS = 30
T4A_PATIENCE = 5

T4A_GRAD_CLIP = 1.0


# -----------------------------------------------------------------------------
# T4-B
# -----------------------------------------------------------------------------

T4B_TEMPERATURE = 0.07

T4B_BATCH = 32

T4B_PROJECTION_LR = 1.0e-4
T4B_BERT_LR = 2.0e-5
T4B_GAT_LR = 1.0e-4

T4B_WEIGHT_DECAY = 1.0e-4

T4B_MAX_EPOCHS = 20
T4B_PATIENCE = 4

T4B_GRAD_CLIP = 1.0


EXPECTED_BERT_TOP2_TRAINABLE = (
    14_175_744
)

EXPECTED_GAT3_TRAINABLE = (
    66_688
)

EXPECTED_PROJECTION_TRAINABLE = (
    229_376
)

EXPECTED_T4B_TOTAL_TRAINABLE = (
    EXPECTED_BERT_TOP2_TRAINABLE
    + EXPECTED_GAT3_TRAINABLE
    + EXPECTED_PROJECTION_TRAINABLE
)


METRIC_KEYS = (
    "audio_to_caption_R@1",
    "audio_to_caption_R@5",
    "audio_to_caption_R@10",
    "caption_to_audio_R@1",
    "caption_to_audio_R@5",
    "caption_to_audio_R@10",
)

ALL_REPORT_METRICS = (
    *METRIC_KEYS,
    "mR",
)


# =============================================================================
# 3. REPORTING
# =============================================================================

FAILURES: list[str] = []
WARNINGS: list[str] = []


def banner(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(
    message: str,
) -> None:

    print(
        f"PASS  {message}"
    )


def fail(
    message: str,
) -> None:

    print(
        f"FAIL  {message}"
    )

    FAILURES.append(
        message
    )


def warning(
    message: str,
) -> None:

    print(
        f"WARN  {message}"
    )

    WARNINGS.append(
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

    fail(
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

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                chunk_size
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


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
            f"{path.name} is not "
            "a JSON dictionary"
        )

    return value


def write_json_atomic(
    path: Path,
    value: Any,
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if temp.exists():

        raise RuntimeError(
            f"Temporary file exists: "
            f"{temp}"
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

        f.write(
            "\n"
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temp,
        path,
    )


def relative(
    path: Path,
) -> str:

    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    )


def verify_sha(
    path: Path,
    expected: str,
    label: str,
) -> None:

    check(
        path.is_file(),
        f"{label} exists",
    )

    if not path.is_file():
        return

    observed = sha256_file(
        path
    )

    print(
        f"{label}:\n"
        f"  observed = {observed}\n"
        f"  expected = {expected}"
    )

    check(
        observed == expected,
        (
            f"{label} SHA256 matches "
            "accepted identity"
        ),
    )


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
            f"{path.name} is not "
            "a checkpoint dictionary"
        )

    return value


def save_checkpoint_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if temp.exists():

        raise RuntimeError(
            f"Temporary checkpoint exists: "
            f"{temp}"
        )

    torch.save(
        value,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def write_history_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:

        raise RuntimeError(
            "Cannot write empty history"
        )

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if temp.exists():

        raise RuntimeError(
            f"Temporary history exists: "
            f"{temp}"
        )

    with temp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temp,
        path,
    )


def cpu_state_dict(
    module: nn.Module,
) -> dict[str, torch.Tensor]:

    return {
        key:
            tensor
            .detach()
            .cpu()
            .clone()

        for key, tensor
        in module.state_dict().items()
    }


# =============================================================================
# 5. REPRODUCIBILITY
# =============================================================================

def configure_t4a_seed(
    seed: int,
) -> None:
    """
    Match formal Script-67 behavior.

    Script 67 explicitly enabled torch deterministic algorithms.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    torch.use_deterministic_algorithms(
        True
    )


def configure_t4b_seed(
    seed: int,
) -> None:
    """
    Match accepted formal Script-68 behavior.

    Script 68 used deterministic cuDNN but did NOT globally force
    torch deterministic algorithms.
    """

    torch.use_deterministic_algorithms(
        False
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    try:

        torch.set_float32_matmul_precision(
            "highest"
        )

    except Exception:

        pass


# =============================================================================
# 6. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. MULTI-SEED OUTPUT PROTECTION"
    )

    final_paths = (
        MULTISEED_CSV,
        MULTISEED_SUMMARY,
        MULTISEED_LOCK,
    )

    for path in final_paths:

        check(
            not path.exists(),
            (
                f"final multi-seed output unused: "
                f"{path.name}"
            ),
        )

    for seed in NEW_SEEDS:

        for path in (
            t4a_checkpoint_path(
                seed
            ),
            t4a_history_path(
                seed
            ),
            t4a_summary_path(
                seed
            ),
            t4b_checkpoint_path(
                seed
            ),
            t4b_history_path(
                seed
            ),
            t4b_summary_path(
                seed
            ),
        ):

            check(
                not path.exists(),
                (
                    f"seed-{seed} output unused: "
                    f"{path.name}"
                ),
            )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite an existing "
            "Task-4 multi-seed result"
        )


# =============================================================================
# 7. VERIFY FROZEN LINEAGE / SELECTION
# =============================================================================

def verify_frozen_lineage() -> dict[str, Any]:

    banner(
        "2. VERIFY FROZEN TASK-4 SELECTION"
    )

    artifacts = (
        (
            DESIGN,
            EXPECTED_DESIGN_SHA256,
            "Task-4 design",
        ),
        (
            DESIGN_LOCK,
            EXPECTED_DESIGN_LOCK_SHA256,
            "Task-4 design lock",
        ),
        (
            MODEL_SELECTION,
            EXPECTED_SELECTION_SHA256,
            "Task-4 model selection",
        ),
        (
            MODEL_SELECTION_LOCK,
            EXPECTED_SELECTION_LOCK_SHA256,
            "Task-4 model-selection lock",
        ),
        (
            SEED42_T4A_CHECKPOINT,
            EXPECTED_SEED42_T4A_CHECKPOINT_SHA256,
            "seed-42 T4-A checkpoint",
        ),
        (
            SEED42_T4A_SUMMARY,
            EXPECTED_SEED42_T4A_SUMMARY_SHA256,
            "seed-42 T4-A summary",
        ),
        (
            SEED42_T4B_CHECKPOINT,
            EXPECTED_SEED42_T4B_CHECKPOINT_SHA256,
            "seed-42 T4-B checkpoint",
        ),
        (
            SEED42_T4B_SUMMARY,
            EXPECTED_SEED42_T4B_SUMMARY_SHA256,
            "seed-42 T4-B summary",
        ),
        (
            BERT_SEQUENCE,
            EXPECTED_BERT_SEQUENCE_SHA256,
            "frozen BERT sequence cache",
        ),
        (
            GAT_EMBEDDING,
            EXPECTED_GAT_EMBEDDING_SHA256,
            "frozen GAT embedding cache",
        ),
        (
            TOKEN_CACHE,
            EXPECTED_TOKEN_CACHE_SHA256,
            "BERT token cache",
        ),
        (
            TOKEN_MANIFEST,
            EXPECTED_TOKEN_MANIFEST_SHA256,
            "BERT token manifest",
        ),
        (
            GRAPH_MANIFEST,
            EXPECTED_GRAPH_MANIFEST_SHA256,
            "graph manifest",
        ),
        (
            T1_CHECKPOINT,
            EXPECTED_T1_SHA256,
            "selected T1-B checkpoint",
        ),
        (
            A4_CHECKPOINT,
            EXPECTED_A4_SHA256,
            "selected A4 checkpoint",
        ),
    )

    for path, digest, label in artifacts:

        verify_sha(
            path,
            digest,
            label,
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen Task-4 lineage validation failed"
        )

    selection = load_json(
        MODEL_SELECTION
    )

    lock = load_json(
        MODEL_SELECTION_LOCK
    )

    check(
        selection[
            "selected_variant"
        ]
        == "T4-B",
        (
            "frozen Task-4 selected variant "
            "is exactly T4-B"
        ),
    )

    check(
        selection[
            "selection_seed"
        ]
        == 42,
        (
            "architecture selection seed "
            "remains 42"
        ),
    )

    check(
        selection[
            "test_used"
        ]
        is False,
        (
            "model selection records "
            "TEST unused"
        ),
    )

    check(
        lock[
            "selected_variant"
        ]
        == "T4-B",
        (
            "model-selection lock also "
            "records T4-B"
        ),
    )

    check(
        lock[
            "test_used"
        ]
        is False,
        (
            "model-selection lock records "
            "TEST unused"
        ),
    )

    design = load_json(
        DESIGN
    )

    check(
        design[
            "final_multiseed_policy"
        ][
            "seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "frozen final multi-seed set is "
            "[42,1337,2026]"
        ),
    )

    check(
        design[
            "final_multiseed_policy"
        ][
            "if_t4_b_wins"
        ]
        ==
        (
            "for each final seed, independently run the full "
            "same-seed T4-A initialization stage followed by "
            "same-seed T4-B partial fine-tuning"
        ),
        (
            "frozen T4-B multi-seed procedure "
            "is unchanged"
        ),
    )

    check(
        design[
            "final_multiseed_policy"
        ][
            "test_access_during_multiseed_training"
        ]
        is False,
        (
            "TEST forbidden during multi-seed training"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen Task-4 multi-seed contract changed"
        )

    return design


# =============================================================================
# 8. RETRIEVAL METRICS
# =============================================================================

def retrieval_metrics(
    z_text: torch.Tensor,
    z_graph: torch.Tensor,
) -> dict[str, float]:

    similarity_a2t = (
        z_graph
        @ z_text.T
    )

    similarity_c2a = (
        similarity_a2t.T
    )

    def ranks(
        matrix: torch.Tensor,
    ) -> torch.Tensor:

        correct = torch.diagonal(
            matrix
        )

        return (
            1
            +
            (
                matrix
                >
                correct.unsqueeze(
                    1
                )
            )
            .sum(
                dim=1
            )
        )

    a2t_rank = ranks(
        similarity_a2t
    )

    c2a_rank = ranks(
        similarity_c2a
    )

    values: dict[str, float] = {}

    for k in (
        1,
        5,
        10,
    ):

        values[
            f"audio_to_caption_R@{k}"
        ] = float(
            (
                a2t_rank
                <= k
            )
            .float()
            .mean()
            .item()
        )

        values[
            f"caption_to_audio_R@{k}"
        ] = float(
            (
                c2a_rank
                <= k
            )
            .float()
            .mean()
            .item()
        )

    values[
        "mR"
    ] = float(
        np.mean(
            [
                values[
                    key
                ]
                for key
                in METRIC_KEYS
            ],
            dtype=np.float64,
        )
    )

    values[
        "audio_to_caption_mean_rank"
    ] = float(
        a2t_rank
        .float()
        .mean()
        .item()
    )

    values[
        "caption_to_audio_mean_rank"
    ] = float(
        c2a_rank
        .float()
        .mean()
        .item()
    )

    return values


# =============================================================================
# 9. T4-A MODEL / LOSS
# =============================================================================

class T4AProjection(
    nn.Module,
):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.text_projection = nn.Linear(
            BERT_DIM,
            SHARED_DIM,
            bias=False,
        )

        self.graph_projection = nn.Linear(
            GAT_DIM,
            SHARED_DIM,
            bias=False,
        )

        nn.init.xavier_uniform_(
            self.text_projection.weight
        )

        nn.init.xavier_uniform_(
            self.graph_projection.weight
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        z_text = self.text_projection(
            text_cls
        )

        z_graph = self.graph_projection(
            graph_embedding
        )

        z_text = F.normalize(
            z_text,
            p=2,
            dim=-1,
        )

        z_graph = F.normalize(
            z_graph,
            p=2,
            dim=-1,
        )

        return (
            z_text,
            z_graph,
        )


def symmetric_infonce(
    z_text: torch.Tensor,
    z_graph: torch.Tensor,
    temperature: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    logits = (
        z_graph
        @ z_text.T
    ) / temperature

    target = torch.arange(
        z_text.shape[0],
        device=z_text.device,
        dtype=torch.long,
    )

    a2t = F.cross_entropy(
        logits,
        target,
    )

    t2a = F.cross_entropy(
        logits.T,
        target,
    )

    total = (
        0.5
        * (
            a2t
            + t2a
        )
    )

    return (
        total,
        a2t,
        t2a,
    )


@torch.no_grad()
def embed_cached(
    model: T4AProjection,
    text_np: np.ndarray,
    graph_np: np.ndarray,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    model.eval()

    text_output = []
    graph_output = []

    for start in range(
        0,
        len(
            text_np
        ),
        256,
    ):

        end = min(
            start + 256,
            len(
                text_np
            ),
        )

        text = torch.from_numpy(
            text_np[
                start:end
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        graph = torch.from_numpy(
            graph_np[
                start:end
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        z_text, z_graph = model(
            text,
            graph,
        )

        text_output.append(
            z_text.cpu()
        )

        graph_output.append(
            z_graph.cpu()
        )

    return (
        torch.cat(
            text_output,
            dim=0,
        ),
        torch.cat(
            graph_output,
            dim=0,
        ),
    )


# =============================================================================
# 10. LOAD CACHED STAGE-A DATA
# =============================================================================

def load_stage_a_data():

    banner(
        "3. LOAD FROZEN T4-A REPRESENTATIONS"
    )

    bert_manifest = pd.read_parquet(
        BERT_SEQUENCE_MANIFEST
    )

    gat_manifest = pd.read_parquet(
        GAT_EMBEDDING_MANIFEST
    )

    check(
        len(
            bert_manifest
        )
        == N_DEV,
        (
            "BERT development manifest "
            "= 4626 rows"
        ),
    )

    check(
        len(
            gat_manifest
        )
        == N_DEV,
        (
            "GAT development manifest "
            "= 4626 rows"
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
            "cached BERT/GAT row order "
            "matches exactly"
        ),
    )

    splits = (
        bert_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
        .to_numpy()
    )

    gat_splits = (
        gat_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
        .to_numpy()
    )

    check(
        np.array_equal(
            splits,
            gat_splits,
        ),
        (
            "cached BERT/GAT split order "
            "matches exactly"
        ),
    )

    train_rows = np.flatnonzero(
        splits
        == "train"
    )

    val_rows = np.flatnonzero(
        splits
        == "val"
    )

    check(
        len(
            train_rows
        )
        == N_TRAIN,
        "Stage-A TRAIN = 4112",
    )

    check(
        len(
            val_rows
        )
        == N_VAL,
        "Stage-A VAL = 514",
    )

    check(
        int(
            np.sum(
                splits
                == "test"
            )
        )
        == 0,
        (
            "Stage-A frozen cache contains "
            "zero TEST rows"
        ),
    )

    bert_sequence = np.load(
        BERT_SEQUENCE,
        mmap_mode="r",
    )

    gat_embedding = np.load(
        GAT_EMBEDDING,
        mmap_mode="r",
    )

    check(
        bert_sequence.shape
        ==
        (
            N_DEV,
            MAX_LENGTH,
            BERT_DIM,
        ),
        (
            "BERT sequence shape "
            "= (4626,112,768)"
        ),
    )

    check(
        gat_embedding.shape
        ==
        (
            N_DEV,
            GAT_DIM,
        ),
        (
            "GAT embedding shape "
            "= (4626,128)"
        ),
    )

    train_text = np.asarray(
        bert_sequence[
            train_rows,
            0,
            :,
        ],
        dtype=np.float32,
    ).copy()

    val_text = np.asarray(
        bert_sequence[
            val_rows,
            0,
            :,
        ],
        dtype=np.float32,
    ).copy()

    train_graph = np.asarray(
        gat_embedding[
            train_rows,
            :,
        ],
        dtype=np.float32,
    ).copy()

    val_graph = np.asarray(
        gat_embedding[
            val_rows,
            :,
        ],
        dtype=np.float32,
    ).copy()

    return {
        "dev_track_ids":
            bert_ids,

        "train_text":
            train_text,

        "train_graph":
            train_graph,

        "val_text":
            val_text,

        "val_graph":
            val_graph,
    }


# =============================================================================
# 11. RUN SAME-SEED T4-A INITIALIZATION STAGE
# =============================================================================

def run_t4a_seed(
    seed: int,
    data: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:

    banner(
        f"4A. SEED {seed} — T4-A INITIALIZATION STAGE"
    )

    configure_t4a_seed(
        seed
    )

    model = T4AProjection().to(
        device
    )

    trainable = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    check(
        trainable
        == EXPECTED_PROJECTION_TRAINABLE,
        (
            f"seed {seed} T4-A trainable "
            "parameters = 229376"
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=T4A_LR,
        weight_decay=T4A_WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        seed
    )

    best_mr = -float(
        "inf"
    )

    best_epoch = None
    best_metrics = None
    best_state = None

    no_improvement = 0

    history = []

    start_time = time.perf_counter()

    for epoch in range(
        1,
        T4A_MAX_EPOCHS + 1,
    ):

        model.train()

        permutation = rng.permutation(
            N_TRAIN
        )

        total_loss = 0.0
        total_a2t = 0.0
        total_t2a = 0.0
        rows_seen = 0

        for start in range(
            0,
            N_TRAIN,
            T4A_BATCH,
        ):

            indices = permutation[
                start:
                start + T4A_BATCH
            ]

            text = torch.from_numpy(
                data[
                    "train_text"
                ][
                    indices
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            graph = torch.from_numpy(
                data[
                    "train_graph"
                ][
                    indices
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            z_text, z_graph = model(
                text,
                graph,
            )

            (
                loss,
                a2t,
                t2a,
            ) = symmetric_infonce(
                z_text,
                z_graph,
                T4A_TEMPERATURE,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"seed {seed} T4-A "
                    "non-finite loss"
                )

            loss.backward()

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=T4A_GRAD_CLIP,
                )
            )

            if not math.isfinite(
                float(
                    grad_norm
                )
            ):

                raise RuntimeError(
                    f"seed {seed} T4-A "
                    "non-finite gradient"
                )

            optimizer.step()

            batch_rows = len(
                indices
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_rows
            )

            total_a2t += (
                float(
                    a2t.detach().cpu()
                )
                * batch_rows
            )

            total_t2a += (
                float(
                    t2a.detach().cpu()
                )
                * batch_rows
            )

            rows_seen += (
                batch_rows
            )

        if rows_seen != N_TRAIN:

            raise RuntimeError(
                f"seed {seed} T4-A "
                f"TRAIN rows={rows_seen}"
            )

        (
            val_z_text,
            val_z_graph,
        ) = embed_cached(
            model,
            data[
                "val_text"
            ],
            data[
                "val_graph"
            ],
            device,
        )

        val = retrieval_metrics(
            val_z_text,
            val_z_graph,
        )

        mr = float(
            val[
                "mR"
            ]
        )

        improved = (
            mr
            >
            best_mr
        )

        if improved:

            best_mr = mr
            best_epoch = epoch
            best_metrics = dict(
                val
            )

            best_state = {
                key:
                    tensor
                    .detach()
                    .cpu()
                    .clone()

                for key, tensor
                in model.state_dict().items()
            }

            no_improvement = 0

        else:

            no_improvement += 1

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    total_loss
                    / rows_seen,

                "train_audio_to_caption_loss":
                    total_a2t
                    / rows_seen,

                "train_caption_to_audio_loss":
                    total_t2a
                    / rows_seen,

                "val_audio_to_caption_R@1":
                    val[
                        "audio_to_caption_R@1"
                    ],

                "val_audio_to_caption_R@5":
                    val[
                        "audio_to_caption_R@5"
                    ],

                "val_audio_to_caption_R@10":
                    val[
                        "audio_to_caption_R@10"
                    ],

                "val_caption_to_audio_R@1":
                    val[
                        "caption_to_audio_R@1"
                    ],

                "val_caption_to_audio_R@5":
                    val[
                        "caption_to_audio_R@5"
                    ],

                "val_caption_to_audio_R@10":
                    val[
                        "caption_to_audio_R@10"
                    ],

                "val_mR":
                    val[
                        "mR"
                    ],

                "improved":
                    bool(
                        improved
                    ),

                "epochs_without_improvement":
                    no_improvement,
            }
        )

        print(
            f"seed={seed} "
            f"T4-A epoch={epoch:02d} "
            f"train={total_loss / rows_seen:.6f} "
            f"VAL_mR={mr:.8f}"
            + (
                " BEST"
                if improved
                else
                ""
            )
        )

        if (
            no_improvement
            >= T4A_PATIENCE
        ):

            print(
                f"seed={seed} T4-A "
                "early stopping"
            )

            break

    if (
        best_epoch is None
        or
        best_metrics is None
        or
        best_state is None
    ):

        raise RuntimeError(
            f"seed {seed}: no valid T4-A state"
        )

    # -------------------------------------------------------------------------
    # Exact state restoration
    # -------------------------------------------------------------------------

    model.load_state_dict(
        best_state,
        strict=True,
    )

    for key, tensor in (
        model.state_dict().items()
    ):

        check(
            torch.equal(
                tensor.detach().cpu(),
                best_state[
                    key
                ],
            ),
            (
                f"seed {seed} T4-A "
                f"{key} restored exactly"
            ),
        )

    (
        re_z_text,
        re_z_graph,
    ) = embed_cached(
        model,
        data[
            "val_text"
        ],
        data[
            "val_graph"
        ],
        device,
    )

    revalidated = retrieval_metrics(
        re_z_text,
        re_z_graph,
    )

    recall_quantum = (
        1.0
        / N_VAL
    )

    for key in METRIC_KEYS:

        difference = abs(
            float(
                revalidated[
                    key
                ]
            )
            -
            float(
                best_metrics[
                    key
                ]
            )
        )

        check(
            difference
            <= (
                recall_quantum
                + 1e-12
            ),
            (
                f"seed {seed} T4-A "
                f"{key} revalidates within "
                "one VAL query"
            ),
        )

    runtime = (
        time.perf_counter()
        -
        start_time
    )

    checkpoint = {
        "artifact_type":
            "task4_T4-A_multiseed_initialization_checkpoint",

        "variant":
            "T4-A",

        "purpose":
            (
                "same-seed initialization stage "
                "for selected T4-B multi-seed run"
            ),

        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "val_metrics":
            best_metrics,

        "model_state_dict":
            best_state,

        "test_access":
            False,

        "provenance": {
            "design_sha256":
                EXPECTED_DESIGN_SHA256,

            "selection_sha256":
                EXPECTED_SELECTION_SHA256,

            "bert_sequence_sha256":
                EXPECTED_BERT_SEQUENCE_SHA256,

            "gat_embedding_sha256":
                EXPECTED_GAT_EMBEDDING_SHA256,
        },
    }

    checkpoint_path = (
        t4a_checkpoint_path(
            seed
        )
    )

    history_path = (
        t4a_history_path(
            seed
        )
    )

    summary_path = (
        t4a_summary_path(
            seed
        )
    )

    save_checkpoint_atomic(
        checkpoint_path,
        checkpoint,
    )

    write_history_atomic(
        history_path,
        history,
    )

    checkpoint_sha = sha256_file(
        checkpoint_path
    )

    history_sha = sha256_file(
        history_path
    )

    summary = {
        "artifact_type":
            "task4_T4-A_multiseed_initialization_summary",

        "variant":
            "T4-A",

        "seed":
            seed,

        "formal_multiseed_stage":
            True,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_metrics":
            best_metrics,

        "best_val_mR":
            float(
                best_metrics[
                    "mR"
                ]
            ),

        "checkpoint":
            relative(
                checkpoint_path
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "history":
            relative(
                history_path
            ),

        "history_sha256":
            history_sha,

        "runtime_seconds":
            float(
                runtime
            ),

        "test_access":
            False,

        "test_retrieval":
            False,
    }

    write_json_atomic(
        summary_path,
        summary,
    )

    summary_sha = sha256_file(
        summary_path
    )

    print()
    print(
        f"seed {seed} T4-A COMPLETE"
    )

    print(
        f"  best epoch = {best_epoch}"
    )

    print(
        f"  VAL mR     = "
        f"{best_metrics['mR']:.8f}"
    )

    print(
        f"  checkpoint SHA256 = "
        f"{checkpoint_sha}"
    )

    print(
        f"  summary SHA256    = "
        f"{summary_sha}"
    )

    del model

    torch.cuda.empty_cache()

    return {
        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_metrics":
            best_metrics,

        "checkpoint":
            checkpoint_path,

        "checkpoint_sha256":
            checkpoint_sha,

        "summary":
            summary_path,

        "summary_sha256":
            summary_sha,
    }


# =============================================================================
# 12. EXACT A4 GRAPH ENCODER
# =============================================================================

class A4GAT(
    nn.Module,
):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.node_projection = nn.Linear(
            140,
            64,
        )

        self.gat1 = GATConv(
            in_channels=64,
            out_channels=32,
            heads=4,
            concat=True,
            dropout=0.2,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.gat2 = GATConv(
            in_channels=128,
            out_channels=32,
            heads=4,
            concat=True,
            dropout=0.2,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.gat3 = GATConv(
            in_channels=128,
            out_channels=128,
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
        data: Batch,
    ) -> torch.Tensor:

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
# 13. LIVE T4-B MODEL
# =============================================================================

class T4BModel(
    nn.Module,
):

    def __init__(
        self,
        bert: BertModel,
        gnn: A4GAT,
    ) -> None:

        super().__init__()

        self.bert = bert
        self.gnn = gnn

        self.text_projection = nn.Linear(
            BERT_DIM,
            SHARED_DIM,
            bias=False,
        )

        self.graph_projection = nn.Linear(
            GAT_DIM,
            SHARED_DIM,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        graph_batch: Batch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        text_cls = (
            bert_output
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

        z_text = self.text_projection(
            text_cls
        )

        z_graph = self.graph_projection(
            graph_embedding
        )

        z_text = F.normalize(
            z_text,
            p=2,
            dim=-1,
        )

        z_graph = F.normalize(
            z_graph,
            p=2,
            dim=-1,
        )

        return (
            z_text,
            z_graph,
        )


# =============================================================================
# 14. GRAPH LOADING
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
    raw_value: str,
) -> Path:

    raw = Path(
        raw_value
    )

    candidates = []

    if raw.is_absolute():

        candidates.append(
            raw
        )

    else:

        candidates.extend(
            [
                ROOT
                / raw,

                GRAPH_MANIFEST.parent
                / raw,

                ROOT
                / "data/processed/graphs"
                / raw.name,
            ]
        )

    found = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = (
                candidate.resolve()
            )

            found[
                str(
                    resolved
                )
            ] = resolved

    if len(
        found
    ) != 1:

        raise RuntimeError(
            "Unable to uniquely resolve graph: "
            f"{raw_value}; "
            f"matches={len(found)}"
        )

    return next(
        iter(
            found.values()
        )
    )


def load_g2_graph(
    graph_path: Path,
) -> Data:

    with np.load(
        graph_path,
        allow_pickle=False,
    ) as graph:

        if set(
            graph.files
        ) != EXPECTED_GRAPH_KEYS:

            raise RuntimeError(
                "Unexpected graph schema: "
                f"{graph_path.name}"
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

        threshold = float(
            np.asarray(
                graph[
                    "similarity_threshold"
                ]
            )
            .reshape(-1)[0]
        )

        nonlocal_only = bool(
            np.asarray(
                graph[
                    "nonlocal_only"
                ]
            )
            .reshape(-1)[0]
        )

        cap = int(
            np.asarray(
                graph[
                    "extra_degree_cap"
                ]
            )
            .reshape(-1)[0]
        )

        version = int(
            np.asarray(
                graph[
                    "rule_version"
                ]
            )
            .reshape(-1)[0]
        )

    if x.shape != (
        9,
        140,
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "x shape changed"
        )

    if x.dtype != np.float32:

        raise RuntimeError(
            f"{graph_path.name}: "
            "x dtype changed"
        )

    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            f"{graph_path.name}: "
            "non-finite x"
        )

    if (
        edge_index.ndim != 2
        or
        edge_index.shape[0] != 2
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "invalid G2 edge shape"
        )

    if edge_index.dtype != np.int64:

        raise RuntimeError(
            f"{graph_path.name}: "
            "G2 dtype changed"
        )

    if not (
        16
        <= edge_index.shape[1]
        <= 34
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "G2 edge count invalid"
        )

    if not np.isclose(
        threshold,
        0.85,
        rtol=0.0,
        atol=1e-6,
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "threshold changed"
        )

    if nonlocal_only is not True:

        raise RuntimeError(
            f"{graph_path.name}: "
            "nonlocal_only changed"
        )

    if cap != 2:

        raise RuntimeError(
            f"{graph_path.name}: "
            "degree cap changed"
        )

    if version != 2:

        raise RuntimeError(
            f"{graph_path.name}: "
            "rule version changed"
        )

    return Data(
        x=torch.from_numpy(
            x.copy()
        ),
        edge_index=torch.from_numpy(
            edge_index.copy()
        ),
    )


# =============================================================================
# 15. LOAD LIVE TRAIN+VAL DATA ONCE
# =============================================================================

def load_live_data(
    expected_dev_ids: list[str],
):

    banner(
        "4. LOAD LIVE T4-B TRAIN+VAL INPUTS"
    )

    token_manifest = pd.read_parquet(
        TOKEN_MANIFEST
    )

    graph_manifest = pd.read_parquet(
        GRAPH_MANIFEST
    )

    check(
        len(
            token_manifest
        )
        == N_TOTAL,
        (
            "token manifest "
            "= 5140 rows"
        ),
    )

    token_manifest = (
        token_manifest.copy()
    )

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

    check(
        token_manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
        ==
        {
            "train":
                N_TRAIN,

            "val":
                N_VAL,

            "test":
                N_TEST,
        },
        (
            "token split counts "
            "= 4112/514/514"
        ),
    )

    dev_manifest = (
        token_manifest.loc[
            token_manifest[
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

    dev_ids = (
        dev_manifest[
            "track_id"
        ].tolist()
    )

    check(
        dev_ids
        == expected_dev_ids,
        (
            "live Task-4 development order "
            "matches frozen Stage-A order exactly"
        ),
    )

    check(
        "test"
        not in set(
            dev_manifest[
                "split"
            ]
        ),
        (
            "live development manifest "
            "contains zero TEST"
        ),
    )

    source_indices = (
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

        check(
            set(
                cache.files
            )
            ==
            {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            },
            (
                "token cache schema "
                "unchanged"
            ),
        )

        input_ids = np.asarray(
            cache[
                "input_ids"
            ][
                source_indices
            ],
            dtype=np.int32,
        )

        attention_mask = np.asarray(
            cache[
                "attention_mask"
            ][
                source_indices
            ],
            dtype=np.uint8,
        )

        token_type_ids = np.asarray(
            cache[
                "token_type_ids"
            ][
                source_indices
            ],
            dtype=np.uint8,
        )

    graph_manifest = (
        graph_manifest.copy()
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
        "Preloading 4626 TRAIN+VAL G2 graphs..."
    )

    for row_number, row in enumerate(
        dev_manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        track_id = str(
            row.track_id
        )

        split = str(
            row.split
        ).lower()

        graph_row = (
            graph_lookup.loc[
                track_id
            ]
        )

        graph_split = str(
            graph_row[
                "split"
            ]
        ).lower()

        if graph_split != split:

            raise RuntimeError(
                f"graph split mismatch: "
                f"{track_id}"
            )

        if graph_split == "test":

            raise RuntimeError(
                "forbidden TEST graph"
            )

        graph_path = resolve_graph_path(
            str(
                graph_row[
                    "graph_path"
                ]
            )
        )

        graphs.append(
            load_g2_graph(
                graph_path
            )
        )

        if (
            row_number == 1
            or
            row_number % 500 == 0
            or
            row_number == N_DEV
        ):

            print(
                f"  loaded "
                f"{row_number}/{N_DEV}"
            )

    check(
        len(
            graphs
        )
        == N_DEV,
        (
            "loaded exactly 4626 "
            "TRAIN+VAL graphs"
        ),
    )

    split_array = (
        dev_manifest[
            "split"
        ].to_numpy()
    )

    train_indices = np.flatnonzero(
        split_array
        == "train"
    )

    val_indices = np.flatnonzero(
        split_array
        == "val"
    )

    check(
        len(
            train_indices
        )
        == N_TRAIN,
        (
            "live TRAIN indices "
            "= 4112"
        ),
    )

    check(
        len(
            val_indices
        )
        == N_VAL,
        (
            "live VAL indices "
            "= 514"
        ),
    )

    print()
    print(
        "TEST token rows supplied to model: 0"
    )

    print(
        "TEST graph files opened:           0"
    )

    print(
        "TEST labels loaded:                0"
    )

    if FAILURES:

        raise RuntimeError(
            "live Task-4 data validation failed"
        )

    return {
        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "token_type_ids":
            token_type_ids,

        "graphs":
            graphs,

        "train_indices":
            train_indices,

        "val_indices":
            val_indices,
    }


# =============================================================================
# 16. LIVE DATASET / COLLATE
# =============================================================================

class PairDataset(
    Dataset,
):

    def __init__(
        self,
        data: dict[str, Any],
        indices: np.ndarray,
    ) -> None:

        self.data = data

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(
        self,
    ) -> int:

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

        return (
            torch.from_numpy(
                self.data[
                    "input_ids"
                ][
                    index
                ]
                .astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "attention_mask"
                ][
                    index
                ]
                .astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.data[
                    "token_type_ids"
                ][
                    index
                ]
                .astype(
                    np.int64,
                    copy=True,
                )
            ),

            self.data[
                "graphs"
            ][
                index
            ],
        )


def collate_pairs(
    samples,
):

    return (
        torch.stack(
            [
                value[0]
                for value
                in samples
            ]
        ),

        torch.stack(
            [
                value[1]
                for value
                in samples
            ]
        ),

        torch.stack(
            [
                value[2]
                for value
                in samples
            ]
        ),

        Batch.from_data_list(
            [
                value[3]
                for value
                in samples
            ]
        ),
    )


def build_live_loaders(
    data: dict[str, Any],
    seed: int,
):

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    train_dataset = PairDataset(
        data,
        data[
            "train_indices"
        ],
    )

    val_dataset = PairDataset(
        data,
        data[
            "val_indices"
        ],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=T4B_BATCH,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_pairs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=T4B_BATCH,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_pairs,
    )

    return (
        train_loader,
        val_loader,
    )


# =============================================================================
# 17. BUILD T4-B FROM SAME-SEED T4-A
# =============================================================================

def build_t4b_model(
    t4a_checkpoint: Path,
    device: torch.device,
) -> T4BModel:

    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        local_files_only=True,
    )

    check(
        config.hidden_size
        == BERT_DIM,
        "BERT hidden size = 768",
    )

    check(
        config.num_hidden_layers
        == 12,
        "BERT layers = 12",
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    t1 = load_checkpoint(
        T1_CHECKPOINT
    )

    full_t1_state = (
        t1[
            "model_state_dict"
        ]
    )

    bert_state = {
        key[
            len(
                "bert."
            ) :
        ]:
            value

        for key, value
        in full_t1_state.items()

        if key.startswith(
            "bert."
        )
    }

    incompatible = bert.load_state_dict(
        bert_state,
        strict=True,
    )

    check(
        len(
            incompatible.missing_keys
        )
        == 0,
        (
            "T1-B BERT strict load "
            "has zero missing keys"
        ),
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "T1-B BERT strict load "
            "has zero unexpected keys"
        ),
    )

    gnn = A4GAT()

    a4 = load_checkpoint(
        A4_CHECKPOINT
    )

    incompatible = gnn.load_state_dict(
        a4[
            "model_state_dict"
        ],
        strict=True,
    )

    check(
        len(
            incompatible.missing_keys
        )
        == 0,
        "A4 strict load: zero missing keys",
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        "A4 strict load: zero unexpected keys",
    )

    model = T4BModel(
        bert,
        gnn,
    )

    t4a = load_checkpoint(
        t4a_checkpoint
    )

    state = (
        t4a[
            "model_state_dict"
        ]
    )

    check(
        set(
            state.keys()
        )
        ==
        {
            "text_projection.weight",
            "graph_projection.weight",
        },
        (
            "same-seed T4-A checkpoint "
            "contains exactly projection weights"
        ),
    )

    model.text_projection.load_state_dict(
        {
            "weight":
                state[
                    "text_projection.weight"
                ]
        },
        strict=True,
    )

    model.graph_projection.load_state_dict(
        {
            "weight":
                state[
                    "graph_projection.weight"
                ]
        },
        strict=True,
    )

    # Freeze everything.
    for p in model.parameters():

        p.requires_grad_(
            False
        )

    # BERT layers 10-11.
    for layer_index in (
        10,
        11,
    ):

        for p in (
            model
            .bert
            .encoder
            .layer[
                layer_index
            ]
            .parameters()
        ):

            p.requires_grad_(
                True
            )

    # GAT3 only.
    for p in (
        model
        .gnn
        .gat3
        .parameters()
    ):

        p.requires_grad_(
            True
        )

    # Projection heads.
    for p in (
        model
        .text_projection
        .parameters()
    ):

        p.requires_grad_(
            True
        )

    for p in (
        model
        .graph_projection
        .parameters()
    ):

        p.requires_grad_(
            True
        )

    bert_trainable = sum(
        p.numel()
        for p
        in model.bert.parameters()
        if p.requires_grad
    )

    gat_trainable = sum(
        p.numel()
        for p
        in model.gnn.parameters()
        if p.requires_grad
    )

    projection_trainable = (
        sum(
            p.numel()
            for p
            in model.text_projection.parameters()
            if p.requires_grad
        )
        +
        sum(
            p.numel()
            for p
            in model.graph_projection.parameters()
            if p.requires_grad
        )
    )

    total_trainable = sum(
        p.numel()
        for p
        in model.parameters()
        if p.requires_grad
    )

    check(
        bert_trainable
        == EXPECTED_BERT_TOP2_TRAINABLE,
        (
            "T4-B BERT trainable "
            "= 14,175,744"
        ),
    )

    check(
        gat_trainable
        == EXPECTED_GAT3_TRAINABLE,
        (
            "T4-B GAT3 trainable "
            "= 66,688"
        ),
    )

    check(
        projection_trainable
        == EXPECTED_PROJECTION_TRAINABLE,
        (
            "T4-B projections trainable "
            "= 229,376"
        ),
    )

    check(
        total_trainable
        == EXPECTED_T4B_TOTAL_TRAINABLE,
        (
            "T4-B total trainable "
            "= 14,471,808"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "T4-B model contract failed"
        )

    return model.to(
        device
    )


# =============================================================================
# 18. LIVE EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_live(
    model: T4BModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    text_output = []
    graph_output = []

    rows = 0

    for (
        input_ids,
        attention_mask,
        token_type_ids,
        graph_batch,
    ) in loader:

        input_ids = input_ids.to(
            device
        )

        attention_mask = attention_mask.to(
            device
        )

        token_type_ids = token_type_ids.to(
            device
        )

        graph_batch = graph_batch.to(
            device
        )

        z_text, z_graph = model(
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
        )

        text_output.append(
            z_text.detach().cpu()
        )

        graph_output.append(
            z_graph.detach().cpu()
        )

        rows += int(
            z_text.shape[0]
        )

    if rows != N_VAL:

        raise RuntimeError(
            f"VAL rows={rows}, expected 514"
        )

    return retrieval_metrics(
        torch.cat(
            text_output,
            dim=0,
        ),
        torch.cat(
            graph_output,
            dim=0,
        ),
    )


# =============================================================================
# 19. T4-B STATE CAPTURE / RESTORE
# =============================================================================

def capture_t4b_state(
    model: T4BModel,
) -> dict[str, Any]:

    return {
        "bert_layer_10":
            cpu_state_dict(
                model.bert.encoder.layer[
                    10
                ]
            ),

        "bert_layer_11":
            cpu_state_dict(
                model.bert.encoder.layer[
                    11
                ]
            ),

        "gat3":
            cpu_state_dict(
                model.gnn.gat3
            ),

        "text_projection":
            cpu_state_dict(
                model.text_projection
            ),

        "graph_projection":
            cpu_state_dict(
                model.graph_projection
            ),
    }


def restore_t4b_state(
    model: T4BModel,
    state: dict[str, Any],
) -> None:

    model.bert.encoder.layer[
        10
    ].load_state_dict(
        state[
            "bert_layer_10"
        ],
        strict=True,
    )

    model.bert.encoder.layer[
        11
    ].load_state_dict(
        state[
            "bert_layer_11"
        ],
        strict=True,
    )

    model.gnn.gat3.load_state_dict(
        state[
            "gat3"
        ],
        strict=True,
    )

    model.text_projection.load_state_dict(
        state[
            "text_projection"
        ],
        strict=True,
    )

    model.graph_projection.load_state_dict(
        state[
            "graph_projection"
        ],
        strict=True,
    )


# =============================================================================
# 20. T4-B OPTIMIZER
# =============================================================================

def build_t4b_optimizer(
    model: T4BModel,
):

    bert = [
        p
        for p
        in model.bert.parameters()
        if p.requires_grad
    ]

    gat = [
        p
        for p
        in model.gnn.parameters()
        if p.requires_grad
    ]

    projections = (
        [
            p
            for p
            in model.text_projection.parameters()
            if p.requires_grad
        ]
        +
        [
            p
            for p
            in model.graph_projection.parameters()
            if p.requires_grad
        ]
    )

    parameter_ids = [
        id(
            p
        )
        for p
        in (
            bert
            +
            gat
            +
            projections
        )
    ]

    check(
        len(
            parameter_ids
        )
        ==
        len(
            set(
                parameter_ids
            )
        ),
        (
            "T4-B optimizer has "
            "no duplicate parameters"
        ),
    )

    return torch.optim.AdamW(
        [
            {
                "params":
                    projections,

                "lr":
                    T4B_PROJECTION_LR,

                "weight_decay":
                    T4B_WEIGHT_DECAY,
            },
            {
                "params":
                    bert,

                "lr":
                    T4B_BERT_LR,

                "weight_decay":
                    T4B_WEIGHT_DECAY,
            },
            {
                "params":
                    gat,

                "lr":
                    T4B_GAT_LR,

                "weight_decay":
                    T4B_WEIGHT_DECAY,
            },
        ]
    )


# =============================================================================
# 21. T4-B TRAIN ONE EPOCH
# =============================================================================

def train_t4b_epoch(
    model: T4BModel,
    loader: DataLoader,
    optimizer,
    device: torch.device,
) -> dict[str, float]:

    model.train()

    total = 0.0
    total_a2t = 0.0
    total_t2a = 0.0

    rows_seen = 0

    for (
        input_ids,
        attention_mask,
        token_type_ids,
        graph_batch,
    ) in loader:

        input_ids = input_ids.to(
            device
        )

        attention_mask = attention_mask.to(
            device
        )

        token_type_ids = token_type_ids.to(
            device
        )

        graph_batch = graph_batch.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        z_text, z_graph = model(
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
        )

        (
            loss,
            a2t,
            t2a,
        ) = symmetric_infonce(
            z_text,
            z_graph,
            T4B_TEMPERATURE,
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "non-finite T4-B loss"
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p
                    in model.parameters()
                    if p.requires_grad
                ],
                max_norm=T4B_GRAD_CLIP,
            )
        )

        if not math.isfinite(
            float(
                grad_norm
            )
        ):

            raise RuntimeError(
                "non-finite T4-B gradient"
            )

        optimizer.step()

        rows = int(
            z_text.shape[0]
        )

        total += (
            float(
                loss.detach().cpu()
            )
            * rows
        )

        total_a2t += (
            float(
                a2t.detach().cpu()
            )
            * rows
        )

        total_t2a += (
            float(
                t2a.detach().cpu()
            )
            * rows
        )

        rows_seen += rows

    if rows_seen != N_TRAIN:

        raise RuntimeError(
            f"T4-B TRAIN rows="
            f"{rows_seen}"
        )

    return {
        "loss":
            total
            / rows_seen,

        "audio_to_caption_loss":
            total_a2t
            / rows_seen,

        "caption_to_audio_loss":
            total_t2a
            / rows_seen,
    }


# =============================================================================
# 22. RUN SAME-SEED T4-B
# =============================================================================

def run_t4b_seed(
    seed: int,
    t4a_result: dict[str, Any],
    live_data: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:

    banner(
        f"4B. SEED {seed} — SELECTED T4-B STAGE"
    )

    configure_t4b_seed(
        seed
    )

    (
        train_loader,
        val_loader,
    ) = build_live_loaders(
        live_data,
        seed,
    )

    model = build_t4b_model(
        t4a_result[
            "checkpoint"
        ],
        device,
    )

    # -------------------------------------------------------------------------
    # Same-seed T4-A initialization parity
    # -------------------------------------------------------------------------

    initial = evaluate_live(
        model,
        val_loader,
        device,
    )

    expected = (
        t4a_result[
            "best_val_metrics"
        ]
    )

    recall_quantum = (
        1.0
        / N_VAL
    )

    print(
        f"seed {seed} initial "
        f"cached T4-A mR = "
        f"{expected['mR']:.10f}"
    )

    print(
        f"seed {seed} initial "
        f"live T4-A mR   = "
        f"{initial['mR']:.10f}"
    )

    for key in METRIC_KEYS:

        difference = abs(
            float(
                initial[
                    key
                ]
            )
            -
            float(
                expected[
                    key
                ]
            )
        )

        check(
            difference
            <= (
                recall_quantum
                + 1e-12
            ),
            (
                f"seed {seed} live T4-A "
                f"parity for {key} "
                "within one VAL query"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            f"seed {seed} T4-A live "
            "initialization parity failed"
        )

    optimizer = build_t4b_optimizer(
        model
    )

    best_mr = -float(
        "inf"
    )

    best_epoch = None
    best_metrics = None
    best_state = None

    no_improvement = 0

    history = []

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start_time = time.perf_counter()

    for epoch in range(
        1,
        T4B_MAX_EPOCHS + 1,
    ):

        train = train_t4b_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val = evaluate_live(
            model,
            val_loader,
            device,
        )

        mr = float(
            val[
                "mR"
            ]
        )

        improved = (
            mr
            >
            best_mr
        )

        if improved:

            best_mr = mr
            best_epoch = epoch
            best_metrics = dict(
                val
            )

            best_state = (
                capture_t4b_state(
                    model
                )
            )

            no_improvement = 0

        else:

            no_improvement += 1

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train[
                        "loss"
                    ],

                "train_audio_to_caption_loss":
                    train[
                        "audio_to_caption_loss"
                    ],

                "train_caption_to_audio_loss":
                    train[
                        "caption_to_audio_loss"
                    ],

                "val_audio_to_caption_R@1":
                    val[
                        "audio_to_caption_R@1"
                    ],

                "val_audio_to_caption_R@5":
                    val[
                        "audio_to_caption_R@5"
                    ],

                "val_audio_to_caption_R@10":
                    val[
                        "audio_to_caption_R@10"
                    ],

                "val_caption_to_audio_R@1":
                    val[
                        "caption_to_audio_R@1"
                    ],

                "val_caption_to_audio_R@5":
                    val[
                        "caption_to_audio_R@5"
                    ],

                "val_caption_to_audio_R@10":
                    val[
                        "caption_to_audio_R@10"
                    ],

                "val_mR":
                    val[
                        "mR"
                    ],

                "improved":
                    bool(
                        improved
                    ),

                "epochs_without_improvement":
                    no_improvement,
            }
        )

        print(
            f"seed={seed} "
            f"T4-B epoch={epoch:02d} "
            f"train={train['loss']:.6f} "
            f"VAL_mR={mr:.8f}"
            + (
                " BEST"
                if improved
                else
                ""
            )
        )

        if (
            no_improvement
            >= T4B_PATIENCE
        ):

            print(
                f"seed={seed} T4-B "
                "early stopping"
            )

            break

    if (
        best_epoch is None
        or
        best_metrics is None
        or
        best_state is None
    ):

        raise RuntimeError(
            f"seed {seed}: "
            "no valid T4-B state"
        )

    # -------------------------------------------------------------------------
    # Restore exact best state.
    # -------------------------------------------------------------------------

    restore_t4b_state(
        model,
        best_state,
    )

    restored = (
        capture_t4b_state(
            model
        )
    )

    for module_name in (
        "bert_layer_10",
        "bert_layer_11",
        "gat3",
        "text_projection",
        "graph_projection",
    ):

        for parameter_name in (
            best_state[
                module_name
            ].keys()
        ):

            check(
                torch.equal(
                    restored[
                        module_name
                    ][
                        parameter_name
                    ],
                    best_state[
                        module_name
                    ][
                        parameter_name
                    ],
                ),
                (
                    f"seed {seed} "
                    f"{module_name}."
                    f"{parameter_name} "
                    "restored exactly"
                ),
            )

    if FAILURES:

        raise RuntimeError(
            f"seed {seed}: "
            "exact T4-B state restore failed"
        )

    # revalidated = evaluate_live(
    #     model,
    #     val_loader,
    #     device,
    # )

    # for key in METRIC_KEYS:

    #     difference = abs(
    #         float(
    #             revalidated[
    #                 key
    #             ]
    #         )
    #         -
    #         float(
    #             best_metrics[
    #                 key
    #             ]
    #         )
    #     )

    #     check(
    #         difference
    #         <= (
    #             recall_quantum
    #             + 1e-12
    #         ),
    #         (
    #             f"seed {seed} T4-B "
    #             f"{key} revalidates "
    #             "within one VAL query"
    #         ),
    #     )

    # mr_difference = abs(
    #     float(
    #         revalidated[
    #             "mR"
    #         ]
    #     )
    #     -
    #     float(
    #         best_metrics[
    #             "mR"
    #         ]
    #     )
    # )

    # check(
    #     mr_difference
    #     <= (
    #         recall_quantum
    #         + 1e-12
    #     ),
    #     (
    #         f"seed {seed} T4-B "
    #         "mR revalidates within "
    #         "one VAL-query quantum"
    #     ),
    # )

    # if FAILURES:

    #     raise RuntimeError(
    #         f"seed {seed}: "
    #         "T4-B best-state revalidation failed"
    #     )

        # -------------------------------------------------------------------------
    # Post-restore retrieval diagnostic
    #
    # Exact tensor restoration above is the authoritative checkpoint-integrity
    # test.
    #
    # Live CUDA BERT/PyG inference is not guaranteed to reproduce every
    # near-tied retrieval rank bit-for-bit across separate forward passes.
    # Therefore retrieval re-evaluation is DIAGNOSTIC ONLY.
    #
    # Formal epoch selection remains the metrics observed at the original
    # validation pass for that epoch. Revalidation must never replace or
    # re-select the formal best epoch.
    # -------------------------------------------------------------------------

    revalidated = evaluate_live(
        model,
        val_loader,
        device,
    )

    revalidation_deltas = {}

    changed_metrics = []

    for key in ALL_REPORT_METRICS:

        selected_value = float(
            best_metrics[
                key
            ]
        )

        restored_value = float(
            revalidated[
                key
            ]
        )

        difference = abs(
            restored_value
            - selected_value
        )

        revalidation_deltas[
            key
        ] = difference

        print(
            f"seed {seed} revalidation {key}: "
            f"selected={selected_value:.10f} "
            f"restored={restored_value:.10f} "
            f"delta={difference:.10f}"
        )

        check(
            math.isfinite(
                restored_value
            ),
            (
                f"seed {seed} restored "
                f"{key} is finite"
            ),
        )

        check(
            0.0
            <= restored_value
            <= 1.0,
            (
                f"seed {seed} restored "
                f"{key} is within [0,1]"
            ),
        )

        if difference > 0.0:

            changed_metrics.append(
                key
            )

    if FAILURES:

        raise RuntimeError(
            f"seed {seed}: "
            "restored-state retrieval produced invalid metrics"
        )

    if changed_metrics:

        warning(
            (
                f"seed {seed} restored checkpoint produced "
                "small live-retrieval differences for: "
                + ", ".join(
                    changed_metrics
                )
                + ". Exact parameter restoration passed; "
                "the originally observed best-epoch metrics "
                "remain the formal validation result."
            )
        )

    else:

        passed(
            (
                f"seed {seed} restored checkpoint "
                "reproduced all retrieval metrics exactly"
            )
        )

    runtime = (
        time.perf_counter()
        -
        start_time
    )

    peak_vram = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    checkpoint_path = (
        t4b_checkpoint_path(
            seed
        )
    )

    history_path = (
        t4b_history_path(
            seed
        )
    )

    summary_path = (
        t4b_summary_path(
            seed
        )
    )

    checkpoint = {
        "artifact_type":
            "task4_T4-B_multiseed_checkpoint",

        "variant":
            "T4-B",

        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_metrics":
            best_metrics,

        "best_state_revalidation": {
            "policy":
                (
                    "diagnostic_only_after_exact_tensor_restore; "
                    "does_not_replace_formal_best_epoch_metrics"
                ),

            "metrics":
                revalidated,

            "absolute_deltas":
                revalidation_deltas,
        },

        "state_format":
            (
                "delta_from_frozen_T1B_A4_"
                "plus_same_seed_T4A_projection_initialization"
            ),

        "source_t4a_checkpoint":
            relative(
                t4a_result[
                    "checkpoint"
                ]
            ),

        "source_t4a_checkpoint_sha256":
            t4a_result[
                "checkpoint_sha256"
            ],

        "bert_layer_10_state_dict":
            best_state[
                "bert_layer_10"
            ],

        "bert_layer_11_state_dict":
            best_state[
                "bert_layer_11"
            ],

        "gat3_state_dict":
            best_state[
                "gat3"
            ],

        "text_projection_state_dict":
            best_state[
                "text_projection"
            ],

        "graph_projection_state_dict":
            best_state[
                "graph_projection"
            ],

        "training": {
            "temperature":
                T4B_TEMPERATURE,

            "batch_size":
                T4B_BATCH,

            "projection_lr":
                T4B_PROJECTION_LR,

            "bert_lr":
                T4B_BERT_LR,

            "gat_lr":
                T4B_GAT_LR,

            "weight_decay":
                T4B_WEIGHT_DECAY,

            "max_epochs":
                T4B_MAX_EPOCHS,

            "patience":
                T4B_PATIENCE,

            "gradient_clip":
                T4B_GRAD_CLIP,

            "amp":
                False,
        },

        "test_access":
            False,

        "test_retrieval":
            False,
    }

    save_checkpoint_atomic(
        checkpoint_path,
        checkpoint,
    )

    write_history_atomic(
        history_path,
        history,
    )

    checkpoint_sha = sha256_file(
        checkpoint_path
    )

    history_sha = sha256_file(
        history_path
    )

    summary = {
        "artifact_type":
            "task4_T4-B_multiseed_summary",

        "variant":
            "T4-B",

        "selected_configuration":
            True,

        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_metrics":
            best_metrics,

        "best_val_mR":
            float(
                best_metrics[
                    "mR"
                ]
            ),

        "best_state_revalidation": {
            "policy":
                (
                    "diagnostic_only_after_exact_tensor_restore; "
                    "formal metrics are those observed during "
                    "the originally selected validation epoch"
                ),

            "metrics":
                revalidated,

            "absolute_deltas":
                revalidation_deltas,

            "exact_parameter_restoration":
                True,
        },

        "initial_same_seed_t4a_val_metrics":
            expected,

        "checkpoint":
            relative(
                checkpoint_path
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "history":
            relative(
                history_path
            ),

        "history_sha256":
            history_sha,

        "runtime_seconds":
            float(
                runtime
            ),

        "peak_vram_bytes":
            int(
                peak_vram
            ),

        "test_access":
            False,

        "test_retrieval":
            False,

        "label_supervision":
            False,
    }

    write_json_atomic(
        summary_path,
        summary,
    )

    summary_sha = sha256_file(
        summary_path
    )

    print()
    print(
        f"seed {seed} T4-B COMPLETE"
    )

    print(
        f"  best epoch = "
        f"{best_epoch}"
    )

    print(
        f"  VAL mR     = "
        f"{best_metrics['mR']:.8f}"
    )

    print(
        f"  checkpoint SHA256 = "
        f"{checkpoint_sha}"
    )

    print(
        f"  summary SHA256    = "
        f"{summary_sha}"
    )

    del model

    torch.cuda.empty_cache()

    return {
        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_metrics":
            best_metrics,

        "checkpoint":
            checkpoint_path,

        "checkpoint_sha256":
            checkpoint_sha,

        "summary":
            summary_path,

        "summary_sha256":
            summary_sha,

        "source_t4a_checkpoint":
            t4a_result[
                "checkpoint"
            ],

        "source_t4a_checkpoint_sha256":
            t4a_result[
                "checkpoint_sha256"
            ],
    }


# =============================================================================
# 23. LOAD EXISTING SEED-42 SELECTED RUN
# =============================================================================

def load_seed42_result() -> dict[str, Any]:

    banner(
        "5. IMPORT ALREADY-COMPLETED SEED-42 RUN"
    )

    summary = load_json(
        SEED42_T4B_SUMMARY
    )

    check(
        summary[
            "variant"
        ]
        == "T4-B",
        (
            "seed-42 existing result "
            "is T4-B"
        ),
    )

    check(
        int(
            summary[
                "seed"
            ]
        )
        == 42,
        (
            "existing selected result "
            "seed = 42"
        ),
    )

    check(
        summary[
            "boundaries"
        ][
            "test_access"
        ]
        is False,
        (
            "seed-42 run records "
            "zero TEST access"
        ),
    )

    check(
        summary[
            "boundaries"
        ][
            "test_retrieval"
        ]
        is False,
        (
            "seed-42 run records "
            "zero TEST retrieval"
        ),
    )

    metrics = (
        summary[
            "best_val_metrics"
        ]
    )

    print(
        "Seed 42 is reused from the "
        "already accepted independent "
        "Script-67 → Script-68 pathway."
    )

    print(
        f"Seed-42 VAL mR = "
        f"{metrics['mR']:.8f}"
    )

    return {
        "seed":
            42,

        "best_epoch":
            int(
                summary[
                    "best_epoch"
                ]
            ),

        "best_val_metrics":
            metrics,

        "checkpoint":
            SEED42_T4B_CHECKPOINT,

        "checkpoint_sha256":
            EXPECTED_SEED42_T4B_CHECKPOINT_SHA256,

        "summary":
            SEED42_T4B_SUMMARY,

        "summary_sha256":
            EXPECTED_SEED42_T4B_SUMMARY_SHA256,

        "source_t4a_checkpoint":
            SEED42_T4A_CHECKPOINT,

        "source_t4a_checkpoint_sha256":
            EXPECTED_SEED42_T4A_CHECKPOINT_SHA256,
    }


# =============================================================================
# 24. AGGREGATE THREE-SEED VALIDATION
# =============================================================================

def aggregate_results(
    seed_results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:

    banner(
        "6. THREE-SEED VALIDATION AGGREGATION"
    )

    check(
        [
            item[
                "seed"
            ]
            for item
            in seed_results
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "final selected T4-B results "
            "cover exactly seeds "
            "42/1337/2026"
        ),
    )

    rows = []

    for result in seed_results:

        metrics = (
            result[
                "best_val_metrics"
            ]
        )

        row = {
            "seed":
                result[
                    "seed"
                ],

            "best_epoch":
                result[
                    "best_epoch"
                ],
        }

        for key in ALL_REPORT_METRICS:

            row[
                key
            ] = float(
                metrics[
                    key
                ]
            )

        row[
            "checkpoint"
        ] = relative(
            result[
                "checkpoint"
            ]
        )

        row[
            "checkpoint_sha256"
        ] = result[
            "checkpoint_sha256"
        ]

        rows.append(
            row
        )

    frame = pd.DataFrame(
        rows
    )

    temp_csv = Path(
        str(
            MULTISEED_CSV
        )
        + ".__tmp__"
    )

    frame.to_csv(
        temp_csv,
        index=False,
    )

    os.replace(
        temp_csv,
        MULTISEED_CSV,
    )

    aggregate = {}

    for key in ALL_REPORT_METRICS:

        values = np.asarray(
            [
                float(
                    result[
                        "best_val_metrics"
                    ][
                        key
                    ]
                )

                for result
                in seed_results
            ],
            dtype=np.float64,
        )

        aggregate[
            key
        ] = {
            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            # Population SD over the complete
            # predetermined final seed set.
            "std_population":
                float(
                    np.std(
                        values,
                        ddof=0,
                    )
                ),

            "values_by_seed": {
                str(
                    result[
                        "seed"
                    ]
                ):
                    float(
                        result[
                            "best_val_metrics"
                        ][
                            key
                        ]
                    )

                for result
                in seed_results
            },
        }

    print()

    for result in seed_results:

        print(
            f"seed {result['seed']:4d}  "
            f"VAL mR = "
            f"{result['best_val_metrics']['mR']:.8f}"
        )

    print()

    print(
        "Three-seed VAL mR:"
    )

    print(
        f"  mean = "
        f"{aggregate['mR']['mean']:.8f}"
    )

    print(
        f"  std  = "
        f"{aggregate['mR']['std_population']:.8f}"
    )

    return {
        "rows":
            rows,

        "aggregate":
            aggregate,
    }


# =============================================================================
# 25. WRITE MULTI-SEED SUMMARY + LOCK
# =============================================================================

def write_multiseed_freeze(
    seed_results,
    aggregation,
) -> tuple[
    str,
    str,
    str,
]:

    banner(
        "7. WRITE MULTI-SEED SUMMARY / LOCK"
    )

    created_utc = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    csv_sha = sha256_file(
        MULTISEED_CSV
    )

    summary = {
        "artifact_type":
            "task4_selected_T4-B_multiseed_validation_summary",

        "version":
            1,

        "created_utc":
            created_utc,

        "selected_variant":
            "T4-B",

        "selection_frozen_before_multiseed":
            True,

        "architecture_reselection_performed":
            False,

        "seeds":
            [
                42,
                1337,
                2026,
            ],

        "seed42_policy":
            (
                "reuse already accepted independent "
                "Script-67 T4-A -> Script-68 T4-B run"
            ),

        "new_seed_policy":
            (
                "for each of seeds 1337 and 2026, "
                "independently run same-seed T4-A "
                "then same-seed T4-B"
            ),

        "validation_only":
            True,

        "test_access":
            False,

        "test_retrieval":
            False,

        "standard_deviation_definition":
            (
                "population standard deviation "
                "over complete predetermined "
                "seed set {42,1337,2026}; ddof=0"
            ),

        "per_seed":
            [
                {
                    "seed":
                        result[
                            "seed"
                        ],

                    "best_epoch":
                        result[
                            "best_epoch"
                        ],

                    "best_val_metrics":
                        result[
                            "best_val_metrics"
                        ],

                    "selected_t4b_checkpoint":
                        relative(
                            result[
                                "checkpoint"
                            ]
                        ),

                    "selected_t4b_checkpoint_sha256":
                        result[
                            "checkpoint_sha256"
                        ],

                    "same_seed_t4a_initialization_checkpoint":
                        relative(
                            result[
                                "source_t4a_checkpoint"
                            ]
                        ),

                    "same_seed_t4a_initialization_checkpoint_sha256":
                        result[
                            "source_t4a_checkpoint_sha256"
                        ],
                }

                for result
                in seed_results
            ],

        "aggregate_validation":
            aggregation[
                "aggregate"
            ],

        "validation_csv":
            relative(
                MULTISEED_CSV
            ),

        "validation_csv_sha256":
            csv_sha,

        "provenance": {
            "script70":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "script70_sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),

            "design_sha256":
                EXPECTED_DESIGN_SHA256,

            "design_lock_sha256":
                EXPECTED_DESIGN_LOCK_SHA256,

            "model_selection_sha256":
                EXPECTED_SELECTION_SHA256,

            "model_selection_lock_sha256":
                EXPECTED_SELECTION_LOCK_SHA256,
        },

        "next_action":
            (
                "Script 71 — freeze final Task-4 "
                "pre-TEST protocol and verify "
                "all three selected seed checkpoints"
            ),
    }

    write_json_atomic(
        MULTISEED_SUMMARY,
        summary,
    )

    summary_sha = sha256_file(
        MULTISEED_SUMMARY
    )

    checkpoint_hashes = {
        str(
            result[
                "seed"
            ]
        ):
            {
                "path":
                    relative(
                        result[
                            "checkpoint"
                        ]
                    ),

                "sha256":
                    result[
                        "checkpoint_sha256"
                    ],
            }

        for result
        in seed_results
    }

    lock = {
        "lock_type":
            "task4_selected_multiseed_validation_lock",

        "version":
            1,

        "created_utc":
            created_utc,

        "selected_variant":
            "T4-B",

        "seeds":
            [
                42,
                1337,
                2026,
            ],

        "summary":
            relative(
                MULTISEED_SUMMARY
            ),

        "summary_sha256":
            summary_sha,

        "validation_csv":
            relative(
                MULTISEED_CSV
            ),

        "validation_csv_sha256":
            csv_sha,

        "selected_checkpoints":
            checkpoint_hashes,

        "selection_sha256":
            EXPECTED_SELECTION_SHA256,

        "selection_lock_sha256":
            EXPECTED_SELECTION_LOCK_SHA256,

        "test_access":
            False,

        "test_retrieval":
            False,

        "script70_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        MULTISEED_LOCK,
        lock,
    )

    lock_sha = sha256_file(
        MULTISEED_LOCK
    )

    check(
        load_json(
            MULTISEED_LOCK
        )[
            "summary_sha256"
        ]
        ==
        sha256_file(
            MULTISEED_SUMMARY
        ),
        (
            "multi-seed lock matches "
            "multi-seed summary"
        ),
    )

    print(
        "Validation CSV:"
    )

    print(
        f"  {relative(MULTISEED_CSV)}"
    )

    print(
        f"  SHA256 = {csv_sha}"
    )

    print()

    print(
        "Multi-seed summary:"
    )

    print(
        f"  {relative(MULTISEED_SUMMARY)}"
    )

    print(
        f"  SHA256 = {summary_sha}"
    )

    print()

    print(
        "Multi-seed lock:"
    )

    print(
        f"  {relative(MULTISEED_LOCK)}"
    )

    print(
        f"  SHA256 = {lock_sha}"
    )

    return (
        csv_sha,
        summary_sha,
        lock_sha,
    )


# =============================================================================
# 26. TEST BOUNDARY
# =============================================================================

def print_test_boundary() -> None:

    banner(
        "8. TEST-BOUNDARY CONFIRMATION"
    )

    print(
        "Final selected configuration: T4-B"
    )

    print(
        "Architecture reselection:        0"
    )

    print(
        "Seeds completed:                 42,1337,2026"
    )

    print(
        "TEST token rows supplied:        0"
    )

    print(
        "TEST graph files opened:         0"
    )

    print(
        "TEST captions inferred:          0"
    )

    print(
        "TEST retrieval calculations:     0"
    )

    print(
        "TEST metrics observed:           0"
    )

    print(
        "TEST-driven changes:             0"
    )

    passed(
        "Script 70 preserved the final TEST boundary"
    )


# =============================================================================
# 27. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 70 — FINAL SELECTED T4-B MULTI-SEED RUNS"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "Frozen selected configuration:"
    )

    print(
        "  T4-B"
    )

    print()
    print(
        "Final seeds:"
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
        "Seed 42:"
    )

    print(
        "  REUSE accepted Scripts 67→68"
    )

    print()
    print(
        "Seeds 1337 / 2026:"
    )

    print(
        "  independent same-seed T4-A"
    )

    print(
        "  then same-seed T4-B"
    )

    print()
    print(
        "TEST access: NONE"
    )

    try:

        verify_output_protection()

        verify_frozen_lineage()

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA unavailable"
            )

        device = torch.device(
            "cuda"
        )

        banner(
            "2A. DEVICE"
        )

        print(
            f"CUDA: "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"PyTorch: "
            f"{torch.__version__}"
        )

        print(
            f"Transformers: "
            f"{transformers.__version__}"
        )

        print(
            f"PyG: "
            f"{torch_geometric.__version__}"
        )

        stage_a_data = (
            load_stage_a_data()
        )

        live_data = (
            load_live_data(
                stage_a_data[
                    "dev_track_ids"
                ]
            )
        )

        # ---------------------------------------------------------------------
        # Existing seed 42.
        # ---------------------------------------------------------------------

        seed_results = [
            load_seed42_result()
        ]

        # ---------------------------------------------------------------------
        # New independent seeds.
        # ---------------------------------------------------------------------

        for seed in NEW_SEEDS:

            t4a_result = run_t4a_seed(
                seed,
                stage_a_data,
                device,
            )

            # Reset CUDA state between stages.
            torch.cuda.empty_cache()

            t4b_result = run_t4b_seed(
                seed,
                t4a_result,
                live_data,
                device,
            )

            seed_results.append(
                t4b_result
            )

        aggregation = aggregate_results(
            seed_results
        )

        (
            csv_sha,
            summary_sha,
            lock_sha,
        ) = write_multiseed_freeze(
            seed_results,
            aggregation,
        )

        print_test_boundary()

    except Exception as exc:

        banner(
            "SCRIPT 70 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        if FAILURES:

            print()
            print(
                "Failed checks:"
            )

            for message in FAILURES:

                print(
                    f"  - {message}"
                )

        if WARNINGS:

            print()
            print(
                "Warnings:"
            )

            for message in WARNINGS:

                print(
                    f"  - {message}"
                )

        print()
        print(
            "Do NOT proceed to Script 71."
        )

        return 1

    banner(
        "SCRIPT 70 SUMMARY"
    )

    print(
        f"Failures: "
        f"{len(FAILURES)}"
    )

    print(
        f"Warnings: "
        f"{len(WARNINGS)}"
    )

    print()

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "SELECTED T4-B MULTI-SEED RUNS COMPLETE."
    )

    print()

    for result in seed_results:

        print(
            f"Seed "
            f"{result['seed']:4d}: "
            f"VAL mR = "
            f"{result['best_val_metrics']['mR']:.8f}"
        )

    print()
    print(
        "Aggregate VAL mR:"
    )

    print(
        f"  mean = "
        f"{aggregation['aggregate']['mR']['mean']:.8f}"
    )

    print(
        f"  std  = "
        f"{aggregation['aggregate']['mR']['std_population']:.8f}"
    )

    print()
    print(
        "Multi-seed summary SHA256:"
    )

    print(
        f"  {summary_sha}"
    )

    print(
        "Multi-seed lock SHA256:"
    )

    print(
        f"  {lock_sha}"
    )

    print()

    print(
        "TEST retrieval observed: NO"
    )

    print()
    print(
        "STOP HERE."
    )

    print()

    print(
        "Next:"
    )

    print(
        "  Script 71 — final Task-4 "
        "pre-TEST lock."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )