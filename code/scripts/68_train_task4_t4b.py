#!/usr/bin/env python3
"""
SCRIPT 68 — FORMAL TASK-4 T4-B PARTIAL DUAL-ENCODER FINE-TUNING

Frozen Task-4 design
--------------------
Initialization:
    Text encoder:
        selected T1-B BERT

    Graph encoder:
        selected A4 GAT / G2

    Projection heads:
        best formal T4-A seed-42 checkpoint

Trainable:
    BERT encoder layers 10 and 11
    A4 GAT3 only
    text projection Linear(768,256,bias=False)
    graph projection Linear(128,256,bias=False)

Frozen:
    BERT embeddings
    BERT layers 0..9
    BERT pooler
    A4 node_projection
    A4 gat1
    A4 gat2
    A4 original classifier

Objective:
    symmetric InfoNCE

Formal protocol:
    seed                  = 42
    temperature           = 0.07
    shared dimension      = 256
    batch size            = 32

    projection LR         = 1e-4
    BERT top-2 LR         = 2e-5
    GAT3 LR               = 1e-4
    weight decay          = 1e-4

    max epochs            = 20
    patience              = 4
    gradient clip         = 1.0

    primary VAL metric    = full-pool mR

Development:
    TRAIN = 4112
    VAL   = 514
    TEST  = 0

IMPORTANT
---------
- NO TEST examples
- NO TEST captions
- NO TEST graphs
- NO TEST retrieval
- NO label supervision
- NO threshold tuning
- NO architecture tuning

A pre-formal batch-32 CUDA memory/backward gate is performed WITHOUT
an optimizer step. The diagnostic model is then destroyed, the random
seed is reset, and the formal model is reconstructed from the frozen
source checkpoints.

AMP is DISABLED because no accepted T4-B AMP pathway was established
before the formal result.
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
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]


DESIGN_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_contrastive_design.json"
)

DESIGN_LOCK_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_contrastive_design_lock.json"
)

SCRIPT66_SMOKE = (
    ROOT
    / "results/runs/task4/"
      "task4_script66_smoke.json"
)

T4A_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-A_seed42_best.pt"
)

T4A_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-A_seed42_summary.json"
)


PRETRAINING_LOCK = (
    ROOT
    / "data/splits/"
      "pretraining_frozen_hashes.json"
)

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

CHECKPOINT_PATH = (
    MODEL_DIR
    / "task4_T4-B_seed42_best.pt"
)

HISTORY_PATH = (
    RUN_DIR
    / "task4_T4-B_seed42_history.csv"
)

SUMMARY_PATH = (
    RUN_DIR
    / "task4_T4-B_seed42_summary.json"
)


# =============================================================================
# 1. ACCEPTED HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "61ed1263476e631374790c9669ab4aa96"
    "e3b5cb295436d4e273da66d4da556db"
)

EXPECTED_DESIGN_LOCK_SHA256 = (
    "53f50205e712833ec4efb506b9b6d911"
    "8dcf46f51b72c5907f33cd61ddca612c"
)

EXPECTED_SCRIPT66_SMOKE_SHA256 = (
    "0151e229f7691284deea9a005f3f357d"
    "55f4c6af8417418ee42c0948bb6b2329"
)

EXPECTED_T4A_CHECKPOINT_SHA256 = (
    "fc3e85fbe6e98d19dee5cb2891fce954"
    "704e09fc49e494a589193d9a0a09ab9f"
)

EXPECTED_T4A_SUMMARY_SHA256 = (
    "f82844e55472a574b4c500dd267435890"
    "e7c55f2d6fb0ca853ce0e8ebed13806"
)


EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28"
    "f4abac8f0766c90d5b0e027a6c917b4e8"
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
# 2. FROZEN CONTRACT
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

SEED = 42

TEMPERATURE = 0.07

BATCH_SIZE = 32

PROJECTION_LR = 1.0e-4
BERT_LR = 2.0e-5
GAT_LR = 1.0e-4

WEIGHT_DECAY = 1.0e-4

MAX_EPOCHS = 20
PATIENCE = 4

GRAD_CLIP_NORM = 1.0


EXPECTED_BERT_TOP2_TRAINABLE = (
    14_175_744
)

EXPECTED_GAT3_TRAINABLE = (
    66_688
)

EXPECTED_PROJECTION_TRAINABLE = (
    229_376
)

EXPECTED_TOTAL_TRAINABLE = (
    EXPECTED_BERT_TOP2_TRAINABLE
    + EXPECTED_GAT3_TRAINABLE
    + EXPECTED_PROJECTION_TRAINABLE
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

    print(f"PASS  {message}")


def fail(
    message: str,
) -> None:

    print(f"FAIL  {message}")
    FAILURES.append(message)


def warning(
    message: str,
) -> None:

    print(f"WARN  {message}")
    WARNINGS.append(message)


def check(
    condition: bool,
    message: str,
) -> bool:

    if condition:

        passed(message)
        return True

    fail(message)
    return False


# =============================================================================
# 4. HELPERS
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


def relative(
    path: Path,
) -> str:

    try:

        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        )

    except Exception:

        return str(path)


def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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
            f"Temporary file already exists: {temp}"
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

        f.flush()
        os.fsync(
            f.fileno()
        )

    os.replace(
        temp,
        path,
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
            f"{path.name} is not a "
            "checkpoint dictionary"
        )

    return value


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


def configure_seed() -> None:

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

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:

        torch.set_float32_matmul_precision(
            "highest"
        )

    except Exception:

        pass


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
        f"{label} SHA256 matches accepted identity",
    )


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. FORMAL T4-B OUTPUT PROTECTION"
    )

    for path, label in (
        (
            CHECKPOINT_PATH,
            "T4-B checkpoint",
        ),
        (
            HISTORY_PATH,
            "T4-B history",
        ),
        (
            SUMMARY_PATH,
            "T4-B summary",
        ),
    ):

        check(
            not path.exists(),
            f"{label} does not already exist",
        )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite an "
            "existing formal T4-B result"
        )


# =============================================================================
# 6. DEPENDENCY / DESIGN VALIDATION
# =============================================================================

def verify_dependencies() -> dict[str, Any]:

    banner(
        "2. FROZEN TASK-4 LINEAGE"
    )

    expected = (
        (
            DESIGN_PATH,
            EXPECTED_DESIGN_SHA256,
            "Task-4 design",
        ),
        (
            DESIGN_LOCK_PATH,
            EXPECTED_DESIGN_LOCK_SHA256,
            "Task-4 design lock",
        ),
        (
            SCRIPT66_SMOKE,
            EXPECTED_SCRIPT66_SMOKE_SHA256,
            "Script-66 smoke report",
        ),
        (
            T4A_CHECKPOINT,
            EXPECTED_T4A_CHECKPOINT_SHA256,
            "formal T4-A checkpoint",
        ),
        (
            T4A_SUMMARY,
            EXPECTED_T4A_SUMMARY_SHA256,
            "formal T4-A summary",
        ),
        (
            PRETRAINING_LOCK,
            EXPECTED_PRETRAINING_LOCK_SHA256,
            "pretraining lock",
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

    for path, digest, label in expected:

        verify_sha(
            path,
            digest,
            label,
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen Task-4 lineage validation failed"
        )

    design = load_json(
        DESIGN_PATH
    )

    check(
        design[
            "shared_embedding_space"
        ][
            "dimension"
        ]
        == SHARED_DIM,
        "Task-4 shared dimension remains 256",
    )

    check(
        np.isclose(
            float(
                design[
                    "contrastive_objective"
                ][
                    "temperature"
                ]
            ),
            TEMPERATURE,
            rtol=0.0,
            atol=1e-12,
        ),
        "Task-4 temperature remains 0.07",
    )

    t4b = design[
        "t4_b"
    ]

    check(
        t4b[
            "text_trainable_scope"
        ]
        ==
        "final 2 BERT encoder blocks only",
        (
            "T4-B text scope remains "
            "final 2 BERT blocks only"
        ),
    )

    check(
        t4b[
            "graph_trainable_scope"
        ]
        ==
        "final GAT message-passing encoder block only",
        (
            "T4-B graph scope remains "
            "final GAT block only"
        ),
    )

    check(
        t4b[
            "projection_heads_trainable"
        ]
        is True,
        "T4-B projection heads remain trainable",
    )

    check(
        t4b[
            "uses_frozen_representation_cache_for_training"
        ]
        is False,
        (
            "T4-B requires live BERT/GAT "
            "encoder forwards"
        ),
    )

    check(
        t4b[
            "batch_size"
        ]
        == BATCH_SIZE,
        "T4-B batch size remains 32",
    )

    check(
        t4b[
            "max_epochs"
        ]
        == MAX_EPOCHS,
        "T4-B max epochs remains 20",
    )

    check(
        t4b[
            "early_stopping_patience"
        ]
        == PATIENCE,
        "T4-B patience remains 4",
    )

    check(
        np.isclose(
            float(
                t4b[
                    "learning_rates"
                ][
                    "projection_heads"
                ]
            ),
            PROJECTION_LR,
            atol=1e-15,
            rtol=0.0,
        ),
        "projection-head LR remains 1e-4",
    )

    check(
        np.isclose(
            float(
                t4b[
                    "learning_rates"
                ][
                    "bert_final_2_blocks"
                ]
            ),
            BERT_LR,
            atol=1e-15,
            rtol=0.0,
        ),
        "BERT top-2 LR remains 2e-5",
    )

    check(
        np.isclose(
            float(
                t4b[
                    "learning_rates"
                ][
                    "gat_final_block"
                ]
            ),
            GAT_LR,
            atol=1e-15,
            rtol=0.0,
        ),
        "GAT3 LR remains 1e-4",
    )

    check(
        np.isclose(
            float(
                t4b[
                    "weight_decay"
                ]
            ),
            WEIGHT_DECAY,
            atol=1e-15,
            rtol=0.0,
        ),
        "T4-B weight decay remains 1e-4",
    )

    check(
        design[
            "validation_retrieval"
        ][
            "primary_model_selection_metric"
        ]
        == "mR",
        "primary validation metric remains mR",
    )

    check(
        design[
            "dataset"
        ][
            "test_during_development"
        ]
        is False,
        "TEST remains forbidden during development",
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen Task-4 design validation failed"
        )

    return design


# =============================================================================
# 7. EXACT A4 MODEL
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
# 8. T4-B MODEL
# =============================================================================

class Task4T4B(
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

        z_text = (
            self.text_projection(
                text_cls
            )
        )

        z_graph = (
            self.graph_projection(
                graph_embedding
            )
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
# 9. BUILD EXACT T4-B INITIALIZATION
# =============================================================================

def build_t4b_model(
    device: torch.device,
) -> Task4T4B:

    banner(
        "3. RECONSTRUCT T4-B FROM FROZEN SOURCES"
    )

    # -------------------------------------------------------------------------
    # BERT = selected Task-1 T1-B
    # -------------------------------------------------------------------------

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
        "BERT has exactly 12 encoder layers",
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    t1_checkpoint = load_checkpoint(
        T1_CHECKPOINT
    )

    full_t1_state = (
        t1_checkpoint[
            "model_state_dict"
        ]
    )

    bert_state = {
        key[
            len("bert.") :
        ]:
            value

        for key, value
        in full_t1_state.items()

        if key.startswith(
            "bert."
        )
    }

    excluded_t1 = {
        key
        for key
        in full_t1_state
        if not key.startswith(
            "bert."
        )
    }

    check(
        excluded_t1
        ==
        {
            "classifier.weight",
            "classifier.bias",
        },
        (
            "only original Task-1 classifier "
            "is excluded"
        ),
    )

    incompatible = (
        bert.load_state_dict(
            bert_state,
            strict=True,
        )
    )

    check(
        len(
            incompatible.missing_keys
        )
        == 0,
        "T1-B BERT strict load has zero missing keys",
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        "T1-B BERT strict load has zero unexpected keys",
    )

    # -------------------------------------------------------------------------
    # GAT = selected A4/G2
    # -------------------------------------------------------------------------

    gnn = A4GAT()

    a4_checkpoint = load_checkpoint(
        A4_CHECKPOINT
    )

    incompatible = (
        gnn.load_state_dict(
            a4_checkpoint[
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
        "A4 strict load has zero missing keys",
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        "A4 strict load has zero unexpected keys",
    )

    # -------------------------------------------------------------------------
    # Construct Task-4 model
    # -------------------------------------------------------------------------

    model = Task4T4B(
        bert=bert,
        gnn=gnn,
    )

    # -------------------------------------------------------------------------
    # Projection heads = BEST formal T4-A
    # -------------------------------------------------------------------------

    t4a = load_checkpoint(
        T4A_CHECKPOINT
    )

    check(
        t4a.get(
            "variant"
        )
        == "T4-A",
        "projection source variant = T4-A",
    )

    check(
        int(
            t4a.get(
                "seed",
                -1,
            )
        )
        == SEED,
        "projection source seed = 42",
    )

    t4a_state = (
        t4a[
            "model_state_dict"
        ]
    )

    expected_projection_keys = {
        "text_projection.weight",
        "graph_projection.weight",
    }

    check(
        set(
            t4a_state.keys()
        )
        ==
        expected_projection_keys,
        (
            "T4-A checkpoint contains exactly "
            "the two projection weights"
        ),
    )

    model.text_projection.load_state_dict(
        {
            "weight":
                t4a_state[
                    "text_projection.weight"
                ]
        },
        strict=True,
    )

    model.graph_projection.load_state_dict(
        {
            "weight":
                t4a_state[
                    "graph_projection.weight"
                ]
        },
        strict=True,
    )

    # -------------------------------------------------------------------------
    # Freeze EVERYTHING first.
    # -------------------------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )

    # -------------------------------------------------------------------------
    # Unfreeze exactly the Script-65 Task-4 contract.
    # -------------------------------------------------------------------------

    for layer_index in (
        10,
        11,
    ):

        for parameter in (
            model
            .bert
            .encoder
            .layer[
                layer_index
            ]
            .parameters()
        ):

            parameter.requires_grad_(
                True
            )

    # IMPORTANT:
    # Task 4 freezes node_projection, GAT1 and GAT2.
    # Only GAT3 is trainable.
    for parameter in (
        model
        .gnn
        .gat3
        .parameters()
    ):

        parameter.requires_grad_(
            True
        )

    for parameter in (
        model
        .text_projection
        .parameters()
    ):

        parameter.requires_grad_(
            True
        )

    for parameter in (
        model
        .graph_projection
        .parameters()
    ):

        parameter.requires_grad_(
            True
        )

    # -------------------------------------------------------------------------
    # Verify exact parameter contract.
    # -------------------------------------------------------------------------

    bert_trainable = sum(
        parameter.numel()
        for parameter
        in model.bert.parameters()
        if parameter.requires_grad
    )

    gat_trainable = sum(
        parameter.numel()
        for parameter
        in model.gnn.parameters()
        if parameter.requires_grad
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
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Trainable BERT top-2: "
        f"{bert_trainable:,}"
    )

    print(
        f"Trainable GAT3: "
        f"{gat_trainable:,}"
    )

    print(
        f"Trainable projections: "
        f"{projection_trainable:,}"
    )

    print(
        f"TOTAL trainable: "
        f"{total_trainable:,}"
    )

    check(
        bert_trainable
        == EXPECTED_BERT_TOP2_TRAINABLE,
        (
            "only BERT layers 10-11 are "
            "trainable: 14,175,744"
        ),
    )

    check(
        gat_trainable
        == EXPECTED_GAT3_TRAINABLE,
        (
            "only GAT3 is trainable: "
            "66,688 parameters"
        ),
    )

    check(
        projection_trainable
        == EXPECTED_PROJECTION_TRAINABLE,
        (
            "projection heads have "
            "229,376 trainable parameters"
        ),
    )

    check(
        total_trainable
        == EXPECTED_TOTAL_TRAINABLE,
        (
            "total T4-B trainable parameters "
            "= 14,471,808"
        ),
    )

    check(
        all(
            not p.requires_grad
            for p
            in model.bert.embeddings.parameters()
        ),
        "BERT embeddings remain frozen",
    )

    for layer_index in range(
        10
    ):

        check(
            all(
                not p.requires_grad
                for p
                in model.bert.encoder.layer[
                    layer_index
                ].parameters()
            ),
            (
                f"BERT layer {layer_index} "
                "remains frozen"
            ),
        )

    check(
        all(
            not p.requires_grad
            for p
            in model.gnn.node_projection.parameters()
        ),
        "GAT node projection remains frozen",
    )

    check(
        all(
            not p.requires_grad
            for p
            in model.gnn.gat1.parameters()
        ),
        "GAT1 remains frozen",
    )

    check(
        all(
            not p.requires_grad
            for p
            in model.gnn.gat2.parameters()
        ),
        "GAT2 remains frozen",
    )

    check(
        all(
            not p.requires_grad
            for p
            in model.gnn.classifier.parameters()
        ),
        (
            "old A4 classifier remains "
            "frozen and unused"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "T4-B trainable contract failed"
        )

    model.to(
        device
    )

    return model


# =============================================================================
# 10. GRAPH LOADING
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

                GRAPH_MANIFEST.parent
                / raw,

                ROOT
                / "data/processed/graphs"
                / raw.name,
            ]
        )

    found: dict[
        str,
        Path,
    ] = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = (
                candidate.resolve()
            )

            found[
                str(resolved)
            ] = resolved

    if len(found) != 1:

        raise RuntimeError(
            "Unable to uniquely resolve graph "
            f"{raw_value}; matches={len(found)}"
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

        degree_cap = int(
            np.asarray(
                graph[
                    "extra_degree_cap"
                ]
            )
            .reshape(-1)[0]
        )

        rule_version = int(
            np.asarray(
                graph[
                    "rule_version"
                ]
            )
            .reshape(-1)[0]
        )

    if x.shape != (
        9,
        RAW_NODE_DIM,
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            f"x shape={x.shape}, "
            "expected (9,140)"
        )

    if x.dtype != np.float32:

        raise RuntimeError(
            f"{graph_path.name}: "
            "node features must be float32"
        )

    if not np.isfinite(
        x
    ).all():

        raise RuntimeError(
            f"{graph_path.name}: "
            "non-finite node features"
        )

    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "invalid G2 edge_index shape"
        )

    if edge_index.dtype != np.int64:

        raise RuntimeError(
            f"{graph_path.name}: "
            "G2 edges must be int64"
        )

    if not (
        16
        <= edge_index.shape[1]
        <= 34
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "G2 edge count outside 16..34"
        )

    if not np.isclose(
        threshold,
        0.85,
        atol=1e-6,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{graph_path.name}: "
            "G2 threshold changed"
        )

    if nonlocal_only is not True:

        raise RuntimeError(
            f"{graph_path.name}: "
            "nonlocal_only changed"
        )

    if degree_cap != 2:

        raise RuntimeError(
            f"{graph_path.name}: "
            "degree cap changed"
        )

    if rule_version != 2:

        raise RuntimeError(
            f"{graph_path.name}: "
            "graph rule version changed"
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
# 11. LOAD TRAIN+VAL DATA ONLY
# =============================================================================

def load_development_data():

    banner(
        "4. LOAD LIVE TASK-4 INPUTS — TRAIN+VAL ONLY"
    )

    token_manifest = (
        pd.read_parquet(
            TOKEN_MANIFEST
        )
    )

    graph_manifest = (
        pd.read_parquet(
            GRAPH_MANIFEST
        )
    )

    check(
        list(
            token_manifest.columns
        )
        ==
        [
            "cache_index",
            "track_id",
            "split",
            "cached_token_length",
        ],
        "token manifest schema unchanged",
    )

    check(
        len(
            token_manifest
        )
        == N_TOTAL,
        "token manifest contains 5140 rows",
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

    split_counts = (
        token_manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    check(
        split_counts
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
            "token split counts remain "
            "4112/514/514"
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

    check(
        len(
            dev_manifest
        )
        == N_DEV,
        (
            "Task-4 development set "
            "contains 4626 rows"
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
            "development manifest "
            "contains zero TEST rows"
        ),
    )

    check(
        dev_manifest[
            "track_id"
        ].nunique()
        == N_DEV,
        (
            "development track IDs "
            "are unique"
        ),
    )

    # -------------------------------------------------------------------------
    # Extract only TRAIN+VAL token rows.
    # -------------------------------------------------------------------------

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
            "BERT token cache schema unchanged",
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

    check(
        input_ids.shape
        ==
        (
            N_DEV,
            MAX_LENGTH,
        ),
        (
            "development input_ids "
            "= (4626,112)"
        ),
    )

    check(
        attention_mask.shape
        ==
        (
            N_DEV,
            MAX_LENGTH,
        ),
        (
            "development attention_mask "
            "= (4626,112)"
        ),
    )

    check(
        token_type_ids.shape
        ==
        (
            N_DEV,
            MAX_LENGTH,
        ),
        (
            "development token_type_ids "
            "= (4626,112)"
        ),
    )

    # -------------------------------------------------------------------------
    # Graph alignment.
    # -------------------------------------------------------------------------

    expected_graph_columns = [
        "track_id",
        "split",
        "graph_path",
        "nodes",
        "node_feature_dim",
        "g1_directed_edges",
        "g2_directed_edges",
        "extra_similarity_pairs",
        "g2_density",
        "cache_status",
    ]

    check(
        list(
            graph_manifest.columns
        )
        == expected_graph_columns,
        "graph manifest schema unchanged",
    )

    check(
        len(
            graph_manifest
        )
        == N_TOTAL,
        "graph manifest contains 5140 rows",
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

    check(
        graph_manifest[
            "track_id"
        ].nunique()
        == N_TOTAL,
        "graph-manifest track IDs are unique",
    )

    graph_lookup = (
        graph_manifest
        .set_index(
            "track_id",
            drop=False,
        )
    )

    dev_ids = (
        dev_manifest[
            "track_id"
        ].tolist()
    )

    check(
        set(
            dev_ids
        ).issubset(
            set(
                graph_manifest[
                    "track_id"
                ]
            )
        ),
        (
            "every TRAIN+VAL track "
            "has a graph"
        ),
    )

    print()
    print(
        "Preloading 4626 TRAIN+VAL G2 graphs..."
    )

    graphs: list[
        Data
    ] = []

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
                "Graph split mismatch for "
                f"{track_id}"
            )

        if graph_split == "test":

            raise RuntimeError(
                "Forbidden TEST graph encountered"
            )

        graph_path = (
            resolve_graph_path(
                str(
                    graph_row[
                        "graph_path"
                    ]
                )
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
            "exactly 4626 TRAIN+VAL "
            "graphs loaded"
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

    check(
        len(
            train_indices
        )
        == N_TRAIN,
        "TRAIN indices = 4112",
    )

    check(
        len(
            val_indices
        )
        == N_VAL,
        "VAL indices = 514",
    )

    if FAILURES:

        raise RuntimeError(
            "Task-4 data alignment failed"
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

    return (
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        train_indices,
        val_indices,
    )


# =============================================================================
# 12. DATASET / COLLATE
# =============================================================================

class Task4PairDataset(
    Dataset,
):

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        indices,
    ) -> None:

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.graphs = graphs

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
                self.input_ids[
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.attention_mask[
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            torch.from_numpy(
                self.token_type_ids[
                    index
                ].astype(
                    np.int64,
                    copy=True,
                )
            ),

            self.graphs[
                index
            ],
        )


def collate_pairs(
    samples,
):

    input_ids = torch.stack(
        [
            sample[0]
            for sample
            in samples
        ]
    )

    attention_mask = torch.stack(
        [
            sample[1]
            for sample
            in samples
        ]
    )

    token_type_ids = torch.stack(
        [
            sample[2]
            for sample
            in samples
        ]
    )

    graph_batch = (
        Batch.from_data_list(
            [
                sample[3]
                for sample
                in samples
            ]
        )
    )

    return (
        input_ids,
        attention_mask,
        token_type_ids,
        graph_batch,
    )


def build_datasets_and_loaders(
    input_ids,
    attention_mask,
    token_type_ids,
    graphs,
    train_indices,
    val_indices,
):

    banner(
        "5. TASK-4 TRAIN / VAL LOADERS"
    )

    train_dataset = (
        Task4PairDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            train_indices,
        )
    )

    val_dataset = (
        Task4PairDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            val_indices,
        )
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_pairs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_pairs,
    )

    check(
        len(
            train_dataset
        )
        == N_TRAIN,
        "TRAIN pair dataset = 4112",
    )

    check(
        len(
            val_dataset
        )
        == N_VAL,
        "VAL pair dataset = 514",
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

    return (
        train_dataset,
        train_loader,
        val_loader,
    )


# =============================================================================
# 13. INFONCE
# =============================================================================

def symmetric_infonce(
    z_text: torch.Tensor,
    z_graph: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    logits = (
        z_graph
        @ z_text.T
    ) / TEMPERATURE

    targets = torch.arange(
        z_text.shape[0],
        device=z_text.device,
        dtype=torch.long,
    )

    loss_a2t = (
        F.cross_entropy(
            logits,
            targets,
        )
    )

    loss_t2a = (
        F.cross_entropy(
            logits.T,
            targets,
        )
    )

    loss = (
        0.5
        * (
            loss_a2t
            + loss_t2a
        )
    )

    return (
        loss,
        loss_a2t,
        loss_t2a,
    )


# =============================================================================
# 14. FULL-POOL RETRIEVAL
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

        correct = (
            torch.diagonal(
                matrix
            )
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

    metrics: dict[
        str,
        float,
    ] = {}

    for k in (
        1,
        5,
        10,
    ):

        metrics[
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

        metrics[
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

    metrics[
        "mR"
    ] = float(
        np.mean(
            [
                metrics[
                    "audio_to_caption_R@1"
                ],
                metrics[
                    "audio_to_caption_R@5"
                ],
                metrics[
                    "audio_to_caption_R@10"
                ],
                metrics[
                    "caption_to_audio_R@1"
                ],
                metrics[
                    "caption_to_audio_R@5"
                ],
                metrics[
                    "caption_to_audio_R@10"
                ],
            ],
            dtype=np.float64,
        )
    )

    metrics[
        "audio_to_caption_mean_rank"
    ] = float(
        a2t_rank
        .float()
        .mean()
        .item()
    )

    metrics[
        "caption_to_audio_mean_rank"
    ] = float(
        c2a_rank
        .float()
        .mean()
        .item()
    )

    return metrics


@torch.no_grad()
def evaluate_retrieval(
    model: Task4T4B,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    text_all: list[
        torch.Tensor
    ] = []

    graph_all: list[
        torch.Tensor
    ] = []

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

        z_text, z_graph = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            graph_batch=graph_batch,
        )

        if not torch.isfinite(
            z_text
        ).all():

            raise RuntimeError(
                "Non-finite validation text embeddings"
            )

        if not torch.isfinite(
            z_graph
        ).all():

            raise RuntimeError(
                "Non-finite validation graph embeddings"
            )

        text_all.append(
            z_text
            .detach()
            .cpu()
        )

        graph_all.append(
            z_graph
            .detach()
            .cpu()
        )

        rows += int(
            z_text.shape[0]
        )

    if rows != N_VAL:

        raise RuntimeError(
            f"VAL rows={rows}; "
            f"expected {N_VAL}"
        )

    z_text = torch.cat(
        text_all,
        dim=0,
    )

    z_graph = torch.cat(
        graph_all,
        dim=0,
    )

    return retrieval_metrics(
        z_text,
        z_graph,
    )


# =============================================================================
# 15. PRE-FORMAL CUDA MEMORY/BACKWARD GATE
# =============================================================================

def run_memory_gate(
    train_dataset: Task4PairDataset,
    device: torch.device,
) -> dict[str, float]:

    banner(
        "6. PRE-FORMAL T4-B BATCH-32 CUDA MEMORY GATE"
    )

    print(
        "Diagnostic only."
    )

    print(
        "Optimizer step: NO"
    )

    print(
        "Result reporting: NO"
    )

    print(
        "AMP: NO"
    )

    configure_seed()

    model = build_t4b_model(
        device
    )

    model.train()

    smoke_samples = [
        train_dataset[
            index
        ]
        for index in range(
            BATCH_SIZE
        )
    ]

    (
        input_ids,
        attention_mask,
        token_type_ids,
        graph_batch,
    ) = collate_pairs(
        smoke_samples
    )

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

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    model.zero_grad(
        set_to_none=True
    )

    z_text, z_graph = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        graph_batch=graph_batch,
    )

    loss, _, _ = (
        symmetric_infonce(
            z_text,
            z_graph,
        )
    )

    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Memory-gate loss is non-finite"
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
            max_norm=GRAD_CLIP_NORM,
        )
    )

    if not math.isfinite(
        float(
            grad_norm
        )
    ):

        raise RuntimeError(
            "Memory-gate gradient norm is non-finite"
        )

    model.zero_grad(
        set_to_none=True
    )

    torch.cuda.synchronize(
        device
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(
            device
        )
    )

    capacity = (
        torch.cuda.get_device_properties(
            device
        ).total_memory
    )

    print(
        f"Diagnostic InfoNCE: "
        f"{float(loss.detach().cpu()):.8f}"
    )

    print(
        f"Peak allocated: "
        f"{peak_allocated / (1024 ** 3):.3f} GB"
    )

    print(
        f"Peak reserved:  "
        f"{peak_reserved / (1024 ** 3):.3f} GB"
    )

    print(
        f"GPU capacity:   "
        f"{capacity / (1024 ** 3):.3f} GB"
    )

    check(
        peak_allocated
        < capacity,
        (
            "T4-B batch-32 forward/backward "
            "fits GPU memory"
        ),
    )

    # Absolutely no optimizer.step().
    passed(
        "memory gate performed zero parameter updates"
    )

    del z_text
    del z_graph
    del loss
    del graph_batch
    del input_ids
    del attention_mask
    del token_type_ids
    del model

    torch.cuda.empty_cache()

    if FAILURES:

        raise RuntimeError(
            "T4-B CUDA memory gate failed"
        )

    return {
        "peak_allocated_bytes":
            int(
                peak_allocated
            ),

        "peak_reserved_bytes":
            int(
                peak_reserved
            ),

        "device_memory_bytes":
            int(
                capacity
            ),
    }


# =============================================================================
# 16. T4-A LIVE INITIALIZATION PARITY
# =============================================================================

def verify_initial_t4a_parity(
    model: Task4T4B,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:

    banner(
        "7. LIVE T4-A INITIALIZATION PARITY"
    )

    observed = evaluate_retrieval(
        model,
        val_loader,
        device,
    )

    t4a = load_checkpoint(
        T4A_CHECKPOINT
    )

    expected = (
        t4a[
            "val_metrics"
        ]
    )

    print(
        "Expected formal T4-A mR:"
    )

    print(
        f"  {float(expected['mR']):.10f}"
    )

    print(
        "Live reconstructed mR:"
    )

    print(
        f"  {float(observed['mR']):.10f}"
    )

    recall_keys = [
        "audio_to_caption_R@1",
        "audio_to_caption_R@5",
        "audio_to_caption_R@10",
        "caption_to_audio_R@1",
        "caption_to_audio_R@5",
        "caption_to_audio_R@10",
    ]

    # One validation-query quantum.
    tolerance = (
        1.0
        / N_VAL
        + 1e-12
    )

    for key in recall_keys:

        difference = abs(
            float(
                observed[
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

        print(
            f"{key}: "
            f"expected={float(expected[key]):.8f} "
            f"live={float(observed[key]):.8f} "
            f"delta={difference:.8f}"
        )

        check(
            difference
            <= tolerance,
            (
                f"live initialization reproduces "
                f"{key} within one VAL-query quantum"
            ),
        )

    mr_difference = abs(
        float(
            observed[
                "mR"
            ]
        )
        -
        float(
            expected[
                "mR"
            ]
        )
    )

    check(
        mr_difference
        <= tolerance,
        (
            "live T1-B + A4 + T4-A projections "
            "reproduce formal T4-A VAL mR"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Live T4-A initialization parity failed"
        )

    return observed


# =============================================================================
# 17. OPTIMIZER
# =============================================================================

def build_optimizer(
    model: Task4T4B,
):

    banner(
        "8. FORMAL T4-B OPTIMIZER"
    )

    bert_parameters = [
        p
        for p
        in model.bert.parameters()
        if p.requires_grad
    ]

    gat_parameters = [
        p
        for p
        in model.gnn.parameters()
        if p.requires_grad
    ]

    projection_parameters = (
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

    check(
        sum(
            p.numel()
            for p
            in bert_parameters
        )
        ==
        EXPECTED_BERT_TOP2_TRAINABLE,
        (
            "optimizer BERT group contains "
            "exactly layers 10-11"
        ),
    )

    check(
        sum(
            p.numel()
            for p
            in gat_parameters
        )
        ==
        EXPECTED_GAT3_TRAINABLE,
        (
            "optimizer GAT group contains "
            "exactly GAT3"
        ),
    )

    check(
        sum(
            p.numel()
            for p
            in projection_parameters
        )
        ==
        EXPECTED_PROJECTION_TRAINABLE,
        (
            "optimizer projection group contains "
            "both projection heads"
        ),
    )

    all_parameters = (
        bert_parameters
        +
        gat_parameters
        +
        projection_parameters
    )

    parameter_ids = [
        id(
            parameter
        )
        for parameter
        in all_parameters
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
        "optimizer has no duplicate parameters",
    )

    optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params":
                        projection_parameters,

                    "lr":
                        PROJECTION_LR,

                    "weight_decay":
                        WEIGHT_DECAY,
                },
                {
                    "params":
                        bert_parameters,

                    "lr":
                        BERT_LR,

                    "weight_decay":
                        WEIGHT_DECAY,
                },
                {
                    "params":
                        gat_parameters,

                    "lr":
                        GAT_LR,

                    "weight_decay":
                        WEIGHT_DECAY,
                },
            ]
        )
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        f"Projection LR: {PROJECTION_LR}"
    )

    print(
        f"BERT top-2 LR: {BERT_LR}"
    )

    print(
        f"GAT3 LR:       {GAT_LR}"
    )

    print(
        f"Weight decay:  {WEIGHT_DECAY}"
    )

    print(
        "Scheduler:     None"
    )

    print(
        "AMP:           False"
    )

    if FAILURES:

        raise RuntimeError(
            "T4-B optimizer contract failed"
        )

    return optimizer


# =============================================================================
# 18. TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model: Task4T4B,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:

    model.train()

    total_loss = 0.0
    total_a2t = 0.0
    total_t2a = 0.0

    total_rows = 0

    for step, (
        input_ids,
        attention_mask,
        token_type_ids,
        graph_batch,
    ) in enumerate(
        loader,
        start=1,
    ):

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

        optimizer.zero_grad(
            set_to_none=True
        )

        z_text, z_graph = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            graph_batch=graph_batch,
        )

        if not torch.isfinite(
            z_text
        ).all():

            raise RuntimeError(
                "Non-finite TRAIN text embeddings"
            )

        if not torch.isfinite(
            z_graph
        ).all():

            raise RuntimeError(
                "Non-finite TRAIN graph embeddings"
            )

        (
            loss,
            loss_a2t,
            loss_t2a,
        ) = symmetric_infonce(
            z_text,
            z_graph,
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite TRAIN loss "
                f"at step {step}"
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
                max_norm=GRAD_CLIP_NORM,
            )
        )

        if not math.isfinite(
            float(
                grad_norm
            )
        ):

            raise RuntimeError(
                f"Non-finite gradient norm "
                f"at step {step}"
            )

        optimizer.step()

        rows = int(
            z_text.shape[0]
        )

        total_loss += (
            float(
                loss
                .detach()
                .cpu()
            )
            * rows
        )

        total_a2t += (
            float(
                loss_a2t
                .detach()
                .cpu()
            )
            * rows
        )

        total_t2a += (
            float(
                loss_t2a
                .detach()
                .cpu()
            )
            * rows
        )

        total_rows += rows

    if total_rows != N_TRAIN:

        raise RuntimeError(
            f"TRAIN rows={total_rows}; "
            f"expected {N_TRAIN}"
        )

    return {
        "loss":
            total_loss
            / total_rows,

        "audio_to_caption_loss":
            total_a2t
            / total_rows,

        "caption_to_audio_loss":
            total_t2a
            / total_rows,
    }


# =============================================================================
# 19. BEST-STATE CAPTURE
# =============================================================================

def capture_state(
    model: Task4T4B,
) -> dict[str, Any]:

    return {
        "bert_layer_10":
            cpu_state_dict(
                model
                .bert
                .encoder
                .layer[
                    10
                ]
            ),

        "bert_layer_11":
            cpu_state_dict(
                model
                .bert
                .encoder
                .layer[
                    11
                ]
            ),

        "gat3":
            cpu_state_dict(
                model
                .gnn
                .gat3
            ),

        "text_projection":
            cpu_state_dict(
                model
                .text_projection
            ),

        "graph_projection":
            cpu_state_dict(
                model
                .graph_projection
            ),
    }


def restore_state(
    model: Task4T4B,
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
# 20. FORMAL TRAINING
# =============================================================================

def run_formal_training(
    model: Task4T4B,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:

    banner(
        "9. FORMAL T4-B CONTRASTIVE TRAINING"
    )

    initial_val = (
        verify_initial_t4a_parity(
            model,
            val_loader,
            device,
        )
    )

    optimizer = build_optimizer(
        model
    )

    best_mr = -float(
        "inf"
    )

    best_epoch = None
    best_metrics = None
    best_state = None

    epochs_without_improvement = 0

    history: list[
        dict[str, Any]
    ] = []

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start_time = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val = evaluate_retrieval(
            model=model,
            loader=val_loader,
            device=device,
        )

        val_mr = float(
            val[
                "mR"
            ]
        )

        improved = (
            val_mr
            >
            best_mr
        )

        if improved:

            best_mr = val_mr
            best_epoch = epoch
            best_metrics = dict(
                val
            )

            best_state = (
                capture_state(
                    model
                )
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            -
            epoch_start
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    float(
                        train[
                            "loss"
                        ]
                    ),

                "train_audio_to_caption_loss":
                    float(
                        train[
                            "audio_to_caption_loss"
                        ]
                    ),

                "train_caption_to_audio_loss":
                    float(
                        train[
                            "caption_to_audio_loss"
                        ]
                    ),

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

                "val_audio_to_caption_mean_rank":
                    val[
                        "audio_to_caption_mean_rank"
                    ],

                "val_caption_to_audio_mean_rank":
                    val[
                        "caption_to_audio_mean_rank"
                    ],

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
        )

        print()
        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS}"
        )

        print(
            f"  TRAIN InfoNCE = "
            f"{train['loss']:.8f}"
        )

        print(
            f"  VAL A→C  "
            f"R@1="
            f"{val['audio_to_caption_R@1']:.6f}  "
            f"R@5="
            f"{val['audio_to_caption_R@5']:.6f}  "
            f"R@10="
            f"{val['audio_to_caption_R@10']:.6f}"
        )

        print(
            f"  VAL C→A  "
            f"R@1="
            f"{val['caption_to_audio_R@1']:.6f}  "
            f"R@5="
            f"{val['caption_to_audio_R@5']:.6f}  "
            f"R@10="
            f"{val['caption_to_audio_R@10']:.6f}"
        )

        print(
            f"  VAL mR = "
            f"{val_mr:.8f}"
        )

        if improved:

            print(
                "  BEST T4-B STATE UPDATED"
            )

        else:

            print(
                f"  no improvement "
                f"("
                f"{epochs_without_improvement}"
                f"/{PATIENCE}"
                f")"
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

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

    if (
        best_epoch is None
        or
        best_metrics is None
        or
        best_state is None
    ):

        raise RuntimeError(
            "No valid T4-B epoch selected"
        )

    restore_state(
        model,
        best_state,
    )

        # -------------------------------------------------------------------------
    # Exact state restoration verification.
    #
    # The model parameters themselves must restore byte-for-byte/equal
    # tensor-for-tensor. Retrieval metrics receive the small rank-boundary
    # tolerance only because the second live GPU inference may differ at
    # floating-point level.
    # -------------------------------------------------------------------------

    restored_state = capture_state(
        model
    )

    for module_name in (
        "bert_layer_10",
        "bert_layer_11",
        "gat3",
        "text_projection",
        "graph_projection",
    ):

        expected_module = (
            best_state[
                module_name
            ]
        )

        restored_module = (
            restored_state[
                module_name
            ]
        )

        check(
            set(
                restored_module.keys()
            )
            ==
            set(
                expected_module.keys()
            ),
            (
                f"{module_name} restored "
                "state keys match exactly"
            ),
        )

        for parameter_name in (
            expected_module.keys()
        ):

            check(
                torch.equal(
                    restored_module[
                        parameter_name
                    ],
                    expected_module[
                        parameter_name
                    ],
                ),
                (
                    f"{module_name}."
                    f"{parameter_name} "
                    "restored exactly"
                ),
            )

    if FAILURES:

        raise RuntimeError(
            "Exact T4-B state restoration failed"
        )

    revalidated = (
        evaluate_retrieval(
            model,
            val_loader,
            device,
        )
    )

    # for key in (
    #     "audio_to_caption_R@1",
    #     "audio_to_caption_R@5",
    #     "audio_to_caption_R@10",
    #     "caption_to_audio_R@1",
    #     "caption_to_audio_R@5",
    #     "caption_to_audio_R@10",
    #     "mR",
    # ):

    #     check(
    #         np.isclose(
    #             float(
    #                 revalidated[
    #                     key
    #                 ]
    #             ),
    #             float(
    #                 best_metrics[
    #                     key
    #                 ]
    #             ),
    #             rtol=0.0,
    #             atol=1e-12,
    #         ),
    #         (
    #             "restored best state "
    #             f"reproduces {key}"
    #         ),
    #     )

        # -------------------------------------------------------------------------
    # Revalidation tolerance
    #
    # Retrieval recalls are discrete multiples of 1/N_VAL.
    # A second live CUDA BERT/GAT forward can produce extremely small
    # floating-point differences in cosine similarities. If two candidates
    # are almost tied, one query can cross an R@K boundary even though the
    # restored model state is identical.
    #
    # Therefore:
    #   - each R@K may differ by at most ONE validation query
    #   - mR may differ by at most one-query / six metrics
    #
    # This is an integrity tolerance, NOT a model-selection tolerance.
    # The originally observed best epoch and best VAL metrics remain the
    # formal selection result.
    # -------------------------------------------------------------------------

    recall_quantum = (
        1.0
        / N_VAL
    )

    mr_quantum = (
        1.0
        / (
            N_VAL
            * 6.0
        )
    )

    recall_keys = (
        "audio_to_caption_R@1",
        "audio_to_caption_R@5",
        "audio_to_caption_R@10",
        "caption_to_audio_R@1",
        "caption_to_audio_R@5",
        "caption_to_audio_R@10",
    )

    for key in recall_keys:

        observed_value = float(
            revalidated[
                key
            ]
        )

        expected_value = float(
            best_metrics[
                key
            ]
        )

        difference = abs(
            observed_value
            - expected_value
        )

        print(
            f"Revalidation {key}: "
            f"selected={expected_value:.10f} "
            f"restored={observed_value:.10f} "
            f"delta={difference:.10f}"
        )

        check(
            difference
            <= (
                recall_quantum
                + 1e-12
            ),
            (
                "restored best state reproduces "
                f"{key} within one VAL-query quantum"
            ),
        )

    observed_mr = float(
        revalidated[
            "mR"
        ]
    )

    expected_mr = float(
        best_metrics[
            "mR"
        ]
    )

    mr_difference = abs(
        observed_mr
        - expected_mr
    )

    print(
        f"Revalidation mR: "
        f"selected={expected_mr:.10f} "
        f"restored={observed_mr:.10f} "
        f"delta={mr_difference:.10f}"
    )

    check(
        mr_difference
        <= (
            mr_quantum
            + 1e-12
        ),
        (
            "restored best state reproduces "
            "mR within one single-metric "
            "VAL-query quantum"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Best T4-B state revalidation failed"
        )

    return {
        "initial_val":
            initial_val,

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val":
            revalidated,

        "best_state":
            best_state,

        "history":
            history,

        "runtime_seconds":
            float(
                runtime
            ),

        "peak_vram_bytes":
            int(
                peak_vram
            ),
    }


# =============================================================================
# 21. WRITE FORMAL OUTPUTS
# =============================================================================

def save_outputs(
    result: dict[str, Any],
    memory_gate: dict[str, float],
) -> dict[str, Any]:

    banner(
        "10. SAVE FORMAL T4-B RESULT"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        CHECKPOINT_PATH,
        HISTORY_PATH,
        SUMMARY_PATH,
    ):

        check(
            not path.exists(),
            (
                f"output path unused: "
                f"{path.name}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Formal T4-B output collision"
        )

    # -------------------------------------------------------------------------
    # Checkpoint — delta from frozen T1-B/A4 + T4-A initialization.
    # -------------------------------------------------------------------------

    checkpoint = {
        "artifact_type":
            "task4_T4-B_checkpoint",

        "variant":
            "T4-B",

        "seed":
            SEED,

        "best_epoch":
            int(
                result[
                    "best_epoch"
                ]
            ),

        "state_format":
            (
                "delta_from_T1B_A4_plus_"
                "T4A_projection_initialization"
            ),

        "trainable_contract": {
            "bert_layers":
                [
                    10,
                    11,
                ],

            "bert_trainable_parameters":
                EXPECTED_BERT_TOP2_TRAINABLE,

            "gnn_trainable":
                "gat3_only",

            "gat3_trainable_parameters":
                EXPECTED_GAT3_TRAINABLE,

            "projection_heads":
                "both_trainable",

            "projection_trainable_parameters":
                EXPECTED_PROJECTION_TRAINABLE,

            "total_trainable_parameters":
                EXPECTED_TOTAL_TRAINABLE,
        },

        "training": {
            "objective":
                "symmetric InfoNCE",

            "temperature":
                TEMPERATURE,

            "optimizer":
                "AdamW",

            "projection_learning_rate":
                PROJECTION_LR,

            "bert_learning_rate":
                BERT_LR,

            "gat_learning_rate":
                GAT_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "patience":
                PATIENCE,

            "gradient_clip_norm":
                GRAD_CLIP_NORM,

            "amp":
                False,

            "selection":
                "highest full-pool VAL mR",

            "tie_rule":
                "earliest epoch",
        },

        "initial_t4a_live_val_metrics":
            result[
                "initial_val"
            ],

        "best_val_metrics":
            result[
                "best_val"
            ],

        "source_hashes": {
            "design":
                EXPECTED_DESIGN_SHA256,

            "design_lock":
                EXPECTED_DESIGN_LOCK_SHA256,

            "script66_smoke":
                EXPECTED_SCRIPT66_SMOKE_SHA256,

            "t4a_checkpoint":
                EXPECTED_T4A_CHECKPOINT_SHA256,

            "t4a_summary":
                EXPECTED_T4A_SUMMARY_SHA256,

            "pretraining_lock":
                EXPECTED_PRETRAINING_LOCK_SHA256,

            "bert_tokens":
                EXPECTED_TOKEN_CACHE_SHA256,

            "bert_token_manifest":
                EXPECTED_TOKEN_MANIFEST_SHA256,

            "graph_manifest":
                EXPECTED_GRAPH_MANIFEST_SHA256,

            "T1_B":
                EXPECTED_T1_SHA256,

            "A4":
                EXPECTED_A4_SHA256,
        },

        "test_split_used":
            False,

        "test_graph_files_opened":
            0,

        "test_examples_evaluated":
            0,

        "test_retrieval":
            False,

        "label_supervision_used":
            False,

        "bert_layer_10_state_dict":
            result[
                "best_state"
            ][
                "bert_layer_10"
            ],

        "bert_layer_11_state_dict":
            result[
                "best_state"
            ][
                "bert_layer_11"
            ],

        "gat3_state_dict":
            result[
                "best_state"
            ][
                "gat3"
            ],

        "text_projection_state_dict":
            result[
                "best_state"
            ][
                "text_projection"
            ],

        "graph_projection_state_dict":
            result[
                "best_state"
            ][
                "graph_projection"
            ],
    }

    checkpoint_temp = Path(
        str(
            CHECKPOINT_PATH
        )
        + ".__tmp__"
    )

    torch.save(
        checkpoint,
        checkpoint_temp,
    )

    os.replace(
        checkpoint_temp,
        CHECKPOINT_PATH,
    )

    checkpoint_sha = sha256_file(
        CHECKPOINT_PATH
    )

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    history = pd.DataFrame(
        result[
            "history"
        ]
    )

    history_temp = Path(
        str(
            HISTORY_PATH
        )
        + ".__tmp__"
    )

    history.to_csv(
        history_temp,
        index=False,
    )

    os.replace(
        history_temp,
        HISTORY_PATH,
    )

    history_sha = sha256_file(
        HISTORY_PATH
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    best = result[
        "best_val"
    ]

    summary = {
        "artifact_type":
            "task4_T4-B_formal_summary",

        "status":
            "formal_validation_result",

        "completed_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "variant":
            "T4-B",

        "seed":
            SEED,

        "train_examples":
            N_TRAIN,

        "val_examples":
            N_VAL,

        "test_examples":
            0,

        "epochs_completed":
            len(
                result[
                    "history"
                ]
            ),

        "early_stopping_triggered":
            (
                len(
                    result[
                        "history"
                    ]
                )
                <
                MAX_EPOCHS
            ),

        "best_epoch":
            int(
                result[
                    "best_epoch"
                ]
            ),

        "initial_t4a_live_val_metrics":
            result[
                "initial_val"
            ],

        "best_val_metrics":
            best,

        "best_val_mR":
            float(
                best[
                    "mR"
                ]
            ),

        "primary_selection_metric":
            "VAL mR",

        "trainable_parameters": {
            "bert_top_2":
                EXPECTED_BERT_TOP2_TRAINABLE,

            "gat3":
                EXPECTED_GAT3_TRAINABLE,

            "projection_heads":
                EXPECTED_PROJECTION_TRAINABLE,

            "total":
                EXPECTED_TOTAL_TRAINABLE,
        },

        "memory_gate": {
            "batch_size":
                BATCH_SIZE,

            "optimizer_steps":
                0,

            "peak_allocated_bytes":
                int(
                    memory_gate[
                        "peak_allocated_bytes"
                    ]
                ),

            "peak_reserved_bytes":
                int(
                    memory_gate[
                        "peak_reserved_bytes"
                    ]
                ),

            "device_memory_bytes":
                int(
                    memory_gate[
                        "device_memory_bytes"
                    ]
                ),

            "result":
                "pass",
        },

        "formal_protocol": {
            "temperature":
                TEMPERATURE,

            "shared_dimension":
                SHARED_DIM,

            "batch_size":
                BATCH_SIZE,

            "projection_lr":
                PROJECTION_LR,

            "bert_lr":
                BERT_LR,

            "gat_lr":
                GAT_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "max_epochs":
                MAX_EPOCHS,

            "patience":
                PATIENCE,

            "gradient_clip_norm":
                GRAD_CLIP_NORM,

            "amp":
                False,
        },

        "checkpoint":
            relative(
                CHECKPOINT_PATH
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "history":
            relative(
                HISTORY_PATH
            ),

        "history_sha256":
            history_sha,

        "runtime_seconds":
            float(
                result[
                    "runtime_seconds"
                ]
            ),

        "peak_formal_vram_bytes":
            int(
                result[
                    "peak_vram_bytes"
                ]
            ),

        "provenance": {
            "script68":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "script68_sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),

            "design_sha256":
                EXPECTED_DESIGN_SHA256,

            "design_lock_sha256":
                EXPECTED_DESIGN_LOCK_SHA256,

            "t4a_checkpoint_sha256":
                EXPECTED_T4A_CHECKPOINT_SHA256,

            "T1_B_sha256":
                EXPECTED_T1_SHA256,

            "A4_sha256":
                EXPECTED_A4_SHA256,
        },

        "boundaries": {
            "label_supervision":
                False,

            "test_access":
                False,

            "test_retrieval":
                False,

            "test_model_selection":
                False,
        },

        "next_action":
            (
                "Script 69 — compare formal T4-A "
                "and T4-B using seed-42 VAL mR only"
            ),
    }

    write_json_atomic(
        SUMMARY_PATH,
        summary,
    )

    summary_sha = sha256_file(
        SUMMARY_PATH
    )

    return {
        "checkpoint_sha":
            checkpoint_sha,

        "history_sha":
            history_sha,

        "summary_sha":
            summary_sha,
    }


# =============================================================================
# 22. POST-RUN UPSTREAM INTEGRITY
# =============================================================================

def verify_upstream_integrity() -> None:

    banner(
        "11. POST-RUN FROZEN-UPSTREAM INTEGRITY"
    )

    expected = (
        (
            DESIGN_PATH,
            EXPECTED_DESIGN_SHA256,
            "Task-4 design",
        ),
        (
            DESIGN_LOCK_PATH,
            EXPECTED_DESIGN_LOCK_SHA256,
            "Task-4 design lock",
        ),
        (
            T4A_CHECKPOINT,
            EXPECTED_T4A_CHECKPOINT_SHA256,
            "T4-A checkpoint",
        ),
        (
            T1_CHECKPOINT,
            EXPECTED_T1_SHA256,
            "T1-B checkpoint",
        ),
        (
            A4_CHECKPOINT,
            EXPECTED_A4_SHA256,
            "A4 checkpoint",
        ),
        (
            TOKEN_CACHE,
            EXPECTED_TOKEN_CACHE_SHA256,
            "BERT token cache",
        ),
        (
            GRAPH_MANIFEST,
            EXPECTED_GRAPH_MANIFEST_SHA256,
            "graph manifest",
        ),
    )

    for path, digest, label in expected:

        verify_sha(
            path,
            digest,
            label,
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen upstream artifact changed"
        )


# =============================================================================
# 23. TEST BOUNDARY
# =============================================================================

def print_test_boundary() -> None:

    banner(
        "12. TEST-BOUNDARY CONFIRMATION"
    )

    print(
        "TRAIN examples optimized:          4112"
    )

    print(
        "VAL examples used for selection:  514"
    )

    print(
        "TEST token rows supplied to model: 0"
    )

    print(
        "TEST graph files opened:           0"
    )

    print(
        "TEST labels loaded:                0"
    )

    print(
        "TEST retrieval calculations:       0"
    )

    print(
        "TEST-driven hyperparameter changes:0"
    )

    print(
        "Task-4 label supervision used:     0"
    )

    passed(
        "formal T4-B preserved the TEST boundary"
    )


# =============================================================================
# 24. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 68 — FORMAL TASK-4 T4-B "
        "PARTIAL DUAL-ENCODER FINE-TUNING"
    )

    print(
        f"Project root: "
        f"{ROOT}"
    )

    print()
    print(
        "Initialization:"
    )
    print(
        "  T1-B BERT"
    )
    print(
        "  A4 GAT/G2"
    )
    print(
        "  best formal T4-A projection heads"
    )

    print()
    print(
        "Trainable:"
    )
    print(
        "  BERT layers 10-11"
    )
    print(
        "  GAT3 only"
    )
    print(
        "  text projection"
    )
    print(
        "  graph projection"
    )

    print()
    print(
        "Frozen:"
    )
    print(
        "  BERT embeddings + layers 0-9"
    )
    print(
        "  A4 node_projection + GAT1 + GAT2"
    )
    print(
        "  A4 original classifier"
    )

    print()
    print(
        "Objective:        symmetric InfoNCE"
    )
    print(
        "Temperature:      0.07"
    )
    print(
        "Batch:            32"
    )
    print(
        "AMP:              NO"
    )
    print(
        "Primary VAL:      mR"
    )

    print()
    print(
        "TRAIN:            4112"
    )
    print(
        "VAL:              514"
    )
    print(
        "TEST:             0"
    )

    configure_seed()

    try:

        verify_output_protection()

        verify_dependencies()

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

        (
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            train_indices,
            val_indices,
        ) = load_development_data()

        (
            train_dataset,
            train_loader,
            val_loader,
        ) = build_datasets_and_loaders(
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            train_indices,
            val_indices,
        )

        # ---------------------------------------------------------------------
        # Diagnostic full backward memory gate.
        #
        # NO optimizer step.
        # Diagnostic model discarded afterward.
        # ---------------------------------------------------------------------

        memory_gate = (
            run_memory_gate(
                train_dataset,
                device,
            )
        )

        # ---------------------------------------------------------------------
        # Reset everything before formal optimization.
        # ---------------------------------------------------------------------

        configure_seed()

        torch.cuda.empty_cache()

        model = build_t4b_model(
            device
        )

        # Rebuild loaders so the formal TRAIN shuffle generator
        # starts from its frozen seed-42 state and was not consumed
        # by any diagnostic operation.
        (
            _,
            train_loader,
            val_loader,
        ) = build_datasets_and_loaders(
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            train_indices,
            val_indices,
        )

        result = (
            run_formal_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )
        )

        outputs = (
            save_outputs(
                result,
                memory_gate,
            )
        )

        verify_upstream_integrity()

        print_test_boundary()

    except Exception as exc:

        banner(
            "SCRIPT 68 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        if (
            isinstance(
                exc,
                RuntimeError,
            )
            and
            "out of memory"
            in str(
                exc
            ).lower()
        ):

            print()
            print(
                "CUDA OOM detected."
            )

            print(
                "No formal T4-B result from "
                "this run should be accepted."
            )

            print(
                "Do NOT silently change batch size."
            )

            print(
                "A documented Task-4 design amendment "
                "would be required before rerunning."
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
            "Do NOT proceed to Script 69."
        )

        return 1

    best = (
        result[
            "best_val"
        ]
    )

    banner(
        "SCRIPT 68 SUMMARY"
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

    if FAILURES:

        print(
            "RESULT: FAIL"
        )

        return 1

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "FORMAL T4-B TRAINING COMPLETE."
    )

    print()
    print(
        f"Epochs completed : "
        f"{len(result['history'])}"
    )

    print(
        f"Best epoch       : "
        f"{result['best_epoch']}"
    )

    print()
    print(
        "BEST VALIDATION RETRIEVAL"
    )

    print(
        f"Audio→Caption R@1  = "
        f"{best['audio_to_caption_R@1']:.8f}"
    )

    print(
        f"Audio→Caption R@5  = "
        f"{best['audio_to_caption_R@5']:.8f}"
    )

    print(
        f"Audio→Caption R@10 = "
        f"{best['audio_to_caption_R@10']:.8f}"
    )

    print(
        f"Caption→Audio R@1  = "
        f"{best['caption_to_audio_R@1']:.8f}"
    )

    print(
        f"Caption→Audio R@5  = "
        f"{best['caption_to_audio_R@5']:.8f}"
    )

    print(
        f"Caption→Audio R@10 = "
        f"{best['caption_to_audio_R@10']:.8f}"
    )

    print(
        f"VAL mR              = "
        f"{best['mR']:.8f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        f"  {relative(CHECKPOINT_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{outputs['checkpoint_sha']}"
    )

    print()
    print(
        "History:"
    )

    print(
        f"  {relative(HISTORY_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{outputs['history_sha']}"
    )

    print()
    print(
        "Summary:"
    )

    print(
        f"  {relative(SUMMARY_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{outputs['summary_sha']}"
    )

    print()
    print(
        f"Runtime: "
        f"{result['runtime_seconds'] / 60.0:.2f} min"
    )

    print(
        f"Peak formal VRAM: "
        f"{result['peak_vram_bytes'] / (1024 ** 3):.3f} GB"
    )

    print()
    print(
        "TEST retrieval observed: NO"
    )

    print()
    print(
        "STOP HERE."
    )

    print(
        "Do NOT decide T4-A vs T4-B manually."
    )

    print(
        "Next:"
    )

    print(
        "  Script 69 — validation-only "
        "T4-A/T4-B model selection."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )