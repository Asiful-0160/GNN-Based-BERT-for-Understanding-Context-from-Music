#!/usr/bin/env python3
"""
Script 54 — Formal Task-3 M4 Cross-Attention Phase-B Joint Fine-Tuning
======================================================================

Starting point:
    selected T1-B BERT
    selected A4 GAT/G2
    best M4 Phase-A fusion checkpoint

Trainable:
    BERT encoder layers 10-11
    complete A4 GNN encoder
    complete M4 fusion module

Frozen / unused:
    BERT embeddings + layers 0-9
    BERT pooler
    original A4 classifier

Pipeline-v6 Phase B:
    LR = 5e-6
    epochs = 5
    batch = 16
    grad accumulation = 1 unless OOM
    BERT WD = 1e-2
    GNN/Fusion WD = 1e-4

Selection:
    highest VAL macro mean per-label Average Precision

TEST:
    labels = 0
    graph files opened = 0
    inference = 0
    metrics = 0
    threshold tuning = 0
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
import torch_geometric
import transformers

from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import BertConfig, BertModel


# =============================================================================
# PATHS
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

GRAPH_MANIFEST = (
    ROOT / "data/processed/graph_manifest.parquet"
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

M4_PHASEA_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M4_phaseA_seed42_20260910T131031Z_best.pt"
)

TASK3_DESIGN = (
    ROOT
    / "results/runs/task3/"
    "task3_phaseA_design.json"
)

PHASEA_RESULTS_FREEZE = (
    ROOT
    / "results/runs/task3/"
    "task3_phaseA_results_freeze.json"
)

M3_PHASEB_SUMMARY = (
    ROOT
    / "results/runs/task3/"
    "task3_M3_phaseB_seed42_20260910T133125Z_summary.json"
)

MODEL_DIR = ROOT / "models/task3_candidates"
RUN_DIR = ROOT / "results/runs/task3"


# =============================================================================
# ACCEPTED HASHES
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

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6ad0492bf90eccc929cadc1aad1da9e158"
)

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e"
)

EXPECTED_M4_PHASEA_SHA256 = (
    "ea210e30a932c124ee07b6a5fc90d068e0201152301998ed1eeb67559dd2e112"
)

EXPECTED_TASK3_DESIGN_SHA256 = (
    "0a9d50413b65aeed58d02d6a2dff70d99363ae25fee115d30bc9749f54e1c9ef"
)

EXPECTED_PHASEA_FREEZE_SHA256 = (
    "f063288047d3d04ce60713180a4b5c5746d8c740080ea78b3cf899702db987be"
)

EXPECTED_M3_PHASEB_SUMMARY_SHA256 = (
    "8fae01740e3bb9ae3e77a107fe4a50e47c11d260b8c514d23e5bb8775ad55dbd"
)


# =============================================================================
# CONTRACT
# =============================================================================

N_TOTAL = 5140
N_TRAIN = 4112
N_VAL = 514
N_TEST = 514
N_DEV = 4626

NUM_LABELS = 20

MAX_LENGTH = 112

SEED = 42

LR = 5e-6

BERT_WEIGHT_DECAY = 1e-2
GNN_FUSION_WEIGHT_DECAY = 1e-4

BATCH_SIZE = 16
EPOCHS = 5
GRAD_ACCUMULATION_STEPS = 1

BERT_DIM = 768
GAT_DIM = 128
SHARED_DIM = 256
ATTENTION_HEADS = 4

M1_VAL_MACRO_AP = 0.8405750022914198
M4_PHASEA_VAL_MACRO_AP = 0.822898704197

EXPECTED_BERT_TRAINABLE = 14_175_744
EXPECTED_GNN_TRAINABLE = 101_056
EXPECTED_M4_TRAINABLE = 629_524

EXPECTED_TOTAL_TRAINABLE = (
    EXPECTED_BERT_TRAINABLE
    + EXPECTED_GNN_TRAINABLE
    + EXPECTED_M4_TRAINABLE
)


# =============================================================================
# REPORTING
# =============================================================================

FAILURES: list[str] = []


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check(condition: bool, message: str) -> None:

    if condition:
        print(f"PASS  {message}")
        return

    print(f"FAIL  {message}")
    FAILURES.append(message)


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_json(path: Path) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_checkpoint(path: Path) -> dict[str, Any]:

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

    if not isinstance(obj, dict):

        raise RuntimeError(
            f"{path.name} is not a checkpoint dictionary"
        )

    return obj


def cpu_state_dict(
    module: nn.Module,
) -> dict[str, torch.Tensor]:

    return {
        key: value.detach().cpu().clone()
        for key, value
        in module.state_dict().items()
    }


def write_json_atomic(
    path: Path,
    obj: Any,
) -> None:

    temp = Path(str(path) + ".__tmp__")

    if temp.exists():

        raise RuntimeError(
            f"Temporary file already exists: {temp}"
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

        f.write("\n")

    os.replace(temp, path)


def seed_everything() -> None:

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)

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
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


# =============================================================================
# MINIMAL DEPENDENCY / OUTPUT CHECKS
# =============================================================================

def verify_starting_state() -> None:

    banner(
        "1. PHASE-B STARTING STATE"
    )

    existing = []

    for base in (
        MODEL_DIR,
        RUN_DIR,
    ):

        if base.exists():

            existing.extend(
                base.glob(
                    "task3_M4_phaseB_seed42_*"
                )
            )

    check(
        len(existing) == 0,
        (
            "no previous formal M4 "
            "Phase-B seed-42 run exists"
        ),
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

        GRAPH_MANIFEST:
            EXPECTED_GRAPH_MANIFEST_SHA256,

        T1_CHECKPOINT:
            EXPECTED_T1_SHA256,

        A4_CHECKPOINT:
            EXPECTED_A4_SHA256,

        M4_PHASEA_CHECKPOINT:
            EXPECTED_M4_PHASEA_SHA256,

        TASK3_DESIGN:
            EXPECTED_TASK3_DESIGN_SHA256,

        PHASEA_RESULTS_FREEZE:
            EXPECTED_PHASEA_FREEZE_SHA256,

        M3_PHASEB_SUMMARY:
            EXPECTED_M3_PHASEB_SUMMARY_SHA256,
    }

    for path, expected_sha in expected.items():

        check(
            path.is_file(),
            (
                "artifact exists: "
                f"{path.relative_to(ROOT)}"
            ),
        )

        if path.is_file():

            check(
                sha256_file(path)
                == expected_sha,
                (
                    f"{path.name} SHA256 "
                    "matches accepted identity"
                ),
            )

    if FAILURES:

        raise RuntimeError(
            "Phase-B starting-state validation failed"
        )

    freeze = load_json(
        PHASEA_RESULTS_FREEZE
    )

    check(
        freeze.get(
            "formal_phase_a_complete"
        )
        is True,
        (
            "Task-3 Phase A is "
            "formally complete"
        ),
    )

    gate = freeze.get(
        "phase_b_gate",
        {}
    )

    check(
        gate.get(
            "M4",
            {}
        ).get(
            "phase_b_required"
        )
        is True,
        (
            "frozen decision requires "
            "M4 Phase B"
        ),
    )

    m3_phase_b = load_json(
        M3_PHASEB_SUMMARY
    )

    check(
        m3_phase_b.get(
            "phase"
        )
        == "B"
        and
        m3_phase_b.get(
            "variant"
        )
        == "M3",
        (
            "formal M3 Phase B "
            "was completed first"
        ),
    )

    check(
        m3_phase_b.get(
            "test_split_used"
        )
        is False,
        (
            "M3 Phase B preserved "
            "TEST boundary"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Phase-B workflow-state validation failed"
        )


# =============================================================================
# SELECTED A4 GNN
# =============================================================================

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
# M4 FUSION
# =============================================================================

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

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
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
        bert_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:

        text_tokens = self.text_projection(
            bert_sequence
        )

        graph_prime = self.graph_projection(
            graph_embedding
        )

        query = graph_prime.unsqueeze(1)

        attended, _ = self.attention(
            query=query,
            key=text_tokens,
            value=text_tokens,
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

        z = self.fusion(z)
        z = F.relu(z)

        return self.classifier(z)


# =============================================================================
# JOINT M4 PHASE-B MODEL
# =============================================================================

class M4PhaseBModel(nn.Module):

    def __init__(
        self,
        bert: BertModel,
        gnn: A4GAT,
        fusion: M4Fusion,
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

        bert_output = self.bert(
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
            bert_sequence=
                bert_output.last_hidden_state,

            attention_mask=
                attention_mask,

            graph_embedding=
                graph_embedding,
        )


# =============================================================================
# RECONSTRUCT EXACT STARTING POINT
# =============================================================================

def build_model(
    device: torch.device,
) -> M4PhaseBModel:

    banner(
        "2. RECONSTRUCT M4 PHASE-B MODEL"
    )

    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        local_files_only=True,
    )

    check(
        config.hidden_size == 768,
        "BERT hidden size = 768",
    )

    check(
        config.num_hidden_layers == 12,
        "BERT has 12 encoder layers",
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    t1 = load_checkpoint(
        T1_CHECKPOINT
    )

    source_state = t1[
        "model_state_dict"
    ]

    bert_state = {
        key[len("bert."):]: value
        for key, value
        in source_state.items()
        if key.startswith("bert.")
    }

    excluded = {
        key
        for key
        in source_state
        if not key.startswith("bert.")
    }

    check(
        excluded
        == {
            "classifier.weight",
            "classifier.bias",
        },
        (
            "only Task-1 classifier "
            "is excluded"
        ),
    )

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
        (
            "A4 strict load has "
            "zero missing keys"
        ),
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "A4 strict load has "
            "zero unexpected keys"
        ),
    )

    fusion = M4Fusion()

    phase_a = load_checkpoint(
        M4_PHASEA_CHECKPOINT
    )

    check(
        phase_a.get(
            "variant"
        )
        == "M4",
        (
            "Phase-A source "
            "variant = M4"
        ),
    )

    check(
        phase_a.get(
            "phase"
        )
        == "A",
        (
            "Phase-A source "
            "phase = A"
        ),
    )

    incompatible = fusion.load_state_dict(
        phase_a[
            "model_state_dict"
        ],
        strict=True,
    )

    check(
        len(
            incompatible.missing_keys
        )
        == 0,
        (
            "M4 Phase-A fusion strict load "
            "has zero missing keys"
        ),
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "M4 Phase-A fusion strict load "
            "has zero unexpected keys"
        ),
    )

    # Freeze everything first.
    for p in bert.parameters():
        p.requires_grad_(False)

    for p in gnn.parameters():
        p.requires_grad_(False)

    # Last two BERT layers.
    for layer_index in (
        10,
        11,
    ):

        for p in (
            bert.encoder.layer[
                layer_index
            ].parameters()
        ):
            p.requires_grad_(True)

    # Whole selected GNN encoder.
    for module in (
        gnn.node_projection,
        gnn.gat1,
        gnn.gat2,
        gnn.gat3,
    ):

        for p in module.parameters():
            p.requires_grad_(True)

    # Old Task-2 classifier stays frozen/unused.
    for p in gnn.classifier.parameters():
        p.requires_grad_(False)

    # Entire M4 fusion remains trainable.
    for p in fusion.parameters():
        p.requires_grad_(True)

    model = M4PhaseBModel(
        bert=bert,
        gnn=gnn,
        fusion=fusion,
    )

    bert_trainable = sum(
        p.numel()
        for p in bert.parameters()
        if p.requires_grad
    )

    gnn_trainable = sum(
        p.numel()
        for p in gnn.parameters()
        if p.requires_grad
    )

    fusion_trainable = sum(
        p.numel()
        for p in fusion.parameters()
        if p.requires_grad
    )

    total_trainable = (
        bert_trainable
        + gnn_trainable
        + fusion_trainable
    )

    print()
    print(
        f"BERT top-2 trainable: "
        f"{bert_trainable:,}"
    )

    print(
        f"GNN encoder trainable: "
        f"{gnn_trainable:,}"
    )

    print(
        f"M4 fusion trainable: "
        f"{fusion_trainable:,}"
    )

    print(
        f"TOTAL trainable: "
        f"{total_trainable:,}"
    )

    check(
        bert_trainable
        == EXPECTED_BERT_TRAINABLE,
        (
            "BERT layers 10-11 contain "
            "14,175,744 trainable parameters"
        ),
    )

    check(
        gnn_trainable
        == EXPECTED_GNN_TRAINABLE,
        (
            "A4 GNN encoder contains "
            "101,056 trainable parameters"
        ),
    )

    check(
        fusion_trainable
        == EXPECTED_M4_TRAINABLE,
        (
            "M4 fusion contains "
            "629,524 trainable parameters"
        ),
    )

    check(
        total_trainable
        == EXPECTED_TOTAL_TRAINABLE,
        (
            "total trainable parameters "
            "= 14,906,324"
        ),
    )

    check(
        all(
            not p.requires_grad
            for p
            in bert.embeddings.parameters()
        ),
        (
            "BERT embeddings remain frozen"
        ),
    )

    for layer_index in range(10):

        check(
            all(
                not p.requires_grad
                for p
                in bert.encoder.layer[
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
            in gnn.classifier.parameters()
        ),
        (
            "old A4 classifier is "
            "frozen and unused"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "M4 Phase-B model reconstruction failed"
        )

    model.to(device)

    return model


# =============================================================================
# GRAPH LOADING
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
                GRAPH_MANIFEST.parent / raw,
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
                str(resolved)
            ] = resolved

    if len(found) != 1:

        raise RuntimeError(
            f"Cannot uniquely resolve graph: "
            f"{value}; matches={len(found)}"
        )

    return next(
        iter(found.values())
    )


def load_graph(path: Path) -> Data:

    with np.load(
        path,
        allow_pickle=False,
    ) as graph:

        if set(graph.files) != EXPECTED_GRAPH_KEYS:

            raise RuntimeError(
                f"Unexpected graph schema: "
                f"{path.name}"
            )

        x = np.asarray(
            graph["x"]
        )

        edge_index = np.asarray(
            graph["edge_index_g2"]
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

    if x.shape != (9, 140):

        raise RuntimeError(
            f"{path.name}: invalid x shape "
            f"{x.shape}"
        )

    if x.dtype != np.float32:

        raise RuntimeError(
            f"{path.name}: x dtype != float32"
        )

    if not np.isfinite(x).all():

        raise RuntimeError(
            f"{path.name}: non-finite x"
        )

    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or not (
            16
            <= edge_index.shape[1]
            <= 34
        )
    ):

        raise RuntimeError(
            f"{path.name}: invalid G2 edges"
        )

    if edge_index.dtype != np.int64:

        raise RuntimeError(
            f"{path.name}: G2 dtype != int64"
        )

    if not np.isclose(
        threshold,
        0.85,
        atol=1e-6,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{path.name}: graph threshold changed"
        )

    if not nonlocal_only:

        raise RuntimeError(
            f"{path.name}: nonlocal_only changed"
        )

    if degree_cap != 2:

        raise RuntimeError(
            f"{path.name}: degree cap changed"
        )

    if rule_version != 2:

        raise RuntimeError(
            f"{path.name}: rule version changed"
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
# TRAIN+VAL LIVE INPUTS
# =============================================================================

def load_development_data():

    banner(
        "3. LOAD TRAIN+VAL LIVE INPUTS"
    )

    token_manifest = pd.read_parquet(
        TOKEN_MANIFEST
    )

    graph_manifest = pd.read_parquet(
        GRAPH_MANIFEST
    )

    check(
        list(
            token_manifest.columns
        )
        == [
            "cache_index",
            "track_id",
            "split",
            "cached_token_length",
        ],
        "token manifest schema unchanged",
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

    counts = (
        token_manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    check(
        counts
        == {
            "train": N_TRAIN,
            "val": N_VAL,
            "test": N_TEST,
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
        len(dev_manifest)
        == N_DEV,
        (
            "development manifest "
            "= 4626 rows"
        ),
    )

    check(
        "test"
        not in set(
            dev_manifest["split"]
        ),
        (
            "development manifest "
            "contains zero TEST rows"
        ),
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
        set(dev_ids).issubset(
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

    # TRAIN+VAL targets only.
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

    dev_filter = (
        (pads.field("split") == "train")
        |
        (pads.field("split") == "val")
    )

    target_table = (
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
        target_table.to_pandas()
    )

    targets[
        "track_id"
    ] = (
        targets[
            "track_id"
        ].astype(str)
    )

    targets[
        "split"
    ] = (
        targets[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    check(
        len(targets)
        == N_DEV,
        (
            "4626 TRAIN+VAL "
            "target rows loaded"
        ),
    )

    check(
        "test"
        not in set(
            targets["split"]
        ),
        (
            "TEST target rows loaded = 0"
        ),
    )

    target_lookup = (
        targets
        .set_index(
            "track_id",
            drop=False,
        )
    )

    check(
        set(dev_ids)
        ==
        set(
            targets[
                "track_id"
            ]
        ),
        (
            "token and target IDs match"
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

    check(
        aligned_targets[
            "split"
        ].tolist()
        ==
        dev_manifest[
            "split"
        ].tolist(),
        (
            "target split aligns "
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
            set(cache.files)
            ==
            {
                "input_ids",
                "attention_mask",
                "token_type_ids",
            },
            (
                "token cache schema unchanged"
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

    check(
        input_ids.shape
        == (
            N_DEV,
            MAX_LENGTH,
        ),
        (
            "input_ids shape "
            "= (4626,112)"
        ),
    )

    check(
        attention_mask.shape
        == (
            N_DEV,
            MAX_LENGTH,
        ),
        (
            "attention_mask shape "
            "= (4626,112)"
        ),
    )

    print()
    print(
        "Preloading 4626 TRAIN+VAL "
        "G2 graphs..."
    )

    graphs = []

    for number, row in enumerate(
        dev_manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        track_id = str(
            row.track_id
        )

        expected_split = str(
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

        if graph_split != expected_split:

            raise RuntimeError(
                f"graph split mismatch: {track_id}"
            )

        if graph_split == "test":

            raise RuntimeError(
                "TEST graph encountered"
            )

        path = resolve_graph_path(
            str(
                graph_row[
                    "graph_path"
                ]
            )
        )

        graphs.append(
            load_graph(path)
        )

        if (
            number == 1
            or number % 500 == 0
            or number == N_DEV
        ):

            print(
                f"  loaded "
                f"{number}/{N_DEV}"
            )

    check(
        len(graphs)
        == N_DEV,
        (
            "exactly 4626 "
            "TRAIN+VAL graphs loaded"
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
        len(train_indices)
        == N_TRAIN,
        "TRAIN indices = 4112",
    )

    check(
        len(val_indices)
        == N_VAL,
        "VAL indices = 514",
    )

    if FAILURES:

        raise RuntimeError(
            "development-data validation failed"
        )

    print()
    print("TEST target rows loaded:  0")
    print("TEST graph files opened:  0")
    print("TEST token rows to model: 0")

    return (
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        y,
        train_indices,
        val_indices,
    )


# =============================================================================
# POS_WEIGHT
# =============================================================================

def compute_pos_weight(
    targets,
    train_indices,
):

    train = (
        targets[
            train_indices
        ].astype(
            np.float64
        )
    )

    positive = train.sum(
        axis=0
    )

    negative = (
        N_TRAIN
        - positive
    )

    weight = np.minimum(
        negative / positive,
        10.0,
    )

    expected = np.asarray(
        [
            2.237795275590551,
            3.5892857142857144,
            5.752052545155993,
            7.602510460251046,
            7.711864406779661,
            7.919739696312364,
            8.117516629711751,
            8.30316742081448,
            9.597938144329897,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ],
        dtype=np.float64,
    )

    check(
        np.allclose(
            weight,
            expected,
            rtol=1e-9,
            atol=1e-9,
        ),
        (
            "TRAIN-only pos_weight "
            "matches frozen vector"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "pos_weight validation failed"
        )

    return weight.astype(
        np.float32
    )


# =============================================================================
# DATASET
# =============================================================================

class PhaseBDataset(Dataset):

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        targets,
        indices,
    ):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.graphs = graphs
        self.targets = targets
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(self, item):

        index = int(
            self.indices[item]
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

            self.graphs[index],

            torch.from_numpy(
                self.targets[
                    index
                ].astype(
                    np.float32,
                    copy=True,
                )
            ),
        )


def collate_batch(samples):

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

        torch.stack(
            [
                x[4]
                for x in samples
            ]
        ),
    )


def build_loaders(
    input_ids,
    attention_mask,
    token_type_ids,
    graphs,
    targets,
    train_indices,
    val_indices,
):

    banner(
        "4. TRAIN / VAL LOADERS"
    )

    train_dataset = PhaseBDataset(
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        targets,
        train_indices,
    )

    val_dataset = PhaseBDataset(
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        targets,
        val_indices,
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )

    check(
        len(train_dataset)
        == N_TRAIN,
        "TRAIN dataset = 4112",
    )

    check(
        len(val_dataset)
        == N_VAL,
        "VAL dataset = 514",
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

    return train_loader, val_loader


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
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

    with torch.inference_mode():

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

            if not torch.isfinite(
                logits
            ).all():

                raise RuntimeError(
                    "non-finite VAL logits"
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

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * rows
            )

            total_rows += rows

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

    if total_rows != N_VAL:

        raise RuntimeError(
            f"VAL rows={total_rows}; "
            f"expected {N_VAL}"
        )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_probability = np.concatenate(
        all_probabilities,
        axis=0,
    )

    per_label_ap = (
        average_precision_score(
            y_true,
            y_probability,
            average=None,
        )
    )

    return {
        "loss":
            total_loss / total_rows,

        "macro_ap":
            float(
                per_label_ap.mean()
            ),

        "micro_ap":
            float(
                average_precision_score(
                    y_true,
                    y_probability,
                    average="micro",
                )
            ),

        "per_label_ap":
            [
                float(x)
                for x
                in per_label_ap
            ],
    }


# =============================================================================
# OPTIMIZER
# =============================================================================

def build_optimizer(model):

    banner(
        "5. PHASE-B OPTIMIZER"
    )

    bert_parameters = [
        p
        for p in model.bert.parameters()
        if p.requires_grad
    ]

    gnn_parameters = [
        p
        for p in model.gnn.parameters()
        if p.requires_grad
    ]

    fusion_parameters = [
        p
        for p in model.fusion.parameters()
        if p.requires_grad
    ]

    check(
        sum(
            p.numel()
            for p in bert_parameters
        )
        == EXPECTED_BERT_TRAINABLE,
        (
            "optimizer BERT group "
            "= layers 10-11 only"
        ),
    )

    check(
        sum(
            p.numel()
            for p in gnn_parameters
        )
        == EXPECTED_GNN_TRAINABLE,
        (
            "optimizer GNN group "
            "= selected encoder"
        ),
    )

    check(
        sum(
            p.numel()
            for p in fusion_parameters
        )
        == EXPECTED_M4_TRAINABLE,
        (
            "optimizer fusion group "
            "= complete M4"
        ),
    )

    all_parameters = (
        bert_parameters
        + gnn_parameters
        + fusion_parameters
    )

    ids = [
        id(p)
        for p in all_parameters
    ]

    check(
        len(ids)
        == len(set(ids)),
        (
            "optimizer has no "
            "duplicate parameters"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "optimizer validation failed"
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    bert_parameters,

                "lr":
                    LR,

                "weight_decay":
                    BERT_WEIGHT_DECAY,
            },

            {
                "params":
                    (
                        gnn_parameters
                        + fusion_parameters
                    ),

                "lr":
                    LR,

                "weight_decay":
                    GNN_FUSION_WEIGHT_DECAY,
            },
        ]
    )

    print("Optimizer: AdamW")
    print("LR: 5e-6")
    print("BERT WD: 0.01")
    print("GNN/Fusion WD: 0.0001")
    print("Scheduler: None")
    print("Epochs: 5")
    print("Batch: 16")
    print("Gradient accumulation: 1")

    return optimizer


# =============================================================================
# SAVE / RESTORE BEST STATE
# =============================================================================

def capture_state(model):

    return {
        "bert_layer_10":
            cpu_state_dict(
                model.bert.encoder.layer[10]
            ),

        "bert_layer_11":
            cpu_state_dict(
                model.bert.encoder.layer[11]
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


def restore_state(
    model,
    state,
):

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

    model.gnn.load_state_dict(
        state["gnn"],
        strict=True,
    )

    model.fusion.load_state_dict(
        state["fusion"],
        strict=True,
    )


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
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

        graph_batch = graph_batch.to(
            device
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

        if not torch.isfinite(
            logits
        ).all():

            raise RuntimeError(
                "non-finite TRAIN logits"
            )

        loss = criterion(
            logits,
            targets,
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                "non-finite TRAIN loss"
            )

        loss.backward()

        optimizer.step()

        rows = int(
            targets.shape[0]
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * rows
        )

        total_rows += rows

    if total_rows != N_TRAIN:

        raise RuntimeError(
            f"TRAIN rows={total_rows}; "
            f"expected {N_TRAIN}"
        )

    return total_loss / total_rows


# =============================================================================
# FORMAL TRAINING
# =============================================================================

def run_training(
    model,
    train_loader,
    val_loader,
    pos_weight,
    device,
):

    banner(
        "6. FORMAL M4 PHASE-B TRAINING"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    # Critical reconstruction check.
    initial_val = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    print(
        "Initial live-model parity:"
    )

    print(
        f"  expected Phase-A AP = "
        f"{M4_PHASEA_VAL_MACRO_AP:.12f}"
    )

    print(
        f"  reconstructed AP    = "
        f"{initial_val['macro_ap']:.12f}"
    )

    check(
        np.isclose(
            initial_val[
                "macro_ap"
            ],
            M4_PHASEA_VAL_MACRO_AP,
            rtol=0.0,
            atol=1e-8,
        ),
        (
            "live T1-B + A4 + M4 "
            "reproduces Phase-A VAL macro AP"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "M4 live reconstruction parity failed"
        )

    optimizer = build_optimizer(
        model
    )

    best_ap = -float("inf")
    best_epoch = None
    best_val = None
    best_state = None

    history = []

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start = time.perf_counter()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val = evaluate(
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
            best_ap
        )

        if improved:

            best_ap = (
                val[
                    "macro_ap"
                ]
            )

            best_epoch = epoch
            best_val = dict(val)
            best_state = capture_state(
                model
            )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    float(
                        train_loss
                    ),

                "val_loss":
                    float(
                        val["loss"]
                    ),

                "val_macro_ap":
                    float(
                        val["macro_ap"]
                    ),

                "val_micro_ap":
                    float(
                        val["micro_ap"]
                    ),

                "improved":
                    bool(
                        improved
                    ),

                "epoch_seconds":
                    float(
                        time.perf_counter()
                        - epoch_start
                    ),
            }
        )

        marker = (
            "  BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val['loss']:.6f}  "
            f"val_macro_AP="
            f"{val['macro_ap']:.8f}  "
            f"val_micro_AP="
            f"{val['micro_ap']:.8f}"
            f"{marker}"
        )

    runtime = (
        time.perf_counter()
        - start
    )

    peak_vram = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    if (
        best_state is None
        or best_epoch is None
        or best_val is None
    ):

        raise RuntimeError(
            "no valid Phase-B epoch selected"
        )

    restore_state(
        model,
        best_state,
    )

    revalidated = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    check(
        np.isclose(
            revalidated[
                "macro_ap"
            ],
            best_val[
                "macro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "restored best state reproduces "
            "VAL macro AP"
        ),
    )

    check(
        np.isclose(
            revalidated[
                "micro_ap"
            ],
            best_val[
                "micro_ap"
            ],
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "restored best state reproduces "
            "VAL micro AP"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "best-state revalidation failed"
        )

    return {
        "initial_val":
            initial_val,

        "best_epoch":
            best_epoch,

        "best_val":
            revalidated,

        "best_state":
            best_state,

        "history":
            history,

        "runtime_seconds":
            runtime,

        "peak_vram_bytes":
            peak_vram,
    }


# =============================================================================
# SAVE FORMAL RESULT
# =============================================================================

def save_result(
    run_id,
    result,
):

    banner(
        "7. SAVE FORMAL M4 PHASE-B RESULT"
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

    epoch_path = (
        RUN_DIR
        / f"{run_id}_epochs.csv"
    )

    summary_path = (
        RUN_DIR
        / f"{run_id}_summary.json"
    )

    for path in (
        checkpoint_path,
        epoch_path,
        summary_path,
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
            "formal output collision"
        )

    best = result[
        "best_val"
    ]

    checkpoint = {
        "task":
            "task3",

        "variant":
            "M4",

        "phase":
            "B",

        "model":
            "CrossAttentionJoint",

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
                "delta_from_selected_T1B_and_A4_"
                "plus_full_M4_fusion"
            ),

        "trainable_contract": {
            "bert_layers":
                [
                    10,
                    11,
                ],

            "bert_trainable_parameters":
                EXPECTED_BERT_TRAINABLE,

            "gnn":
                (
                    "node_projection + "
                    "gat1 + gat2 + gat3"
                ),

            "gnn_trainable_parameters":
                EXPECTED_GNN_TRAINABLE,

            "a4_original_classifier":
                "frozen_unused",

            "fusion_trainable_parameters":
                EXPECTED_M4_TRAINABLE,

            "total_trainable_parameters":
                EXPECTED_TOTAL_TRAINABLE,
        },

        "training": {
            "optimizer":
                "AdamW",

            "learning_rate":
                LR,

            "bert_weight_decay":
                BERT_WEIGHT_DECAY,

            "gnn_fusion_weight_decay":
                GNN_FUSION_WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "scheduler":
                None,

            "early_stopping":
                None,

            "gradient_accumulation_steps":
                GRAD_ACCUMULATION_STEPS,

            "selection":
                (
                    "highest VAL macro mean "
                    "per-label Average Precision "
                    "across epochs 1-5"
                ),
        },

        "initial_phaseA_live_val_macro_ap":
            float(
                result[
                    "initial_val"
                ][
                    "macro_ap"
                ]
            ),

        "val_loss":
            float(
                best["loss"]
            ),

        "val_auc_pr_macro_mean_ap":
            float(
                best["macro_ap"]
            ),

        "val_micro_ap":
            float(
                best["micro_ap"]
            ),

        "val_per_label_ap":
            best[
                "per_label_ap"
            ],

        "source_hashes": {
            "pretraining_lock":
                EXPECTED_PRETRAINING_LOCK_SHA256,

            "targets":
                EXPECTED_TARGETS_SHA256,

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

            "M4_phaseA":
                EXPECTED_M4_PHASEA_SHA256,

            "task3_design":
                EXPECTED_TASK3_DESIGN_SHA256,

            "phaseA_results_freeze":
                EXPECTED_PHASEA_FREEZE_SHA256,
        },

        "test_split_used":
            False,

        "test_target_rows_loaded":
            0,

        "test_graph_files_opened":
            0,

        "test_examples_evaluated":
            0,

        "threshold_tuning_used":
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

        "gnn_state_dict":
            result[
                "best_state"
            ][
                "gnn"
            ],

        "fusion_state_dict":
            result[
                "best_state"
            ][
                "fusion"
            ],
    }

    checkpoint_temp = Path(
        str(checkpoint_path)
        + ".__tmp__"
    )

    if checkpoint_temp.exists():

        raise RuntimeError(
            f"temporary checkpoint exists: "
            f"{checkpoint_temp}"
        )

    torch.save(
        checkpoint,
        checkpoint_temp,
    )

    os.replace(
        checkpoint_temp,
        checkpoint_path,
    )

    checkpoint_sha = sha256_file(
        checkpoint_path
    )

    history = pd.DataFrame(
        result["history"]
    )

    epoch_temp = Path(
        str(epoch_path)
        + ".__tmp__"
    )

    history.to_csv(
        epoch_temp,
        index=False,
    )

    os.replace(
        epoch_temp,
        epoch_path,
    )

    epoch_sha = sha256_file(
        epoch_path
    )

    best_ap = float(
        best["macro_ap"]
    )

    summary = {
        "task":
            "task3",

        "variant":
            "M4",

        "phase":
            "B",

        "formal_run":
            True,

        "run_id":
            run_id,

        "seed":
            SEED,

        "best_epoch":
            int(
                result[
                    "best_epoch"
                ]
            ),

        "best_val_auc_pr_macro_mean_ap":
            best_ap,

        "best_val_micro_ap":
            float(
                best["micro_ap"]
            ),

        "best_val_loss":
            float(
                best["loss"]
            ),

        "phase_a_reference":
            M4_PHASEA_VAL_MACRO_AP,

        "delta_vs_m4_phase_a":
            (
                best_ap
                - M4_PHASEA_VAL_MACRO_AP
            ),

        "m1_bert_reference":
            M1_VAL_MACRO_AP,

        "delta_vs_m1":
            (
                best_ap
                - M1_VAL_MACRO_AP
            ),

        "beats_m4_phase_a":
            bool(
                best_ap
                >
                M4_PHASEA_VAL_MACRO_AP
            ),

        "beats_m1_bert":
            bool(
                best_ap
                >
                M1_VAL_MACRO_AP
            ),

        "checkpoint":
            str(
                checkpoint_path
                .relative_to(ROOT)
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "epoch_log":
            str(
                epoch_path
                .relative_to(ROOT)
            ),

        "epoch_log_sha256":
            epoch_sha,

        "runtime_seconds":
            float(
                result[
                    "runtime_seconds"
                ]
            ),

        "peak_vram_bytes":
            int(
                result[
                    "peak_vram_bytes"
                ]
            ),

        "test_split_used":
            False,

        "test_target_rows_loaded":
            0,

        "test_graph_files_opened":
            0,

        "test_examples_evaluated":
            0,

        "threshold_tuning_used":
            False,

        "next_action":
            (
                "Select Task-3 configurations "
                "using validation only."
            ),

        "environment": {
            "python":
                sys.version.split()[0],

            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,

            "torch":
                torch.__version__,

            "torch_geometric":
                torch_geometric.__version__,

            "transformers":
                transformers.__version__,

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

    summary_sha = sha256_file(
        summary_path
    )

    return {
        "checkpoint_path":
            checkpoint_path,

        "checkpoint_sha":
            checkpoint_sha,

        "epoch_path":
            epoch_path,

        "epoch_sha":
            epoch_sha,

        "summary_path":
            summary_path,

        "summary_sha":
            summary_sha,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 54 — FORMAL TASK-3 M4 PHASE-B JOINT FINE-TUNING"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "BERT layers 10-11: TRAINABLE"
    )
    print(
        "BERT layers 0-9:   FROZEN"
    )
    print(
        "A4 GNN encoder:     TRAINABLE"
    )
    print(
        "A4 classifier:      FROZEN / UNUSED"
    )
    print(
        "M4 fusion:          TRAINABLE"
    )

    print()
    print(
        "Live BERT:           YES"
    )
    print(
        "Raw G2 graphs:       YES"
    )
    print(
        "LR:                  5e-6"
    )
    print(
        "Batch:               16"
    )
    print(
        "Epochs:              5"
    )
    print(
        "Threshold tuning:    NO"
    )
    print(
        "TEST inference:      NO"
    )

    seed_everything()

    try:

        verify_starting_state()

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

        model = build_model(
            device
        )

        (
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            targets,
            train_indices,
            val_indices,
        ) = load_development_data()

        pos_weight = compute_pos_weight(
            targets,
            train_indices,
        )

        train_loader, val_loader = (
            build_loaders(
                input_ids,
                attention_mask,
                token_type_ids,
                graphs,
                targets,
                train_indices,
                val_indices,
            )
        )

        result = run_training(
            model,
            train_loader,
            val_loader,
            pos_weight,
            device,
        )

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

        run_id = (
            "task3_M4_phaseB_"
            f"seed{SEED}_"
            f"{timestamp}"
        )

        outputs = save_result(
            run_id,
            result,
        )

    except Exception as exc:

        banner(
            "SCRIPT 54 — ABORTED"
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
            in str(exc).lower()
        ):

            print()
            print(
                "CUDA OOM detected."
            )
            print(
                "Do not accept this run."
            )
            print(
                "V6 permits gradient "
                "accumulation x2 if required."
            )

        if FAILURES:

            print()
            print("Failed checks:")

            for failure in FAILURES:
                print(
                    f"  - {failure}"
                )

        print()
        print(
            "Do NOT begin Task-3 selection."
        )

        return 1

    best = result[
        "best_val"
    ]

    banner(
        "SCRIPT 54 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "Formal model:"
    )
    print(
        "  Task 3 / M4 / Phase B"
    )

    print()
    print(
        f"Best epoch: "
        f"{result['best_epoch']}"
    )

    print()
    print(
        "Validation:"
    )

    print(
        f"  Phase A M4 = "
        f"{M4_PHASEA_VAL_MACRO_AP:.8f}"
    )

    print(
        f"  Phase B M4 = "
        f"{best['macro_ap']:.8f}"
    )

    print(
        f"  delta       = "
        f"{best['macro_ap'] - M4_PHASEA_VAL_MACRO_AP:+.8f}"
    )

    print()

    print(
        f"  M1 BERT     = "
        f"{M1_VAL_MACRO_AP:.8f}"
    )

    print(
        f"  delta vs M1 = "
        f"{best['macro_ap'] - M1_VAL_MACRO_AP:+.8f}"
    )

    print()

    print(
        f"  VAL micro AP = "
        f"{best['micro_ap']:.8f}"
    )

    print(
        f"  VAL loss     = "
        f"{best['loss']:.8f}"
    )

    print()
    print(
        "Trainable parameters:"
    )

    print(
        f"  BERT top 2 = "
        f"{EXPECTED_BERT_TRAINABLE:,}"
    )

    print(
        f"  GNN encoder = "
        f"{EXPECTED_GNN_TRAINABLE:,}"
    )

    print(
        f"  M4 fusion   = "
        f"{EXPECTED_M4_TRAINABLE:,}"
    )

    print(
        f"  TOTAL       = "
        f"{EXPECTED_TOTAL_TRAINABLE:,}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        f"  "
        f"{outputs['checkpoint_path'].relative_to(ROOT)}"
    )

    print(
        "Checkpoint SHA256:"
    )

    print(
        f"  {outputs['checkpoint_sha']}"
    )

    print()
    print(
        "Epoch log SHA256:"
    )

    print(
        f"  {outputs['epoch_sha']}"
    )

    print()
    print(
        "Summary SHA256:"
    )

    print(
        f"  {outputs['summary_sha']}"
    )

    print()
    print(
        f"Runtime: "
        f"{result['runtime_seconds'] / 60.0:.2f} min"
    )

    print(
        f"Peak VRAM: "
        f"{result['peak_vram_bytes'] / (1024 ** 3):.3f} GB"
    )

    print()
    print(
        "TEST target rows loaded: 0"
    )
    print(
        "TEST graph files opened: 0"
    )
    print(
        "TEST inference examples:  0"
    )
    print(
        "Threshold tuning:         0"
    )

    print()
    print(
        "NEXT:"
    )
    print(
        "  Validation-only Task-3 "
        "model selection."
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