#!/usr/bin/env python3

"""
Script 59 — Freeze Thresholds for Remaining Final-Report Models
===============================================================

Already frozen by Script 57:
    T1-B BERT
    A1 CNN
    M3 Early Fusion Phase B

Remaining final comparison-table models:
    A3 GraphSAGE / G2
    A4 GAT / G2
    M4 Cross-Attention / Phase B

Purpose:
    - reproduce each accepted seed-42 VAL macro AP
    - sweep threshold 0.30..0.70 on VAL only
    - use the exact same frozen threshold-selection rule
    - combine all report-model thresholds into one final lock

NO training.
NO TEST targets.
NO TEST inference.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    Dataset,
    DataLoader,
)

from torch_geometric.data import (
    Data,
    Batch,
)

from torch_geometric.nn import (
    SAGEConv,
    GATConv,
    global_mean_pool,
)

from transformers import (
    BertConfig,
    BertModel,
)


# =============================================================================
# PATHS
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

GRAPH_MANIFEST = (
    ROOT
    / "data/processed/graph_manifest.parquet"
)

WINNER_THRESHOLDS = (
    ROOT
    / "results/runs/final_eval/"
    "winner_thresholds_validation.json"
)

FINAL_MULTISEED_MANIFEST = (
    ROOT
    / "results/runs/final_eval/multiseed/"
    "final_multiseed_training_manifest.json"
)


# -----------------------------------------------------------------------------
# Frozen seed-42 report models
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

T1_CHECKPOINT = (
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


OUTPUT = (
    ROOT
    / "results/runs/final_eval/"
    "final_report_thresholds_validation.json"
)

OUTPUT_TEMP = Path(
    str(OUTPUT) + ".__tmp__"
)


# =============================================================================
# ACCEPTED HASHES
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

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6ad0492bf90eccc929cadc1aad1da9e158"
)

EXPECTED_WINNER_THRESHOLDS_SHA256 = (
    "441f3bf1229afd4a578a5485111d1a5ad9ab049bd4b163d7bd21c44f64040403"
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

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_M4_PHASEA_SHA256 = (
    "ea210e30a932c124ee07b6a5fc90d068e0201152301998ed1eeb67559dd2e112"
)

EXPECTED_M4_PHASEB_SHA256 = (
    "7a27a270074ef4e50143aec94b8b5d7ada001de0c0708c18b3aea406d159c165"
)


# =============================================================================
# ACCEPTED RESULTS
# =============================================================================

N_VAL = 514
NUM_LABELS = 20
MAX_LENGTH = 112

EXPECTED_A3_AP = 0.18372526
EXPECTED_A4_AP = 0.18752693895679443
EXPECTED_M4_AP = 0.832154727509

AP_TOL = 1e-8

THRESHOLDS = np.arange(
    0.30,
    0.7001,
    0.05,
).round(2)


# =============================================================================
# HELPERS
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
        raise RuntimeError(message)

    print(f"PASS  {message}")


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

    if not isinstance(obj, dict):

        raise RuntimeError(
            f"{path.name} is not a checkpoint dictionary"
        )

    return obj


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        obj = json.load(f)

    if not isinstance(obj, dict):

        raise RuntimeError(
            f"{path.name} is not a JSON object"
        )

    return obj


def atomic_json(
    obj: dict[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():

        raise RuntimeError(
            f"output already exists: {path}"
        )

    if OUTPUT_TEMP.exists():

        raise RuntimeError(
            f"temporary output exists: {OUTPUT_TEMP}"
        )

    with OUTPUT_TEMP.open(
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

    os.replace(
        OUTPUT_TEMP,
        path,
    )


def compute_macro_ap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    return float(
        np.mean(
            average_precision_score(
                y_true,
                probabilities,
                average=None,
            )
        )
    )


def sweep_threshold(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
):

    banner(
        f"THRESHOLD SWEEP — {name}"
    )

    rows = []

    for threshold in THRESHOLDS:

        prediction = (
            probabilities >= threshold
        ).astype(np.int64)

        macro_f1 = float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        )

        micro_f1 = float(
            f1_score(
                y_true,
                prediction,
                average="micro",
                zero_division=0,
            )
        )

        per_label = f1_score(
            y_true,
            prediction,
            average=None,
            zero_division=0,
        )

        row = {
            "threshold":
                float(threshold),

            "macro_f1":
                macro_f1,

            "micro_f1":
                micro_f1,

            "per_label_f1": [
                float(x)
                for x in per_label
            ],
        }

        rows.append(row)

        print(
            f"threshold={threshold:.2f}  "
            f"macro_F1={macro_f1:.8f}  "
            f"micro_F1={micro_f1:.8f}"
        )

    best = sorted(
        rows,
        key=lambda x: (
            -x["macro_f1"],
            -x["micro_f1"],
            abs(
                x["threshold"]
                - 0.50
            ),
            x["threshold"],
        ),
    )[0]

    print()
    print(
        f"Selected {name}: "
        f"{best['threshold']:.2f}"
    )

    return rows, best


# =============================================================================
# GRAPH MODELS
# =============================================================================

class GraphSAGEModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.node_projection = nn.Linear(
            140,
            64,
        )

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

        return self.classifier(
            x
        )


class GATModel(nn.Module):

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

    def forward(
        self,
        data,
    ):

        return self.classifier(
            self.encode_graph(data)
        )


# =============================================================================
# M4 CROSS-ATTENTION
# =============================================================================

class M4Fusion(nn.Module):

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

    matches = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = candidate.resolve()

            matches[
                str(resolved)
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            f"cannot uniquely resolve graph: {value}"
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

        if set(graph.files) != EXPECTED_GRAPH_KEYS:

            raise RuntimeError(
                f"unexpected graph schema: {path.name}"
            )

        x = np.asarray(
            graph["x"]
        )

        edge = np.asarray(
            graph["edge_index_g2"]
        )

    if x.shape != (
        9,
        140,
    ):

        raise RuntimeError(
            f"invalid x shape: {path.name}"
        )

    if (
        edge.ndim != 2
        or edge.shape[0] != 2
        or edge.dtype != np.int64
    ):

        raise RuntimeError(
            f"invalid G2 edges: {path.name}"
        )

    return Data(
        x=torch.from_numpy(
            x.astype(
                np.float32,
                copy=True,
            )
        ),
        edge_index=torch.from_numpy(
            edge.copy()
        ),
    )


# =============================================================================
# VALIDATION DATASET
# =============================================================================

class ValidationDataset(Dataset):

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        targets,
    ):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.graphs = graphs
        self.targets = targets

    def __len__(self):

        return N_VAL

    def __getitem__(
        self,
        index,
    ):

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


def collate(samples):

    return (
        torch.stack(
            [x[0] for x in samples]
        ),

        torch.stack(
            [x[1] for x in samples]
        ),

        torch.stack(
            [x[2] for x in samples]
        ),

        Batch.from_data_list(
            [x[3] for x in samples]
        ),

        torch.stack(
            [x[4] for x in samples]
        ),
    )


# =============================================================================
# LOAD VALIDATION ONLY
# =============================================================================

def load_validation():

    banner(
        "LOAD VALIDATION DATA ONLY"
    )

    token_manifest = pd.read_parquet(
        TOKEN_MANIFEST
    ).copy()

    token_manifest["track_id"] = (
        token_manifest[
            "track_id"
        ].astype(str)
    )

    token_manifest["split"] = (
        token_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    val_manifest = (
        token_manifest.loc[
            token_manifest[
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
        "validation token rows = 514",
    )

    val_ids = (
        val_manifest[
            "track_id"
        ].tolist()
    )

    y_columns = [
        f"y_{i:02d}"
        for i in range(
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
            pads.field("split")
            == "val"
        ),
    )

    target_frame = table.to_pandas()

    target_frame["track_id"] = (
        target_frame[
            "track_id"
        ].astype(str)
    )

    target_frame["split"] = (
        target_frame[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    require(
        len(target_frame)
        == N_VAL,
        "VAL target rows loaded = 514",
    )

    require(
        set(
            target_frame["split"]
        )
        == {"val"},
        "VAL labels only",
    )

    target_lookup = (
        target_frame.set_index(
            "track_id",
            drop=False,
        )
    )

    require(
        set(val_ids)
        ==
        set(
            target_frame[
                "track_id"
            ]
        ),
        "VAL IDs align",
    )

    target_frame = (
        target_lookup.loc[
            val_ids
        ]
        .reset_index(
            drop=True
        )
    )

    y_true = (
        target_frame[
            y_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

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

        input_ids = np.asarray(
            cache["input_ids"][
                cache_indices
            ],
            dtype=np.int32,
        )

        attention_mask = np.asarray(
            cache["attention_mask"][
                cache_indices
            ],
            dtype=np.uint8,
        )

        token_type_ids = np.asarray(
            cache["token_type_ids"][
                cache_indices
            ],
            dtype=np.uint8,
        )

    require(
        input_ids.shape
        == (
            N_VAL,
            MAX_LENGTH,
        ),
        "VAL tokens = (514,112)",
    )

    graph_manifest = pd.read_parquet(
        GRAPH_MANIFEST
    ).copy()

    graph_manifest["track_id"] = (
        graph_manifest[
            "track_id"
        ].astype(str)
    )

    graph_manifest["split"] = (
        graph_manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
    )

    lookup = graph_manifest.set_index(
        "track_id",
        drop=False,
    )

    graphs = []

    print(
        "Loading 514 VAL G2 graphs..."
    )

    for i, track_id in enumerate(
        val_ids,
        start=1,
    ):

        row = lookup.loc[
            track_id
        ]

        if row["split"] != "val":

            raise RuntimeError(
                "non-VAL graph encountered"
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
            i == 1
            or i % 100 == 0
            or i == N_VAL
        ):

            print(
                f"  loaded {i}/{N_VAL}"
            )

    require(
        len(graphs)
        == N_VAL,
        "exactly 514 VAL graphs loaded",
    )

    print()
    print(
        "TEST target rows loaded: 0"
    )

    print(
        "TEST graph files opened: 0"
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

        "targets":
            y_true,
    }


# =============================================================================
# GRAPH-ONLY INFERENCE
# =============================================================================

def graph_probabilities(
    model,
    graphs,
    device,
    *,
    use_amp: bool,
):

    model.to(device)
    model.eval()

    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            N_VAL,
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

            probabilities.append(
                probability
                .detach()
                .cpu()
                .numpy()
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
        "graph probability matrix = (514,20)",
    )

    return result


def choose_graph_numerical_path(
    name,
    model,
    graphs,
    y_true,
    expected_ap,
    device,
):

    print()
    print(
        f"Checking {name} FP32 parity..."
    )

    fp32 = graph_probabilities(
        model,
        graphs,
        device,
        use_amp=False,
    )

    fp32_ap = compute_macro_ap(
        y_true,
        fp32,
    )

    print(
        f"  FP32 AP = "
        f"{fp32_ap:.12f}"
    )

    print(
        f"Checking {name} FP16-autocast parity..."
    )

    amp = graph_probabilities(
        model,
        graphs,
        device,
        use_amp=True,
    )

    amp_ap = compute_macro_ap(
        y_true,
        amp,
    )

    print(
        f"  AMP AP  = "
        f"{amp_ap:.12f}"
    )

    choices = [
        (
            abs(
                fp32_ap
                - expected_ap
            ),
            "FP32",
            fp32,
            fp32_ap,
        ),

        (
            abs(
                amp_ap
                - expected_ap
            ),
            "FP16_autocast",
            amp,
            amp_ap,
        ),
    ]

    choices.sort(
        key=lambda x: x[0]
    )

    error, mode, probability, ap = (
        choices[0]
    )

    require(
        error <= AP_TOL,
        (
            f"{name} reproduces accepted "
            "VAL macro AP"
        ),
    )

    print(
        f"Selected numerical path: "
        f"{mode}"
    )

    return probability, ap, mode


# =============================================================================
# M4 RECONSTRUCTION
# =============================================================================

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
        "required Phase-B checkpoint "
        f"state missing: {preferred}"
    )


def reconstruct_m4(
    device,
):

    banner(
        "RECONSTRUCT M4 PHASE B"
    )

    # BERT source.
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        local_files_only=True,
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    t1 = load_checkpoint(
        T1_CHECKPOINT
    )

    source = t1[
        "model_state_dict"
    ]

    bert_state = {
        key[len("bert."):]:
            value

        for key, value
        in source.items()

        if key.startswith(
            "bert."
        )
    }

    bert.load_state_dict(
        bert_state,
        strict=True,
    )

    # GNN source.
    gnn = GATModel()

    a4 = load_checkpoint(
        A4_CHECKPOINT
    )

    gnn.load_state_dict(
        a4[
            "model_state_dict"
        ],
        strict=True,
    )

    # Phase-A fusion.
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

    # Phase-B learned states.
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


def m4_probabilities(
    model,
    data,
    device,
    *,
    use_amp,
):

    dataset = ValidationDataset(
        data[
            "input_ids"
        ],
        data[
            "attention_mask"
        ],
        data[
            "token_type_ids"
        ],
        data[
            "graphs"
        ],
        data[
            "targets"
        ],
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate,
    )

    probabilities = []

    with torch.no_grad():

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
            _,
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

            probabilities.append(
                torch.sigmoid(
                    logits.float()
                )
                .detach()
                .cpu()
                .numpy()
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
        "M4 probability matrix = (514,20)",
    )

    return result


def choose_m4_numerical_path(
    model,
    data,
    device,
):

    print()
    print(
        "Checking M4 FP32 parity..."
    )

    fp32 = m4_probabilities(
        model,
        data,
        device,
        use_amp=False,
    )

    fp32_ap = compute_macro_ap(
        data[
            "targets"
        ],
        fp32,
    )

    print(
        f"  FP32 AP = {fp32_ap:.12f}"
    )

    print(
        "Checking M4 FP16-autocast parity..."
    )

    amp = m4_probabilities(
        model,
        data,
        device,
        use_amp=True,
    )

    amp_ap = compute_macro_ap(
        data[
            "targets"
        ],
        amp,
    )

    print(
        f"  AMP AP  = {amp_ap:.12f}"
    )

    candidates = [
        (
            abs(
                fp32_ap
                - EXPECTED_M4_AP
            ),
            "FP32",
            fp32,
            fp32_ap,
        ),

        (
            abs(
                amp_ap
                - EXPECTED_M4_AP
            ),
            "FP16_autocast",
            amp,
            amp_ap,
        ),
    ]

    candidates.sort(
        key=lambda x: x[0]
    )

    error, mode, probability, ap = (
        candidates[0]
    )

    require(
        error <= AP_TOL,
        (
            "M4 reproduces accepted "
            "VAL macro AP"
        ),
    )

    print(
        f"Selected numerical path: "
        f"{mode}"
    )

    return probability, ap, mode


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 59 — FREEZE FINAL REPORT MODEL THRESHOLDS"
    )

    print(
        "VAL threshold tuning only."
    )

    print(
        "TEST access: ZERO"
    )

    try:

        require(
            not OUTPUT.exists(),
            "final report threshold lock does not already exist",
        )

        require(
            not OUTPUT_TEMP.exists(),
            "temporary threshold lock does not exist",
        )

        expected = {
            TARGETS:
                EXPECTED_TARGETS_SHA256,

            TOKEN_CACHE:
                EXPECTED_TOKEN_CACHE_SHA256,

            TOKEN_MANIFEST:
                EXPECTED_TOKEN_MANIFEST_SHA256,

            GRAPH_MANIFEST:
                EXPECTED_GRAPH_MANIFEST_SHA256,

            WINNER_THRESHOLDS:
                EXPECTED_WINNER_THRESHOLDS_SHA256,

            FINAL_MULTISEED_MANIFEST:
                EXPECTED_MULTISEED_MANIFEST_SHA256,

            A3_CHECKPOINT:
                EXPECTED_A3_SHA256,

            A4_CHECKPOINT:
                EXPECTED_A4_SHA256,

            T1_CHECKPOINT:
                EXPECTED_T1_SHA256,

            M4_PHASEA_CHECKPOINT:
                EXPECTED_M4_PHASEA_SHA256,

            M4_PHASEB_CHECKPOINT:
                EXPECTED_M4_PHASEB_SHA256,
        }

        banner(
            "1. FROZEN ARTIFACT IDENTITIES"
        )

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

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA unavailable"
            )

        device = torch.device(
            "cuda"
        )

        data = load_validation()

        # ---------------------------------------------------------------------
        # A3 GraphSAGE / G2
        # ---------------------------------------------------------------------

        banner(
            "2. A3 GRAPHSAGE/G2"
        )

        a3_checkpoint = (
            load_checkpoint(
                A3_CHECKPOINT
            )
        )

        require(
            a3_checkpoint.get(
                "variant"
            )
            == "A3",
            "A3 checkpoint variant is correct",
        )

        a3 = GraphSAGEModel()

        a3.load_state_dict(
            a3_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        (
            a3_probability,
            a3_ap,
            a3_mode,
        ) = choose_graph_numerical_path(
            "A3 GraphSAGE/G2",
            a3,
            data[
                "graphs"
            ],
            data[
                "targets"
            ],
            float(
                a3_checkpoint[
                    "val_auc_pr_macro_mean_ap"
                ]
            ),
            device,
        )

        (
            a3_sweep,
            a3_best,
        ) = sweep_threshold(
            "A3 GraphSAGE/G2",
            data[
                "targets"
            ],
            a3_probability,
        )

        # ---------------------------------------------------------------------
        # A4 GAT / G2
        # ---------------------------------------------------------------------

        banner(
            "3. A4 GAT/G2"
        )

        a4_checkpoint = (
            load_checkpoint(
                A4_CHECKPOINT
            )
        )

        a4 = GATModel()

        a4.load_state_dict(
            a4_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        (
            a4_probability,
            a4_ap,
            a4_mode,
        ) = choose_graph_numerical_path(
            "A4 GAT/G2",
            a4,
            data[
                "graphs"
            ],
            data[
                "targets"
            ],
            EXPECTED_A4_AP,
            device,
        )

        (
            a4_sweep,
            a4_best,
        ) = sweep_threshold(
            "A4 GAT/G2",
            data[
                "targets"
            ],
            a4_probability,
        )

        # ---------------------------------------------------------------------
        # M4 Cross-Attention Phase B
        # ---------------------------------------------------------------------

        banner(
            "4. M4 CROSS-ATTENTION PHASE B"
        )

        m4 = reconstruct_m4(
            device
        )

        (
            m4_probability,
            m4_ap,
            m4_mode,
        ) = choose_m4_numerical_path(
            m4,
            data,
            device,
        )

        (
            m4_sweep,
            m4_best,
        ) = sweep_threshold(
            "M4 Cross-Attention Phase B",
            data[
                "targets"
            ],
            m4_probability,
        )

        # ---------------------------------------------------------------------
        # Import already-frozen winner thresholds.
        # ---------------------------------------------------------------------

        winners = load_json(
            WINNER_THRESHOLDS
        )

        require(
            winners.get(
                "artifact_type"
            )
            ==
            "final_winner_validation_thresholds",
            (
                "Script-57 threshold "
                "artifact is valid"
            ),
        )

        # ---------------------------------------------------------------------
        # Final report threshold lock
        # ---------------------------------------------------------------------

        banner(
            "5. FREEZE COMPLETE REPORT THRESHOLDS"
        )

        output = {
            "artifact_type":
                "final_report_validation_thresholds",

            "version":
                1,

            "selection_basis":
                "VALIDATION only",

            "threshold_candidates":
                [
                    float(x)
                    for x in THRESHOLDS
                ],

            "selection_rule": {
                "primary":
                    "maximum validation Macro-F1",

                "tie_break_1":
                    "higher validation Micro-F1",

                "tie_break_2":
                    "threshold closest to 0.50",

                "tie_break_3":
                    "lower threshold",
            },

            "models": {
                "B1_majority":
                    {
                        "threshold":
                            None,
                    },

                "B2_A1_CNN":
                    {
                        "threshold":
                            float(
                                winners[
                                    "task2"
                                ][
                                    "selected_threshold"
                                ]
                            ),
                    },

                "BERT_T1B":
                    {
                        "threshold":
                            float(
                                winners[
                                    "task1"
                                ][
                                    "selected_threshold"
                                ]
                            ),
                    },

                "GraphSAGE_A3_G2":
                    {
                        "threshold":
                            float(
                                a3_best[
                                    "threshold"
                                ]
                            ),

                        "val_macro_ap":
                            a3_ap,

                        "val_macro_f1":
                            a3_best[
                                "macro_f1"
                            ],

                        "val_micro_f1":
                            a3_best[
                                "micro_f1"
                            ],

                        "numerical_path":
                            a3_mode,

                        "checkpoint_sha256":
                            EXPECTED_A3_SHA256,

                        "sweep":
                            a3_sweep,
                    },

                "GAT_A4_G2":
                    {
                        "threshold":
                            float(
                                a4_best[
                                    "threshold"
                                ]
                            ),

                        "val_macro_ap":
                            a4_ap,

                        "val_macro_f1":
                            a4_best[
                                "macro_f1"
                            ],

                        "val_micro_f1":
                            a4_best[
                                "micro_f1"
                            ],

                        "numerical_path":
                            a4_mode,

                        "checkpoint_sha256":
                            EXPECTED_A4_SHA256,

                        "sweep":
                            a4_sweep,
                    },

                "EarlyFusion_M3B":
                    {
                        "threshold":
                            float(
                                winners[
                                    "task3"
                                ][
                                    "selected_threshold"
                                ]
                            ),
                    },

                "CrossAttention_M4B":
                    {
                        "threshold":
                            float(
                                m4_best[
                                    "threshold"
                                ]
                            ),

                        "val_macro_ap":
                            m4_ap,

                        "val_macro_f1":
                            m4_best[
                                "macro_f1"
                            ],

                        "val_micro_f1":
                            m4_best[
                                "micro_f1"
                            ],

                        "numerical_path":
                            m4_mode,

                        "checkpoint_sha256":
                            EXPECTED_M4_PHASEB_SHA256,

                        "sweep":
                            m4_sweep,
                    },
            },

            "winner_threshold_lock_sha256":
                EXPECTED_WINNER_THRESHOLDS_SHA256,

            "final_multiseed_manifest_sha256":
                EXPECTED_MULTISEED_MANIFEST_SHA256,

            "test_target_rows_loaded":
                0,

            "test_graph_files_opened":
                0,

            "test_examples_evaluated":
                0,

            "threshold_tuned_on_test":
                False,

            "ready_for_final_test":
                True,
        }

        atomic_json(
            output,
            OUTPUT,
        )

        output_sha = sha256_file(
            OUTPUT
        )

    except Exception as exc:

        banner(
            "SCRIPT 59 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "Do NOT begin TEST evaluation."
        )

        return 1

    banner(
        "SCRIPT 59 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Complete frozen report thresholds:"
    )

    print(
        "  B1 Majority       : n/a"
    )

    print(
        f"  A1 CNN            : "
        f"{winners['task2']['selected_threshold']:.2f}"
    )

    print(
        f"  T1-B BERT         : "
        f"{winners['task1']['selected_threshold']:.2f}"
    )

    print(
        f"  A3 GraphSAGE/G2   : "
        f"{a3_best['threshold']:.2f}"
    )

    print(
        f"  A4 GAT/G2         : "
        f"{a4_best['threshold']:.2f}"
    )

    print(
        f"  M3 Early Fusion   : "
        f"{winners['task3']['selected_threshold']:.2f}"
    )

    print(
        f"  M4 Cross-Attention: "
        f"{m4_best['threshold']:.2f}"
    )

    print()

    print(
        "TEST target rows loaded: 0"
    )

    print(
        "TEST inference examples:  0"
    )

    print()

    print(
        "Final threshold lock:"
    )

    print(
        f"  {OUTPUT.relative_to(ROOT)}"
    )

    print()

    print(
        "SHA256:"
    )

    print(
        f"  {output_sha}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  FINAL TEST evaluation — exactly once."
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