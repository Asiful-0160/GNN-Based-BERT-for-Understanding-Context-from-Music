#!/usr/bin/env python3
"""
Script 56 — Validation-Only Threshold Tuning for Selected Task-3 M3
===================================================================

Selected configuration:
    M3 Early Fusion / Phase B / seed 42

Selection artifact:
    task3_model_selection.json

Threshold sweep:
    {0.30, 0.35, ..., 0.70}

Threshold-selection rule — frozen before observing sweep results:
    1. highest validation Macro-F1
    2. if tied: highest validation Micro-F1
    3. if tied: threshold closest to 0.50
    4. if tied: lower threshold

Important:
    Model selection remains based on threshold-free VAL macro AP.
    Threshold tuning is only for F1 reporting.

TEST:
    target rows loaded     = 0
    graph files opened     = 0
    inference examples     = 0
    metrics                = 0
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

from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
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

TOKEN_CACHE = (
    ROOT
    / "data/processed/bert_tokens/bert_tokens.npz"
)

TOKEN_MANIFEST = (
    ROOT
    / "data/processed/bert_tokens/bert_token_manifest.parquet"
)

TARGETS = (
    ROOT
    / "data/processed/labels/musiccaps_targets.parquet"
)

GRAPH_MANIFEST = (
    ROOT
    / "data/processed/graph_manifest.parquet"
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

M3_PHASEA_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M3_phaseA_seed42_20260910T130250Z_best.pt"
)

M3_PHASEB_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M3_phaseB_seed42_20260910T133125Z_best.pt"
)

TASK3_SELECTION = (
    ROOT
    / "results/runs/task3/"
    "task3_model_selection.json"
)

OUTPUT = (
    ROOT
    / "results/runs/task3/"
    "task3_M3_phaseB_seed42_threshold.json"
)

OUTPUT_TEMP = Path(
    str(OUTPUT) + ".__tmp__"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_SELECTION_SHA256 = (
    "1d02aafbc0aef9702194874ba02de7c0cbf7483dfac4b15d191fbfdeaac31e5e"
)

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e"
)

EXPECTED_M3_PHASEA_SHA256 = (
    "f23a0a5f8bb604672de640eca5cbfbb189e99febd32cab02f082a1de9a1fa5b8"
)

EXPECTED_M3_PHASEB_SHA256 = (
    "e853c64b22a20f9000ecb3260fd45f8c09a60379eee04294606193e17353dd89"
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

EXPECTED_TARGETS_SHA256 = (
    "a6f80bb42b0b48b2c47dac5f27ad2fd66f8893d83bd5d4e552d356cee3547056"
)


# =============================================================================
# FROZEN MODEL RESULT
# =============================================================================

N_VAL = 514
NUM_LABELS = 20
MAX_LENGTH = 112

EXPECTED_M3_VAL_MACRO_AP = (
    0.839664696610
)

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


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:

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


def write_json_atomic(
    path: Path,
    obj: dict[str, Any],
) -> None:

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


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:

        raise RuntimeError(message)

    print(
        f"PASS  {message}"
    )


# =============================================================================
# A4 GAT — EXACT SELECTED GNN
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
# M3 FUSION
# =============================================================================

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
        text_cls,
        graph_embedding,
    ):

        text_prime = self.text_projection(
            text_cls
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


# =============================================================================
# FULL SELECTED M3 PHASE-B MODEL
# =============================================================================

class M3Model(nn.Module):

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

        return self.fusion(
            text_cls,
            graph_embedding,
        )


# =============================================================================
# RECONSTRUCT SELECTED M3-B
# =============================================================================

def build_selected_model(
    device: torch.device,
) -> M3Model:

    banner(
        "2. RECONSTRUCT SELECTED M3 PHASE-B"
    )

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

    t1_state = t1[
        "model_state_dict"
    ]

    bert_state = {
        key[len("bert."):]: value
        for key, value
        in t1_state.items()
        if key.startswith("bert.")
    }

    incompatible = bert.load_state_dict(
        bert_state,
        strict=True,
    )

    require(
        len(
            incompatible.missing_keys
        ) == 0,
        "T1-B BERT strict load has zero missing keys",
    )

    require(
        len(
            incompatible.unexpected_keys
        ) == 0,
        "T1-B BERT strict load has zero unexpected keys",
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

    require(
        len(
            incompatible.missing_keys
        ) == 0,
        "A4 GAT strict load has zero missing keys",
    )

    require(
        len(
            incompatible.unexpected_keys
        ) == 0,
        "A4 GAT strict load has zero unexpected keys",
    )

    fusion = M3Fusion()

    phase_a = load_checkpoint(
        M3_PHASEA_CHECKPOINT
    )

    incompatible = fusion.load_state_dict(
        phase_a[
            "model_state_dict"
        ],
        strict=True,
    )

    require(
        len(
            incompatible.missing_keys
        ) == 0,
        "M3 Phase-A fusion strict load passes",
    )

    # -------------------------------------------------------------------------
    # Apply the selected Phase-B delta.
    # -------------------------------------------------------------------------

    phase_b = load_checkpoint(
        M3_PHASEB_CHECKPOINT
    )

    require(
        phase_b.get("variant") == "M3",
        "selected checkpoint variant = M3",
    )

    require(
        phase_b.get("phase") == "B",
        "selected checkpoint phase = B",
    )

    bert.encoder.layer[
        10
    ].load_state_dict(
        phase_b[
            "bert_layer_10_state_dict"
        ],
        strict=True,
    )

    bert.encoder.layer[
        11
    ].load_state_dict(
        phase_b[
            "bert_layer_11_state_dict"
        ],
        strict=True,
    )

    gnn.load_state_dict(
        phase_b[
            "gnn_state_dict"
        ],
        strict=True,
    )

    fusion.load_state_dict(
        phase_b[
            "fusion_state_dict"
        ],
        strict=True,
    )

    model = M3Model(
        bert,
        gnn,
        fusion,
    )

    model.to(device)

    model.eval()

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

    matches = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = candidate.resolve()

            matches[
                str(resolved)
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            f"cannot uniquely resolve graph: "
            f"{value}; matches={len(matches)}"
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
                f"unexpected graph schema: "
                f"{path.name}"
            )

        x = np.asarray(
            graph["x"]
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

    if edge_index.dtype != np.int64:

        raise RuntimeError(
            f"{path.name}: invalid edge dtype"
        )

    if not (
        16
        <= edge_index.shape[1]
        <= 34
    ):

        raise RuntimeError(
            f"{path.name}: invalid G2 edge count"
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
            x.astype(
                np.float32,
                copy=True,
            )
        ),
        edge_index=torch.from_numpy(
            edge_index.copy()
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

            self.graphs[
                index
            ],

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
# LOAD VAL ONLY
# =============================================================================

def load_validation_data():

    banner(
        "3. LOAD VALIDATION DATA ONLY"
    )

    token_manifest = pd.read_parquet(
        TOKEN_MANIFEST
    )

    token_manifest = token_manifest.copy()

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

    # -------------------------------------------------------------------------
    # VAL labels only.
    # -------------------------------------------------------------------------

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

    targets = table.to_pandas()

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

    require(
        len(targets)
        == N_VAL,
        "VAL target rows loaded = 514",
    )

    require(
        set(
            targets["split"]
        )
        == {"val"},
        "target-value load contains VAL only",
    )

    lookup = targets.set_index(
        "track_id",
        drop=False,
    )

    require(
        set(val_ids)
        ==
        set(
            targets[
                "track_id"
            ]
        ),
        "VAL token and target IDs match",
    )

    targets = (
        lookup.loc[
            val_ids
        ]
        .reset_index(
            drop=True
        )
    )

    y_true = (
        targets[
            y_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    source_indices = (
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

    require(
        input_ids.shape
        == (
            N_VAL,
            MAX_LENGTH,
        ),
        "VAL input_ids shape = (514,112)",
    )

    # -------------------------------------------------------------------------
    # VAL G2 graphs only.
    # -------------------------------------------------------------------------

    graph_manifest = pd.read_parquet(
        GRAPH_MANIFEST
    )

    graph_manifest = graph_manifest.copy()

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

    graph_lookup = graph_manifest.set_index(
        "track_id",
        drop=False,
    )

    print(
        "Loading 514 VAL G2 graphs..."
    )

    graphs = []

    for number, track_id in enumerate(
        val_ids,
        start=1,
    ):

        row = graph_lookup.loc[
            track_id
        ]

        if str(
            row["split"]
        ).lower() != "val":

            raise RuntimeError(
                f"non-VAL graph encountered: "
                f"{track_id}"
            )

        path = resolve_graph_path(
            str(
                row[
                    "graph_path"
                ]
            )
        )

        graphs.append(
            load_graph(path)
        )

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

    print(
        "TEST examples scheduled:  0"
    )

    return (
        input_ids,
        attention_mask,
        token_type_ids,
        graphs,
        y_true,
    )


# =============================================================================
# VALIDATION INFERENCE
# =============================================================================

def get_probabilities(
    model,
    loader,
    device,
):

    banner(
        "4. SELECTED-MODEL VAL INFERENCE"
    )

    model.eval()

    probabilities = []
    targets = []

    with torch.inference_mode():

        for (
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
            y_true,
        ) in loader:

            logits = model(
                input_ids.to(
                    device
                ),
                attention_mask.to(
                    device
                ),
                token_type_ids.to(
                    device
                ),
                graph_batch.to(
                    device
                ),
            )

            probability = torch.sigmoid(
                logits
            )

            if not torch.isfinite(
                probability
            ).all():

                raise RuntimeError(
                    "non-finite VAL probabilities"
                )

            probabilities.append(
                probability
                .cpu()
                .numpy()
            )

            targets.append(
                y_true.numpy()
            )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    require(
        probabilities.shape
        == (
            N_VAL,
            NUM_LABELS,
        ),
        (
            "VAL probability matrix "
            "= (514,20)"
        ),
    )

    macro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average=None,
        ).mean()
    )

    print(
        f"Stored M3-B VAL macro AP: "
        f"{EXPECTED_M3_VAL_MACRO_AP:.12f}"
    )

    print(
        f"Reconstructed VAL macro AP: "
        f"{macro_ap:.12f}"
    )

    require(
        np.isclose(
            macro_ap,
            EXPECTED_M3_VAL_MACRO_AP,
            rtol=0.0,
            atol=1e-8,
        ),
        (
            "selected M3-B checkpoint "
            "reproduces accepted VAL macro AP"
        ),
    )

    return (
        targets,
        probabilities,
        macro_ap,
    )


# =============================================================================
# THRESHOLD SWEEP
# =============================================================================

def tune_threshold(
    y_true,
    probabilities,
):

    banner(
        "5. VALIDATION-ONLY THRESHOLD SWEEP"
    )

    rows = []

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
                float(threshold),

            "macro_f1":
                macro_f1,

            "micro_f1":
                micro_f1,

            "per_label_f1":
                [
                    float(value)
                    for value
                    in per_label_f1
                ],
        }

        rows.append(row)

        print(
            f"threshold={threshold:.2f}  "
            f"macro_F1={macro_f1:.8f}  "
            f"micro_F1={micro_f1:.8f}"
        )

    # -------------------------------------------------------------------------
    # Frozen selection rule:
    #
    # 1. Macro-F1 descending
    # 2. Micro-F1 descending
    # 3. distance to 0.50 ascending
    # 4. threshold ascending
    # -------------------------------------------------------------------------

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
        "Selected threshold:"
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
# MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 56 — TASK-3 M3 VALIDATION THRESHOLD TUNING"
    )

    print(
        "Model:"
    )

    print(
        "  selected M3 Phase B / seed42"
    )

    print()

    print(
        "Thresholds:"
    )

    print(
        "  0.30, 0.35, ..., 0.70"
    )

    print()

    print(
        "Primary threshold criterion:"
    )

    print(
        "  VAL Macro-F1"
    )

    print()

    print(
        "Model selection metric remains:"
    )

    print(
        "  VAL macro mean per-label AP"
    )

    print()

    print(
        "Training:          NO"
    )

    print(
        "TEST inference:    NO"
    )

    print(
        "TEST labels:       NO"
    )

    try:

        # ---------------------------------------------------------------------
        # Prevent accidental repeated threshold selection.
        # ---------------------------------------------------------------------

        banner(
            "1. SELECTED MODEL IDENTITY"
        )

        require(
            not OUTPUT.exists(),
            (
                "threshold artifact does not "
                "already exist"
            ),
        )

        require(
            not OUTPUT_TEMP.exists(),
            (
                "temporary threshold artifact "
                "does not exist"
            ),
        )

        expected = {
            TASK3_SELECTION:
                EXPECTED_SELECTION_SHA256,

            T1_CHECKPOINT:
                EXPECTED_T1_SHA256,

            A4_CHECKPOINT:
                EXPECTED_A4_SHA256,

            M3_PHASEA_CHECKPOINT:
                EXPECTED_M3_PHASEA_SHA256,

            M3_PHASEB_CHECKPOINT:
                EXPECTED_M3_PHASEB_SHA256,

            TOKEN_CACHE:
                EXPECTED_TOKEN_CACHE_SHA256,

            TOKEN_MANIFEST:
                EXPECTED_TOKEN_MANIFEST_SHA256,

            GRAPH_MANIFEST:
                EXPECTED_GRAPH_MANIFEST_SHA256,

            TARGETS:
                EXPECTED_TARGETS_SHA256,
        }

        for path, expected_sha in (
            expected.items()
        ):

            require(
                path.is_file(),
                (
                    f"artifact exists: "
                    f"{path.relative_to(ROOT)}"
                ),
            )

            require(
                sha256_file(path)
                == expected_sha,
                (
                    f"{path.name} SHA256 "
                    "matches accepted identity"
                ),
            )

        selection = load_json(
            TASK3_SELECTION
        )

        require(
            selection.get(
                "selected_task3_fusion"
            )
            == "M3_phaseB",
            (
                "Task-3 selected fusion "
                "is M3 Phase B"
            ),
        )

        require(
            selection.get(
                "selected_checkpoint_sha256"
            )
            == EXPECTED_M3_PHASEB_SHA256,
            (
                "selection references accepted "
                "M3 Phase-B checkpoint"
            ),
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA unavailable"
            )

        device = torch.device(
            "cuda"
        )

        model = build_selected_model(
            device
        )

        (
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            targets,
        ) = load_validation_data()

        dataset = ValidationDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            graphs,
            targets,
        )

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_batch,
        )

        (
            y_true,
            probabilities,
            macro_ap,
        ) = get_probabilities(
            model,
            loader,
            device,
        )

        (
            sweep,
            best,
        ) = tune_threshold(
            y_true,
            probabilities,
        )

        artifact = {
            "artifact_type":
                "task3_threshold_selection",

            "version":
                1,

            "variant":
                "M3",

            "phase":
                "B",

            "seed":
                42,

            "selection_basis":
                "VALIDATION only",

            "checkpoint_sha256":
                EXPECTED_M3_PHASEB_SHA256,

            "task3_model_selection_sha256":
                EXPECTED_SELECTION_SHA256,

            "threshold_candidates":
                [
                    float(x)
                    for x in THRESHOLDS
                ],

            "threshold_selection_rule": {
                "primary":
                    "maximum validation Macro-F1",

                "tie_break_1":
                    "higher validation Micro-F1",

                "tie_break_2":
                    (
                        "threshold closest "
                        "to 0.50"
                    ),

                "tie_break_3":
                    "lower threshold",
            },

            "threshold_independent_validation": {
                "macro_mean_per_label_AP":
                    macro_ap,
            },

            "sweep":
                sweep,

            "selected_threshold":
                best[
                    "threshold"
                ],

            "selected_validation_macro_f1":
                best[
                    "macro_f1"
                ],

            "selected_validation_micro_f1":
                best[
                    "micro_f1"
                ],

            "selected_validation_per_label_f1":
                best[
                    "per_label_f1"
                ],

            "test_target_rows_loaded":
                0,

            "test_graph_files_opened":
                0,

            "test_examples_evaluated":
                0,

            "threshold_tuned_on_test":
                False,

            "next_stage":
                (
                    "three-seed winning-"
                    "configuration reruns"
                ),
        }

        write_json_atomic(
            OUTPUT,
            artifact,
        )

        output_sha = sha256_file(
            OUTPUT
        )

    except Exception as exc:

        banner(
            "SCRIPT 56 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()
        print(
            "Do NOT proceed to multi-seed "
            "reruns or TEST."
        )

        return 1

    banner(
        "SCRIPT 56 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Selected model:"
    )

    print(
        "  M3 Early Fusion / Phase B / seed42"
    )

    print()

    print(
        f"VAL macro AP: "
        f"{macro_ap:.8f}"
    )

    print()

    print(
        "Frozen threshold:"
    )

    print(
        f"  {best['threshold']:.2f}"
    )

    print()

    print(
        f"VAL Macro-F1: "
        f"{best['macro_f1']:.8f}"
    )

    print(
        f"VAL Micro-F1: "
        f"{best['micro_f1']:.8f}"
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

    print()

    print(
        "Threshold artifact:"
    )

    print(
        f"  {OUTPUT.relative_to(ROOT)}"
    )

    print()

    print(
        "Threshold artifact SHA256:"
    )

    print(
        f"  {output_sha}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  Three-seed winner reruns:"
    )

    print(
        "  seeds {42, 1337, 2026}"
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