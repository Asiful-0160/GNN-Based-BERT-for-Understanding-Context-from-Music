#!/usr/bin/env python3

"""
Script 44 — Formal Task-2 A4 GAT/G2 model-selection run

Pipeline Plan v6
----------------
A4:
    GAT using the graph topology selected from A2/A3.

Script 42 froze:
    selected graph = G2

Frozen GAT interpretation established before formal A4 results:
    raw x (N,140)
    -> Linear(140,64)

    -> GATConv(
           64,
           32,
           heads=4,
           concat=True,
           dropout=0.2
       )
       output = 4*32 = 128
    -> ReLU

    -> GATConv(
           128,
           32,
           heads=4,
           concat=True,
           dropout=0.2
       )
       output = 128
    -> ReLU

    -> GATConv(
           128,
           128,
           heads=4,
           concat=False,
           dropout=0.2
       )
       output = 128

    -> global_mean_pool
    -> Linear(128,20)

Additional fixed implementation details:
    add_self_loops=True
    bias=True
    residual=False
    GAT dropout=0.2 is attention-coefficient dropout
    no separate feature dropout

Training:
    AdamW
    LR = 1e-3
    weight decay = 1e-4
    batch size = 32
    maximum epochs = 15
    early stopping patience = 3
    scheduler = none

Loss:
    BCEWithLogitsLoss using the same TRAIN-only pos_weight
    used in earlier Task-2 models.

Selection:
    validation macro mean per-label average precision

Restrictions:
    TEST targets/examples/inference are never loaded.
    No threshold sweep.
    Frozen artifacts are never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric

from sklearn.metrics import average_precision_score

from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GRAPH_MANIFEST_PATH = (
    ROOT
    / "data/processed/graph_manifest.parquet"
)

TARGETS_PATH = (
    ROOT
    / "data/processed/labels/musiccaps_targets.parquet"
)

VOCAB_PATH = (
    ROOT
    / "data/splits/label_vocab.json"
)

PRETRAINING_LOCK_PATH = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

GRAPH_SELECTION_PATH = (
    ROOT
    / "results/runs/task2/task2_graph_selection.json"
)

TRIALS_PATH = (
    ROOT
    / "results/csv/hyperparameter_trials.csv"
)

EPOCH_METRICS_PATH = (
    ROOT
    / "results/csv/epoch_metrics.csv"
)

RUNS_DIR = (
    ROOT
    / "results/runs/task2"
)

MODEL_DIR = (
    ROOT
    / "models/task2_candidates"
)


# ============================================================
# FROZEN IDENTITIES
# ============================================================

EXPECTED_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_DATASET_TRACKS = 5140
EXPECTED_TRAIN_TRACKS = 4112
EXPECTED_VAL_TRACKS = 514

EXPECTED_NODES = 9

NODE_INPUT_DIM = 140
NODE_EMBED_DIM = 64

GNN_HIDDEN_DIM = 128

HEADS = 4
HIDDEN_PER_HEAD = 32

GAT_DROPOUT = 0.2

EXPECTED_G2_MIN_EDGES = 16
EXPECTED_G2_MAX_EDGES = 34

NUM_LABELS = 20

TARGET_COLUMNS = [
    f"y_{i:02d}"
    for i in range(NUM_LABELS)
]

EXPECTED_GRAPH_KEYS = {
    "x",
    "edge_index_g1",
    "edge_index_g2",
    "similarity_threshold",
    "nonlocal_only",
    "extra_degree_cap",
    "rule_version",
}


# ============================================================
# FORMAL A4 SETTINGS
# ============================================================

TASK = "task2"
VARIANT = "A4"

MODEL_NAME = "GAT"
GRAPH_TYPE = "G2"

SEED = 42

BATCH_SIZE = 32

MAX_EPOCHS = 15

LEARNING_RATE = 1.0e-3

WEIGHT_DECAY = 1.0e-4

EARLY_STOP_PATIENCE = 3

POS_WEIGHT_CAP = 10.0

SELECTION_METRIC_NAME = (
    "val_auc_pr_macro_mean_ap"
)

IMPROVEMENT_EPSILON = 1.0e-12


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(block)

    return digest.hexdigest()


def resolve_path(
    raw_path: str,
) -> Path:

    path = Path(
        str(raw_path)
    )

    if not path.is_absolute():
        path = ROOT / path

    return path


def relative(
    path: Path,
) -> str:

    try:

        return str(
            path.relative_to(ROOT)
        )

    except ValueError:

        return str(path)


def set_seed(
    seed: int,
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


def get_git_commit() -> str:

    try:

        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        value = (
            result.stdout.strip()
        )

        if value:
            return value

    except Exception:
        pass

    return "UNAVAILABLE"


# ============================================================
# SAFE CSV HELPERS
# ============================================================

def write_dataframe_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )


def append_rows_to_csv(
    rows: list[dict],
    path: Path,
) -> None:

    new_frame = pd.DataFrame(
        rows
    )

    if path.exists():

        old_frame = pd.read_csv(
            path
        )

        combined = pd.concat(
            [
                old_frame,
                new_frame,
            ],
            ignore_index=True,
            sort=False,
        )

    else:

        combined = new_frame

    write_dataframe_atomic(
        combined,
        path,
    )


# ============================================================
# DUPLICATE FORMAL-RUN PROTECTION
# ============================================================

def assert_no_completed_duplicate() -> None:

    if not TRIALS_PATH.exists():
        return

    trials = pd.read_csv(
        TRIALS_PATH
    )

    required = {
        "task",
        "variant",
        "seed",
        "status",
    }

    if not required.issubset(
        trials.columns
    ):
        return

    duplicate = trials.loc[
        (trials["task"] == TASK)
        & (trials["variant"] == VARIANT)
        & (trials["seed"] == SEED)
        & (trials["status"] == "COMPLETE")
    ]

    require(
        len(duplicate) == 0,
        (
            "A completed formal A4 "
            "GAT/G2 seed-42 run already "
            "exists. Refusing accidental "
            "rerun."
        ),
    )


# ============================================================
# VERIFY INPUTS / GRAPH SELECTION
# ============================================================

def verify_inputs():

    section(
        "1. FROZEN ARTIFACT VERIFICATION"
    )

    lock_sha = sha256_file(
        PRETRAINING_LOCK_PATH
    )

    print(
        "Expected lock SHA256:",
        EXPECTED_LOCK_SHA256,
    )

    print(
        "Actual lock SHA256:  ",
        lock_sha,
    )

    require(
        lock_sha
        == EXPECTED_LOCK_SHA256,
        "Pretraining lock changed.",
    )

    with PRETRAINING_LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        lock = json.load(file)

    require(
        lock.get(
            "dataset_tracks"
        )
        == EXPECTED_DATASET_TRACKS,
        "Frozen dataset size changed.",
    )

    critical_files = lock.get(
        "critical_files"
    )

    require(
        isinstance(
            critical_files,
            dict,
        ),
        "Malformed critical_files.",
    )

    for key in [
        "graph_manifest",
        "targets",
        "config",
        "original_dataset_lock",
    ]:

        record = critical_files.get(
            key
        )

        require(
            isinstance(record, dict),
            f"Missing frozen record {key}.",
        )

        path = ROOT / record["path"]

        require(
            path.is_file(),
            (
                "Missing frozen artifact: "
                f"{path}"
            ),
        )

        require(
            sha256_file(path)
            == record["sha256"],
            (
                "Frozen artifact changed: "
                f"{key}"
            ),
        )

        print(
            f"PASS  {key:<24} "
            f"{record['path']}"
        )

    print()
    print(
        "Frozen preprocessing identity: PASS"
    )


    # ========================================================
    # Script-42 selection
    # ========================================================

    require(
        GRAPH_SELECTION_PATH.is_file(),
        (
            "Missing Script-42 "
            "graph-selection artifact."
        ),
    )

    with GRAPH_SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        selection = json.load(file)

    require(
        selection[
            "status"
        ]
        == "FROZEN",
        "Graph selection not frozen.",
    )

    require(
        selection[
            "selection_stage"
        ]
        == "graph_topology_for_A4_GAT",
        (
            "Unexpected graph-selection "
            "stage."
        ),
    )

    require(
        selection[
            "selection_basis"
        ]
        == "validation_only",
        (
            "Graph selection was not "
            "validation-only."
        ),
    )

    require(
        selection[
            "test_split_used"
        ]
        is False,
        (
            "Graph selection indicates "
            "TEST usage."
        ),
    )

    require(
        selection[
            "pretraining_lock_sha256"
        ]
        == lock_sha,
        (
            "Graph selection references "
            "different frozen artifacts."
        ),
    )

    require(
        selection[
            "a4_graph"
        ]
        == GRAPH_TYPE,
        (
            "Frozen A4 graph is not G2."
        ),
    )

    winner = selection[
        "winner"
    ]

    require(
        winner[
            "variant"
        ]
        == "A3",
        "Selected graph winner != A3.",
    )

    require(
        winner[
            "model"
        ]
        == "GraphSAGE",
        (
            "Selected graph winner "
            "is not GraphSAGE."
        ),
    )

    require(
        winner[
            "graph"
        ]
        == "G2",
        (
            "Selected graph winner "
            "is not G2."
        ),
    )

    selected_checkpoint = (
        ROOT
        / winner[
            "checkpoint_path"
        ]
    )

    require(
        selected_checkpoint.is_file(),
        (
            "Selected A3 checkpoint "
            "is missing."
        ),
    )

    require(
        sha256_file(
            selected_checkpoint
        )
        == winner[
            "checkpoint_sha256"
        ],
        (
            "Selected A3 checkpoint "
            "SHA256 changed."
        ),
    )

    print()
    print(
        "Script-42 graph selection: PASS"
    )

    print(
        "Selected A4 graph:",
        selection[
            "a4_graph"
        ],
    )

    print(
        "Selection winner:",
        winner[
            "variant"
        ],
    )

    print(
        "Selected topology VAL macro AP:",
        f"{winner['best_val_auc_pr_macro']:.8f}",
    )

    print(
        "Selected A3 checkpoint:",
        relative(
            selected_checkpoint
        ),
    )

    print(
        "Selected A3 checkpoint SHA256:",
        winner[
            "checkpoint_sha256"
        ],
    )

    print()
    print(
        "TEST targets/examples loaded: 0"
    )

    return {
        "pretraining_lock_sha256":
            lock_sha,

        "graph_selection_path":
            relative(
                GRAPH_SELECTION_PATH
            ),

        "selected_graph":
            selection[
                "a4_graph"
            ],

        "a3_checkpoint_path":
            relative(
                selected_checkpoint
            ),

        "a3_checkpoint_sha256":
            winner[
                "checkpoint_sha256"
            ],

        "a3_best_val_auc_pr_macro":
            float(
                winner[
                    "best_val_auc_pr_macro"
                ]
            ),
    }


# ============================================================
# LOAD TRAIN + VAL
# ============================================================

def load_development_data():

    section(
        "2. TRAIN + VAL GRAPH ALIGNMENT"
    )

    manifest = pd.read_parquet(
        GRAPH_MANIFEST_PATH,
        columns=[
            "track_id",
            "split",
            "graph_path",
            "nodes",
            "node_feature_dim",
            "g2_directed_edges",
            "cache_status",
        ],
    )

    require(
        len(manifest)
        == EXPECTED_DATASET_TRACKS,
        "Graph manifest size mismatch.",
    )

    require(
        manifest[
            "track_id"
        ].is_unique,
        "Duplicate graph track_id.",
    )

    require(
        (
            manifest[
                "nodes"
            ]
            == EXPECTED_NODES
        ).all(),
        "Unexpected node count.",
    )

    require(
        (
            manifest[
                "node_feature_dim"
            ]
            == NODE_INPUT_DIM
        ).all(),
        (
            "Unexpected node-feature "
            "dimension."
        ),
    )

    require(
        manifest[
            "g2_directed_edges"
        ].between(
            EXPECTED_G2_MIN_EDGES,
            EXPECTED_G2_MAX_EDGES,
            inclusive="both",
        ).all(),
        (
            "Invalid G2 edge count "
            "in manifest."
        ),
    )

    print(
        "cache_status values:",
        sorted(
            manifest[
                "cache_status"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    train_targets = pd.read_parquet(
        TARGETS_PATH,
        filters=[
            (
                "split",
                "==",
                "train",
            )
        ],
    )

    val_targets = pd.read_parquet(
        TARGETS_PATH,
        filters=[
            (
                "split",
                "==",
                "val",
            )
        ],
    )

    require(
        len(train_targets)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN count mismatch.",
    )

    require(
        len(val_targets)
        == EXPECTED_VAL_TRACKS,
        "VAL count mismatch.",
    )

    train_frame = (
        train_targets.merge(
            manifest,
            on="track_id",
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_target",
                "_manifest",
            ),
        )
        .sort_values(
            "track_id"
        )
        .reset_index(
            drop=True
        )
    )

    val_frame = (
        val_targets.merge(
            manifest,
            on="track_id",
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_target",
                "_manifest",
            ),
        )
        .sort_values(
            "track_id"
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(train_frame)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN mapping failed.",
    )

    require(
        len(val_frame)
        == EXPECTED_VAL_TRACKS,
        "VAL mapping failed.",
    )

    require(
        (
            train_frame[
                "split_target"
            ]
            == train_frame[
                "split_manifest"
            ]
        ).all(),
        "TRAIN split mismatch.",
    )

    require(
        (
            val_frame[
                "split_target"
            ]
            == val_frame[
                "split_manifest"
            ]
        ).all(),
        "VAL split mismatch.",
    )

    require(
        set(
            train_frame[
                "track_id"
            ]
        ).isdisjoint(
            set(
                val_frame[
                    "track_id"
                ]
            )
        ),
        "TRAIN/VAL overlap.",
    )

    print(
        "TRAIN:",
        len(train_frame),
    )

    print(
        "VAL:  ",
        len(val_frame),
    )

    print(
        "TEST target rows loaded: 0"
    )

    print(
        "Graph/target alignment: PASS"
    )

    return (
        train_frame,
        val_frame,
    )


# ============================================================
# SHARED POS_WEIGHT
# ============================================================

def compute_pos_weight(
    train_frame: pd.DataFrame,
    a3_checkpoint_path: Path,
):

    section(
        "3. SHARED TRAIN-ONLY CLASS WEIGHTS"
    )

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        vocab = json.load(file)

    labels = vocab[
        "labels"
    ]

    require(
        len(labels)
        == NUM_LABELS,
        "Expected 20 labels.",
    )

    label_names = [
        item[
            "label"
        ]
        for item
        in labels
    ]

    train_y = train_frame[
        TARGET_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    require(
        set(
            np.unique(
                train_y
            ).tolist()
        ).issubset(
            {
                0.0,
                1.0,
            }
        ),
        "TRAIN targets not binary.",
    )

    support = (
        train_y
        .sum(axis=0)
        .astype(np.int64)
    )

    frozen_support = np.asarray(
        [
            int(
                item[
                    "train_support"
                ]
            )
            for item
            in labels
        ],
        dtype=np.int64,
    )

    require(
        np.array_equal(
            support,
            frozen_support,
        ),
        (
            "TRAIN support differs "
            "from frozen vocabulary."
        ),
    )

    negatives = (
        EXPECTED_TRAIN_TRACKS
        - support
    )

    pos_weight = np.minimum(
        negatives / support,
        POS_WEIGHT_CAP,
    ).astype(
        np.float32
    )

    require(
        np.isfinite(
            pos_weight
        ).all(),
        "Non-finite pos_weight.",
    )

    a3_checkpoint = torch.load(
        a3_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        a3_checkpoint[
            "variant"
        ]
        == "A3",
        "Selected checkpoint != A3.",
    )

    require(
        a3_checkpoint[
            "graph"
        ]
        == "G2",
        "Selected checkpoint != G2.",
    )

    a3_pos_weight = np.asarray(
        a3_checkpoint[
            "pos_weight"
        ],
        dtype=np.float32,
    )

    require(
        np.array_equal(
            pos_weight,
            a3_pos_weight,
        ),
        (
            "A4 pos_weight differs "
            "from selected A3."
        ),
    )

    print(
        "TRAIN support == frozen vocabulary: PASS"
    )

    print(
        "Formula min(negative/positive,10): PASS"
    )

    print(
        "A3 pos_weight == A4 pos_weight: PASS"
    )

    return (
        pos_weight,
        label_names,
    )


# ============================================================
# G2 LOADER
# ============================================================

def load_g2_graph(
    path: Path,
    target: np.ndarray,
) -> Data:

    require(
        path.is_file(),
        f"Graph missing: {path}",
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as graph:

        require(
            set(
                graph.files
            )
            == EXPECTED_GRAPH_KEYS,
            (
                "Frozen graph schema "
                f"changed: {path}"
            ),
        )

        x = np.array(
            graph["x"],
            dtype=np.float32,
            copy=True,
        )

        edge_g1 = np.array(
            graph[
                "edge_index_g1"
            ],
            dtype=np.int64,
            copy=True,
        )

        edge_g2 = np.array(
            graph[
                "edge_index_g2"
            ],
            dtype=np.int64,
            copy=True,
        )

        threshold = float(
            graph[
                "similarity_threshold"
            ][0]
        )

        nonlocal_only = bool(
            graph[
                "nonlocal_only"
            ][0]
        )

        degree_cap = int(
            graph[
                "extra_degree_cap"
            ][0]
        )

        rule_version = int(
            graph[
                "rule_version"
            ][0]
        )

    require(
        x.shape
        == (
            EXPECTED_NODES,
            NODE_INPUT_DIM,
        ),
        (
            "Unexpected node shape "
            f"in {path}: {x.shape}"
        ),
    )

    require(
        np.isfinite(x).all(),
        (
            "Non-finite node features "
            f"in {path}"
        ),
    )

    require(
        np.isclose(
            threshold,
            0.85,
            rtol=0.0,
            atol=1e-6,
        ),
        (
            "Unexpected G2 threshold "
            f"in {path}"
        ),
    )

    require(
        nonlocal_only is True,
        (
            "Unexpected G2 "
            f"nonlocal rule in {path}"
        ),
    )

    require(
        degree_cap == 2,
        (
            "Unexpected G2 degree cap "
            f"in {path}"
        ),
    )

    require(
        rule_version == 2,
        (
            "Unexpected G2 rule version "
            f"in {path}"
        ),
    )

    require(
        edge_g2.ndim == 2
        and edge_g2.shape[0] == 2,
        (
            "Malformed G2 edge_index "
            f"in {path}"
        ),
    )

    edge_count = int(
        edge_g2.shape[1]
    )

    require(
        EXPECTED_G2_MIN_EDGES
        <= edge_count
        <= EXPECTED_G2_MAX_EDGES,
        (
            "Unexpected G2 edge count "
            f"in {path}"
        ),
    )

    g1_set = {
        (
            int(src),
            int(dst),
        )
        for src, dst
        in edge_g1.T
    }

    g2_set = {
        (
            int(src),
            int(dst),
        )
        for src, dst
        in edge_g2.T
    }

    require(
        len(g2_set)
        == edge_count,
        (
            "Duplicate G2 edges "
            f"in {path}"
        ),
    )

    require(
        g1_set.issubset(
            g2_set
        ),
        (
            "G1 is not subset of "
            f"G2 in {path}"
        ),
    )

    for src, dst in g2_set:

        require(
            0 <= src < EXPECTED_NODES,
            (
                "Out-of-range source "
                f"in {path}"
            ),
        )

        require(
            0 <= dst < EXPECTED_NODES,
            (
                "Out-of-range target "
                f"in {path}"
            ),
        )

        require(
            src != dst,
            (
                "Self-loop found in "
                f"frozen G2: {path}"
            ),
        )

        require(
            (
                dst,
                src,
            )
            in g2_set,
            (
                "Non-bidirectional G2 "
                f"edge in {path}"
            ),
        )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    require(
        target.shape
        == (
            NUM_LABELS,
        ),
        "Target dimension != 20.",
    )

    return Data(
        x=torch.from_numpy(
            x
        ),

        edge_index=torch.from_numpy(
            edge_g2
        ),

        y=torch.tensor(
            target,
            dtype=torch.float32,
        ).unsqueeze(0),

        source_edge_count=torch.tensor(
            [
                edge_count
            ],
            dtype=torch.long,
        ),
    )


def build_graph_dataset(
    frame: pd.DataFrame,
):

    dataset = []

    for row in frame.itertuples(
        index=False
    ):

        path = resolve_path(
            getattr(
                row,
                "graph_path"
            )
        )

        target = np.asarray(
            [
                getattr(
                    row,
                    column
                )
                for column
                in TARGET_COLUMNS
            ],
            dtype=np.float32,
        )

        dataset.append(
            load_g2_graph(
                path,
                target,
            )
        )

    return dataset


# ============================================================
# GAT MODEL — IDENTICAL TO SCRIPT 43
# ============================================================

class GATG2(nn.Module):

    def __init__(self):

        super().__init__()

        self.node_projection = nn.Linear(
            NODE_INPUT_DIM,
            NODE_EMBED_DIM,
        )

        self.gat1 = GATConv(
            NODE_EMBED_DIM,
            HIDDEN_PER_HEAD,
            heads=HEADS,
            concat=True,
            dropout=GAT_DROPOUT,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.gat2 = GATConv(
            GNN_HIDDEN_DIM,
            HIDDEN_PER_HEAD,
            heads=HEADS,
            concat=True,
            dropout=GAT_DROPOUT,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.gat3 = GATConv(
            GNN_HIDDEN_DIM,
            GNN_HIDDEN_DIM,
            heads=HEADS,
            concat=False,
            dropout=GAT_DROPOUT,
            add_self_loops=True,
            bias=True,
            residual=False,
        )

        self.classifier = nn.Linear(
            GNN_HIDDEN_DIM,
            NUM_LABELS,
        )

    def forward(
        self,
        x,
        edge_index,
        batch,
    ):

        x = self.node_projection(
            x
        )

        x = self.gat1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.gat2(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.gat3(
            x,
            edge_index,
        )

        graph_embedding = (
            global_mean_pool(
                x,
                batch,
            )
        )

        logits = self.classifier(
            graph_embedding
        )

        return (
            logits,
            graph_embedding,
        )


# ============================================================
# METRICS
# ============================================================

def compute_ap_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
):

    require(
        y_true.shape
        == y_probability.shape,
        "Metric shape mismatch.",
    )

    require(
        y_true.shape[1]
        == NUM_LABELS,
        (
            "Metric label "
            "dimension != 20."
        ),
    )

    per_label_ap = (
        average_precision_score(
            y_true,
            y_probability,
            average=None,
        )
    )

    require(
        len(per_label_ap)
        == NUM_LABELS,
        "Expected 20 AP values.",
    )

    require(
        np.isfinite(
            per_label_ap
        ).all(),
        (
            "Non-finite per-label "
            "AP values."
        ),
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

    require(
        math.isfinite(
            macro_ap
        ),
        "Non-finite macro AP.",
    )

    require(
        math.isfinite(
            micro_ap
        ),
        "Non-finite micro AP.",
    )

    return (
        macro_ap,
        micro_ap,
        per_label_ap.astype(
            np.float64
        ),
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in loader:

        batch = batch.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        require(
            logits.shape
            == batch.y.shape,
            (
                "TRAIN logit/target "
                "shape mismatch."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            (
                "Non-finite "
                "TRAIN logits."
            ),
        )

        loss = criterion(
            logits,
            batch.y,
        )

        require(
            bool(
                torch.isfinite(
                    loss
                )
            ),
            "Non-finite TRAIN loss.",
        )

        loss.backward()

        for name, parameter in (
            model.named_parameters()
        ):

            if parameter.grad is None:
                continue

            require(
                bool(
                    torch.isfinite(
                        parameter.grad
                    ).all()
                ),
                (
                    "Non-finite gradient: "
                    f"{name}"
                ),
            )

        optimizer.step()

        batch_graphs = int(
            batch.num_graphs
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
            )
            * batch_graphs
        )

        total_graphs += (
            batch_graphs
        )

    require(
        total_graphs
        == EXPECTED_TRAIN_TRACKS,
        (
            "TRAIN epoch did not "
            "process exactly 4112 graphs."
        ),
    )

    return (
        total_loss
        / total_graphs
    )


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_graphs = 0

    target_batches = []
    probability_batches = []

    for batch in loader:

        batch = batch.to(
            device
        )

        logits, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        require(
            logits.shape
            == batch.y.shape,
            (
                "VAL logit/target "
                "shape mismatch."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            (
                "Non-finite "
                "VAL logits."
            ),
        )

        loss = criterion(
            logits,
            batch.y,
        )

        require(
            bool(
                torch.isfinite(
                    loss
                )
            ),
            "Non-finite VAL loss.",
        )

        probability = torch.sigmoid(
            logits
        )

        require(
            bool(
                torch.isfinite(
                    probability
                ).all()
            ),
            (
                "Non-finite VAL "
                "probabilities."
            ),
        )

        batch_graphs = int(
            batch.num_graphs
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
            )
            * batch_graphs
        )

        total_graphs += (
            batch_graphs
        )

        target_batches.append(
            batch.y.detach()
            .cpu()
            .numpy()
        )

        probability_batches.append(
            probability.detach()
            .cpu()
            .numpy()
        )

    require(
        total_graphs
        == EXPECTED_VAL_TRACKS,
        (
            "VAL did not process "
            "exactly 514 graphs."
        ),
    )

    y_true = np.concatenate(
        target_batches,
        axis=0,
    )

    y_probability = np.concatenate(
        probability_batches,
        axis=0,
    )

    (
        macro_ap,
        micro_ap,
        per_label_ap,
    ) = compute_ap_metrics(
        y_true,
        y_probability,
    )

    return {
        "loss":
            total_loss
            / total_graphs,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,

        "per_label_ap":
            per_label_ap,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    val_macro_ap: float,
    val_loss: float,
    run_id: str,
    pos_weight: np.ndarray,
    lock_sha: str,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dict = {
        key:
            value.detach()
            .cpu()
        for key, value
        in model.state_dict().items()
    }

    checkpoint = {
        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "graph":
            GRAPH_TYPE,

        "run_id":
            run_id,

        "seed":
            SEED,

        "epoch":
            epoch,

        "val_auc_pr_macro_mean_ap":
            val_macro_ap,

        "val_loss":
            val_loss,

        "architecture": {
            "raw_node_dim":
                NODE_INPUT_DIM,

            "node_projection":
                "Linear(140,64)",

            "gat1": {
                "in_channels":
                    64,

                "out_channels_per_head":
                    32,

                "heads":
                    HEADS,

                "concat":
                    True,

                "output_dim":
                    128,
            },

            "gat2": {
                "in_channels":
                    128,

                "out_channels_per_head":
                    32,

                "heads":
                    HEADS,

                "concat":
                    True,

                "output_dim":
                    128,
            },

            "gat3": {
                "in_channels":
                    128,

                "out_channels_per_head":
                    128,

                "heads":
                    HEADS,

                "concat":
                    False,

                "output_dim":
                    128,
            },

            "attention_dropout":
                GAT_DROPOUT,

            "extra_feature_dropout":
                None,

            "add_self_loops":
                True,

            "bias":
                True,

            "residual":
                False,

            "pool":
                "global_mean_pool",

            "classifier":
                "Linear(128,20)",

            "pyg_version":
                torch_geometric.__version__,
        },

        "graph_rule": {
            "similarity_threshold":
                0.85,

            "nonlocal_only":
                True,

            "extra_degree_cap":
                2,

            "rule_version":
                2,
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
                EARLY_STOP_PATIENCE,

            "scheduler":
                None,
        },

        "pos_weight":
            pos_weight.tolist(),

        "graph_selection_path":
            relative(
                GRAPH_SELECTION_PATH
            ),

        "pretraining_lock_sha256":
            lock_sha,

        "model_state_dict":
            state_dict,
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "SCRIPT 44 — FORMAL TASK-2 "
        "A4 GAT / G2"
    )
    print("=" * 78)

    print()
    print(
        "TRAIN -> optimization"
    )

    print(
        "VAL -> early stopping / "
        "checkpoint selection"
    )

    print(
        "TEST -> NOT LOADED / "
        "NOT EVALUATED"
    )

    print(
        "Threshold sweep -> NOT RUN"
    )

    print()

    print(
        "Graph topology:"
    )

    print(
        "G2 — selected and frozen "
        "by Script 42 before A4."
    )

    assert_no_completed_duplicate()


    # ========================================================
    # INPUT VERIFICATION
    # ========================================================

    artifact_info = (
        verify_inputs()
    )

    (
        train_frame,
        val_frame,
    ) = load_development_data()

    a3_checkpoint_path = (
        ROOT
        / artifact_info[
            "a3_checkpoint_path"
        ]
    )

    (
        pos_weight_np,
        label_names,
    ) = compute_pos_weight(
        train_frame,
        a3_checkpoint_path,
    )


    # ========================================================
    # LOAD G2
    # ========================================================

    section(
        "4. LOAD FROZEN G2 DATASET"
    )

    train_dataset = (
        build_graph_dataset(
            train_frame
        )
    )

    val_dataset = (
        build_graph_dataset(
            val_frame
        )
    )

    require(
        len(train_dataset)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN G2 count mismatch.",
    )

    require(
        len(val_dataset)
        == EXPECTED_VAL_TRACKS,
        "VAL G2 count mismatch.",
    )

    train_edge_counts = np.asarray(
        [
            int(
                graph[
                    "source_edge_count"
                ].item()
            )
            for graph
            in train_dataset
        ],
        dtype=np.int64,
    )

    val_edge_counts = np.asarray(
        [
            int(
                graph[
                    "source_edge_count"
                ].item()
            )
            for graph
            in val_dataset
        ],
        dtype=np.int64,
    )

    print(
        "TRAIN G2 graphs:",
        len(train_dataset),
    )

    print(
        "VAL G2 graphs:  ",
        len(val_dataset),
    )

    print(
        "TRAIN G2 edges min/mean/max:",
        int(
            train_edge_counts.min()
        ),
        f"{train_edge_counts.mean():.3f}",
        int(
            train_edge_counts.max()
        ),
    )

    print(
        "VAL G2 edges min/mean/max:  ",
        int(
            val_edge_counts.min()
        ),
        f"{val_edge_counts.mean():.3f}",
        int(
            val_edge_counts.max()
        ),
    )

    print(
        "Frozen G2 loading: PASS"
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    section(
        "5. ENVIRONMENT + REPRODUCIBILITY"
    )

    require(
        torch.cuda.is_available(),
        "CUDA required.",
    )

    device = torch.device(
        "cuda"
    )

    set_seed(SEED)

    gpu_name = (
        torch.cuda
        .get_device_name(0)
    )

    git_commit = (
        get_git_commit()
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "PyG:",
        torch_geometric.__version__,
    )

    print(
        "scikit-learn:",
        sklearn.__version__,
    )

    print(
        "GPU:",
        gpu_name,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "cudnn.deterministic:",
        torch.backends
        .cudnn.deterministic,
    )

    print(
        "cudnn.benchmark:",
        torch.backends
        .cudnn.benchmark,
    )

    print(
        "Git commit:",
        git_commit,
    )


    # ========================================================
    # RUN PATHS
    # ========================================================

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    run_id = (
        f"task2_A4_GAT_G2_"
        f"seed{SEED}_{timestamp}"
    )

    run_dir = (
        RUNS_DIR
        / run_id
    )

    checkpoint_path = (
        MODEL_DIR
        / f"{run_id}_best.pt"
    )

    require(
        not run_dir.exists(),
        "Run directory exists.",
    )

    require(
        not checkpoint_path.exists(),
        "Checkpoint path exists.",
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    section(
        "6. DATALOADERS"
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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    require(
        len(train_loader)
        == 129,
        "TRAIN batch count mismatch.",
    )

    require(
        len(val_loader)
        == 17,
        "VAL batch count mismatch.",
    )

    first_batch = next(
        iter(
            train_loader
        )
    )

    require(
        first_batch.x.shape
        == (
            BATCH_SIZE
            * EXPECTED_NODES,
            NODE_INPUT_DIM,
        ),
        (
            "Unexpected first-batch "
            "node shape."
        ),
    )

    require(
        first_batch.y.shape
        == (
            BATCH_SIZE,
            NUM_LABELS,
        ),
        (
            "Unexpected first-batch "
            "target shape."
        ),
    )

    source_edge_total = int(
        first_batch[
            "source_edge_count"
        ].sum().item()
    )

    batched_edge_total = int(
        first_batch.edge_index
        .shape[1]
    )

    require(
        source_edge_total
        == batched_edge_total,
        (
            "Variable-edge batching "
            "failed."
        ),
    )

    print(
        "TRAIN batches:",
        len(train_loader),
    )

    print(
        "VAL batches:",
        len(val_loader),
    )

    print(
        "First batch nodes:",
        tuple(
            first_batch.x.shape
        ),
    )

    print(
        "First batch G2 edges:",
        tuple(
            first_batch.edge_index.shape
        ),
    )

    print(
        "Variable-edge batching: PASS"
    )


    # ========================================================
    # MODEL
    # ========================================================

    section(
        "7. MODEL"
    )

    # Reset initialization RNG after
    # DataLoader inspection.
    set_seed(SEED)

    model = GATG2()

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    require(
        total_parameters
        == 103636,
        (
            "GAT parameter count "
            "differs from Script 43."
        ),
    )

    require(
        trainable_parameters
        == total_parameters,
        (
            "Unexpected frozen "
            "GAT parameters."
        ),
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Graph:",
        GRAPH_TYPE,
    )

    print(
        "Heads:",
        HEADS,
    )

    print(
        "Attention dropout:",
        GAT_DROPOUT,
    )

    print(
        "Final concat:",
        False,
    )

    print(
        "add_self_loops:",
        True,
    )

    model = model.to(
        device
    )


    # ========================================================
    # OPTIMIZATION
    # ========================================================

    section(
        "8. FORMAL OPTIMIZATION CONFIG"
    )

    pos_weight = torch.tensor(
        pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
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
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Maximum epochs:",
        MAX_EPOCHS,
    )

    print(
        "Early-stop patience:",
        EARLY_STOP_PATIENCE,
    )

    print(
        "Scheduler: NONE"
    )

    print(
        "Selection metric:",
        SELECTION_METRIC_NAME,
    )

    print(
        "Loss: BCEWithLogitsLoss("
        "shared TRAIN-only pos_weight)"
    )


    # ========================================================
    # RUN SPEC
    # ========================================================

    run_spec = {
        "run_id":
            run_id,

        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "graph":
            GRAPH_TYPE,

        "formal_model_selection_run":
            True,

        "graph_selection_source":
            artifact_info[
                "graph_selection_path"
            ],

        "graph_selected_before_A4":
            True,

        "seed":
            SEED,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "architecture": {
            "raw_node_dim":
                NODE_INPUT_DIM,

            "node_projection":
                "Linear(140,64)",

            "gat1": {
                "in_channels":
                    64,

                "out_channels_per_head":
                    32,

                "heads":
                    4,

                "concat":
                    True,

                "output_dim":
                    128,
            },

            "gat2": {
                "in_channels":
                    128,

                "out_channels_per_head":
                    32,

                "heads":
                    4,

                "concat":
                    True,

                "output_dim":
                    128,
            },

            "gat3": {
                "in_channels":
                    128,

                "out_channels_per_head":
                    128,

                "heads":
                    4,

                "concat":
                    False,

                "output_dim":
                    128,
            },

            "attention_dropout":
                GAT_DROPOUT,

            "extra_feature_dropout":
                None,

            "add_self_loops":
                True,

            "bias":
                True,

            "residual":
                False,

            "pool":
                "global_mean_pool",

            "classifier":
                "Linear(128,20)",

            "parameter_count":
                total_parameters,
        },

        "implementation_clarification": [
            (
                "The hidden per-head width "
                "was frozen before formal "
                "A4 results."
            ),

            (
                "First two layers use "
                "32 dimensions/head x "
                "4 concatenated heads = "
                "128 hidden dimensions."
            ),

            (
                "Final layer uses heads=4 "
                "with concat=False and "
                "128 output dimensions."
            ),

            (
                "dropout=0.2 is implemented "
                "inside GATConv as attention "
                "dropout; no separate feature "
                "dropout is added."
            ),
        ],

        "training": {
            "optimizer":
                "torch.optim.AdamW",

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "early_stop_patience":
                EARLY_STOP_PATIENCE,

            "scheduler":
                None,
        },

        "loss": {
            "name":
                "BCEWithLogitsLoss",

            "pos_weight_source":
                "TRAIN only",

            "formula":
                "min(negative/positive,10)",

            "pos_weight":
                pos_weight_np.tolist(),

            "verified_equal_to_A3":
                True,
        },

        "selection": {
            "metric":
                SELECTION_METRIC_NAME,

            "aggregation":
                "mean over 20 labels",

            "implementation":
                (
                    "sklearn.metrics."
                    "average_precision_score"
                    "(average=None), then mean"
                ),

            "threshold_free":
                True,
        },

        "frozen_artifacts": {
            "pretraining_lock_sha256":
                artifact_info[
                    "pretraining_lock_sha256"
                ],

            "a3_checkpoint_path":
                artifact_info[
                    "a3_checkpoint_path"
                ],

            "a3_checkpoint_sha256":
                artifact_info[
                    "a3_checkpoint_sha256"
                ],
        },

        "environment": {
            "torch":
                torch.__version__,

            "torch_geometric":
                torch_geometric.__version__,

            "sklearn":
                sklearn.__version__,

            "gpu":
                gpu_name,

            "git_commit":
                git_commit,
        },
    }

    with (
        run_dir
        / "run_spec.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            run_spec,
            file,
            indent=2,
        )


    # ========================================================
    # FORMAL TRAINING
    # ========================================================

    section(
        "9. FORMAL TRAINING"
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start_time = (
        time.perf_counter()
    )

    best_metric = -math.inf

    best_epoch = None

    best_val_loss = None

    epochs_without_improvement = 0

    stopped_early = False

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = float(
            validation[
                "loss"
            ]
        )

        val_macro_ap = float(
            validation[
                "macro_ap"
            ]
        )

        val_micro_ap = float(
            validation[
                "micro_ap"
            ]
        )

        improved = (
            val_macro_ap
            > (
                best_metric
                + IMPROVEMENT_EPSILON
            )
        )

        if improved:

            best_metric = (
                val_macro_ap
            )

            best_epoch = (
                epoch
            )

            best_val_loss = (
                val_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                epoch=epoch,
                val_macro_ap=val_macro_ap,
                val_loss=val_loss,
                run_id=run_id,
                pos_weight=pos_weight_np,
                lock_sha=(
                    artifact_info[
                        "pretraining_lock_sha256"
                    ]
                ),
            )

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history.append(
            {
                "run_id":
                    run_id,

                "task":
                    TASK,

                "variant":
                    VARIANT,

                "model":
                    MODEL_NAME,

                "graph":
                    GRAPH_TYPE,

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "val_auc_pr_macro":
                    val_macro_ap,

                "val_auc_pr_micro_diagnostic":
                    val_micro_ap,

                "learning_rate_end":
                    LEARNING_RATE,

                "improved":
                    improved,

                "best_val_auc_pr_so_far":
                    best_metric,

                "epochs_without_improvement":
                    epochs_without_improvement,

                "epoch_seconds":
                    epoch_seconds,

                "test_split_used":
                    False,
            }
        )

        write_dataframe_atomic(
            pd.DataFrame(
                history
            ),
            run_dir
            / "history.csv",
        )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss="
            f"{train_loss:.6f} | "
            f"val_loss="
            f"{val_loss:.6f} | "
            f"val_macro_AP="
            f"{val_macro_ap:.6f} | "
            f"val_micro_AP="
            f"{val_micro_ap:.6f} | "
            f"time="
            f"{epoch_seconds:.2f}s"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            stopped_early = True

            print()
            print(
                "Early stopping triggered "
                "after "
                f"{EARLY_STOP_PATIENCE} "
                "consecutive non-improving "
                "epochs."
            )

            break


    # ========================================================
    # BEST CHECKPOINT REVALIDATION
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    require(
        best_epoch is not None,
        "No best epoch selected.",
    )

    require(
        checkpoint_path.is_file(),
        "Best checkpoint missing.",
    )

    section(
        "10. BEST CHECKPOINT REVALIDATION"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    require(
        checkpoint[
            "task"
        ]
        == TASK,
        "Checkpoint task mismatch.",
    )

    require(
        checkpoint[
            "variant"
        ]
        == VARIANT,
        "Checkpoint variant mismatch.",
    )

    require(
        checkpoint[
            "model"
        ]
        == MODEL_NAME,
        "Checkpoint model mismatch.",
    )

    require(
        checkpoint[
            "graph"
        ]
        == GRAPH_TYPE,
        "Checkpoint graph mismatch.",
    )

    require(
        int(
            checkpoint[
                "seed"
            ]
        )
        == SEED,
        "Checkpoint seed mismatch.",
    )

    require(
        checkpoint[
            "graph_selection_path"
        ]
        == relative(
            GRAPH_SELECTION_PATH
        ),
        (
            "Checkpoint graph-selection "
            "record mismatch."
        ),
    )

    require(
        checkpoint[
            "pretraining_lock_sha256"
        ]
        == artifact_info[
            "pretraining_lock_sha256"
        ],
        (
            "Checkpoint frozen-lock "
            "identity mismatch."
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    revalidated = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    reproduced_metric = float(
        revalidated[
            "macro_ap"
        ]
    )

    require(
        abs(
            reproduced_metric
            - best_metric
        )
        <= 1.0e-10,
        (
            "Reloaded GAT checkpoint "
            "failed to reproduce "
            "best validation AP."
        ),
    )

    require(
        abs(
            float(
                checkpoint[
                    "val_auc_pr_macro_mean_ap"
                ]
            )
            - best_metric
        )
        <= 1.0e-12,
        (
            "Checkpoint metric "
            "metadata mismatch."
        ),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Recorded macro AP:",
        f"{best_metric:.8f}",
    )

    print(
        "Reloaded macro AP:",
        f"{reproduced_metric:.8f}",
    )

    print(
        "Checkpoint revalidation: PASS"
    )


    # ========================================================
    # PER-LABEL AP
    # ========================================================

    per_label_frame = pd.DataFrame(
        {
            "index":
                np.arange(
                    NUM_LABELS
                ),

            "label":
                label_names,

            "val_average_precision":
                revalidated[
                    "per_label_ap"
                ],
        }
    )

    write_dataframe_atomic(
        per_label_frame,
        run_dir
        / "best_val_per_label_ap.csv",
    )


    # ========================================================
    # SAVE FORMAL RESULT
    # ========================================================

    epochs_completed = len(
        history
    )

    peak_vram_gb = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / (1024 ** 3)
    )

    checkpoint_sha256 = (
        sha256_file(
            checkpoint_path
        )
    )

    result = {
        "status":
            "COMPLETE",

        "run_id":
            run_id,

        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "graph":
            GRAPH_TYPE,

        "seed":
            SEED,

        "best_epoch":
            best_epoch,

        "best_val_auc_pr_macro":
            best_metric,

        "best_val_loss":
            best_val_loss,

        "epochs_completed":
            epochs_completed,

        "stopped_early":
            stopped_early,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "checkpoint_path":
            relative(
                checkpoint_path
            ),

        "checkpoint_sha256":
            checkpoint_sha256,

        "runtime_seconds":
            runtime,

        "peak_vram_gb":
            peak_vram_gb,

        "pretraining_lock_sha256":
            artifact_info[
                "pretraining_lock_sha256"
            ],

        "graph_selection_path":
            artifact_info[
                "graph_selection_path"
            ],

        "selected_graph":
            GRAPH_TYPE,
    }

    with (
        run_dir
        / "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # Global records only after successful
    # checkpoint revalidation.
    append_rows_to_csv(
        history,
        EPOCH_METRICS_PATH,
    )

    trial_row = {
        "run_id":
            run_id,

        "status":
            "COMPLETE",

        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "graph":
            GRAPH_TYPE,

        "strategy":
            "GAT audio-graph G2",

        "seed":
            SEED,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "epochs_completed":
            epochs_completed,

        "dropout":
            GAT_DROPOUT,

        "heads":
            HEADS,

        "weight_decay":
            WEIGHT_DECAY,

        "scheduler":
            "NONE",

        "early_stop_patience":
            EARLY_STOP_PATIENCE,

        "selection_metric":
            SELECTION_METRIC_NAME,

        "best_epoch":
            best_epoch,

        "best_val_auc_pr_macro":
            best_metric,

        "best_val_loss":
            best_val_loss,

        "stopped_early":
            stopped_early,

        "trainable_parameters":
            trainable_parameters,

        "total_parameters":
            total_parameters,

        "runtime_seconds":
            runtime,

        "peak_vram_gb":
            peak_vram_gb,

        "checkpoint_path":
            relative(
                checkpoint_path
            ),

        "checkpoint_sha256":
            checkpoint_sha256,

        "pretraining_lock_sha256":
            artifact_info[
                "pretraining_lock_sha256"
            ],

        "graph_selection_path":
            artifact_info[
                "graph_selection_path"
            ],

        "git_commit":
            git_commit,

        "test_split_used":
            False,
    }

    append_rows_to_csv(
        [
            trial_row
        ],
        TRIALS_PATH,
    )


    # ========================================================
    # FINISH
    # ========================================================

    section(
        "FORMAL TASK-2 A4 "
        "GAT/G2 COMPLETE"
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation macro mean AP:",
        f"{best_metric:.8f}",
    )

    print(
        "Epochs completed:",
        epochs_completed,
    )

    print(
        "Stopped early:",
        stopped_early,
    )

    print(
        "Runtime:",
        f"{runtime / 60:.2f} min",
    )

    print(
        "Peak allocated VRAM:",
        f"{peak_vram_gb:.3f} GB",
    )

    print(
        "Checkpoint:",
        relative(
            checkpoint_path
        ),
    )

    print(
        "Checkpoint SHA256:",
        checkpoint_sha256,
    )

    print(
        "Run directory:",
        relative(
            run_dir
        ),
    )

    print()

    print(
        "TEST examples used for "
        "labels/inference: 0"
    )

    print(
        "Threshold tuning performed: NO"
    )

    print()

    print(
        "DO NOT proceed beyond Task-2 "
        "model selection until A1/A2/A3/A4 "
        "have been reviewed together."
    )


if __name__ == "__main__":
    main()