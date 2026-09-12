#!/usr/bin/env python3

"""
Script 39 — Formal Task-2 A2 GraphSAGE/G1 model-selection run

Pipeline Plan v6
----------------
A2:
    GraphSAGE on frozen G1 temporal graphs.

Architecture:
    raw node x: 140
    Linear(140 -> 64)
    SAGEConv(64 -> 128)
    ReLU
    Dropout(0.2)
    SAGEConv(128 -> 128)
    ReLU
    Dropout(0.2)
    SAGEConv(128 -> 128)
    global_mean_pool
    Linear(128 -> 20)

Training:
    AdamW
    LR = 1e-3
    weight decay = 1e-4
    batch size = 32 graphs
    early stopping patience = 3
    Task-2 operational ceiling = 15 epochs
    shared TRAIN-only BCEWithLogitsLoss(pos_weight)
    model selection = validation macro mean per-label AP

Important:
- TRAIN only optimizes parameters.
- VAL only selects checkpoints / early stopping.
- TEST targets/inference are NOT used.
- No threshold sweep is performed.
- Frozen graph artifacts are never modified.
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
    SAGEConv,
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
# FROZEN VALUES
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

EXPECTED_G1_EDGES = 16

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
# FORMAL A2 SETTINGS
# ============================================================

TASK = "task2"
VARIANT = "A2"
MODEL_NAME = "GraphSAGE"
GRAPH_TYPE = "G1"

SEED = 42

BATCH_SIZE = 32

MAX_EPOCHS = 15

LEARNING_RATE = 1.0e-3

WEIGHT_DECAY = 1.0e-4

DROPOUT = 0.2

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

        value = result.stdout.strip()

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

        existing = pd.read_csv(
            path
        )

        combined = pd.concat(
            [
                existing,
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
# DUPLICATE FORMAL RUN PROTECTION
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
            "A completed formal "
            "A2 GraphSAGE/G1 seed-42 "
            "run already exists. "
            "Refusing accidental rerun."
        ),
    )


# ============================================================
# G1 TOPOLOGY
# ============================================================

def expected_g1_edge_set():

    edges = set()

    for node in range(
        EXPECTED_NODES - 1
    ):

        edges.add(
            (
                node,
                node + 1,
            )
        )

        edges.add(
            (
                node + 1,
                node,
            )
        )

    require(
        len(edges)
        == EXPECTED_G1_EDGES,
        "Internal G1 topology error.",
    )

    return edges


EXPECTED_G1_SET = (
    expected_g1_edge_set()
)


# ============================================================
# VERIFY FROZEN ARTIFACTS + A1
# ============================================================

def verify_inputs():

    section(
        "1. FROZEN ARTIFACT VERIFICATION"
    )

    actual_lock_sha = sha256_file(
        PRETRAINING_LOCK_PATH
    )

    print(
        "Expected lock SHA256:",
        EXPECTED_LOCK_SHA256,
    )

    print(
        "Actual lock SHA256:  ",
        actual_lock_sha,
    )

    require(
        actual_lock_sha
        == EXPECTED_LOCK_SHA256,
        "Pretraining lock changed.",
    )

    with PRETRAINING_LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        lock = json.load(file)

    require(
        lock.get("dataset_tracks")
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
            f"Missing frozen artifact {path}.",
        )

        actual_sha = sha256_file(
            path
        )

        require(
            actual_sha
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

    # --------------------------------------------------------
    # A1 must already exist
    # --------------------------------------------------------

    require(
        TRIALS_PATH.is_file(),
        "Missing trial CSV.",
    )

    trials = pd.read_csv(
        TRIALS_PATH
    )

    required_columns = {
        "status",
        "task",
        "variant",
        "seed",
        "best_val_auc_pr_macro",
        "checkpoint_path",
    }

    require(
        required_columns.issubset(
            trials.columns
        ),
        "Trial CSV schema incomplete.",
    )

    a1 = trials.loc[
        (trials["task"] == "task2")
        & (trials["variant"] == "A1")
        & (trials["status"] == "COMPLETE")
        & (trials["seed"] == SEED)
    ]

    require(
        len(a1) == 1,
        (
            "Expected exactly one "
            "completed A1 seed-42 run."
        ),
    )

    a1_row = a1.iloc[0]

    a1_checkpoint_path = (
        ROOT
        / str(
            a1_row[
                "checkpoint_path"
            ]
        )
    )

    require(
        a1_checkpoint_path.is_file(),
        "A1 CNN checkpoint missing.",
    )

    print()
    print(
        "A1 CNN prerequisite: PASS"
    )

    print(
        "A1 VAL macro AP:",
        float(
            a1_row[
                "best_val_auc_pr_macro"
            ]
        ),
    )

    print(
        "A1 checkpoint:",
        relative(
            a1_checkpoint_path
        ),
    )

    print()
    print(
        "Frozen-artifact verification: PASS"
    )

    print(
        "TEST targets/examples loaded: 0"
    )

    return {
        "pretraining_lock_sha256":
            actual_lock_sha,

        "a1_checkpoint_path":
            relative(
                a1_checkpoint_path
            ),

        "a1_checkpoint_sha256":
            sha256_file(
                a1_checkpoint_path
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
            "g1_directed_edges",
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
            manifest["nodes"]
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
        "Unexpected feature dimension.",
    )

    require(
        (
            manifest[
                "g1_directed_edges"
            ]
            == EXPECTED_G1_EDGES
        ).all(),
        "Unexpected G1 edge count.",
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
        "TRAIN target count mismatch.",
    )

    require(
        len(val_targets)
        == EXPECTED_VAL_TRACKS,
        "VAL target count mismatch.",
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
        .sort_values("track_id")
        .reset_index(drop=True)
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
        .sort_values("track_id")
        .reset_index(drop=True)
    )

    require(
        len(train_frame)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN graph merge failed.",
    )

    require(
        len(val_frame)
        == EXPECTED_VAL_TRACKS,
        "VAL graph merge failed.",
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
            train_frame["track_id"]
        ).isdisjoint(
            set(
                val_frame["track_id"]
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
# SHARED TRAIN-ONLY CLASS WEIGHTS
# ============================================================

def compute_pos_weight(
    train_frame: pd.DataFrame,
    a1_checkpoint_path: Path,
):

    section(
        "3. SHARED TRAIN-ONLY CLASS WEIGHTS"
    )

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocab = json.load(file)

    labels = vocab["labels"]

    require(
        len(labels) == NUM_LABELS,
        "Expected 20 labels.",
    )

    label_names = [
        item["label"]
        for item in labels
    ]

    train_y = (
        train_frame[
            TARGET_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    require(
        set(
            np.unique(
                train_y
            ).tolist()
        ).issubset(
            {0.0, 1.0}
        ),
        "TRAIN targets are not binary.",
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
            for item in labels
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

    # --------------------------------------------------------
    # Strong fairness check:
    # current GraphSAGE weighting must be bitwise/numerically
    # the same values saved in A1 CNN.
    # --------------------------------------------------------

    a1_checkpoint = torch.load(
        a1_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        "pos_weight"
        in a1_checkpoint,
        (
            "A1 checkpoint lacks "
            "saved pos_weight."
        ),
    )

    a1_pos_weight = np.asarray(
        a1_checkpoint[
            "pos_weight"
        ],
        dtype=np.float32,
    )

    require(
        a1_pos_weight.shape
        == (
            NUM_LABELS,
        ),
        (
            "A1 pos_weight "
            "dimension mismatch."
        ),
    )

    require(
        np.array_equal(
            pos_weight,
            a1_pos_weight,
        ),
        (
            "GraphSAGE pos_weight "
            "differs from A1 CNN."
        ),
    )

    print(
        "TRAIN support == frozen vocabulary: PASS"
    )

    print(
        "Formula min(negative/positive,10): PASS"
    )

    print(
        "A1 CNN pos_weight == A2 GraphSAGE pos_weight: PASS"
    )

    return (
        pos_weight,
        label_names,
    )


# ============================================================
# FROZEN G1 LOADER
# ============================================================

def load_g1_graph(
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

        keys = set(
            graph.files
        )

        require(
            keys
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

        edge_index = np.array(
            graph[
                "edge_index_g1"
            ],
            dtype=np.int64,
            copy=True,
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
        edge_index.shape
        == (
            2,
            EXPECTED_G1_EDGES,
        ),
        (
            "Unexpected G1 shape "
            f"in {path}: "
            f"{edge_index.shape}"
        ),
    )

    edges = {
        (
            int(src),
            int(dst),
        )
        for src, dst
        in edge_index.T
    }

    require(
        len(edges)
        == EXPECTED_G1_EDGES,
        (
            "Duplicate G1 edges "
            f"in {path}"
        ),
    )

    require(
        edges == EXPECTED_G1_SET,
        (
            "G1 temporal topology "
            f"changed in {path}"
        ),
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    require(
        target.shape
        == (NUM_LABELS,),
        "Target dimension != 20.",
    )

    return Data(
        x=torch.from_numpy(x),

        edge_index=torch.from_numpy(
            edge_index
        ),

        y=torch.tensor(
            target,
            dtype=torch.float32,
        ).unsqueeze(0),
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
            load_g1_graph(
                path,
                target,
            )
        )

    return dataset


# ============================================================
# MODEL — IDENTICAL TO SCRIPT 38
# ============================================================

class GraphSAGEG1(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        self.node_projection = nn.Linear(
            NODE_INPUT_DIM,
            NODE_EMBED_DIM,
        )

        # Keep the exact PyG SAGEConv constructor
        # already smoke-tested by Script 38.
        # Do not introduce non-v6 options here.
        self.conv1 = SAGEConv(
            NODE_EMBED_DIM,
            GNN_HIDDEN_DIM,
        )

        self.conv2 = SAGEConv(
            GNN_HIDDEN_DIM,
            GNN_HIDDEN_DIM,
        )

        self.conv3 = SAGEConv(
            GNN_HIDDEN_DIM,
            GNN_HIDDEN_DIM,
        )

        self.dropout = nn.Dropout(
            p=DROPOUT
        )

        self.classifier = nn.Linear(
            GNN_HIDDEN_DIM,
            NUM_LABELS,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ):

        x = self.node_projection(
            x
        )

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv3(
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
        "Metric label dimension != 20.",
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
        "Non-finite per-label AP.",
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
        math.isfinite(macro_ap),
        "Non-finite macro AP.",
    )

    require(
        math.isfinite(micro_ap),
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
# TRAIN ONE EPOCH
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

        require(
            batch.y.ndim == 2
            and batch.y.shape[1]
            == NUM_LABELS,
            (
                "Unexpected TRAIN "
                "target shape."
            ),
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
                "TRAIN logits/targets "
                "shape mismatch."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            "Non-finite TRAIN logits.",
        )

        loss = criterion(
            logits,
            batch.y,
        )

        require(
            bool(
                torch.isfinite(loss)
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
                loss.detach().cpu()
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
                "VAL logits/targets "
                "shape mismatch."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            "Non-finite VAL logits.",
        )

        loss = criterion(
            logits,
            batch.y,
        )

        require(
            bool(
                torch.isfinite(loss)
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
                loss.detach().cpu()
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
    pretraining_lock_sha: str,
) -> None:

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

            "sage_layers": [
                "SAGEConv(64,128)",
                "SAGEConv(128,128)",
                "SAGEConv(128,128)",
            ],

            "dropout":
                DROPOUT,

            "global_pool":
                "global_mean_pool",

            "classifier":
                "Linear(128,20)",

            "pyg_version":
                torch_geometric.__version__,
        },

        "training": {
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

        "pretraining_lock_sha256":
            pretraining_lock_sha,

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
        "SCRIPT 39 — FORMAL TASK-2 "
        "A2 GRAPHSAGE / G1"
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

    assert_no_completed_duplicate()


    # ========================================================
    # 1-3. INPUT VALIDATION
    # ========================================================

    artifact_info = verify_inputs()

    (
        train_frame,
        val_frame,
    ) = load_development_data()

    a1_checkpoint_path = (
        ROOT
        / artifact_info[
            "a1_checkpoint_path"
        ]
    )

    (
        pos_weight_np,
        label_names,
    ) = compute_pos_weight(
        train_frame,
        a1_checkpoint_path,
    )


    # ========================================================
    # 4. LOAD FROZEN G1 DATASET
    # ========================================================

    section(
        "4. LOAD FROZEN G1 DATASET"
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
        "TRAIN dataset count mismatch.",
    )

    require(
        len(val_dataset)
        == EXPECTED_VAL_TRACKS,
        "VAL dataset count mismatch.",
    )

    print(
        "TRAIN G1 graphs:",
        len(train_dataset),
    )

    print(
        "VAL G1 graphs:  ",
        len(val_dataset),
    )

    print(
        "Frozen G1 loading: PASS"
    )


    # ========================================================
    # 5. ENVIRONMENT
    # ========================================================

    section(
        "5. ENVIRONMENT + REPRODUCIBILITY"
    )

    require(
        torch.cuda.is_available(),
        "CUDA is required.",
    )

    device = torch.device(
        "cuda"
    )

    set_seed(
        SEED
    )

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
    # 6. RUN PATHS
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
        f"task2_A2_GraphSAGE_G1_"
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
        (
            "Run directory "
            "already exists."
        ),
    )

    require(
        not checkpoint_path.exists(),
        (
            "Checkpoint path "
            "already exists."
        ),
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
    # 7. DATALOADERS
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
        (
            "Unexpected TRAIN "
            "batch count."
        ),
    )

    require(
        len(val_loader)
        == 17,
        (
            "Unexpected VAL "
            "batch count."
        ),
    )

    first_batch = next(
        iter(train_loader)
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
        first_batch.edge_index.shape
        == (
            2,
            BATCH_SIZE
            * EXPECTED_G1_EDGES,
        ),
        (
            "Unexpected first-batch "
            "G1 edge shape."
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
        "First batch edges:",
        tuple(
            first_batch.edge_index.shape
        ),
    )


    # ========================================================
    # 8. MODEL
    # ========================================================

    section(
        "7. MODEL"
    )

    # Reset again because fetching the first shuffled batch
    # should not influence model initialization RNG.
    set_seed(
        SEED
    )

    model = GraphSAGEG1()

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
        total_parameters == 93908,
        (
            "GraphSAGE parameter count "
            "differs from Script 38."
        ),
    )

    require(
        trainable_parameters
        == total_parameters,
        (
            "Unexpected frozen "
            "GraphSAGE parameters."
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
        "Dropout:",
        DROPOUT,
    )

    model = model.to(
        device
    )


    # ========================================================
    # 9. OPTIMIZATION
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
    # 10. RUN SPEC
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

            "sage_layers": [
                "SAGEConv(64,128)",
                "SAGEConv(128,128)",
                "SAGEConv(128,128)",
            ],

            "sageconv_constructor":
                (
                    "torch_geometric.nn."
                    "SAGEConv(in_channels,"
                    "out_channels)"
                ),

            "dropout":
                DROPOUT,

            "activation_after_sage":
                [
                    "ReLU",
                    "ReLU",
                    None,
                ],

            "pool":
                "global_mean_pool",

            "classifier":
                "Linear(128,20)",

            "parameter_count":
                total_parameters,
        },

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

            "epoch_ceiling_basis":
                (
                    "Task-2 operational "
                    "ceiling fixed at 15 "
                    "before formal Task-2 "
                    "model results."
                ),

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

            "verified_equal_to_A1_CNN":
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

        "frozen_artifacts":
            artifact_info,

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
    # 11. FORMAL TRAINING
    # ========================================================

    section(
        "9. FORMAL TRAINING"
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start_time = time.perf_counter()

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
            validation["loss"]
        )

        val_macro_ap = float(
            validation["macro_ap"]
        )

        val_micro_ap = float(
            validation["micro_ap"]
        )

        improved = (
            val_macro_ap
            > (
                best_metric
                + IMPROVEMENT_EPSILON
            )
        )

        if improved:

            best_metric = val_macro_ap
            best_epoch = epoch
            best_val_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                epoch=epoch,
                val_macro_ap=val_macro_ap,
                val_loss=val_loss,
                run_id=run_id,
                pos_weight=pos_weight_np,
                pretraining_lock_sha=(
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

        row = {
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

        history.append(row)

        write_dataframe_atomic(
            pd.DataFrame(history),
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
    # 12. BEST CHECKPOINT REVALIDATION
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
        checkpoint["task"] == TASK,
        "Checkpoint task mismatch.",
    )

    require(
        checkpoint["variant"]
        == VARIANT,
        "Checkpoint variant mismatch.",
    )

    require(
        checkpoint["model"]
        == MODEL_NAME,
        "Checkpoint model mismatch.",
    )

    require(
        checkpoint["graph"]
        == GRAPH_TYPE,
        "Checkpoint graph mismatch.",
    )

    require(
        int(
            checkpoint["seed"]
        )
        == SEED,
        "Checkpoint seed mismatch.",
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
        revalidated["macro_ap"]
    )

    require(
        abs(
            reproduced_metric
            - best_metric
        )
        <= 1.0e-10,
        (
            "Reloaded GraphSAGE "
            "checkpoint did not reproduce "
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
            "Checkpoint metadata "
            "metric mismatch."
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
    # 13. PER-LABEL AP
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
    # 14. SAVE SUCCESSFUL FORMAL RESULT
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

    # Update global CSVs only after the entire
    # run and checkpoint revalidation succeed.
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
            "GraphSAGE audio-graph G1",

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
            DROPOUT,

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

        "git_commit":
            git_commit,

        "test_split_used":
            False,
    }

    append_rows_to_csv(
        [trial_row],
        TRIALS_PATH,
    )


    # ========================================================
    # FINISH
    # ========================================================

    section(
        "FORMAL TASK-2 A2 "
        "GRAPHSAGE/G1 COMPLETE"
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
        "DO NOT begin A3 GraphSAGE/G2 "
        "until this run has been reviewed."
    )


if __name__ == "__main__":
    main()