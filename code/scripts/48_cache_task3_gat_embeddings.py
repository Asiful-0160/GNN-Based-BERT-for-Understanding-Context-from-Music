#!/usr/bin/env python3
"""
Script 48 — Cache Frozen Task-3 A4 GAT/G2 Representations
=========================================================

Purpose
-------
Generate the frozen 128-D graph representation from the selected Task-2
A4 GAT/G2 checkpoint for Task-3 Phase A.

STRICT DEVELOPMENT BOUNDARY
---------------------------
Included:
    TRAIN
    VAL

Excluded:
    TEST

The Script-47 BERT sequence manifest is used as the canonical row order.
Therefore:

    graph_embedding_trainval[i]

and

    bert_sequence_trainval[i]

must correspond to the exact same track_id.

This script performs NO:
    - model training
    - gradient updates
    - optimizer steps
    - threshold tuning
    - TEST graph inference
    - TEST metric calculation
    - TEST label access
    - modification of frozen preprocessing artifacts

Output
------
data/processed/gat_g2_frozen/

    gat_g2_embedding_trainval.npy
        shape = (4626, 128)
        dtype = float32

    gat_g2_manifest.parquet

    gat_g2_cache_metadata.json

    gat_g2_frozen_hashes.json
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
)


# =============================================================================
# 0. PROJECT CONTRACT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Frozen preprocessing / selections
# -----------------------------------------------------------------------------

PRETRAINING_LOCK = (
    PROJECT_ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

TASK2_GRAPH_SELECTION = (
    PROJECT_ROOT
    / "results/runs/task2/task2_graph_selection.json"
)

TASK2_MODEL_SELECTION = (
    PROJECT_ROOT
    / "results/runs/task2/task2_model_selection.json"
)

A4_CHECKPOINT = (
    PROJECT_ROOT
    / "models/task2_candidates/"
    "task2_A4_GAT_G2_seed42_20260828T193949Z_best.pt"
)

GRAPH_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/graph_manifest.parquet"
)


# -----------------------------------------------------------------------------
# Script-47 frozen BERT cache
#
# We use its manifest as the canonical Task-3 TRAIN+VAL ordering.
# -----------------------------------------------------------------------------

BERT_SEQ_DIR = (
    PROJECT_ROOT
    / "data/processed/bert_seq_frozen"
)

BERT_SEQ_FILE = (
    BERT_SEQ_DIR
    / "bert_sequence_trainval.npy"
)

BERT_SEQ_MASK = (
    BERT_SEQ_DIR
    / "attention_mask_trainval.npy"
)

BERT_SEQ_MANIFEST = (
    BERT_SEQ_DIR
    / "bert_seq_manifest.parquet"
)

BERT_SEQ_METADATA = (
    BERT_SEQ_DIR
    / "bert_seq_cache_metadata.json"
)

BERT_SEQ_LOCK = (
    BERT_SEQ_DIR
    / "bert_seq_frozen_hashes.json"
)


# -----------------------------------------------------------------------------
# Script-48 outputs
# -----------------------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/gat_g2_frozen"
)

TEMP_DIR = (
    PROJECT_ROOT
    / "data/processed/gat_g2_frozen.__tmp__"
)


# =============================================================================
# 1. FROZEN IDENTITIES
# =============================================================================

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_A4_CHECKPOINT_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e"
)

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6ad0492bf90eccc929cadc1aad1da9e158"
)


# Script-47 cache hashes produced by the accepted Script-47 run.

EXPECTED_BERT_SEQUENCE_SHA256 = (
    "a1c723ec939392b8b0d9bafadb8327d8a0967ecb9bd8e89ffcd559306633e133"
)

EXPECTED_BERT_MASK_SHA256 = (
    "633b11d5ca9146000da84b282f9081c6ff33c1279486c2e3112e4c89137d197b"
)

EXPECTED_BERT_MANIFEST_SHA256 = (
    "ebd86077489b716abc8768247f550890ca8e5a26b8f0967c32882c22c4e85d2f"
)

EXPECTED_BERT_METADATA_SHA256 = (
    "d50d25fbea4cd0ea83a946ae57e3c8673bd53303efd58a2c2e9bf3998e792c73"
)


# =============================================================================
# 2. FROZEN DATA / MODEL DIMENSIONS
# =============================================================================

EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_TEST = 514

EXPECTED_TOTAL = 5140

EXPECTED_DEVELOPMENT = (
    EXPECTED_TRAIN
    + EXPECTED_VAL
)

EXPECTED_NODES = 9
EXPECTED_RAW_NODE_DIM = 140
EXPECTED_NODE_PROJECTION = 64

EXPECTED_GAT_HIDDEN = 128
EXPECTED_LABELS = 20

EXPECTED_G1_EDGES = 16

EXPECTED_G2_MIN_EDGES = 16
EXPECTED_G2_MAX_EDGES = 34

EXPECTED_SIMILARITY_THRESHOLD = 0.85
EXPECTED_NONLOCAL_ONLY = True
EXPECTED_EXTRA_DEGREE_CAP = 2
EXPECTED_GRAPH_RULE_VERSION = 2

BATCH_SIZE = 32
SEED = 42


# =============================================================================
# 3. REPORTING
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
# 4. BASIC HELPERS
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


def load_json(
    path: Path,
) -> Any:

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


def relative(
    path: Path,
) -> str:

    try:
        return str(
            path.resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )

    except Exception:
        return str(path)


def load_checkpoint(
    path: Path,
) -> dict[str, Any]:

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

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "A4 checkpoint is not a dictionary"
        )

    return checkpoint


# =============================================================================
# 5. REPRODUCIBILITY
# =============================================================================

def configure_reproducibility() -> None:

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

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

        torch.backends.cudnn.benchmark = False

    try:

        torch.set_float32_matmul_precision(
            "highest"
        )

    except Exception:
        pass


# =============================================================================
# 6. FROZEN INPUT VALIDATION
# =============================================================================

def verify_frozen_inputs() -> None:

    banner(
        "1. FROZEN INPUT ARTIFACTS"
    )

    required = (
        PRETRAINING_LOCK,
        TASK2_GRAPH_SELECTION,
        TASK2_MODEL_SELECTION,
        A4_CHECKPOINT,
        GRAPH_MANIFEST,
        BERT_SEQ_FILE,
        BERT_SEQ_MASK,
        BERT_SEQ_MANIFEST,
        BERT_SEQ_METADATA,
        BERT_SEQ_LOCK,
    )

    for path in required:

        check(
            path.is_file(),
            (
                "required artifact exists: "
                f"{relative(path)}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Required frozen artifact is missing"
        )

    # -------------------------------------------------------------------------
    # Pretraining lock
    # -------------------------------------------------------------------------

    lock_sha = sha256_file(
        PRETRAINING_LOCK
    )

    print(
        f"Pretraining lock SHA256: "
        f"{lock_sha}"
    )

    check(
        lock_sha
        == EXPECTED_PRETRAINING_LOCK_SHA256,
        (
            "pretraining lock remains "
            "byte-for-byte unchanged"
        ),
    )

    # -------------------------------------------------------------------------
    # A4 checkpoint
    # -------------------------------------------------------------------------

    a4_sha = sha256_file(
        A4_CHECKPOINT
    )

    print(
        f"A4 checkpoint SHA256: "
        f"{a4_sha}"
    )

    check(
        a4_sha
        == EXPECTED_A4_CHECKPOINT_SHA256,
        (
            "selected A4 checkpoint SHA256 "
            "remains unchanged"
        ),
    )

    # -------------------------------------------------------------------------
    # Graph manifest
    # -------------------------------------------------------------------------

    graph_manifest_sha = sha256_file(
        GRAPH_MANIFEST
    )

    print(
        f"Graph manifest SHA256: "
        f"{graph_manifest_sha}"
    )

    check(
        graph_manifest_sha
        == EXPECTED_GRAPH_MANIFEST_SHA256,
        (
            "frozen graph manifest SHA256 "
            "remains unchanged"
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen graph topology decision
    # -------------------------------------------------------------------------

    graph_selection = load_json(
        TASK2_GRAPH_SELECTION
    )

    graph_selection_text = json.dumps(
        graph_selection,
        sort_keys=True,
    ).lower()

    check(
        "g2" in graph_selection_text,
        (
            "frozen Task-2 graph selection "
            "still selects G2"
        ),
    )

    if (
        isinstance(
            graph_selection,
            dict,
        )
        and
        "test_split_used"
        in graph_selection
    ):

        check(
            graph_selection[
                "test_split_used"
            ] is False,
            (
                "graph selection records "
                "TEST unused"
            ),
        )

    # -------------------------------------------------------------------------
    # Frozen A4 model selection
    # -------------------------------------------------------------------------

    model_selection = load_json(
        TASK2_MODEL_SELECTION
    )

    model_selection_text = json.dumps(
        model_selection,
        sort_keys=True,
    ).lower()

    check(
        "a4" in model_selection_text,
        (
            "Task-2 frozen model selection "
            "still references A4"
        ),
    )

    check(
        "gat" in model_selection_text,
        (
            "Task-2 frozen model selection "
            "still references GAT"
        ),
    )

    check(
        "g2" in model_selection_text,
        (
            "Task-2 frozen model selection "
            "still references G2"
        ),
    )

    check(
        A4_CHECKPOINT.name.lower()
        in model_selection_text,
        (
            "Task-2 frozen selection still "
            "references exact A4 checkpoint"
        ),
    )

    check(
        EXPECTED_A4_CHECKPOINT_SHA256
        in model_selection_text,
        (
            "Task-2 frozen selection still "
            "records exact A4 SHA256"
        ),
    )

    if (
        isinstance(
            model_selection,
            dict,
        )
        and
        "test_split_used"
        in model_selection
    ):

        check(
            model_selection[
                "test_split_used"
            ] is False,
            (
                "Task-2 model selection records "
                "TEST unused"
            ),
        )

    if (
        isinstance(
            model_selection,
            dict,
        )
        and
        "threshold_tuning_used"
        in model_selection
    ):

        check(
            model_selection[
                "threshold_tuning_used"
            ] is False,
            (
                "Task-2 model selection records "
                "threshold tuning unused"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen model-selection validation failed"
        )


# =============================================================================
# 7. VERIFY SCRIPT-47 CACHE
# =============================================================================

def verify_bert_cache() -> pd.DataFrame:

    banner(
        "2. VERIFY SCRIPT-47 BERT CACHE"
    )

    # -------------------------------------------------------------------------
    # Direct hashes from accepted Script-47 output
    # -------------------------------------------------------------------------

    expected_hashes = {
        BERT_SEQ_FILE:
            EXPECTED_BERT_SEQUENCE_SHA256,

        BERT_SEQ_MASK:
            EXPECTED_BERT_MASK_SHA256,

        BERT_SEQ_MANIFEST:
            EXPECTED_BERT_MANIFEST_SHA256,

        BERT_SEQ_METADATA:
            EXPECTED_BERT_METADATA_SHA256,
    }

    for path, expected in (
        expected_hashes.items()
    ):

        observed = sha256_file(
            path
        )

        print(
            f"{path.name} SHA256:"
        )

        print(
            f"  observed = {observed}"
        )

        print(
            f"  expected = {expected}"
        )

        check(
            observed == expected,
            (
                f"{path.name} matches "
                "the accepted Script-47 hash"
            ),
        )

    # -------------------------------------------------------------------------
    # Verify the Script-47 internal lock too
    # -------------------------------------------------------------------------

    bert_lock = load_json(
        BERT_SEQ_LOCK
    )

    check(
        bert_lock.get(
            "lock_type"
        )
        == "task3_bert_sequence_cache_lock",
        (
            "Script-47 cache lock has expected "
            "lock_type"
        ),
    )

    check(
        bert_lock.get(
            "construction_scope"
        )
        == "train_val_only",
        (
            "Script-47 cache lock records "
            "TRAIN+VAL-only construction"
        ),
    )

    check(
        bert_lock.get(
            "test_examples_inferred"
        )
        == 0,
        (
            "Script-47 cache lock records "
            "zero TEST inference"
        ),
    )

    # -------------------------------------------------------------------------
    # Lightweight shape verification
    # -------------------------------------------------------------------------

    sequence = np.load(
        BERT_SEQ_FILE,
        mmap_mode="r",
        allow_pickle=False,
    )

    mask = np.load(
        BERT_SEQ_MASK,
        mmap_mode="r",
        allow_pickle=False,
    )

    check(
        sequence.shape
        == (
            EXPECTED_DEVELOPMENT,
            112,
            768,
        ),
        (
            "Script-47 sequence shape remains "
            "(4626, 112, 768)"
        ),
    )

    check(
        sequence.dtype
        == np.float32,
        (
            "Script-47 sequence dtype remains "
            "float32"
        ),
    )

    check(
        mask.shape
        == (
            EXPECTED_DEVELOPMENT,
            112,
        ),
        (
            "Script-47 attention-mask shape "
            "remains (4626, 112)"
        ),
    )

    check(
        mask.dtype
        == np.uint8,
        (
            "Script-47 attention-mask dtype "
            "remains uint8"
        ),
    )

    del sequence
    del mask

    # -------------------------------------------------------------------------
    # Canonical development ordering
    # -------------------------------------------------------------------------

    manifest = pd.read_parquet(
        BERT_SEQ_MANIFEST
    )

    print(
        f"BERT sequence manifest shape: "
        f"{manifest.shape}"
    )

    required_columns = {
        "cache_index",
        "track_id",
        "split",
        "source_token_cache_index",
        "cached_token_length",
    }

    check(
        required_columns.issubset(
            manifest.columns
        ),
        (
            "Script-47 manifest contains "
            "all required columns"
        ),
    )

    check(
        len(manifest)
        == EXPECTED_DEVELOPMENT,
        (
            "Script-47 manifest contains "
            "exactly 4626 rows"
        ),
    )

    check(
        manifest[
            "track_id"
        ].nunique()
        == EXPECTED_DEVELOPMENT,
        (
            "Script-47 manifest track IDs "
            "are unique"
        ),
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

    print(
        f"BERT development split counts: "
        f"{split_counts}"
    )

    check(
        split_counts
        == {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
        },
        (
            "Script-47 manifest contains "
            "4112 TRAIN + 514 VAL only"
        ),
    )

    check(
        "test"
        not in set(
            manifest[
                "split"
            ]
        ),
        (
            "Script-47 canonical development "
            "manifest contains zero TEST rows"
        ),
    )

    expected_indices = np.arange(
        EXPECTED_DEVELOPMENT,
        dtype=np.int64,
    )

    actual_indices = pd.to_numeric(
        manifest[
            "cache_index"
        ],
        errors="raise",
    ).astype(
        np.int64
    ).to_numpy()

    check(
        np.array_equal(
            actual_indices,
            expected_indices,
        ),
        (
            "Script-47 canonical cache indices "
            "are contiguous 0..4625"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Script-47 cache verification failed"
        )

    return manifest


# =============================================================================
# 8. GRAPH MANIFEST VALIDATION + ALIGNMENT
# =============================================================================

def load_and_align_graph_manifest(
    bert_manifest: pd.DataFrame,
) -> pd.DataFrame:

    banner(
        "3. GRAPH MANIFEST VALIDATION + CROSS-MODAL ALIGNMENT"
    )

    graph_manifest = pd.read_parquet(
        GRAPH_MANIFEST
    )

    expected_columns = [
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

    print(
        f"Graph manifest shape: "
        f"{graph_manifest.shape}"
    )

    print(
        f"Graph manifest columns: "
        f"{list(graph_manifest.columns)}"
    )

    check(
        list(
            graph_manifest.columns
        )
        == expected_columns,
        (
            "graph manifest has exact "
            "frozen schema"
        ),
    )

    check(
        len(
            graph_manifest
        )
        == EXPECTED_TOTAL,
        (
            "graph manifest contains "
            "exactly 5140 tracks"
        ),
    )

    check(
        graph_manifest[
            "track_id"
        ].nunique()
        == EXPECTED_TOTAL,
        (
            "graph manifest track IDs "
            "are unique"
        ),
    )

    graph_manifest = (
        graph_manifest
        .copy()
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

    split_counts = (
        graph_manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"Graph split counts: "
        f"{split_counts}"
    )

    check(
        split_counts
        == {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },
        (
            "graph manifest split counts "
            "remain exactly 4112/514/514"
        ),
    )

    check(
        bool(
            (
                graph_manifest[
                    "nodes"
                ]
                == EXPECTED_NODES
            ).all()
        ),
        (
            "all graph-manifest rows "
            "record exactly 9 nodes"
        ),
    )

    check(
        bool(
            (
                graph_manifest[
                    "node_feature_dim"
                ]
                == EXPECTED_RAW_NODE_DIM
            ).all()
        ),
        (
            "all graph-manifest rows record "
            "140-D node features"
        ),
    )

    check(
        bool(
            (
                graph_manifest[
                    "g1_directed_edges"
                ]
                == EXPECTED_G1_EDGES
            ).all()
        ),
        (
            "all graph-manifest rows "
            "record exactly 16 G1 edges"
        ),
    )

    g2_min = int(
        graph_manifest[
            "g2_directed_edges"
        ].min()
    )

    g2_max = int(
        graph_manifest[
            "g2_directed_edges"
        ].max()
    )

    print(
        f"G2 directed-edge range: "
        f"{g2_min}..{g2_max}"
    )

    check(
        g2_min
        >= EXPECTED_G2_MIN_EDGES,
        (
            "graph manifest G2 minimum "
            "edge count is valid"
        ),
    )

    check(
        g2_max
        <= EXPECTED_G2_MAX_EDGES,
        (
            "graph manifest G2 maximum "
            "edge count is valid"
        ),
    )

    # -------------------------------------------------------------------------
    # Development rows only
    # -------------------------------------------------------------------------

    graph_dev = (
        graph_manifest.loc[
            graph_manifest[
                "split"
            ].isin(
                [
                    "train",
                    "val",
                ]
            )
        ]
        .copy()
    )

    check(
        len(graph_dev)
        == EXPECTED_DEVELOPMENT,
        (
            "graph development subset "
            "contains exactly 4626 rows"
        ),
    )

    check(
        "test"
        not in set(
            graph_dev[
                "split"
            ]
        ),
        (
            "graph development subset "
            "contains zero TEST rows"
        ),
    )

    # -------------------------------------------------------------------------
    # Cross-modal identity equality
    # -------------------------------------------------------------------------

    bert_ids = set(
        bert_manifest[
            "track_id"
        ].astype(str)
    )

    graph_ids = set(
        graph_dev[
            "track_id"
        ].astype(str)
    )

    check(
        bert_ids == graph_ids,
        (
            "BERT and GAT TRAIN+VAL track-ID "
            "sets are exactly identical"
        ),
    )

    # -------------------------------------------------------------------------
    # Reorder graph rows to the Script-47 BERT cache order.
    # -------------------------------------------------------------------------

    graph_dev = (
        graph_dev
        .reset_index()
        .rename(
            columns={
                "index":
                    "source_graph_manifest_row"
            }
        )
    )

    graph_lookup = (
        graph_dev
        .set_index(
            "track_id",
            drop=False,
        )
    )

    aligned_rows = []

    for bert_row in (
        bert_manifest
        .itertuples(
            index=False
        )
    ):

        track_id = str(
            bert_row.track_id
        )

        if track_id not in graph_lookup.index:

            raise RuntimeError(
                "BERT track missing from graph "
                f"manifest: {track_id}"
            )

        graph_row = (
            graph_lookup
            .loc[
                track_id
            ]
        )

        # Unique track IDs guarantee Series,
        # not a multi-row DataFrame.

        if (
            str(
                graph_row[
                    "split"
                ]
            ).lower()
            !=
            str(
                bert_row.split
            ).lower()
        ):

            raise RuntimeError(
                "Split mismatch for track "
                f"{track_id}: "
                f"BERT={bert_row.split}, "
                f"graph={graph_row['split']}"
            )

        aligned_rows.append(
            {
                "cache_index":
                    int(
                        bert_row.cache_index
                    ),

                "track_id":
                    track_id,

                "split":
                    str(
                        bert_row.split
                    ).lower(),

                "bert_seq_cache_index":
                    int(
                        bert_row.cache_index
                    ),

                "source_graph_manifest_row":
                    int(
                        graph_row[
                            "source_graph_manifest_row"
                        ]
                    ),

                "graph_path":
                    str(
                        graph_row[
                            "graph_path"
                        ]
                    ),

                "nodes":
                    int(
                        graph_row[
                            "nodes"
                        ]
                    ),

                "node_feature_dim":
                    int(
                        graph_row[
                            "node_feature_dim"
                        ]
                    ),

                "g2_directed_edges":
                    int(
                        graph_row[
                            "g2_directed_edges"
                        ]
                    ),
            }
        )

    aligned = pd.DataFrame(
        aligned_rows
    )

    check(
        len(aligned)
        == EXPECTED_DEVELOPMENT,
        (
            "aligned GAT manifest contains "
            "exactly 4626 rows"
        ),
    )

    check(
        aligned[
            "track_id"
        ].tolist()
        ==
        bert_manifest[
            "track_id"
        ].astype(str).tolist(),
        (
            "GAT cache order exactly matches "
            "Script-47 BERT cache order"
        ),
    )

    check(
        aligned[
            "split"
        ].tolist()
        ==
        bert_manifest[
            "split"
        ].astype(str).str.lower().tolist(),
        (
            "GAT split order exactly matches "
            "Script-47 BERT split order"
        ),
    )

    check(
        np.array_equal(
            aligned[
                "cache_index"
            ].to_numpy(
                dtype=np.int64
            ),
            np.arange(
                EXPECTED_DEVELOPMENT,
                dtype=np.int64,
            ),
        ),
        (
            "aligned GAT cache indices "
            "are contiguous 0..4625"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Graph-manifest alignment failed"
        )

    return aligned


# =============================================================================
# 9. GRAPH PATH RESOLUTION
# =============================================================================

def resolve_graph_path(
    raw_value: str,
) -> Path:

    raw = Path(
        raw_value
    )

    candidates: list[Path] = []

    if raw.is_absolute():

        candidates.append(
            raw
        )

    else:

        candidates.append(
            PROJECT_ROOT
            / raw
        )

        candidates.append(
            GRAPH_MANIFEST.parent
            / raw
        )

        candidates.append(
            PROJECT_ROOT
            / "data/processed/graphs"
            / raw.name
        )

    existing: dict[str, Path] = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = (
                candidate.resolve()
            )

            existing[
                str(resolved)
            ] = resolved

    if len(existing) == 0:

        raise FileNotFoundError(
            "Could not resolve graph path "
            f"from manifest value: {raw_value}"
        )

    if len(existing) > 1:

        paths = "\n".join(
            f"  - {path}"
            for path in existing.values()
        )

        raise RuntimeError(
            "Multiple distinct graph paths "
            "resolved from manifest value "
            f"{raw_value}:\n{paths}"
        )

    return next(
        iter(
            existing.values()
        )
    )


# =============================================================================
# 10. GRAPH DATASET
# =============================================================================

class FrozenG2Dataset:
    """
    Read-only TRAIN+VAL graph dataset.

    Important:
        The dataframe supplied here was already filtered to TRAIN+VAL
        and aligned to the Script-47 BERT cache.

        TEST graph files therefore cannot be opened by this dataset.
    """

    EXPECTED_KEYS = {
        "x",
        "edge_index_g1",
        "edge_index_g2",
        "similarity_threshold",
        "nonlocal_only",
        "extra_degree_cap",
        "rule_version",
    }

    def __init__(
        self,
        aligned_manifest: pd.DataFrame,
    ) -> None:

        self.manifest = (
            aligned_manifest
            .reset_index(
                drop=True
            )
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.manifest
        )

    def __getitem__(
        self,
        index: int,
    ) -> Data:

        row = (
            self.manifest
            .iloc[index]
        )

        split = str(
            row[
                "split"
            ]
        ).lower()

        if split not in {
            "train",
            "val",
        }:

            raise RuntimeError(
                "Forbidden split encountered "
                f"inside GAT dataset: {split}"
            )

        graph_path = (
            resolve_graph_path(
                str(
                    row[
                        "graph_path"
                    ]
                )
            )
        )

        with np.load(
            graph_path,
            allow_pickle=False,
        ) as graph:

            actual_keys = set(
                graph.files
            )

            if (
                actual_keys
                !=
                self.EXPECTED_KEYS
            ):

                raise RuntimeError(
                    "Unexpected graph NPZ schema "
                    f"for {graph_path.name}.\n"
                    f"Expected: "
                    f"{sorted(self.EXPECTED_KEYS)}\n"
                    f"Actual: "
                    f"{sorted(actual_keys)}"
                )

            x = np.asarray(
                graph[
                    "x"
                ]
            )

            edge_index_g1 = np.asarray(
                graph[
                    "edge_index_g1"
                ]
            )

            edge_index_g2 = np.asarray(
                graph[
                    "edge_index_g2"
                ]
            )

            similarity_threshold = float(
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

            extra_degree_cap = int(
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

        # ---------------------------------------------------------------------
        # Exact graph contract
        # ---------------------------------------------------------------------

        if x.shape != (
            EXPECTED_NODES,
            EXPECTED_RAW_NODE_DIM,
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                f"x shape {x.shape}, "
                "expected (9, 140)"
            )

        if x.dtype != np.float32:

            raise RuntimeError(
                f"{graph_path.name}: "
                f"x dtype {x.dtype}, "
                "expected float32"
            )

        if not np.isfinite(
            x
        ).all():

            raise RuntimeError(
                f"{graph_path.name}: "
                "non-finite node features"
            )

        if edge_index_g1.shape != (
            2,
            EXPECTED_G1_EDGES,
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "unexpected G1 edge shape "
                f"{edge_index_g1.shape}"
            )

        if (
            edge_index_g1.dtype
            != np.int64
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "G1 edge dtype must be int64"
            )

        if (
            edge_index_g2.ndim != 2
            or
            edge_index_g2.shape[0] != 2
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "G2 edge_index must have "
                "shape (2, E)"
            )

        if (
            edge_index_g2.dtype
            != np.int64
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "G2 edge dtype must be int64"
            )

        g2_edges = int(
            edge_index_g2.shape[1]
        )

        if not (
            EXPECTED_G2_MIN_EDGES
            <=
            g2_edges
            <=
            EXPECTED_G2_MAX_EDGES
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                f"G2 edges={g2_edges}, "
                "outside frozen range 16..34"
            )

        if g2_edges != int(
            row[
                "g2_directed_edges"
            ]
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "G2 edge count disagrees "
                "with graph manifest"
            )

        if not np.isclose(
            similarity_threshold,
            EXPECTED_SIMILARITY_THRESHOLD,
            rtol=0.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "similarity threshold changed"
            )

        if (
            nonlocal_only
            !=
            EXPECTED_NONLOCAL_ONLY
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "nonlocal_only metadata changed"
            )

        if (
            extra_degree_cap
            !=
            EXPECTED_EXTRA_DEGREE_CAP
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "extra_degree_cap metadata changed"
            )

        if (
            rule_version
            !=
            EXPECTED_GRAPH_RULE_VERSION
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "graph rule version changed"
            )

        # Node index safety.
        if (
            np.min(
                edge_index_g2
            ) < 0
            or
            np.max(
                edge_index_g2
            ) >= EXPECTED_NODES
        ):

            raise RuntimeError(
                f"{graph_path.name}: "
                "G2 edge contains invalid "
                "node index"
            )

        return Data(
            x=torch.from_numpy(
                x.copy()
            ),
            edge_index=torch.from_numpy(
                edge_index_g2.copy()
            ),
        )


# =============================================================================
# 11. EXACT A4 GAT ARCHITECTURE
# =============================================================================

class A4GAT(
    torch.nn.Module
):
    """
    Frozen Task-2 A4 architecture.

    Contract established before/through Scripts 43-46:

        Linear(140 -> 64)

        GAT1:
            64 -> 32/head
            heads = 4
            concat = True
            output = 128
            ReLU

        GAT2:
            128 -> 32/head
            heads = 4
            concat = True
            output = 128
            ReLU

        GAT3:
            128 -> 128/head
            heads = 4
            concat = False
            output = 128

        global_mean_pool

        Linear(128 -> 20)

    Attention dropout:
        0.2

    add_self_loops:
        True

    bias:
        True

    residual:
        False

    Extra feature dropout:
        None
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.node_projection = (
            torch.nn.Linear(
                EXPECTED_RAW_NODE_DIM,
                EXPECTED_NODE_PROJECTION,
            )
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

        self.classifier = (
            torch.nn.Linear(
                EXPECTED_GAT_HIDDEN,
                EXPECTED_LABELS,
            )
        )

    def encode_graph(
        self,
        data: Data,
    ) -> torch.Tensor:

        x = data.x

        edge_index = (
            data.edge_index
        )

        batch = data.batch

        x = self.node_projection(
            x
        )

        x = self.gat1(
            x,
            edge_index,
        )

        x = F.relu(
            x
        )

        x = self.gat2(
            x,
            edge_index,
        )

        x = F.relu(
            x
        )

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

        return graph_embedding

    def forward(
        self,
        data: Data,
    ) -> torch.Tensor:

        graph_embedding = (
            self.encode_graph(
                data
            )
        )

        return self.classifier(
            graph_embedding
        )


# =============================================================================
# 12. LOAD EXACT SELECTED A4 CHECKPOINT
# =============================================================================

def build_frozen_a4(
    device: torch.device,
) -> tuple[
    A4GAT,
    dict[str, Any],
]:

    banner(
        "4. LOAD SELECTED A4 GAT/G2"
    )

    checkpoint = load_checkpoint(
        A4_CHECKPOINT
    )

    required_fields = (
        "task",
        "variant",
        "model",
        "graph",
        "run_id",
        "seed",
        "epoch",
        "val_auc_pr_macro_mean_ap",
        "architecture",
        "graph_rule",
        "training",
        "pretraining_lock_sha256",
        "model_state_dict",
    )

    for field in (
        required_fields
    ):

        check(
            field in checkpoint,
            (
                "A4 checkpoint contains "
                f"field '{field}'"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "A4 checkpoint metadata incomplete"
        )

    check(
        checkpoint[
            "task"
        ]
        == "task2",
        (
            "checkpoint task is "
            "exactly task2"
        ),
    )

    check(
        checkpoint[
            "variant"
        ]
        == "A4",
        (
            "checkpoint variant is "
            "exactly A4"
        ),
    )

    check(
        checkpoint[
            "model"
        ]
        == "GAT",
        (
            "checkpoint model is "
            "exactly GAT"
        ),
    )

    check(
        checkpoint[
            "graph"
        ]
        == "G2",
        (
            "checkpoint graph is "
            "exactly G2"
        ),
    )

    check(
        int(
            checkpoint[
                "seed"
            ]
        )
        == 42,
        (
            "A4 checkpoint seed "
            "is exactly 42"
        ),
    )

    check(
        int(
            checkpoint[
                "epoch"
            ]
        )
        == 11,
        (
            "selected A4 checkpoint epoch "
            "is exactly 11"
        ),
    )

    check(
        checkpoint[
            "pretraining_lock_sha256"
        ]
        ==
        EXPECTED_PRETRAINING_LOCK_SHA256,
        (
            "A4 checkpoint records the "
            "expected preprocessing lock"
        ),
    )

    architecture = (
        checkpoint[
            "architecture"
        ]
    )

    check(
        int(
            architecture[
                "raw_node_dim"
            ]
        )
        == 140,
        (
            "A4 metadata raw-node "
            "dimension is 140"
        ),
    )

    check(
        architecture[
            "node_projection"
        ]
        == "Linear(140,64)",
        (
            "A4 metadata node projection "
            "is Linear(140,64)"
        ),
    )

    check(
        int(
            architecture[
                "gat1"
            ][
                "output_dim"
            ]
        )
        == 128,
        (
            "A4 GAT1 metadata output "
            "dimension is 128"
        ),
    )

    check(
        int(
            architecture[
                "gat2"
            ][
                "output_dim"
            ]
        )
        == 128,
        (
            "A4 GAT2 metadata output "
            "dimension is 128"
        ),
    )

    check(
        int(
            architecture[
                "gat3"
            ][
                "output_dim"
            ]
        )
        == 128,
        (
            "A4 GAT3 metadata output "
            "dimension is 128"
        ),
    )

    check(
        architecture[
            "classifier"
        ]
        == "Linear(128,20)",
        (
            "A4 classifier metadata "
            "is Linear(128,20)"
        ),
    )

    check(
        architecture[
            "pool"
        ]
        == "global_mean_pool",
        (
            "A4 pooling metadata is "
            "global_mean_pool"
        ),
    )

    check(
        float(
            architecture[
                "attention_dropout"
            ]
        )
        == 0.2,
        (
            "A4 attention dropout "
            "metadata is 0.2"
        ),
    )

    check(
        architecture[
            "extra_feature_dropout"
        ]
        is None,
        (
            "A4 metadata confirms no "
            "extra feature dropout"
        ),
    )

    check(
        architecture[
            "add_self_loops"
        ]
        is True,
        (
            "A4 metadata confirms "
            "add_self_loops=True"
        ),
    )

    check(
        architecture[
            "bias"
        ]
        is True,
        (
            "A4 metadata confirms "
            "bias=True"
        ),
    )

    check(
        architecture[
            "residual"
        ]
        is False,
        (
            "A4 metadata confirms "
            "residual=False"
        ),
    )

    graph_rule = (
        checkpoint[
            "graph_rule"
        ]
    )

    check(
        np.isclose(
            float(
                graph_rule[
                    "similarity_threshold"
                ]
            ),
            0.85,
            atol=1e-8,
            rtol=0.0,
        ),
        (
            "A4 graph-rule threshold "
            "is exactly 0.85"
        ),
    )

    check(
        graph_rule[
            "nonlocal_only"
        ]
        is True,
        (
            "A4 graph rule records "
            "nonlocal_only=True"
        ),
    )

    check(
        int(
            graph_rule[
                "extra_degree_cap"
            ]
        )
        == 2,
        (
            "A4 graph rule records "
            "extra_degree_cap=2"
        ),
    )

    check(
        int(
            graph_rule[
                "rule_version"
            ]
        )
        == 2,
        (
            "A4 graph rule records "
            "rule_version=2"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "A4 architecture metadata "
            "validation failed"
        )

    model = A4GAT()

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

    if not isinstance(
        state_dict,
        dict,
    ):

        raise RuntimeError(
            "A4 model_state_dict "
            "is not a dictionary"
        )

    try:

        incompatible = (
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "A4 checkpoint does not strictly "
            "match the reconstructed frozen "
            "A4 architecture"
        ) from exc

    check(
        len(
            incompatible.missing_keys
        )
        == 0,
        (
            "strict A4 load has "
            "zero missing keys"
        ),
    )

    check(
        len(
            incompatible.unexpected_keys
        )
        == 0,
        (
            "strict A4 load has "
            "zero unexpected keys"
        ),
    )

    # -------------------------------------------------------------------------
    # Parameter-shape contract
    # -------------------------------------------------------------------------

    state = (
        model.state_dict()
    )

    check(
        tuple(
            state[
                "node_projection.weight"
            ].shape
        )
        == (
            64,
            140,
        ),
        (
            "loaded A4 node projection "
            "weight is (64,140)"
        ),
    )

    check(
        tuple(
            state[
                "classifier.weight"
            ].shape
        )
        == (
            20,
            128,
        ),
        (
            "loaded A4 classifier weight "
            "is (20,128)"
        ),
    )

    # -------------------------------------------------------------------------
    # Freeze everything.
    # -------------------------------------------------------------------------

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    model.eval()

    check(
        not model.training,
        (
            "A4 GAT is in "
            "eval mode"
        ),
    )

    check(
        all(
            not parameter.requires_grad
            for parameter
            in model.parameters()
        ),
        (
            "all A4 parameters "
            "are frozen"
        ),
    )

    model.to(
        device
    )

    if FAILURES:

        raise RuntimeError(
            "A4 model loading validation failed"
        )

    return (
        model,
        checkpoint,
    )


# =============================================================================
# 13. OUTPUT COLLISION PROTECTION
# =============================================================================

def verify_clean_output_location() -> None:

    banner(
        "5. OUTPUT COLLISION PROTECTION"
    )

    check(
        not OUTPUT_DIR.exists(),
        (
            "final gat_g2_frozen output "
            "directory does not already exist"
        ),
    )

    check(
        not TEMP_DIR.exists(),
        (
            "temporary gat_g2_frozen.__tmp__ "
            "directory does not already exist"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Output collision detected; "
            "refusing to overwrite"
        )


# =============================================================================
# 14. BUILD DATALOADER
# =============================================================================

def build_dataloader(
    aligned_manifest: pd.DataFrame,
) -> DataLoader:

    banner(
        "6. BUILD TRAIN+VAL G2 DATALOADER"
    )

    dataset = FrozenG2Dataset(
        aligned_manifest
    )

    check(
        len(dataset)
        == EXPECTED_DEVELOPMENT,
        (
            "G2 dataset contains exactly "
            "4626 TRAIN+VAL graphs"
        ),
    )

    check(
        "test"
        not in set(
            aligned_manifest[
                "split"
            ]
        ),
        (
            "G2 dataset manifest contains "
            "zero TEST graphs"
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    expected_batches = (
        math.ceil(
            EXPECTED_DEVELOPMENT
            / BATCH_SIZE
        )
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Expected batches: "
        f"{expected_batches}"
    )

    print(
        "Shuffle: False"
    )

    print(
        "Workers: 0"
    )

    print(
        "TEST graphs scheduled: 0"
    )

    return loader


# =============================================================================
# 15. CACHE GENERATION
# =============================================================================

def generate_cache(
    model: A4GAT,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:

    banner(
        "7. GENERATE TRAIN+VAL FROZEN GAT REPRESENTATIONS"
    )

    TEMP_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    embedding_path = (
        TEMP_DIR
        / "gat_g2_embedding_trainval.npy"
    )

    expected_shape = (
        EXPECTED_DEVELOPMENT,
        EXPECTED_GAT_HIDDEN,
    )

    cache = np.lib.format.open_memmap(
        embedding_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape,
    )

    total_batches = len(
        loader
    )

    print(
        f"Embedding shape: "
        f"{expected_shape}"
    )

    print(
        "Embedding dtype: float32"
    )

    print(
        f"Total batches: "
        f"{total_batches}"
    )

    print(
        "Inference splits: TRAIN + VAL only"
    )

    print(
        "TEST graph inference examples: 0"
    )

    rows_written = 0

    global_abs_max = 0.0

    with torch.inference_mode():

        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):

            batch = batch.to(
                device
            )

            embedding = (
                model.encode_graph(
                    batch
                )
            )

            expected_batch_rows = (
                int(
                    batch.num_graphs
                )
            )

            expected_batch_shape = (
                expected_batch_rows,
                EXPECTED_GAT_HIDDEN,
            )

            if (
                tuple(
                    embedding.shape
                )
                !=
                expected_batch_shape
            ):

                raise RuntimeError(
                    "Unexpected A4 graph "
                    "representation shape. "
                    f"Expected "
                    f"{expected_batch_shape}, "
                    f"got "
                    f"{tuple(embedding.shape)}"
                )

            if (
                embedding.dtype
                != torch.float32
            ):

                raise RuntimeError(
                    "Unexpected A4 embedding dtype: "
                    f"{embedding.dtype}; "
                    "expected torch.float32"
                )

            if not torch.isfinite(
                embedding
            ).all():

                raise RuntimeError(
                    "Non-finite A4 graph "
                    f"embedding detected in "
                    f"batch {batch_number}"
                )

            # -------------------------------------------------------------
            # Classifier-input contract check.
            #
            # The cached vector must be exactly the tensor consumed by
            # classifier Linear(128,20).
            # -------------------------------------------------------------

            logits = (
                model.classifier(
                    embedding
                )
            )

            expected_logits_shape = (
                expected_batch_rows,
                EXPECTED_LABELS,
            )

            if (
                tuple(
                    logits.shape
                )
                !=
                expected_logits_shape
            ):

                raise RuntimeError(
                    "Classifier compatibility "
                    "check failed. "
                    f"Expected logits "
                    f"{expected_logits_shape}, "
                    f"got "
                    f"{tuple(logits.shape)}"
                )

            if not torch.isfinite(
                logits
            ).all():

                raise RuntimeError(
                    "Non-finite classifier logits "
                    "during embedding-contract "
                    "validation"
                )

            embedding_np = (
                embedding
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            start = (
                rows_written
            )

            end = (
                start
                + expected_batch_rows
            )

            cache[
                start:end
            ] = embedding_np

            rows_written = end

            batch_abs_max = float(
                np.max(
                    np.abs(
                        embedding_np
                    )
                )
            )

            global_abs_max = max(
                global_abs_max,
                batch_abs_max,
            )

            if (
                batch_number == 1
                or
                batch_number % 10 == 0
                or
                batch_number == total_batches
            ):

                print(
                    f"Batch "
                    f"{batch_number:3d}/"
                    f"{total_batches}: "
                    f"rows "
                    f"{start:4d}.."
                    f"{end - 1:4d} "
                    "cached"
                )

            del embedding
            del logits
            del batch

    cache.flush()

    del cache

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    print()

    print(
        f"Rows written: "
        f"{rows_written}"
    )

    print(
        "Maximum absolute GAT "
        f"embedding value: "
        f"{global_abs_max:.6f}"
    )

    check(
        rows_written
        == EXPECTED_DEVELOPMENT,
        (
            "exactly 4626 TRAIN+VAL "
            "graph embeddings were written"
        ),
    )

    return {
        "embedding_path":
            embedding_path,

        "rows_written":
            rows_written,

        "global_abs_max":
            global_abs_max,
    }


# =============================================================================
# 16. WRITE MANIFEST + METADATA
# =============================================================================

def write_supporting_artifacts(
    aligned_manifest: pd.DataFrame,
    checkpoint: dict[str, Any],
    generation_info: dict[str, Any],
    device: torch.device,
) -> dict[str, Path]:

    banner(
        "8. WRITE GAT CACHE MANIFEST + PROVENANCE"
    )

    manifest_path = (
        TEMP_DIR
        / "gat_g2_manifest.parquet"
    )

    metadata_path = (
        TEMP_DIR
        / "gat_g2_cache_metadata.json"
    )

    aligned_manifest.to_parquet(
        manifest_path,
        index=False,
    )

    print(
        f"Wrote: "
        f"{relative(manifest_path)}"
    )

    cuda_name = None

    if (
        device.type
        == "cuda"
    ):

        cuda_name = (
            torch.cuda.get_device_name(
                device
            )
        )

    metadata = {
        "artifact_type":
            "task3_frozen_gat_g2_embedding_cache",

        "version": 1,

        "construction_scope":
            "train_val_only",

        "test_examples_inferred": 0,

        "test_graph_files_opened": 0,

        "test_labels_accessed":
            False,

        "threshold_tuning_used":
            False,

        "training_performed":
            False,

        "encoder_frozen":
            True,

        "source_task":
            "task2",

        "source_variant":
            "A4",

        "source_model":
            "GAT",

        "source_graph":
            "G2",

        "source_checkpoint":
            relative(
                A4_CHECKPOINT
            ),

        "source_checkpoint_sha256":
            sha256_file(
                A4_CHECKPOINT
            ),

        "source_checkpoint_seed":
            int(
                checkpoint[
                    "seed"
                ]
            ),

        "source_checkpoint_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "source_checkpoint_val_auc_pr_macro_mean_ap":
            float(
                checkpoint[
                    "val_auc_pr_macro_mean_ap"
                ]
            ),

        "pretraining_lock":
            relative(
                PRETRAINING_LOCK
            ),

        "pretraining_lock_sha256":
            sha256_file(
                PRETRAINING_LOCK
            ),

        "graph_manifest":
            relative(
                GRAPH_MANIFEST
            ),

        "graph_manifest_sha256":
            sha256_file(
                GRAPH_MANIFEST
            ),

        "graph_rule": {
            "similarity_threshold":
                EXPECTED_SIMILARITY_THRESHOLD,

            "nonlocal_only":
                EXPECTED_NONLOCAL_ONLY,

            "extra_degree_cap":
                EXPECTED_EXTRA_DEGREE_CAP,

            "rule_version":
                EXPECTED_GRAPH_RULE_VERSION,
        },

        "split_counts": {
            "train":
                EXPECTED_TRAIN,

            "val":
                EXPECTED_VAL,

            "test":
                0,

            "total_cached":
                EXPECTED_DEVELOPMENT,
        },

        "representation": {
            "location":
                (
                    "after GAT3 and "
                    "global_mean_pool, "
                    "before classifier"
                ),

            "dimension":
                EXPECTED_GAT_HIDDEN,

            "dtype":
                "float32",

            "classifier_input":
                "Linear(128,20)",
        },

        "cross_modal_alignment": {
            "canonical_order_source":
                relative(
                    BERT_SEQ_MANIFEST
                ),

            "canonical_manifest_sha256":
                sha256_file(
                    BERT_SEQ_MANIFEST
                ),

            "alignment_key":
                "track_id",

            "rowwise_alignment":
                True,

            "rows":
                EXPECTED_DEVELOPMENT,
        },

        "cache_generation": {
            "batch_size":
                BATCH_SIZE,

            "torch_inference_mode":
                True,

            "model_eval_mode":
                True,

            "autocast":
                False,

            "tf32_allowed":
                False,

            "output_dtype":
                "float32",

            "rows_written":
                int(
                    generation_info[
                        "rows_written"
                    ]
                ),

            "maximum_absolute_embedding_value":
                float(
                    generation_info[
                        "global_abs_max"
                    ]
                ),
        },

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

            "cuda_available":
                torch.cuda.is_available(),

            "device":
                str(
                    device
                ),

            "cuda_device_name":
                cuda_name,
        },
    }

    write_json(
        metadata_path,
        metadata,
    )

    print(
        f"Wrote: "
        f"{relative(metadata_path)}"
    )

    return {
        "manifest":
            manifest_path,

        "metadata":
            metadata_path,
    }


# =============================================================================
# 17. POST-GENERATION VALIDATION
# =============================================================================

def validate_completed_cache(
    generation_info: dict[str, Any],
    supporting: dict[str, Path],
    bert_manifest: pd.DataFrame,
) -> None:

    banner(
        "9. POST-GENERATION CACHE VALIDATION"
    )

    embedding_path = (
        generation_info[
            "embedding_path"
        ]
    )

    manifest_path = (
        supporting[
            "manifest"
        ]
    )

    metadata_path = (
        supporting[
            "metadata"
        ]
    )

    for path in (
        embedding_path,
        manifest_path,
        metadata_path,
    ):

        check(
            path.is_file(),
            (
                "generated artifact exists: "
                f"{path.name}"
            ),
        )

    embedding = np.load(
        embedding_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    check(
        embedding.shape
        == (
            EXPECTED_DEVELOPMENT,
            EXPECTED_GAT_HIDDEN,
        ),
        (
            "saved GAT embedding shape "
            "is exactly (4626,128)"
        ),
    )

    check(
        embedding.dtype
        == np.float32,
        (
            "saved GAT embedding "
            "dtype is float32"
        ),
    )

    # Small enough to validate every value.
    check(
        np.isfinite(
            embedding
        ).all(),
        (
            "all saved GAT embeddings "
            "are finite"
        ),
    )

    max_abs = float(
        np.max(
            np.abs(
                embedding
            )
        )
    )

    std = float(
        np.std(
            embedding
        )
    )

    print(
        f"Saved embedding max |value|: "
        f"{max_abs:.6f}"
    )

    print(
        f"Saved embedding global std: "
        f"{std:.6f}"
    )

    check(
        std > 0.0,
        (
            "saved GAT embedding cache "
            "is not constant"
        ),
    )

    del embedding

    saved_manifest = (
        pd.read_parquet(
            manifest_path
        )
    )

    check(
        len(
            saved_manifest
        )
        == EXPECTED_DEVELOPMENT,
        (
            "saved GAT manifest contains "
            "exactly 4626 rows"
        ),
    )

    check(
        saved_manifest[
            "track_id"
        ].tolist()
        ==
        bert_manifest[
            "track_id"
        ].astype(str).tolist(),
        (
            "saved GAT manifest row order "
            "exactly matches BERT cache"
        ),
    )

    check(
        saved_manifest[
            "split"
        ].astype(str).str.lower().tolist()
        ==
        bert_manifest[
            "split"
        ].astype(str).str.lower().tolist(),
        (
            "saved GAT split order exactly "
            "matches BERT cache"
        ),
    )

    check(
        "test"
        not in set(
            saved_manifest[
                "split"
            ]
            .astype(str)
            .str.lower()
        ),
        (
            "saved GAT manifest contains "
            "zero TEST rows"
        ),
    )

    metadata = load_json(
        metadata_path
    )

    check(
        metadata[
            "test_examples_inferred"
        ]
        == 0,
        (
            "metadata records zero "
            "TEST graph inference"
        ),
    )

    check(
        metadata[
            "test_graph_files_opened"
        ]
        == 0,
        (
            "metadata records zero "
            "TEST graph files opened"
        ),
    )

    check(
        metadata[
            "test_labels_accessed"
        ]
        is False,
        (
            "metadata records zero "
            "TEST-label access"
        ),
    )

    check(
        metadata[
            "training_performed"
        ]
        is False,
        (
            "metadata records "
            "no model training"
        ),
    )

    check(
        metadata[
            "threshold_tuning_used"
        ]
        is False,
        (
            "metadata records "
            "no threshold tuning"
        ),
    )

    check(
        metadata[
            "representation"
        ][
            "dimension"
        ]
        == EXPECTED_GAT_HIDDEN,
        (
            "metadata records 128-D "
            "GAT representation"
        ),
    )

    check(
        metadata[
            "cross_modal_alignment"
        ][
            "rowwise_alignment"
        ]
        is True,
        (
            "metadata records explicit "
            "rowwise BERT/GAT alignment"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Post-generation GAT cache "
            "validation failed"
        )


# =============================================================================
# 18. HASH CACHE
# =============================================================================

def write_output_hashes(
    generation_info: dict[str, Any],
    supporting: dict[str, Path],
) -> Path:

    banner(
        "10. HASH GENERATED GAT CACHE"
    )

    files_to_hash = {
        "gat_g2_embedding_trainval":
            generation_info[
                "embedding_path"
            ],

        "gat_g2_manifest":
            supporting[
                "manifest"
            ],

        "gat_g2_cache_metadata":
            supporting[
                "metadata"
            ],
    }

    lock = {
        "lock_type":
            "task3_gat_g2_embedding_cache_lock",

        "version":
            1,

        "construction_scope":
            "train_val_only",

        "test_examples_inferred":
            0,

        "files":
            {},
    }

    for name, path in (
        files_to_hash.items()
    ):

        print(
            f"Hashing "
            f"{path.name}..."
        )

        digest = sha256_file(
            path
        )

        lock[
            "files"
        ][
            name
        ] = {
            "path":
                (
                    "data/processed/"
                    "gat_g2_frozen/"
                    f"{path.name}"
                ),

            "sha256":
                digest,

            "size_bytes":
                int(
                    path.stat().st_size
                ),
        }

        print(
            f"  SHA256: "
            f"{digest}"
        )

    lock_path = (
        TEMP_DIR
        / "gat_g2_frozen_hashes.json"
    )

    write_json(
        lock_path,
        lock,
    )

    print(
        f"Wrote: "
        f"{relative(lock_path)}"
    )

    return lock_path


# =============================================================================
# 19. FINAL ATOMIC COMMIT
# =============================================================================

def commit_cache(
    lock_path: Path,
) -> None:

    banner(
        "11. FINAL ATOMIC COMMIT"
    )

    check(
        lock_path.is_file(),
        (
            "GAT cache lock/hash "
            "artifact exists"
        ),
    )

    check(
        not OUTPUT_DIR.exists(),
        (
            "final output location "
            "remains absent before commit"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Final GAT cache commit "
            "safety validation failed"
        )

    TEMP_DIR.rename(
        OUTPUT_DIR
    )

    check(
        OUTPUT_DIR.is_dir(),
        (
            "gat_g2_frozen cache "
            "committed successfully"
        ),
    )

    check(
        not TEMP_DIR.exists(),
        (
            "temporary GAT cache "
            "directory no longer exists"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Final GAT cache commit failed"
        )


# =============================================================================
# 20. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 48 — CACHE FROZEN TASK-3 A4 GAT/G2 REPRESENTATIONS"
    )

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print()

    print(
        "Source encoder:"
    )

    print(
        "  Task-2 A4 GAT / G2"
    )

    print()

    print(
        "Representation:"
    )

    print(
        "  after GAT3"
    )

    print(
        "  after global_mean_pool"
    )

    print(
        "  before Linear(128,20)"
    )

    print(
        "  dimension = 128"
    )

    print()

    print(
        "Canonical Task-3 row order:"
    )

    print(
        "  Script-47 BERT sequence manifest"
    )

    print(
        "  alignment key = track_id"
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
        # Immutable project inputs
        # ---------------------------------------------------------------------

        verify_frozen_inputs()

        # ---------------------------------------------------------------------
        # Verify Script-47 text cache before using its order
        # ---------------------------------------------------------------------

        bert_manifest = (
            verify_bert_cache()
        )

        # ---------------------------------------------------------------------
        # Graph manifest + cross-modal row alignment
        # ---------------------------------------------------------------------

        aligned_manifest = (
            load_and_align_graph_manifest(
                bert_manifest
            )
        )

        # ---------------------------------------------------------------------
        # Protect outputs
        # ---------------------------------------------------------------------

        verify_clean_output_location()

        # ---------------------------------------------------------------------
        # Device
        # ---------------------------------------------------------------------

        banner(
            "3A. INFERENCE DEVICE"
        )

        check(
            torch.cuda.is_available(),
            (
                "CUDA is available"
            ),
        )

        if FAILURES:

            raise RuntimeError(
                "Expected CUDA environment "
                "is unavailable"
            )

        device = torch.device(
            "cuda"
        )

        print(
            "CUDA device: "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"PyTorch: "
            f"{torch.__version__}"
        )

        print(
            f"PyG: "
            f"{torch_geometric.__version__}"
        )

        # ---------------------------------------------------------------------
        # Selected A4 encoder
        # ---------------------------------------------------------------------

        model, checkpoint = (
            build_frozen_a4(
                device
            )
        )

        # ---------------------------------------------------------------------
        # Development-only graph loader
        # ---------------------------------------------------------------------

        loader = (
            build_dataloader(
                aligned_manifest
            )
        )

        # ---------------------------------------------------------------------
        # Cache generation
        # ---------------------------------------------------------------------

        generation_info = (
            generate_cache(
                model=model,
                loader=loader,
                device=device,
            )
        )

        del model
        del loader

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------
        # Manifest + provenance
        # ---------------------------------------------------------------------

        supporting = (
            write_supporting_artifacts(
                aligned_manifest=
                    aligned_manifest,

                checkpoint=
                    checkpoint,

                generation_info=
                    generation_info,

                device=
                    device,
            )
        )

        # ---------------------------------------------------------------------
        # Validate complete temp cache
        # ---------------------------------------------------------------------

        validate_completed_cache(
            generation_info=
                generation_info,

            supporting=
                supporting,

            bert_manifest=
                bert_manifest,
        )

        # ---------------------------------------------------------------------
        # Freeze cache hashes
        # ---------------------------------------------------------------------

        lock_path = (
            write_output_hashes(
                generation_info=
                    generation_info,

                supporting=
                    supporting,
            )
        )

        # ---------------------------------------------------------------------
        # Atomic commit
        # ---------------------------------------------------------------------

        commit_cache(
            lock_path
        )

    except Exception as exc:

        banner(
            "SCRIPT 48 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        if FAILURES:

            print(
                f"Failed checks: "
                f"{len(FAILURES)}"
            )

            for message in (
                FAILURES
            ):

                print(
                    f"  - {message}"
                )

        print()

        if TEMP_DIR.exists():

            print(
                "IMPORTANT:"
            )

            print(
                "A temporary GAT cache "
                "directory exists:"
            )

            print(
                f"  {TEMP_DIR}"
            )

            print()

            print(
                "It has NOT been committed "
                "as the final cache."
            )

            print(
                "Do not reuse or delete it "
                "until inspected."
            )

        print()

        print(
            "Final gat_g2_frozen cache "
            "was NOT successfully frozen."
        )

        print(
            "Do not proceed to fusion training."
        )

        return 1

    # =========================================================================
    # Final summary
    # =========================================================================

    banner(
        "SCRIPT 48 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Frozen encoder:"
    )

    print(
        "  A4 GAT / G2"
    )

    print(
        "  checkpoint SHA256:"
    )

    print(
        f"  "
        f"{EXPECTED_A4_CHECKPOINT_SHA256}"
    )

    print()

    print(
        "Cached examples:"
    )

    print(
        f"  TRAIN = "
        f"{EXPECTED_TRAIN}"
    )

    print(
        f"  VAL   = "
        f"{EXPECTED_VAL}"
    )

    print(
        "  TEST  = 0"
    )

    print(
        f"  TOTAL = "
        f"{EXPECTED_DEVELOPMENT}"
    )

    print()

    print(
        "Frozen GAT representation:"
    )

    print(
        "  gat_g2_embedding_trainval.npy"
    )

    print(
        "  shape = (4626, 128)"
    )

    print(
        "  dtype = float32"
    )

    print()

    print(
        "Representation location:"
    )

    print(
        "  GAT3"
    )

    print(
        "    -> global_mean_pool"
    )

    print(
        "    -> [CACHED 128-D VECTOR]"
    )

    print(
        "    -> original classifier "
        "Linear(128,20)"
    )

    print()

    print(
        "Cross-modal alignment:"
    )

    print(
        "  GAT row i == BERT row i"
    )

    print(
        "  alignment key = track_id"
    )

    print()

    print(
        "Model training performed: 0"
    )

    print(
        "TEST graph inference:      0"
    )

    print(
        "TEST graph files opened:   0"
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
        "Review Script 48 output "
        "before creating any fusion model."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )