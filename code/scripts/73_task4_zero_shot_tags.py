#!/usr/bin/env python3
"""
SCRIPT 73 — FROZEN TASK-4 ZERO-SHOT TAG EVALUATION
==================================================

Primary Task-4 zero-shot mode
-----------------------------
Caption embedding -> canonical label-text embeddings

Additional analysis
-------------------
Audio graph embedding -> canonical label-text embeddings

Selected contrastive configuration
----------------------------------
T4-B

Predetermined seeds
-------------------
42
1337
2026

Frozen labels
-------------
Exactly the 20 canonical TRAIN-only labels.

NO prompt engineering:
    label phrase = exact canonical label text

Scoring
-------
cosine similarity in the learned 256-D Task-4 space

Primary metric
--------------
Macro Average Precision (threshold-free)

Additional metrics
------------------
Micro AP
Macro F1
Micro F1

F1 threshold protocol
---------------------
For each predetermined seed and each zero-shot mode independently:

    - select ONE global threshold across all 20 labels
    - use VALIDATION only
    - fixed grid [-1.0, 1.0] with step 0.005
    - maximize VAL Macro F1
    - exact tie -> lowest threshold

CRITICAL ORDER
--------------
1. Compute all VAL scores.
2. Freeze + hash all VAL-selected thresholds.
3. ONLY THEN load TEST tag targets.
4. Evaluate TEST once with frozen thresholds.

TEST retrieval is NOT rerun.
Script-72 frozen TEST embeddings are reused.

Task-3 supervised comparison is NOT recomputed here.
It will be joined with the already frozen Task-3 result in the final
Task-4 reporting stage, avoiding any reopening/retraining of Task-3.
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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
    GATConv,
    global_mean_pool,
)

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
)


# =============================================================================
# 0. ROOT / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Frozen Task-4 protocol / retrieval
# -----------------------------------------------------------------------------

PRETEST_PROTOCOL = (
    ROOT
    / "results/runs/task4/"
      "task4_pretest_protocol.json"
)

PRETEST_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_pretest_protocol_lock.json"
)

TEST_RETRIEVAL_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_test_retrieval_summary.json"
)

TEST_RETRIEVAL_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_test_retrieval_lock.json"
)

TEST_RETRIEVAL_MANIFEST = (
    ROOT
    / "results/runs/task4/"
      "task4_test_retrieval_manifest.csv"
)


# -----------------------------------------------------------------------------
# Frozen model/data
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


CHECKPOINTS = {
    42:
        ROOT
        / "models/task4_candidates/"
          "task4_T4-B_seed42_best.pt",

    1337:
        ROOT
        / "models/task4_candidates/"
          "task4_T4-B_seed1337_multiseed_best.pt",

    2026:
        ROOT
        / "models/task4_candidates/"
          "task4_T4-B_seed2026_multiseed_best.pt",
}


# -----------------------------------------------------------------------------
# Script-72 TEST embedding caches
# -----------------------------------------------------------------------------

TEST_EMBED_DIR = (
    ROOT
    / "data/processed/task4_test_retrieval"
)


def test_text_embedding_path(
    seed: int,
) -> Path:

    return (
        TEST_EMBED_DIR
        / f"seed{seed}_text_embeddings.npy"
    )


def test_graph_embedding_path(
    seed: int,
) -> Path:

    return (
        TEST_EMBED_DIR
        / f"seed{seed}_graph_embeddings.npy"
    )


# -----------------------------------------------------------------------------
# Script-73 outputs
# -----------------------------------------------------------------------------

OUTPUT_DATA_DIR = (
    ROOT
    / "data/processed/task4_zero_shot_tags"
)

RUN_DIR = (
    ROOT
    / "results/runs/task4"
)

VAL_FREEZE_PATH = (
    RUN_DIR
    / "task4_zero_shot_val_freeze.json"
)

PER_SEED_MODE_CSV = (
    RUN_DIR
    / "task4_zero_shot_test_per_seed_mode.csv"
)

PER_LABEL_CSV = (
    RUN_DIR
    / "task4_zero_shot_test_per_label.csv"
)

SUMMARY_PATH = (
    RUN_DIR
    / "task4_zero_shot_test_summary.json"
)

LOCK_PATH = (
    RUN_DIR
    / "task4_zero_shot_test_lock.json"
)


def score_path(
    seed: int,
    mode: str,
) -> Path:

    return (
        OUTPUT_DATA_DIR
        / f"seed{seed}_{mode}_test_scores.npy"
    )


# =============================================================================
# 1. ACCEPTED HASHES
# =============================================================================

EXPECTED_PRETEST_PROTOCOL_SHA256 = (
    "f1f4651c9bffab2295ec8ef44d17f208"
    "4430386585fb2e9fd489d0bc2af93f7c"
)

EXPECTED_PRETEST_LOCK_SHA256 = (
    "dddbd32013dd7e9912e0df0fa72d0352"
    "49abae8f4c83f6313b57a99931f00836"
)

EXPECTED_TEST_RETRIEVAL_SUMMARY_SHA256 = (
    "77e08ce11af9547319515137197ae122c"
    "83224d1a445619c6daef799a722af90"
)

EXPECTED_TEST_RETRIEVAL_LOCK_SHA256 = (
    "45a3adc75581b5ddf0f28fe08ecb70c0"
    "0ddb3bd6d3c665f82fe4e4b556d40504"
)

EXPECTED_TEST_MANIFEST_SHA256 = (
    "e3e726814fdd890af0951fd1c4726a4d"
    "e5f506180f4ae6982db51e89e1d39431"
)


EXPECTED_TOKEN_CACHE_SHA256 = (
    "0df45ab266884abbc2f3f7f96a5ba789"
    "bbf92bf8da49ac4cb57a1e4eb476188f"
)

EXPECTED_TOKEN_MANIFEST_SHA256 = (
    "4be874ab26375ce63bb0f4b55706d178b"
    "fac8a951fee96fd75d2d49b28a390b8"
)

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6a"
    "d0492bf90eccc929cadc1aad1da9e158"
)

EXPECTED_TARGETS_SHA256 = (
    "a6f80bb42b0b48b2c47dac5f27ad2fd6"
    "6f8893d83bd5d4e552d356cee3547056"
)

EXPECTED_T1_SHA256 = (
    "3b675acbacb7191d514e3cc513669442"
    "71e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_A4_SHA256 = (
    "c4e3084f6f7f6030caf157623663d311"
    "daf74ea0a24f6b02e78e5513078fae1e"
)


EXPECTED_CHECKPOINT_SHA256 = {
    42:
        (
            "2be12c7e6fdc851b2b1efae0a7bed2c4"
            "95cbefb5ae90a4f491c54a72ec8e9ace"
        ),

    1337:
        (
            "824a98d52327d513f67cfa21cf702a459"
            "7f51a91c0b17cea1ea7d10628b4b918"
        ),

    2026:
        (
            "a328d029b46ef14a4199cff0604f190f8"
            "06cf953c7c4ad75a0f884210bb743b9"
        ),
}


# =============================================================================
# 2. FROZEN ZERO-SHOT CONTRACT
# =============================================================================

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

N_TOTAL = 5140
N_VAL = 514
N_TEST = 514

MAX_LENGTH = 112

BERT_DIM = 768
GAT_DIM = 128
SHARED_DIM = 256

BATCH_SIZE = 32

N_LABELS = 20


CANONICAL_LABELS = (
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
)


Y_COLUMNS = tuple(
    f"y_{index:02d}"
    for index in range(
        N_LABELS
    )
)


ZERO_SHOT_MODES = (
    "caption",
    "audio",
)


# Fixed BEFORE TEST label access.
THRESHOLD_GRID = np.linspace(
    -1.0,
    1.0,
    401,
    dtype=np.float64,
)


EXPECTED_GRAPH_KEYS = {
    "x",
    "edge_index_g1",
    "edge_index_g2",
    "similarity_threshold",
    "nonlocal_only",
    "extra_degree_cap",
    "rule_version",
}


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


def relative(
    path: Path,
) -> str:

    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    )


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

    temporary = Path(
        str(
            path
        )
        + ".__tmp__"
    )

    if temporary.exists():

        raise RuntimeError(
            f"Temporary file exists: "
            f"{temporary}"
        )

    with temporary.open(
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
        temporary,
        path,
    )


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:

        raise RuntimeError(
            f"Cannot write empty CSV: "
            f"{path.name}"
        )

    temporary = Path(
        str(
            path
        )
        + ".__tmp__"
    )

    with temporary.open(
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
        temporary,
        path,
    )


def save_npy_atomic(
    path: Path,
    array: np.ndarray,
) -> None:

    temporary = Path(
        str(
            path
        )
        + ".__tmp__"
    )

    with temporary.open(
        "wb"
    ) as f:

        np.save(
            f,
            array,
            allow_pickle=False,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temporary,
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
            f"{path.name} is not "
            "a checkpoint dictionary"
        )

    return value


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
        observed
        == expected,
        (
            f"{label} SHA256 matches "
            "frozen identity"
        ),
    )


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. ZERO-SHOT OUTPUT PROTECTION"
    )

    paths = [
        VAL_FREEZE_PATH,
        PER_SEED_MODE_CSV,
        PER_LABEL_CSV,
        SUMMARY_PATH,
        LOCK_PATH,
    ]

    for seed in FINAL_SEEDS:

        for mode in ZERO_SHOT_MODES:

            paths.append(
                score_path(
                    seed,
                    mode,
                )
            )

    for path in paths:

        check(
            not path.exists(),
            (
                f"output unused: "
                f"{path.name}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite an "
            "existing zero-shot result"
        )


# =============================================================================
# 6. VERIFY FROZEN LINEAGE
# =============================================================================

def verify_lineage() -> dict[str, Any]:

    banner(
        "2. VERIFY FROZEN TASK-4 / TEST LINEAGE"
    )

    frozen = (
        (
            PRETEST_PROTOCOL,
            EXPECTED_PRETEST_PROTOCOL_SHA256,
            "pre-TEST protocol",
        ),
        (
            PRETEST_LOCK,
            EXPECTED_PRETEST_LOCK_SHA256,
            "pre-TEST lock",
        ),
        (
            TEST_RETRIEVAL_SUMMARY,
            EXPECTED_TEST_RETRIEVAL_SUMMARY_SHA256,
            "final TEST retrieval summary",
        ),
        (
            TEST_RETRIEVAL_LOCK,
            EXPECTED_TEST_RETRIEVAL_LOCK_SHA256,
            "final TEST retrieval lock",
        ),
        (
            TEST_RETRIEVAL_MANIFEST,
            EXPECTED_TEST_MANIFEST_SHA256,
            "final TEST retrieval manifest",
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
            "T1-B checkpoint",
        ),
        (
            A4_CHECKPOINT,
            EXPECTED_A4_SHA256,
            "A4 checkpoint",
        ),
    )

    for path, digest, label in frozen:

        verify_sha(
            path,
            digest,
            label,
        )

    for seed in FINAL_SEEDS:

        verify_sha(
            CHECKPOINTS[
                seed
            ],
            EXPECTED_CHECKPOINT_SHA256[
                seed
            ],
            (
                f"seed-{seed} "
                "T4-B checkpoint"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen zero-shot lineage failed"
        )

    protocol = load_json(
        PRETEST_PROTOCOL
    )

    test_summary = load_json(
        TEST_RETRIEVAL_SUMMARY
    )

    test_lock = load_json(
        TEST_RETRIEVAL_LOCK
    )

    check(
        protocol[
            "zero_shot_tags"
        ][
            "canonical_label_text"
        ]
        ==
        (
            "exact frozen canonical label phrase; "
            "no prompt engineering"
        ),
        (
            "zero-shot label text remains "
            "exact canonical phrase"
        ),
    )

    check(
        protocol[
            "zero_shot_tags"
        ][
            "primary_metric"
        ]
        ==
        "Macro Average Precision",
        (
            "zero-shot primary metric "
            "remains Macro AP"
        ),
    )

    check(
        protocol[
            "zero_shot_tags"
        ][
            "no_best_seed_selection"
        ]
        is True,
        (
            "zero-shot best-seed "
            "selection forbidden"
        ),
    )

    check(
        test_summary[
            "selected_variant"
        ]
        == "T4-B",
        (
            "TEST retrieval used selected T4-B"
        ),
    )

    check(
        test_summary[
            "seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "TEST retrieval used "
            "42/1337/2026"
        ),
    )

    check(
        test_summary[
            "best_seed_selection"
        ]
        is False,
        (
            "TEST retrieval performed "
            "no best-seed selection"
        ),
    )

    check(
        test_lock[
            "test_labels_loaded"
        ]
        is False,
        (
            "Script 72 opened zero TEST labels"
        ),
    )

    # -------------------------------------------------------------------------
    # Verify Script-72 TEST embedding caches dynamically against its lock.
    # -------------------------------------------------------------------------

    artifact_hashes = (
        test_lock[
            "seed_artifact_hashes"
        ]
    )

    for seed in FINAL_SEEDS:

        seed_key = str(
            seed
        )

        check(
            seed_key
            in artifact_hashes,
            (
                f"Script-72 lock contains "
                f"seed {seed} artifacts"
            ),
        )

        if seed_key not in artifact_hashes:
            continue

        verify_sha(
            test_text_embedding_path(
                seed
            ),
            artifact_hashes[
                seed_key
            ][
                "text_embeddings"
            ],
            (
                f"seed-{seed} TEST "
                "text embeddings"
            ),
        )

        verify_sha(
            test_graph_embedding_path(
                seed
            ),
            artifact_hashes[
                seed_key
            ][
                "graph_embeddings"
            ],
            (
                f"seed-{seed} TEST "
                "graph embeddings"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Script-72 embedding-cache "
            "verification failed"
        )

    return {
        "protocol":
            protocol,

        "test_summary":
            test_summary,

        "test_lock":
            test_lock,
    }


# =============================================================================
# 7. LOCATE EXACT FROZEN TARGET FILE BY HASH
# =============================================================================

def resolve_targets_path() -> Path:

    banner(
        "3. RESOLVE FROZEN TARGET ARTIFACT "
        "WITHOUT OPENING TEST LABELS"
    )

    required_columns = {
        "track_id",
        "split",
        *Y_COLUMNS,
    }

    schema_matches = []

    for path in (
        ROOT
        / "data"
    ).rglob(
        "*.parquet"
    ):

        try:

            schema = set(
                pq.ParquetFile(
                    path
                ).schema.names
            )

        except Exception:

            continue

        if required_columns.issubset(
            schema
        ):

            schema_matches.append(
                path
            )

    print(
        f"Schema-compatible target "
        f"parquets found: "
        f"{len(schema_matches)}"
    )

    hash_matches = []

    for path in schema_matches:

        observed = sha256_file(
            path
        )

        if (
            observed
            ==
            EXPECTED_TARGETS_SHA256
        ):

            hash_matches.append(
                path
            )

    check(
        len(
            hash_matches
        )
        == 1,
        (
            "exactly one target parquet "
            "matches frozen target SHA256"
        ),
    )

    if len(
        hash_matches
    ) != 1:

        raise RuntimeError(
            "Unable to uniquely resolve "
            "frozen targets"
        )

    path = hash_matches[
        0
    ]

    print(
        f"Frozen targets:"
    )

    print(
        f"  {relative(path)}"
    )

    print(
        f"  SHA256 = "
        f"{EXPECTED_TARGETS_SHA256}"
    )

    print()
    print(
        "TEST target rows opened so far: 0"
    )

    return path


# =============================================================================
# 8. EXACT A4 MODEL
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
            N_LABELS,
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
# 9. T4-B MODEL
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

        z_text = F.normalize(
            self.text_projection(
                text_cls
            ),
            p=2,
            dim=-1,
        )

        z_graph = F.normalize(
            self.graph_projection(
                graph_embedding
            ),
            p=2,
            dim=-1,
        )

        return (
            z_text,
            z_graph,
        )


# =============================================================================
# 10. INFERENCE SETTINGS
# =============================================================================

def configure_inference() -> None:

    torch.use_deterministic_algorithms(
        False
    )

    if torch.cuda.is_available():

        torch.backends.cudnn.deterministic = (
            True
        )

        torch.backends.cudnn.benchmark = (
            False
        )

        if hasattr(
            torch.backends.cuda.matmul,
            "allow_tf32",
        ):

            torch.backends.cuda.matmul.allow_tf32 = (
                False
            )

        if hasattr(
            torch.backends.cudnn,
            "allow_tf32",
        ):

            torch.backends.cudnn.allow_tf32 = (
                False
            )

    try:

        torch.set_float32_matmul_precision(
            "highest"
        )

    except Exception:

        pass


# =============================================================================
# 11. GRAPH LOADING
# =============================================================================

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
            "Unable to uniquely resolve "
            f"graph {raw_value}; "
            f"matches={len(found)}"
        )

    return next(
        iter(
            found.values()
        )
    )


def load_g2_graph(
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
                f"Unexpected graph schema: "
                f"{path.name}"
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

    if (
        x.shape
        !=
        (
            9,
            140,
        )
        or
        x.dtype
        !=
        np.float32
        or
        not np.isfinite(
            x
        ).all()
    ):

        raise RuntimeError(
            f"Invalid graph x: "
            f"{path.name}"
        )

    if (
        edge_index.ndim
        != 2
        or
        edge_index.shape[0]
        != 2
        or
        edge_index.dtype
        != np.int64
    ):

        raise RuntimeError(
            f"Invalid G2 edge_index: "
            f"{path.name}"
        )

    if (
        not np.isclose(
            threshold,
            0.85,
            rtol=0.0,
            atol=1e-6,
        )
        or
        nonlocal_only
        is not True
        or
        cap
        != 2
        or
        version
        != 2
    ):

        raise RuntimeError(
            f"G2 rule changed: "
            f"{path.name}"
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
# 12. LOAD VALIDATION INPUTS ONLY
# =============================================================================

def load_val_inputs() -> dict[str, Any]:

    banner(
        "4. LOAD VALIDATION INPUTS ONLY"
    )

    token_manifest = (
        pd.read_parquet(
            TOKEN_MANIFEST
        )
        .copy()
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

    val_manifest = (
        token_manifest.loc[
            token_manifest[
                "split"
            ]
            ==
            "val"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    check(
        len(
            val_manifest
        )
        ==
        N_VAL,
        (
            "VAL token rows = 514"
        ),
    )

    check(
        val_manifest[
            "track_id"
        ].nunique()
        ==
        N_VAL,
        (
            "VAL track IDs unique"
        ),
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
            cache[
                "input_ids"
            ][
                cache_indices
            ],
            dtype=np.int32,
        ).copy()

        attention_mask = np.asarray(
            cache[
                "attention_mask"
            ][
                cache_indices
            ],
            dtype=np.uint8,
        ).copy()

        token_type_ids = np.asarray(
            cache[
                "token_type_ids"
            ][
                cache_indices
            ],
            dtype=np.uint8,
        ).copy()

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
        graph_manifest
        .set_index(
            "track_id",
            drop=False,
        )
    )

    graphs = []

    print()
    print(
        "Loading 514 VAL G2 graphs..."
    )

    for index, row in enumerate(
        val_manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        track_id = str(
            row.track_id
        )

        graph_row = (
            graph_lookup.loc[
                track_id
            ]
        )

        if (
            str(
                graph_row[
                    "split"
                ]
            ).lower()
            !=
            "val"
        ):

            raise RuntimeError(
                f"VAL graph split mismatch: "
                f"{track_id}"
            )

        graphs.append(
            load_g2_graph(
                resolve_graph_path(
                    str(
                        graph_row[
                            "graph_path"
                        ]
                    )
                )
            )
        )

        if (
            index == 1
            or
            index % 100 == 0
            or
            index == N_VAL
        ):

            print(
                f"  loaded "
                f"{index}/{N_VAL}"
            )

    check(
        len(
            graphs
        )
        ==
        N_VAL,
        (
            "loaded exactly 514 "
            "VAL graphs"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "VAL input loading failed"
        )

    return {
        "track_ids":
            val_manifest[
                "track_id"
            ].tolist(),

        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "token_type_ids":
            token_type_ids,

        "graphs":
            graphs,
    }


# =============================================================================
# 13. LOAD VALIDATION TARGETS ONLY
# =============================================================================

def load_val_targets(
    targets_path: Path,
    val_track_ids: list[str],
) -> np.ndarray:

    banner(
        "5. OPEN VALIDATION TARGETS ONLY"
    )

    columns = [
        "track_id",
        "split",
        *Y_COLUMNS,
    ]

    table = pq.read_table(
        targets_path,
        columns=columns,
        filters=[
            (
                "split",
                "==",
                "val",
            )
        ],
    )

    frame = (
        table
        .to_pandas()
        .copy()
    )

    frame[
        "track_id"
    ] = (
        frame[
            "track_id"
        ].astype(str)
    )

    check(
        len(
            frame
        )
        ==
        N_VAL,
        (
            "VAL target rows = 514"
        ),
    )

    check(
        frame[
            "track_id"
        ].nunique()
        ==
        N_VAL,
        (
            "VAL target IDs unique"
        ),
    )

    target_ids = set(
        frame[
            "track_id"
        ]
    )

    check(
        target_ids
        ==
        set(
            val_track_ids
        ),
        (
            "VAL target IDs exactly match "
            "VAL model-input IDs"
        ),
    )

    aligned = (
        frame
        .set_index(
            "track_id"
        )
        .loc[
            val_track_ids
        ]
    )

    y = (
        aligned[
            list(
                Y_COLUMNS
            )
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    check(
        y.shape
        ==
        (
            N_VAL,
            N_LABELS,
        ),
        (
            "VAL targets = (514,20)"
        ),
    )

    check(
        np.isin(
            y,
            [
                0,
                1,
            ],
        ).all(),
        (
            "VAL targets are binary"
        ),
    )

    val_support = y.sum(
        axis=0
    )

    print()
    print(
        "VAL positive support:"
    )

    for index, label in enumerate(
        CANONICAL_LABELS
    ):

        print(
            f"  {index:02d} "
            f"{label:<20s} "
            f"{int(val_support[index])}"
        )

    check(
        np.all(
            val_support
            > 0
        ),
        (
            "every canonical label has "
            "at least one VAL positive"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "VAL target validation failed"
        )

    print()
    print(
        "TEST target rows opened so far: 0"
    )

    return y


# =============================================================================
# 14. DATASET / LOADER
# =============================================================================

class PairDataset(
    Dataset,
):

    def __init__(
        self,
        data: dict[str, Any],
    ) -> None:

        self.data = data

    def __len__(
        self,
    ) -> int:

        return N_VAL

    def __getitem__(
        self,
        index: int,
    ):

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
                item[0]
                for item
                in samples
            ]
        ),

        torch.stack(
            [
                item[1]
                for item
                in samples
            ]
        ),

        torch.stack(
            [
                item[2]
                for item
                in samples
            ]
        ),

        Batch.from_data_list(
            [
                item[3]
                for item
                in samples
            ]
        ),
    )


def build_val_loader(
    data: dict[str, Any],
) -> DataLoader:

    return DataLoader(
        PairDataset(
            data
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_pairs,
    )


# =============================================================================
# 15. RECONSTRUCT SELECTED T4-B
# =============================================================================

def build_model(
    seed: int,
    device: torch.device,
) -> T4BModel:

    config = (
        BertConfig.from_pretrained(
            "bert-base-uncased",
            local_files_only=True,
        )
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    t1 = load_checkpoint(
        T1_CHECKPOINT
    )

    full_state = (
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
        in full_state.items()

        if key.startswith(
            "bert."
        )
    }

    bert.load_state_dict(
        bert_state,
        strict=True,
    )

    gnn = A4GAT()

    gnn.load_state_dict(
        load_checkpoint(
            A4_CHECKPOINT
        )[
            "model_state_dict"
        ],
        strict=True,
    )

    model = T4BModel(
        bert,
        gnn,
    )

    checkpoint = load_checkpoint(
        CHECKPOINTS[
            seed
        ]
    )

    check(
        checkpoint[
            "variant"
        ]
        ==
        "T4-B",
        (
            f"seed {seed} checkpoint "
            "variant = T4-B"
        ),
    )

    check(
        int(
            checkpoint[
                "seed"
            ]
        )
        ==
        seed,
        (
            f"seed {seed} checkpoint "
            "seed exact"
        ),
    )

    model.bert.encoder.layer[
        10
    ].load_state_dict(
        checkpoint[
            "bert_layer_10_state_dict"
        ],
        strict=True,
    )

    model.bert.encoder.layer[
        11
    ].load_state_dict(
        checkpoint[
            "bert_layer_11_state_dict"
        ],
        strict=True,
    )

    model.gnn.gat3.load_state_dict(
        checkpoint[
            "gat3_state_dict"
        ],
        strict=True,
    )

    model.text_projection.load_state_dict(
        checkpoint[
            "text_projection_state_dict"
        ],
        strict=True,
    )

    model.graph_projection.load_state_dict(
        checkpoint[
            "graph_projection_state_dict"
        ],
        strict=True,
    )

    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )

    model.to(
        device
    )

    model.eval()

    return model


# =============================================================================
# 16. EMBED VALIDATION PAIRS
# =============================================================================

@torch.inference_mode()
def embed_val(
    model: T4BModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    text_outputs = []
    graph_outputs = []

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
            input_ids,
            attention_mask,
            token_type_ids,
            graph_batch,
        )

        text_outputs.append(
            z_text.cpu()
        )

        graph_outputs.append(
            z_graph.cpu()
        )

        rows += int(
            z_text.shape[0]
        )

    if rows != N_VAL:

        raise RuntimeError(
            f"VAL embedded rows="
            f"{rows}; expected 514"
        )

    return (
        torch.cat(
            text_outputs,
            dim=0,
        ),

        torch.cat(
            graph_outputs,
            dim=0,
        ),
    )


# =============================================================================
# 17. EMBED EXACT CANONICAL LABEL PHRASES
# =============================================================================

@torch.inference_mode()
def embed_labels(
    model: T4BModel,
    tokenizer: BertTokenizerFast,
    device: torch.device,
) -> torch.Tensor:

    encoded = tokenizer(
        list(
            CANONICAL_LABELS
        ),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_token_type_ids=True,
    )

    input_ids = (
        encoded[
            "input_ids"
        ].to(
            device
        )
    )

    attention_mask = (
        encoded[
            "attention_mask"
        ].to(
            device
        )
    )

    token_type_ids = (
        encoded[
            "token_type_ids"
        ].to(
            device
        )
    )

    output = model.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )

    cls = (
        output
        .last_hidden_state[
            :,
            0,
            :
        ]
    )

    z_label = F.normalize(
        model.text_projection(
            cls
        ),
        p=2,
        dim=-1,
    )

    z_label = (
        z_label
        .detach()
        .cpu()
    )

    check(
        z_label.shape
        ==
        (
            N_LABELS,
            SHARED_DIM,
        ),
        (
            "canonical label embeddings "
            "= (20,256)"
        ),
    )

    check(
        bool(
            torch.isfinite(
                z_label
            ).all()
        ),
        (
            "canonical label embeddings finite"
        ),
    )

    return z_label


# =============================================================================
# 18. ZERO-SHOT METRICS
# =============================================================================

def calculate_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    return {
        "macro_ap":
            float(
                average_precision_score(
                    y_true,
                    scores,
                    average="macro",
                )
            ),

        "micro_ap":
            float(
                average_precision_score(
                    y_true,
                    scores,
                    average="micro",
                )
            ),
    }


def calculate_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    prediction = (
        scores
        >= threshold
    ).astype(
        np.int64
    )

    return {
        "macro_f1":
            float(
                f1_score(
                    y_true,
                    prediction,
                    average="macro",
                    zero_division=0,
                )
            ),

        "micro_f1":
            float(
                f1_score(
                    y_true,
                    prediction,
                    average="micro",
                    zero_division=0,
                )
            ),
    }


def select_global_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:

    best_threshold = None
    best_macro_f1 = -float(
        "inf"
    )

    best_micro_f1 = None

    for threshold in THRESHOLD_GRID:

        values = calculate_f1(
            y_true,
            scores,
            float(
                threshold
            ),
        )

        macro_f1 = (
            values[
                "macro_f1"
            ]
        )

        # Grid is ascending.
        # Strict > therefore freezes the
        # LOWEST threshold on exact tie.
        if (
            macro_f1
            >
            best_macro_f1
        ):

            best_threshold = float(
                threshold
            )

            best_macro_f1 = float(
                macro_f1
            )

            best_micro_f1 = float(
                values[
                    "micro_f1"
                ]
            )

    if best_threshold is None:

        raise RuntimeError(
            "Global threshold selection failed"
        )

    return {
        "threshold":
            best_threshold,

        "macro_f1":
            best_macro_f1,

        "micro_f1":
            best_micro_f1,
    }


def per_label_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:

    values = []

    for index in range(
        N_LABELS
    ):

        values.append(
            float(
                average_precision_score(
                    y_true[
                        :,
                        index
                    ],
                    scores[
                        :,
                        index
                    ],
                )
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


# =============================================================================
# 19. VALIDATION ZERO-SHOT STAGE
# =============================================================================

def run_validation_stage(
    val_loader: DataLoader,
    y_val: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[int, dict[str, Any]],
    str,
]:

    banner(
        "6. ZERO-SHOT VALIDATION / "
        "THRESHOLD SELECTION"
    )

    tokenizer = (
        BertTokenizerFast.from_pretrained(
            "bert-base-uncased",
            local_files_only=True,
        )
    )

    seed_results = {}

    for seed in FINAL_SEEDS:

        print()
        print(
            f"Seed {seed}"
        )

        model = build_model(
            seed,
            device,
        )

        (
            val_text,
            val_graph,
        ) = embed_val(
            model,
            val_loader,
            device,
        )

        label_embedding = embed_labels(
            model,
            tokenizer,
            device,
        )

        caption_scores = (
            val_text
            @ label_embedding.T
        ).numpy()

        audio_scores = (
            val_graph
            @ label_embedding.T
        ).numpy()

        result = {}

        for (
            mode,
            scores,
        ) in (
            (
                "caption",
                caption_scores,
            ),
            (
                "audio",
                audio_scores,
            ),
        ):

            ap = calculate_ap(
                y_val,
                scores,
            )

            threshold = (
                select_global_threshold(
                    y_val,
                    scores,
                )
            )

            label_ap = per_label_ap(
                y_val,
                scores,
            )

            result[
                mode
            ] = {
                "macro_ap":
                    ap[
                        "macro_ap"
                    ],

                "micro_ap":
                    ap[
                        "micro_ap"
                    ],

                "global_threshold":
                    threshold[
                        "threshold"
                    ],

                "macro_f1":
                    threshold[
                        "macro_f1"
                    ],

                "micro_f1":
                    threshold[
                        "micro_f1"
                    ],

                "per_label_ap":
                    label_ap.tolist(),
            }

            print(
                f"  {mode:<7s} "
                f"VAL MacroAP="
                f"{ap['macro_ap']:.8f}  "
                f"MicroAP="
                f"{ap['micro_ap']:.8f}  "
                f"threshold="
                f"{threshold['threshold']:+.3f}  "
                f"MacroF1="
                f"{threshold['macro_f1']:.8f}"
            )

        seed_results[
            seed
        ] = result

        del model

        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Freeze all decisions BEFORE TEST tag labels are opened.
    # -------------------------------------------------------------------------

    val_freeze = {
        "artifact_type":
            "task4_zero_shot_validation_freeze",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "frozen_before_zero_shot_test_label_access",

        "selected_variant":
            "T4-B",

        "seeds":
            [
                42,
                1337,
                2026,
            ],

        "canonical_labels":
            list(
                CANONICAL_LABELS
            ),

        "label_text_policy":
            (
                "exact canonical phrase only; "
                "no prompt template; "
                "no prompt engineering"
            ),

        "primary_mode":
            "caption",

        "secondary_mode":
            "audio",

        "primary_metric":
            "Macro Average Precision",

        "score":
            (
                "cosine similarity via "
                "L2-normalized Task-4 embeddings"
            ),

        "threshold_protocol": {
            "purpose":
                "secondary F1 reporting only",

            "selection_split":
                "validation",

            "one_global_threshold":
                True,

            "per_label_thresholds":
                False,

            "grid_start":
                -1.0,

            "grid_end":
                1.0,

            "grid_points":
                401,

            "grid_step":
                0.005,

            "objective":
                "maximize validation Macro F1",

            "exact_tie_rule":
                "lowest threshold",
        },

        "per_seed_validation":
            {
                str(
                    seed
                ):
                    seed_results[
                        seed
                    ]

                for seed
                in FINAL_SEEDS
            },

        "test_tag_labels_opened":
            False,

        "test_zero_shot_metrics_observed":
            False,

        "test_threshold_tuning":
            False,

        "pretest_protocol_sha256":
            EXPECTED_PRETEST_PROTOCOL_SHA256,

        "test_retrieval_summary_sha256":
            EXPECTED_TEST_RETRIEVAL_SUMMARY_SHA256,

        "script73_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        VAL_FREEZE_PATH,
        val_freeze,
    )

    val_freeze_sha = sha256_file(
        VAL_FREEZE_PATH
    )

    print()
    print(
        "ZERO-SHOT VAL DECISIONS FROZEN "
        "BEFORE TEST LABEL ACCESS."
    )

    print(
        f"VAL freeze:"
    )

    print(
        f"  {relative(VAL_FREEZE_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{val_freeze_sha}"
    )

    print()
    print(
        "TEST tag labels opened so far: 0"
    )

    return (
        seed_results,
        val_freeze_sha,
    )


# =============================================================================
# 20. LOAD SCRIPT-72 TEST MANIFEST / EMBEDDINGS
# =============================================================================

def load_test_embedding_inputs(
    lineage: dict[str, Any],
) -> tuple[
    list[str],
    dict[int, dict[str, np.ndarray]],
]:

    banner(
        "7. LOAD FROZEN SCRIPT-72 "
        "TEST EMBEDDINGS"
    )

    manifest = pd.read_csv(
        TEST_RETRIEVAL_MANIFEST
    )

    check(
        list(
            manifest.columns
        )
        ==
        [
            "row_index",
            "track_id",
        ],
        (
            "Script-72 TEST manifest "
            "schema exact"
        ),
    )

    check(
        len(
            manifest
        )
        ==
        N_TEST,
        (
            "Script-72 TEST manifest "
            "= 514 rows"
        ),
    )

    check(
        manifest[
            "row_index"
        ].tolist()
        ==
        list(
            range(
                N_TEST
            )
        ),
        (
            "Script-72 TEST row_index "
            "= 0..513"
        ),
    )

    test_track_ids = (
        manifest[
            "track_id"
        ]
        .astype(str)
        .tolist()
    )

    check(
        len(
            set(
                test_track_ids
            )
        )
        ==
        N_TEST,
        (
            "Script-72 TEST track IDs unique"
        ),
    )

    embeddings = {}

    for seed in FINAL_SEEDS:

        text = np.load(
            test_text_embedding_path(
                seed
            ),
            allow_pickle=False,
        )

        graph = np.load(
            test_graph_embedding_path(
                seed
            ),
            allow_pickle=False,
        )

        check(
            text.shape
            ==
            (
                N_TEST,
                SHARED_DIM,
            ),
            (
                f"seed {seed} TEST text "
                "= (514,256)"
            ),
        )

        check(
            graph.shape
            ==
            (
                N_TEST,
                SHARED_DIM,
            ),
            (
                f"seed {seed} TEST graph "
                "= (514,256)"
            ),
        )

        check(
            np.isfinite(
                text
            ).all(),
            (
                f"seed {seed} TEST text finite"
            ),
        )

        check(
            np.isfinite(
                graph
            ).all(),
            (
                f"seed {seed} TEST graph finite"
            ),
        )

        embeddings[
            seed
        ] = {
            "text":
                np.asarray(
                    text,
                    dtype=np.float32,
                ),

            "graph":
                np.asarray(
                    graph,
                    dtype=np.float32,
                ),
        }

    if FAILURES:

        raise RuntimeError(
            "Frozen Script-72 TEST "
            "embedding loading failed"
        )

    return (
        test_track_ids,
        embeddings,
    )


# =============================================================================
# 21. FIRST ZERO-SHOT TEST LABEL ACCESS
# =============================================================================

def load_test_targets(
    targets_path: Path,
    test_track_ids: list[str],
) -> np.ndarray:

    banner(
        "8. OPEN ZERO-SHOT TEST TAG TARGETS "
        "— FIRST ACCESS AFTER VAL FREEZE"
    )

    check(
        VAL_FREEZE_PATH.is_file(),
        (
            "VAL threshold freeze exists "
            "before TEST target access"
        ),
    )

    columns = [
        "track_id",
        "split",
        *Y_COLUMNS,
    ]

    table = pq.read_table(
        targets_path,
        columns=columns,
        filters=[
            (
                "split",
                "==",
                "test",
            )
        ],
    )

    frame = (
        table
        .to_pandas()
        .copy()
    )

    frame[
        "track_id"
    ] = (
        frame[
            "track_id"
        ].astype(str)
    )

    check(
        len(
            frame
        )
        ==
        N_TEST,
        (
            "TEST target rows = 514"
        ),
    )

    check(
        frame[
            "track_id"
        ].nunique()
        ==
        N_TEST,
        (
            "TEST target IDs unique"
        ),
    )

    check(
        set(
            frame[
                "track_id"
            ]
        )
        ==
        set(
            test_track_ids
        ),
        (
            "TEST target IDs exactly match "
            "Script-72 retrieval IDs"
        ),
    )

    aligned = (
        frame
        .set_index(
            "track_id"
        )
        .loc[
            test_track_ids
        ]
    )

    y = (
        aligned[
            list(
                Y_COLUMNS
            )
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    check(
        y.shape
        ==
        (
            N_TEST,
            N_LABELS,
        ),
        (
            "TEST targets = (514,20)"
        ),
    )

    check(
        np.isin(
            y,
            [
                0,
                1,
            ],
        ).all(),
        (
            "TEST targets binary"
        ),
    )

    support = y.sum(
        axis=0
    )

    print()
    print(
        "TEST positive support:"
    )

    for index, label in enumerate(
        CANONICAL_LABELS
    ):

        print(
            f"  {index:02d} "
            f"{label:<20s} "
            f"{int(support[index])}"
        )

    check(
        np.all(
            support
            > 0
        ),
        (
            "every canonical label has "
            "at least one TEST positive"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "TEST target validation failed"
        )

    passed(
        (
            "TEST targets opened only after "
            "VAL zero-shot decisions were frozen"
        )
    )

    return y


# =============================================================================
# 22. TEST ZERO-SHOT EVALUATION
# =============================================================================

def run_test_stage(
    val_results: dict[int, dict[str, Any]],
    test_embeddings: dict[int, dict[str, np.ndarray]],
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:

    banner(
        "9. FROZEN ZERO-SHOT TEST EVALUATION"
    )

    tokenizer = (
        BertTokenizerFast.from_pretrained(
            "bert-base-uncased",
            local_files_only=True,
        )
    )

    mode_rows = []
    label_rows = []

    per_seed_scores = {}

    for seed in FINAL_SEEDS:

        print()
        print(
            f"Seed {seed}"
        )

        # Only label phrases need live text encoding.
        # TEST music embeddings are reused from Script 72.
        model = build_model(
            seed,
            device,
        )

        label_embedding = (
            embed_labels(
                model,
                tokenizer,
                device,
            )
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        caption_scores = (
            test_embeddings[
                seed
            ][
                "text"
            ]
            @ label_embedding.T
        )

        audio_scores = (
            test_embeddings[
                seed
            ][
                "graph"
            ]
            @ label_embedding.T
        )

        per_seed_scores[
            seed
        ] = {
            "caption":
                caption_scores,

            "audio":
                audio_scores,
        }

        for (
            mode,
            scores,
        ) in (
            (
                "caption",
                caption_scores,
            ),
            (
                "audio",
                audio_scores,
            ),
        ):

            frozen_threshold = float(
                val_results[
                    seed
                ][
                    mode
                ][
                    "global_threshold"
                ]
            )

            ap = calculate_ap(
                y_test,
                scores,
            )

            f1 = calculate_f1(
                y_test,
                scores,
                frozen_threshold,
            )

            label_ap = per_label_ap(
                y_test,
                scores,
            )

            mode_rows.append(
                {
                    "seed":
                        seed,

                    "mode":
                        mode,

                    "val_selected_global_threshold":
                        frozen_threshold,

                    "test_macro_ap":
                        ap[
                            "macro_ap"
                        ],

                    "test_micro_ap":
                        ap[
                            "micro_ap"
                        ],

                    "test_macro_f1":
                        f1[
                            "macro_f1"
                        ],

                    "test_micro_f1":
                        f1[
                            "micro_f1"
                        ],
                }
            )

            for label_index, label in enumerate(
                CANONICAL_LABELS
            ):

                label_rows.append(
                    {
                        "seed":
                            seed,

                        "mode":
                            mode,

                        "label_index":
                            label_index,

                        "label":
                            label,

                        "test_support":
                            int(
                                y_test[
                                    :,
                                    label_index
                                ].sum()
                            ),

                        "val_ap":
                            float(
                                val_results[
                                    seed
                                ][
                                    mode
                                ][
                                    "per_label_ap"
                                ][
                                    label_index
                                ]
                            ),

                        "test_ap":
                            float(
                                label_ap[
                                    label_index
                                ]
                            ),
                    }
                )

            print(
                f"  {mode:<7s} "
                f"TEST MacroAP="
                f"{ap['macro_ap']:.8f}  "
                f"MicroAP="
                f"{ap['micro_ap']:.8f}  "
                f"MacroF1="
                f"{f1['macro_f1']:.8f}  "
                f"threshold="
                f"{frozen_threshold:+.3f}"
            )

        del model

        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate predetermined seeds.
    # -------------------------------------------------------------------------

    aggregate = {}

    for mode in ZERO_SHOT_MODES:

        aggregate[
            mode
        ] = {}

        for metric in (
            "test_macro_ap",
            "test_micro_ap",
            "test_macro_f1",
            "test_micro_f1",
        ):

            values = np.asarray(
                [
                    float(
                        row[
                            metric
                        ]
                    )

                    for row
                    in mode_rows

                    if row[
                        "mode"
                    ]
                    ==
                    mode
                ],
                dtype=np.float64,
            )

            aggregate[
                mode
            ][
                metric
            ] = {
                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "std_population":
                    float(
                        np.std(
                            values,
                            ddof=0,
                        )
                    ),

                "values_by_seed": {
                    str(
                        row[
                            "seed"
                        ]
                    ):
                        float(
                            row[
                                metric
                            ]
                        )

                    for row
                    in mode_rows

                    if row[
                        "mode"
                    ]
                    ==
                    mode
                },
            }

    # Save score caches after all predetermined seeds complete.
    OUTPUT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in FINAL_SEEDS:

        for mode in ZERO_SHOT_MODES:

            save_npy_atomic(
                score_path(
                    seed,
                    mode,
                ),
                np.asarray(
                    per_seed_scores[
                        seed
                    ][
                        mode
                    ],
                    dtype=np.float32,
                ),
            )

    return (
        mode_rows,
        label_rows,
        aggregate,
    )


# =============================================================================
# 23. WRITE FINAL ZERO-SHOT ARTIFACTS
# =============================================================================

def write_outputs(
    targets_path: Path,
    val_freeze_sha: str,
    mode_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> tuple[
    str,
    str,
    str,
    str,
]:

    banner(
        "10. WRITE ZERO-SHOT RESULT / LOCK"
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        PER_SEED_MODE_CSV,
        mode_rows,
    )

    write_csv_atomic(
        PER_LABEL_CSV,
        label_rows,
    )

    per_seed_sha = sha256_file(
        PER_SEED_MODE_CSV
    )

    per_label_sha = sha256_file(
        PER_LABEL_CSV
    )

    score_hashes = {}

    for seed in FINAL_SEEDS:

        score_hashes[
            str(
                seed
            )
        ] = {}

        for mode in ZERO_SHOT_MODES:

            path = score_path(
                seed,
                mode,
            )

            score_hashes[
                str(
                    seed
                )
            ][
                mode
            ] = {
                "path":
                    relative(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }

    summary = {
        "artifact_type":
            "task4_zero_shot_tag_test_summary",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "final_zero_shot_test_complete",

        "selected_variant":
            "T4-B",

        "seeds":
            [
                42,
                1337,
                2026,
            ],

        "canonical_labels":
            list(
                CANONICAL_LABELS
            ),

        "number_of_labels":
            N_LABELS,

        "primary_zero_shot_mode":
            "caption",

        "secondary_zero_shot_mode":
            "audio",

        "primary_metric":
            "Macro Average Precision",

        "label_text":
            (
                "exact canonical phrase only; "
                "no prompt engineering"
            ),

        "score":
            (
                "cosine similarity via "
                "L2-normalized 256-D Task-4 space"
            ),

        "per_seed_mode":
            mode_rows,

        "aggregate_three_seed":
            aggregate,

        "f1_threshold_policy": {
            "one_global_threshold_per_seed_and_mode":
                True,

            "selection_split":
                "validation",

            "test_threshold_tuning":
                False,

            "grid":
                "[-1,1] inclusive, step 0.005",

            "objective":
                "VAL Macro F1",

            "tie_rule":
                "lowest threshold",
        },

        "validation_freeze": {
            "path":
                relative(
                    VAL_FREEZE_PATH
                ),

            "sha256":
                val_freeze_sha,
        },

        "targets": {
            "path":
                relative(
                    targets_path
                ),

            "sha256":
                EXPECTED_TARGETS_SHA256,
        },

        "script72_embedding_reuse":
            True,

        "test_retrieval_rerun":
            False,

        "contrastive_training_used_tag_labels":
            False,

        "best_seed_selection":
            False,

        "prompt_engineering":
            False,

        "score_artifacts":
            score_hashes,

        "per_seed_mode_csv": {
            "path":
                relative(
                    PER_SEED_MODE_CSV
                ),

            "sha256":
                per_seed_sha,
        },

        "per_label_csv": {
            "path":
                relative(
                    PER_LABEL_CSV
                ),

            "sha256":
                per_label_sha,
        },

        "task3_supervised_comparison": {
            "status":
                "deferred_to_final_reporting",

            "reason":
                (
                    "Script 73 does not rerun or "
                    "reselect frozen Task-3 models. "
                    "The already frozen Task-3 "
                    "supervised TEST result will be "
                    "joined in final reporting."
                ),
        },

        "provenance": {
            "script73":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "script73_sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),

            "pretest_protocol_sha256":
                EXPECTED_PRETEST_PROTOCOL_SHA256,

            "test_retrieval_summary_sha256":
                EXPECTED_TEST_RETRIEVAL_SUMMARY_SHA256,

            "test_retrieval_lock_sha256":
                EXPECTED_TEST_RETRIEVAL_LOCK_SHA256,
        },

        "next_script":
            74,
    }

    write_json_atomic(
        SUMMARY_PATH,
        summary,
    )

    summary_sha = sha256_file(
        SUMMARY_PATH
    )

    lock = {
        "lock_type":
            "task4_zero_shot_tag_test_lock",

        "version":
            1,

        "created_utc":
            summary[
                "created_utc"
            ],

        "summary":
            relative(
                SUMMARY_PATH
            ),

        "summary_sha256":
            summary_sha,

        "validation_freeze_sha256":
            val_freeze_sha,

        "per_seed_mode_csv_sha256":
            per_seed_sha,

        "per_label_csv_sha256":
            per_label_sha,

        "score_hashes":
            score_hashes,

        "canonical_labels":
            list(
                CANONICAL_LABELS
            ),

        "test_threshold_tuning":
            False,

        "best_seed_selection":
            False,

        "prompt_engineering":
            False,

        "script73_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        LOCK_PATH,
        lock,
    )

    lock_sha = sha256_file(
        LOCK_PATH
    )

    check(
        load_json(
            LOCK_PATH
        )[
            "summary_sha256"
        ]
        ==
        sha256_file(
            SUMMARY_PATH
        ),
        (
            "zero-shot lock matches "
            "zero-shot summary"
        ),
    )

    return (
        per_seed_sha,
        per_label_sha,
        summary_sha,
        lock_sha,
    )


# =============================================================================
# 24. FINAL DISCIPLINE CHECK
# =============================================================================

def print_boundary() -> None:

    banner(
        "11. ZERO-SHOT DISCIPLINE CONFIRMATION"
    )

    print(
        "Contrastive retraining:               0"
    )

    print(
        "Task-4 retrieval rerun:               0"
    )

    print(
        "Prompt engineering:                   0"
    )

    print(
        "Per-label threshold tuning:           0"
    )

    print(
        "TEST-selected thresholds:             0"
    )

    print(
        "TEST-driven checkpoint selection:     0"
    )

    print(
        "TEST-driven seed selection:           0"
    )

    print(
        "Zero-shot TEST labels opened only "
        "after VAL threshold freeze: YES"
    )

    passed(
        (
            "Script 73 preserved the "
            "frozen zero-shot protocol"
        )
    )


# =============================================================================
# 25. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 73 — FROZEN TASK-4 "
        "ZERO-SHOT TAG EVALUATION"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "Selected model: T4-B"
    )

    print(
        "Seeds:          42 / 1337 / 2026"
    )

    print(
        "Labels:         20 canonical phrases"
    )

    print()

    print(
        "Primary mode:"
    )

    print(
        "  Caption → canonical label text"
    )

    print()

    print(
        "Additional mode:"
    )

    print(
        "  Audio → canonical label text"
    )

    print()

    print(
        "Primary metric:"
    )

    print(
        "  Macro Average Precision"
    )

    print()

    print(
        "Prompt engineering: NO"
    )

    print(
        "TEST threshold tuning: NO"
    )

    try:

        verify_output_protection()

        lineage = verify_lineage()

        targets_path = (
            resolve_targets_path()
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA unavailable"
            )

        configure_inference()

        device = torch.device(
            "cuda"
        )

        banner(
            "3A. DEVICE"
        )

        print(
            "CUDA:",
            torch.cuda.get_device_name(
                device
            ),
        )

        print(
            "PyTorch:",
            torch.__version__,
        )

        # ---------------------------------------------------------------------
        # VAL ONLY.
        # ---------------------------------------------------------------------

        val_inputs = (
            load_val_inputs()
        )

        y_val = load_val_targets(
            targets_path,
            val_inputs[
                "track_ids"
            ],
        )

        val_loader = (
            build_val_loader(
                val_inputs
            )
        )

        (
            val_results,
            val_freeze_sha,
        ) = run_validation_stage(
            val_loader,
            y_val,
            device,
        )

        # ---------------------------------------------------------------------
        # Frozen Script-72 TEST embeddings.
        # Still no TEST labels at this point.
        # ---------------------------------------------------------------------

        (
            test_track_ids,
            test_embeddings,
        ) = load_test_embedding_inputs(
            lineage
        )

        # ---------------------------------------------------------------------
        # FIRST TEST TAG LABEL ACCESS.
        # This occurs only AFTER the threshold freeze exists.
        # ---------------------------------------------------------------------

        y_test = load_test_targets(
            targets_path,
            test_track_ids,
        )

        (
            mode_rows,
            label_rows,
            aggregate,
        ) = run_test_stage(
            val_results,
            test_embeddings,
            y_test,
            device,
        )

        (
            per_seed_sha,
            per_label_sha,
            summary_sha,
            lock_sha,
        ) = write_outputs(
            targets_path,
            val_freeze_sha,
            mode_rows,
            label_rows,
            aggregate,
        )

        print_boundary()

    except Exception as exc:

        banner(
            "SCRIPT 73 — ABORTED"
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
            "Do NOT alter label phrases, "
            "threshold grid, seeds, or metrics."
        )

        print(
            "Preserve any partial attempt "
            "before rerunning."
        )

        print()
        print(
            "Do NOT proceed to Script 74."
        )

        return 1

    # =========================================================================
    # Summary
    # =========================================================================

    banner(
        "SCRIPT 73 SUMMARY"
    )

    print(
        f"Failures: {len(FAILURES)}"
    )

    print(
        f"Warnings: {len(WARNINGS)}"
    )

    print()

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "FINAL TASK-4 ZERO-SHOT "
        "TAG EVALUATION COMPLETE."
    )

    print()
    print(
        "PRIMARY — CAPTION → LABEL"
    )

    for row in mode_rows:

        if (
            row[
                "mode"
            ]
            !=
            "caption"
        ):

            continue

        print(
            f"Seed {row['seed']:4d}: "
            f"MacroAP="
            f"{row['test_macro_ap']:.8f}  "
            f"MicroAP="
            f"{row['test_micro_ap']:.8f}  "
            f"MacroF1="
            f"{row['test_macro_f1']:.8f}"
        )

    print()

    print(
        "Caption zero-shot "
        "three-seed Macro AP:"
    )

    print(
        f"  mean = "
        f"{aggregate['caption']['test_macro_ap']['mean']:.8f}"
    )

    print(
        f"  std  = "
        f"{aggregate['caption']['test_macro_ap']['std_population']:.8f}"
    )

    print()
    print(
        "SECONDARY — AUDIO → LABEL"
    )

    for row in mode_rows:

        if (
            row[
                "mode"
            ]
            !=
            "audio"
        ):

            continue

        print(
            f"Seed {row['seed']:4d}: "
            f"MacroAP="
            f"{row['test_macro_ap']:.8f}  "
            f"MicroAP="
            f"{row['test_micro_ap']:.8f}  "
            f"MacroF1="
            f"{row['test_macro_f1']:.8f}"
        )

    print()

    print(
        "Audio zero-shot "
        "three-seed Macro AP:"
    )

    print(
        f"  mean = "
        f"{aggregate['audio']['test_macro_ap']['mean']:.8f}"
    )

    print(
        f"  std  = "
        f"{aggregate['audio']['test_macro_ap']['std_population']:.8f}"
    )

    print()
    print(
        "VAL threshold freeze:"
    )

    print(
        f"  {relative(VAL_FREEZE_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{val_freeze_sha}"
    )

    print()

    print(
        "Per-seed/mode CSV SHA256:"
    )

    print(
        f"  {per_seed_sha}"
    )

    print(
        "Per-label CSV SHA256:"
    )

    print(
        f"  {per_label_sha}"
    )

    print(
        "Summary SHA256:"
    )

    print(
        f"  {summary_sha}"
    )

    print(
        "Lock SHA256:"
    )

    print(
        f"  {lock_sha}"
    )

    print()
    print(
        "Prompt engineering: NO"
    )

    print(
        "TEST threshold tuning: NO"
    )

    print(
        "Best-seed selection: NO"
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
        "  Script 74 — frozen 10-query "
        "qualitative caption→top-3 retrieval."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )