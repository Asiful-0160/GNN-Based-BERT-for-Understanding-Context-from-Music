#!/usr/bin/env python3
"""
SCRIPT 72 — ONE-TIME FINAL TASK-4 TEST RETRIEVAL
================================================

Selected configuration:
    T4-B

Predetermined seeds:
    42
    1337
    2026

Frozen TEST protocol:
    514 paired TEST items

For EACH seed:
    - reconstruct selected T4-B model
    - perform one complete TEST inference pass
    - generate normalized text/audio embeddings
    - calculate full 514 x 514 similarity matrix
    - calculate:
        Audio -> Caption R@1/R@5/R@10
        Caption -> Audio R@1/R@5/R@10
        mR

Final reporting:
    - per-seed metrics
    - mean across all 3 predetermined seeds
    - population SD across all 3 seeds

STRICTLY FORBIDDEN:
    - TEST labels
    - tag targets
    - best-seed selection
    - checkpoint selection
    - retraining
    - architecture changes
    - temperature tuning
    - candidate-pool filtering
    - seed ensemble

TEST outputs are generated only after ALL three seed inference
passes complete successfully.
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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


CANONICAL_DATASET = (
    ROOT
    / "data/interim/"
      "musiccaps_canonical.parquet"
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


OUT_DATA_DIR = (
    ROOT
    / "data/processed/task4_test_retrieval"
)

RUN_DIR = (
    ROOT
    / "results/runs/task4"
)


TEST_MANIFEST_OUT = (
    RUN_DIR
    / "task4_test_retrieval_manifest.csv"
)

PER_SEED_CSV = (
    RUN_DIR
    / "task4_T4-B_test_retrieval_per_seed.csv"
)

SUMMARY_OUT = (
    RUN_DIR
    / "task4_T4-B_test_retrieval_summary.json"
)

LOCK_OUT = (
    RUN_DIR
    / "task4_T4-B_test_retrieval_lock.json"
)


def text_out(
    seed: int,
) -> Path:

    return (
        OUT_DATA_DIR
        / f"seed{seed}_text_embeddings.npy"
    )


def graph_out(
    seed: int,
) -> Path:

    return (
        OUT_DATA_DIR
        / f"seed{seed}_graph_embeddings.npy"
    )


def sim_out(
    seed: int,
) -> Path:

    return (
        OUT_DATA_DIR
        / f"seed{seed}_audio_by_caption_similarity.npy"
    )


def ranks_out(
    seed: int,
) -> Path:

    return (
        OUT_DATA_DIR
        / f"seed{seed}_retrieval_ranks.npz"
    )


# =============================================================================
# 1. FROZEN HASHES
# =============================================================================

EXPECTED_PROTOCOL_SHA256 = (
    "f1f4651c9bffab2295ec8ef44d17f208"
    "4430386585fb2e9fd489d0bc2af93f7c"
)

EXPECTED_PROTOCOL_LOCK_SHA256 = (
    "dddbd32013dd7e9912e0df0fa72d0352"
    "49abae8f4c83f6313b57a99931f00836"
)


EXPECTED_CANONICAL_SHA256 = (
    "704e057ba3807886b35000ff23d533d4"
    "759c6a65fdf5d7a8912e618ba523c4d3"
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
# 2. FINAL FROZEN CONTRACT
# =============================================================================

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

N_TOTAL = 5140
N_TEST = 514

MAX_LENGTH = 112

BERT_DIM = 768
GAT_DIM = 128
SHARED_DIM = 256

BATCH_SIZE = 32


METRIC_KEYS = (
    "audio_to_caption_R@1",
    "audio_to_caption_R@5",
    "audio_to_caption_R@10",
    "caption_to_audio_R@1",
    "caption_to_audio_R@5",
    "caption_to_audio_R@10",
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

    print(
        "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


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
# 4. HELPERS
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
            "a JSON object"
        )

    return value


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
                rows[
                    0
                ].keys()
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


def save_npz_atomic(
    path: Path,
    **arrays,
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

        np.savez(
            f,
            **arrays,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. FINAL TEST OUTPUT PROTECTION"
    )

    paths = [
        TEST_MANIFEST_OUT,
        PER_SEED_CSV,
        SUMMARY_OUT,
        LOCK_OUT,
    ]

    for seed in FINAL_SEEDS:

        paths.extend(
            [
                text_out(
                    seed
                ),

                graph_out(
                    seed
                ),

                sim_out(
                    seed
                ),

                ranks_out(
                    seed
                ),
            ]
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
            "Refusing to overwrite final "
            "Task-4 TEST retrieval outputs"
        )


# =============================================================================
# 6. VERIFY PRE-TEST LOCK
# =============================================================================

def verify_lineage_and_protocol() -> dict[str, Any]:

    banner(
        "2. VERIFY PRE-TEST LOCK / FROZEN INPUTS"
    )

    artifacts = (
        (
            PRETEST_PROTOCOL,
            EXPECTED_PROTOCOL_SHA256,
            "pre-TEST protocol",
        ),
        (
            PRETEST_LOCK,
            EXPECTED_PROTOCOL_LOCK_SHA256,
            "pre-TEST lock",
        ),
        (
            CANONICAL_DATASET,
            EXPECTED_CANONICAL_SHA256,
            "canonical dataset (hash only)",
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

    for (
        path,
        digest,
        label,
    ) in artifacts:

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
            "Frozen input identity "
            "verification failed"
        )

    protocol = load_json(
        PRETEST_PROTOCOL
    )

    lock = load_json(
        PRETEST_LOCK
    )

    check(
        lock[
            "protocol_sha256"
        ]
        ==
        EXPECTED_PROTOCOL_SHA256,
        (
            "pre-TEST lock points to "
            "exact protocol"
        ),
    )

    check(
        protocol[
            "status"
        ]
        ==
        (
            "frozen_before_first_"
            "task4_test_retrieval"
        ),
        (
            "protocol status is "
            "pre-TEST frozen"
        ),
    )

    selected = (
        protocol[
            "selected_configuration"
        ]
    )

    check(
        selected[
            "variant"
        ]
        == "T4-B",
        (
            "selected variant = T4-B"
        ),
    )

    check(
        selected[
            "final_seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "final seeds = "
            "42/1337/2026"
        ),
    )

    check(
        selected[
            "best_seed_selection"
        ]
        is False,
        (
            "best-seed selection forbidden"
        ),
    )

    check(
        selected[
            "seed_ensemble"
        ]
        is False,
        (
            "seed ensemble forbidden"
        ),
    )

    inference = (
        protocol[
            "test_inference"
        ]
    )

    check(
        inference[
            "batch_size"
        ]
        == BATCH_SIZE,
        (
            "TEST batch size = 32"
        ),
    )

    check(
        inference[
            "dtype"
        ]
        == "float32",
        (
            "TEST inference dtype = float32"
        ),
    )

    check(
        inference[
            "AMP"
        ]
        is False,
        (
            "AMP disabled"
        ),
    )

    check(
        inference[
            "TF32"
        ]
        is False,
        (
            "TF32 disabled"
        ),
    )

    check(
        inference[
            "similarity_matrix_orientation"
        ]
        ==
        "rows=audio, columns=caption",
        (
            "similarity orientation frozen"
        ),
    )

    metric_protocol = (
        protocol[
            "retrieval_metrics"
        ]
    )

    check(
        "strictly greater"
        in metric_protocol[
            "rank_definition"
        ],
        (
            "strict-greater ranking "
            "rule frozen"
        ),
    )

    forbidden = (
        protocol[
            "script72_forbidden"
        ]
    )

    check(
        forbidden[
            "test_labels"
        ]
        is True,
        (
            "TEST labels forbidden"
        ),
    )

    check(
        forbidden[
            "seed_selection"
        ]
        is True,
        (
            "TEST seed selection forbidden"
        ),
    )

    check(
        protocol[
            "test_population"
        ][
            "expected_pairs"
        ]
        == N_TEST,
        (
            "expected TEST pool = 514"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Pre-TEST protocol "
            "validation failed"
        )

    return protocol


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
# 8. T4-B MODEL
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

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        text_embedding = (
            output
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
                text_embedding
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
# 9. INFERENCE SETTINGS
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
# 10. G2 GRAPH LOADING
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
                "Unexpected graph schema: "
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

    if (
        x.shape != (9, 140)
        or
        x.dtype != np.float32
        or
        not np.isfinite(
            x
        ).all()
    ):

        raise RuntimeError(
            f"Invalid graph nodes: "
            f"{path.name}"
        )

    if (
        edge_index.ndim != 2
        or
        edge_index.shape[0] != 2
        or
        edge_index.dtype != np.int64
        or
        not (
            16
            <= edge_index.shape[1]
            <= 34
        )
    ):

        raise RuntimeError(
            f"Invalid G2 edges: "
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
        nonlocal_only is not True
        or
        degree_cap != 2
        or
        rule_version != 2
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
# 11. FIRST FORMAL TEST INPUT ACCESS
# =============================================================================

def load_test_inputs() -> dict[str, Any]:

    banner(
        "3. OPEN FROZEN TASK-4 TEST INPUTS "
        "— FIRST FORMAL ACCESS"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # No target/label artifact is referenced anywhere in Script 72.
    #
    # TEST captions are represented only through the already frozen
    # BERT token cache.
    #
    # The canonical dataset is NOT deserialized here; its frozen hash
    # was checked only as provenance.
    # -------------------------------------------------------------------------

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
        (
            "token manifest schema unchanged"
        ),
    )

    check(
        len(
            token_manifest
        )
        == N_TOTAL,
        (
            "token manifest rows = 5140"
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

    check(
        len(
            test_manifest
        )
        == N_TEST,
        (
            "TEST token rows = 514"
        ),
    )

    check(
        test_manifest[
            "track_id"
        ].nunique()
        == N_TEST,
        (
            "TEST track IDs unique"
        ),
    )

    source_indices = (
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
        ).copy()

        attention_mask = np.asarray(
            cache[
                "attention_mask"
            ][
                source_indices
            ],
            dtype=np.uint8,
        ).copy()

        token_type_ids = np.asarray(
            cache[
                "token_type_ids"
            ][
                source_indices
            ],
            dtype=np.uint8,
        ).copy()

    check(
        input_ids.shape
        ==
        (
            N_TEST,
            MAX_LENGTH,
        ),
        (
            "TEST input_ids = (514,112)"
        ),
    )

    check(
        attention_mask.shape
        ==
        (
            N_TEST,
            MAX_LENGTH,
        ),
        (
            "TEST attention_mask = (514,112)"
        ),
    )

    check(
        token_type_ids.shape
        ==
        (
            N_TEST,
            MAX_LENGTH,
        ),
        (
            "TEST token_type_ids = (514,112)"
        ),
    )

    # -------------------------------------------------------------------------
    # Graph alignment:
    # graph list is constructed in EXACT token-manifest TEST order.
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
        (
            "graph manifest schema unchanged"
        ),
    )

    check(
        len(
            graph_manifest
        )
        == N_TOTAL,
        (
            "graph manifest rows = 5140"
        ),
    )

    check(
        graph_manifest[
            "track_id"
        ].nunique()
        == N_TOTAL,
        (
            "graph track IDs unique"
        ),
    )

    graph_lookup = (
        graph_manifest
        .set_index(
            "track_id",
            drop=False,
        )
    )

    graphs = []

    for (
        row_number,
        row,
    ) in enumerate(
        test_manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        track_id = str(
            row.track_id
        )

        if (
            track_id
            not in graph_lookup.index
        ):

            raise RuntimeError(
                f"Missing TEST graph for "
                f"{track_id}"
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
            != "test"
        ):

            raise RuntimeError(
                "TEST split mismatch for "
                f"{track_id}"
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
            row_number % 100 == 0
            or
            row_number == N_TEST
        ):

            print(
                f"  loaded TEST graph "
                f"{row_number}/{N_TEST}"
            )

    check(
        len(
            graphs
        )
        == N_TEST,
        (
            "opened exactly 514 TEST graphs"
        ),
    )

    manifest_rows = [
        {
            "row_index":
                index,

            "track_id":
                track_id,
        }

        for (
            index,
            track_id,
        ) in enumerate(
            test_manifest[
                "track_id"
            ].tolist()
        )
    ]

    if FAILURES:

        raise RuntimeError(
            "TEST input alignment failed"
        )

    print()
    print(
        "TEST labels loaded: 0"
    )

    print(
        "Canonical dataset deserialized: NO"
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

        "manifest_rows":
            manifest_rows,
    }


# =============================================================================
# 12. TEST DATASET / LOADER
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

        return N_TEST

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
                sample[0]
                for sample
                in samples
            ]
        ),

        torch.stack(
            [
                sample[1]
                for sample
                in samples
            ]
        ),

        torch.stack(
            [
                sample[2]
                for sample
                in samples
            ]
        ),

        Batch.from_data_list(
            [
                sample[3]
                for sample
                in samples
            ]
        ),
    )


def build_loader(
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
# 13. RECONSTRUCT FROZEN T4-B
# =============================================================================

def build_model(
    seed: int,
    device: torch.device,
) -> tuple[
    T4BModel,
    int,
]:

    config = (
        BertConfig.from_pretrained(
            "bert-base-uncased",
            local_files_only=True,
        )
    )

    check(
        config.hidden_size
        == BERT_DIM,
        (
            f"seed {seed} "
            "BERT hidden size = 768"
        ),
    )

    check(
        config.num_hidden_layers
        == 12,
        (
            f"seed {seed} "
            "BERT layers = 12"
        ),
    )

    bert = BertModel(
        config,
        add_pooling_layer=True,
    )

    task1 = load_checkpoint(
        T1_CHECKPOINT
    )

    full_state = (
        task1[
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

        for (
            key,
            value,
        ) in full_state.items()

        if key.startswith(
            "bert."
        )
    }

    excluded = {
        key
        for key
        in full_state

        if not key.startswith(
            "bert."
        )
    }

    check(
        excluded
        ==
        {
            "classifier.weight",
            "classifier.bias",
        },
        (
            f"seed {seed} only "
            "T1 classifier excluded"
        ),
    )

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
        checkpoint.get(
            "variant"
        )
        == "T4-B",
        (
            f"seed {seed} checkpoint "
            "variant = T4-B"
        ),
    )

    check(
        int(
            checkpoint.get(
                "seed",
                -1,
            )
        )
        == seed,
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

    check(
        sum(
            parameter.numel()

            for parameter
            in model.parameters()

            if parameter.requires_grad
        )
        == 0,
        (
            f"seed {seed} TEST model "
            "has zero trainable parameters"
        ),
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        int(
            checkpoint[
                "best_epoch"
            ]
        ),
    )


# =============================================================================
# 14. RETRIEVAL
# =============================================================================

def ranks(
    similarity: torch.Tensor,
) -> torch.Tensor:

    positive_similarity = (
        torch.diagonal(
            similarity
        )
    )

    return (
        1
        +
        (
            similarity
            >
            positive_similarity.unsqueeze(
                1
            )
        )
        .sum(
            dim=1
        )
    ).to(
        torch.int64
    )


def metrics_from_ranks(
    audio_to_caption_rank: torch.Tensor,
    caption_to_audio_rank: torch.Tensor,
) -> dict[str, float]:

    metrics = {}

    for k in (
        1,
        5,
        10,
    ):

        metrics[
            f"audio_to_caption_R@{k}"
        ] = float(
            (
                audio_to_caption_rank
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
                caption_to_audio_rank
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
                    key
                ]

                for key
                in METRIC_KEYS
            ],
            dtype=np.float64,
        )
    )

    metrics[
        "audio_to_caption_mean_rank"
    ] = float(
        audio_to_caption_rank
        .float()
        .mean()
        .item()
    )

    metrics[
        "caption_to_audio_mean_rank"
    ] = float(
        caption_to_audio_rank
        .float()
        .mean()
        .item()
    )

    return metrics


# =============================================================================
# 15. ONE COMPLETE INFERENCE PASS PER SEED
# =============================================================================

def run_seed(
    seed: int,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:

    model, best_epoch = (
        build_model(
            seed,
            device,
        )
    )

    text_outputs = []
    graph_outputs = []

    rows = 0

    with torch.inference_mode():

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

            z_text, z_graph = (
                model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    graph_batch,
                )
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

    if rows != N_TEST:

        raise RuntimeError(
            f"seed {seed} TEST rows="
            f"{rows}; expected 514"
        )

    z_text = (
        torch.cat(
            text_outputs,
            dim=0,
        )
        .contiguous()
    )

    z_graph = (
        torch.cat(
            graph_outputs,
            dim=0,
        )
        .contiguous()
    )

    check(
        tuple(
            z_text.shape
        )
        ==
        (
            N_TEST,
            SHARED_DIM,
        ),
        (
            f"seed {seed} text embedding "
            "= (514,256)"
        ),
    )

    check(
        tuple(
            z_graph.shape
        )
        ==
        (
            N_TEST,
            SHARED_DIM,
        ),
        (
            f"seed {seed} graph embedding "
            "= (514,256)"
        ),
    )

    check(
        bool(
            torch.isfinite(
                z_text
            ).all()
        ),
        (
            f"seed {seed} "
            "text embeddings finite"
        ),
    )

    check(
        bool(
            torch.isfinite(
                z_graph
            ).all()
        ),
        (
            f"seed {seed} "
            "graph embeddings finite"
        ),
    )

    text_norm = z_text.norm(
        dim=1
    )

    graph_norm = z_graph.norm(
        dim=1
    )

    check(
        torch.allclose(
            text_norm,
            torch.ones_like(
                text_norm
            ),
            rtol=0.0,
            atol=1e-5,
        ),
        (
            f"seed {seed} text embeddings "
            "L2-normalized"
        ),
    )

    check(
        torch.allclose(
            graph_norm,
            torch.ones_like(
                graph_norm
            ),
            rtol=0.0,
            atol=1e-5,
        ),
        (
            f"seed {seed} graph embeddings "
            "L2-normalized"
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen Script-71 convention:
    #
    # rows    = audio
    # columns = caption
    #
    # CPU calculation.
    # Temperature is NOT used for retrieval.
    # -------------------------------------------------------------------------

    similarity = (
        z_graph
        @ z_text.T
    ).contiguous()

    check(
        tuple(
            similarity.shape
        )
        ==
        (
            N_TEST,
            N_TEST,
        ),
        (
            f"seed {seed} similarity "
            "= (514,514)"
        ),
    )

    check(
        bool(
            torch.isfinite(
                similarity
            ).all()
        ),
        (
            f"seed {seed} similarity finite"
        ),
    )

    audio_to_caption_rank = (
        ranks(
            similarity
        )
    )

    caption_to_audio_rank = (
        ranks(
            similarity.T
        )
    )

    metrics = (
        metrics_from_ranks(
            audio_to_caption_rank,
            caption_to_audio_rank,
        )
    )

    arrays = {
        "text":
            z_text
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "graph":
            z_graph
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "similarity":
            similarity
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "audio_to_caption_rank":
            audio_to_caption_rank
            .numpy()
            .astype(
                np.int64,
                copy=False,
            ),

        "caption_to_audio_rank":
            caption_to_audio_rank
            .numpy()
            .astype(
                np.int64,
                copy=False,
            ),
    }

    del model

    torch.cuda.empty_cache()

    return {
        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "metrics":
            metrics,

        "arrays":
            arrays,
    }


# =============================================================================
# 16. WRITE ALL RESULTS ONLY AFTER ALL THREE SEEDS SUCCEED
# =============================================================================

def write_outputs(
    results: list[dict[str, Any]],
    test_data: dict[str, Any],
    protocol: dict[str, Any],
):

    banner(
        "5. WRITE FROZEN FINAL TEST "
        "RETRIEVAL ARTIFACTS"
    )

    OUT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        TEST_MANIFEST_OUT,
        test_data[
            "manifest_rows"
        ],
    )

    manifest_sha = sha256_file(
        TEST_MANIFEST_OUT
    )

    output_hashes = {}

    metric_rows = []

    for result in results:

        seed = int(
            result[
                "seed"
            ]
        )

        arrays = (
            result[
                "arrays"
            ]
        )

        save_npy_atomic(
            text_out(
                seed
            ),
            arrays[
                "text"
            ],
        )

        save_npy_atomic(
            graph_out(
                seed
            ),
            arrays[
                "graph"
            ],
        )

        save_npy_atomic(
            sim_out(
                seed
            ),
            arrays[
                "similarity"
            ],
        )

        save_npz_atomic(
            ranks_out(
                seed
            ),

            audio_to_caption_rank=
                arrays[
                    "audio_to_caption_rank"
                ],

            caption_to_audio_rank=
                arrays[
                    "caption_to_audio_rank"
                ],
        )

        output_hashes[
            str(
                seed
            )
        ] = {
            "text_embeddings":
                sha256_file(
                    text_out(
                        seed
                    )
                ),

            "graph_embeddings":
                sha256_file(
                    graph_out(
                        seed
                    )
                ),

            "similarity":
                sha256_file(
                    sim_out(
                        seed
                    )
                ),

            "ranks":
                sha256_file(
                    ranks_out(
                        seed
                    )
                ),
        }

        row = {
            "seed":
                seed,

            "best_epoch":
                int(
                    result[
                        "best_epoch"
                    ]
                ),
        }

        for key in (
            *METRIC_KEYS,
            "mR",
            "audio_to_caption_mean_rank",
            "caption_to_audio_mean_rank",
        ):

            row[
                key
            ] = float(
                result[
                    "metrics"
                ][
                    key
                ]
            )

        metric_rows.append(
            row
        )

    write_csv_atomic(
        PER_SEED_CSV,
        metric_rows,
    )

    per_seed_sha = sha256_file(
        PER_SEED_CSV
    )

    aggregate = {}

    for key in (
        *METRIC_KEYS,
        "mR",
    ):

        values = np.asarray(
            [
                float(
                    result[
                        "metrics"
                    ][
                        key
                    ]
                )

                for result
                in results
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
                            "metrics"
                        ][
                            key
                        ]
                    )

                for result
                in results
            },
        }

    summary = {
        "artifact_type":
            "task4_final_test_retrieval_summary",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "final_test_retrieval_complete",

        "selected_variant":
            "T4-B",

        "seeds":
            [
                42,
                1337,
                2026,
            ],

        "test_pairs":
            N_TEST,

        "best_seed_selection":
            False,

        "seed_ensemble":
            False,

        "test_labels_loaded":
            False,

        "candidate_pool_per_direction":
            N_TEST,

        "similarity":
            (
                "cosine_via_l2_normalized_"
                "dot_product"
            ),

        "temperature_used_for_retrieval":
            False,

        "rank_rule":
            (
                "1 + count("
                "candidate_similarity > "
                "positive_similarity)"
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

                    "metrics":
                        result[
                            "metrics"
                        ],

                    "checkpoint":
                        relative(
                            CHECKPOINTS[
                                result[
                                    "seed"
                                ]
                            ]
                        ),

                    "checkpoint_sha256":
                        EXPECTED_CHECKPOINT_SHA256[
                            result[
                                "seed"
                            ]
                        ],

                    "artifact_hashes":
                        output_hashes[
                            str(
                                result[
                                    "seed"
                                ]
                            )
                        ],
                }

                for result
                in results
            ],

        "aggregate_three_seed":
            aggregate,

        "random_baseline": {
            "R@1":
                1.0
                / N_TEST,

            "R@5":
                5.0
                / N_TEST,

            "R@10":
                10.0
                / N_TEST,
        },

        "test_manifest": {
            "path":
                relative(
                    TEST_MANIFEST_OUT
                ),

            "sha256":
                manifest_sha,
        },

        "per_seed_csv": {
            "path":
                relative(
                    PER_SEED_CSV
                ),

            "sha256":
                per_seed_sha,
        },

        "pretest_protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "pretest_lock_sha256":
            EXPECTED_PROTOCOL_LOCK_SHA256,

        "qualitative_query_indices":
            protocol[
                "qualitative_retrieval"
            ][
                "frozen_test_row_indices"
            ],

        "qualitative_model_seed":
            protocol[
                "qualitative_retrieval"
            ][
                "model_seed"
            ],

        "test_result_used_for_selection":
            False,

        "post_test_retraining_allowed":
            False,

        "next_script":
            73,

        "provenance": {
            "script72":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "script72_sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),
        },
    }

    write_json_atomic(
        SUMMARY_OUT,
        summary,
    )

    summary_sha = sha256_file(
        SUMMARY_OUT
    )

    lock = {
        "lock_type":
            "task4_final_test_retrieval_lock",

        "version":
            1,

        "created_utc":
            summary[
                "created_utc"
            ],

        "summary":
            relative(
                SUMMARY_OUT
            ),

        "summary_sha256":
            summary_sha,

        "test_manifest_sha256":
            manifest_sha,

        "per_seed_csv_sha256":
            per_seed_sha,

        "seed_artifact_hashes":
            output_hashes,

        "checkpoint_hashes": {
            str(
                seed
            ):
                EXPECTED_CHECKPOINT_SHA256[
                    seed
                ]

            for seed
            in FINAL_SEEDS
        },

        "pretest_protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "pretest_lock_sha256":
            EXPECTED_PROTOCOL_LOCK_SHA256,

        "test_labels_loaded":
            False,

        "best_seed_selection":
            False,

        "seed_ensemble":
            False,

        "script72_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        LOCK_OUT,
        lock,
    )

    lock_sha = sha256_file(
        LOCK_OUT
    )

    check(
        load_json(
            LOCK_OUT
        )[
            "summary_sha256"
        ]
        ==
        sha256_file(
            SUMMARY_OUT
        ),
        (
            "final TEST lock matches summary"
        ),
    )

    return (
        metric_rows,
        aggregate,
        manifest_sha,
        per_seed_sha,
        summary_sha,
        lock_sha,
    )


# =============================================================================
# 17. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 72 — ONE-TIME FINAL "
        "TASK-4 TEST RETRIEVAL"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()

    print(
        "Selected variant: T4-B"
    )

    print(
        "Final seeds:      42 / 1337 / 2026"
    )

    print(
        "TEST pairs:       514"
    )

    print()

    print(
        "TEST labels:          FORBIDDEN"
    )

    print(
        "Best-seed selection:  NO"
    )

    print(
        "Seed ensemble:        NO"
    )

    try:

        verify_output_protection()

        protocol = (
            verify_lineage_and_protocol()
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
            "2A. DEVICE"
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

        test_data = (
            load_test_inputs()
        )

        loader = build_loader(
            test_data
        )

        banner(
            "4. RUN PREDETERMINED THREE-SEED "
            "FINAL TEST RETRIEVAL"
        )

        results = []

        # ---------------------------------------------------------------------
        # IMPORTANT:
        # No metrics are printed until all 3 predetermined seeds finish.
        # ---------------------------------------------------------------------

        for seed in FINAL_SEEDS:

            print(
                f"Running frozen seed "
                f"{seed}..."
            )

            result = run_seed(
                seed,
                loader,
                device,
            )

            results.append(
                result
            )

            passed(
                (
                    f"seed {seed} completed "
                    "one full TEST "
                    "inference/retrieval pass"
                )
            )

        (
            metric_rows,
            aggregate,
            manifest_sha,
            per_seed_sha,
            summary_sha,
            lock_sha,
        ) = write_outputs(
            results,
            test_data,
            protocol,
        )

    except Exception as exc:

        banner(
            "SCRIPT 72 — ABORTED"
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

        print()
        print(
            "Preserve any partial artifacts."
        )

        print(
            "Rerun only the unchanged locked "
            "protocol after fixing a technical "
            "implementation failure."
        )

        print()
        print(
            "Do NOT tune, retrain, replace seeds, "
            "or alter metric definitions."
        )

        return 1

    # =========================================================================
    # Final boundary confirmation
    # =========================================================================

    banner(
        "6. FINAL TEST-BOUNDARY / "
        "DISCIPLINE CONFIRMATION"
    )

    print(
        "TEST pairs evaluated:               514"
    )

    print(
        "TEST labels loaded:                  0"
    )

    print(
        "TEST-driven training:                0"
    )

    print(
        "TEST-driven checkpoint selection:    0"
    )

    print(
        "TEST-driven seed selection:          0"
    )

    print(
        "Seed ensemble:                       0"
    )

    print(
        "Candidate-pool filtering:            0"
    )

    passed(
        (
            "final Task-4 TEST retrieval "
            "followed the frozen Script-71 protocol"
        )
    )

    # =========================================================================
    # Final result
    # =========================================================================

    banner(
        "SCRIPT 72 SUMMARY"
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
        "FINAL TASK-4 TEST RETRIEVAL COMPLETE."
    )

    print()

    for row in metric_rows:

        print(
            f"Seed {row['seed']:4d}: "
            f"mR = {row['mR']:.8f}"
        )

        print(
            "  Audio→Caption "
            f"R@1={row['audio_to_caption_R@1']:.8f}  "
            f"R@5={row['audio_to_caption_R@5']:.8f}  "
            f"R@10={row['audio_to_caption_R@10']:.8f}"
        )

        print(
            "  Caption→Audio "
            f"R@1={row['caption_to_audio_R@1']:.8f}  "
            f"R@5={row['caption_to_audio_R@5']:.8f}  "
            f"R@10={row['caption_to_audio_R@10']:.8f}"
        )

        print()

    print(
        "Three-seed TEST mR:"
    )

    print(
        f"  mean = "
        f"{aggregate['mR']['mean']:.8f}"
    )

    print(
        f"  std  = "
        f"{aggregate['mR']['std_population']:.8f}"
    )

    print()

    print(
        "TEST manifest SHA256:"
    )

    print(
        f"  {manifest_sha}"
    )

    print(
        "Per-seed CSV SHA256:"
    )

    print(
        f"  {per_seed_sha}"
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
        "No best seed was selected."
    )

    print(
        "No model was retrained or "
        "changed after TEST."
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
        "  Script 73 — frozen zero-shot "
        "tag evaluation."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )