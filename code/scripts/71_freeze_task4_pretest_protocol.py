#!/usr/bin/env python3
"""
SCRIPT 71 — FINAL TASK-4 PRE-TEST PROTOCOL LOCK
==============================================

Purpose
-------
Freeze the complete Task-4 final TEST retrieval protocol BEFORE any
Task-4 TEST example, caption, graph, retrieval result, or metric is observed.

This script performs:
    - provenance verification
    - selected-checkpoint verification
    - checkpoint-schema verification
    - final TEST protocol specification
    - deterministic qualitative-query INDEX selection
    - final pre-TEST protocol + lock writing

This script DOES NOT:
    - instantiate BERT
    - instantiate GAT
    - run model inference
    - load TEST token rows
    - open TEST graph files
    - read TEST captions
    - read TEST labels
    - calculate TEST similarities
    - calculate TEST retrieval
    - observe TEST metrics

Final selected configuration
----------------------------
T4-B

Predetermined final seeds
-------------------------
42
1337
2026

Final TEST reporting
--------------------
For EACH seed independently:
    Audio -> Caption:
        R@1
        R@5
        R@10

    Caption -> Audio:
        R@1
        R@5
        R@10

    mR:
        arithmetic mean of all six recalls

Then report:
    per-seed values
    mean across all 3 seeds
    population SD across all 3 seeds

No:
    best seed
    test-based seed selection
    embedding ensemble
    similarity ensemble
    post-TEST hyperparameter changes
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


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

MULTISEED_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_multiseed_validation.csv"
)

MULTISEED_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_multiseed_summary.json"
)

MULTISEED_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_multiseed_lock.json"
)


PRETRAINING_LOCK = (
    ROOT
    / "data/splits/"
      "pretraining_frozen_hashes.json"
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


SEED42_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-B_seed42_best.pt"
)

SEED1337_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-B_seed1337_multiseed_best.pt"
)

SEED2026_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-B_seed2026_multiseed_best.pt"
)


SEED42_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_seed42_summary.json"
)

SEED1337_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_seed1337_multiseed_summary.json"
)

SEED2026_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_seed2026_multiseed_summary.json"
)


OUTPUT_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_pretest_protocol.json"
)

LOCK_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_pretest_protocol_lock.json"
)


# =============================================================================
# 1. EXPECTED HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "61ed1263476e631374790c9669ab4aa96"
    "e3b5cb295436d4e273da66d4da556db"
)

EXPECTED_DESIGN_LOCK_SHA256 = (
    "53f50205e712833ec4efb506b9b6d911"
    "8dcf46f51b72c5907f33cd61ddca612c"
)

EXPECTED_MODEL_SELECTION_SHA256 = (
    "831e1535cfe6f45a646f1732ecf268db"
    "02c5214aa48404fd07fb3ae8ff5e41c5"
)

EXPECTED_MODEL_SELECTION_LOCK_SHA256 = (
    "a1aa9d2420a97c86b0f3709227ff22d1"
    "ee3ccf111010733b2bf705dee5eb6bb6"
)

EXPECTED_MULTISEED_CSV_SHA256 = (
    "f3b09fe49470fb351fcdf9b0fe5f7293"
    "0ddeaeecc9b9e327ad49af599b190b47"
)

EXPECTED_MULTISEED_SUMMARY_SHA256 = (
    "0bd0bb281448706cd86a124295be8ae68"
    "2d5b32b74d7e517742caec486959297"
)

EXPECTED_MULTISEED_LOCK_SHA256 = (
    "d6ab94b262903c33f28c8eb57c2de2e6"
    "c1e7fd974ea232186eef8ee19a27c674"
)


EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28"
    "f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_CANONICAL_DATASET_SHA256 = (
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


EXPECTED_SEED42_CHECKPOINT_SHA256 = (
    "2be12c7e6fdc851b2b1efae0a7bed2c4"
    "95cbefb5ae90a4f491c54a72ec8e9ace"
)

EXPECTED_SEED1337_CHECKPOINT_SHA256 = (
    "824a98d52327d513f67cfa21cf702a459"
    "7f51a91c0b17cea1ea7d10628b4b918"
)

EXPECTED_SEED2026_CHECKPOINT_SHA256 = (
    "a328d029b46ef14a4199cff0604f190f8"
    "06cf953c7c4ad75a0f884210bb743b9"
)


EXPECTED_SEED42_SUMMARY_SHA256 = (
    "d4300c9f5089ca3efa9626bb60fc10e1"
    "23336604808e99eba3cc7cfd0839e6aa"
)

EXPECTED_SEED1337_SUMMARY_SHA256 = (
    "abecd215adde30e282dbc63e15ea5954"
    "00b1089585521a616f4b1c75542df6ec"
)

EXPECTED_SEED2026_SUMMARY_SHA256 = (
    "300d0ee81afffec10dacc91e0feb0259"
    "8950ec153a0bb1820ef65f04604f2b36"
)


# =============================================================================
# 2. FINAL PRE-TEST CONTRACT
# =============================================================================

SELECTED_VARIANT = "T4-B"

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

EXPECTED_TEST_PAIRS = 514

BERT_DIM = 768
GAT_DIM = 128
SHARED_DIM = 256

MAX_LENGTH = 112

TEMPERATURE = 0.07

INFERENCE_BATCH_SIZE = 32

QUALITATIVE_QUERY_COUNT = 10
QUALITATIVE_QUERY_SEED = 42

QUALITATIVE_MODEL_SEED = 42

HUMAN_EVAL_MODEL_SEED = 42

MIN_HUMAN_LISTENERS = 5


METRIC_KEYS = (
    "audio_to_caption_R@1",
    "audio_to_caption_R@5",
    "audio_to_caption_R@10",
    "caption_to_audio_R@1",
    "caption_to_audio_R@5",
    "caption_to_audio_R@10",
)

REQUIRED_DELTA_KEYS = (
    "bert_layer_10_state_dict",
    "bert_layer_11_state_dict",
    "gat3_state_dict",
    "text_projection_state_dict",
    "graph_projection_state_dict",
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

    temp = Path(
        str(path)
        + ".__tmp__"
    )

    if temp.exists():

        raise RuntimeError(
            f"Temporary file already exists: "
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


def validate_state_dict(
    state: Any,
    label: str,
) -> None:

    check(
        isinstance(
            state,
            dict,
        ),
        f"{label} is a state dictionary",
    )

    if not isinstance(
        state,
        dict,
    ):
        return

    check(
        len(
            state
        )
        > 0,
        f"{label} is non-empty",
    )

    for key, value in state.items():

        check(
            isinstance(
                value,
                torch.Tensor,
            ),
            (
                f"{label}.{key} "
                "is a tensor"
            ),
        )

        if not isinstance(
            value,
            torch.Tensor,
        ):
            continue

        check(
            bool(
                torch.isfinite(
                    value
                ).all()
            ),
            (
                f"{label}.{key} "
                "is finite"
            ),
        )


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. PRE-TEST OUTPUT PROTECTION"
    )

    check(
        not OUTPUT_PATH.exists(),
        (
            "pre-TEST protocol does not "
            "already exist"
        ),
    )

    check(
        not LOCK_PATH.exists(),
        (
            "pre-TEST protocol lock does "
            "not already exist"
        ),
    )

    # -------------------------------------------------------------------------
    # There must not already be a Task-4 TEST retrieval result.
    # This examines file NAMES only.
    # -------------------------------------------------------------------------

    suspicious = []

    run_dir = (
        ROOT
        / "results/runs/task4"
    )

    if run_dir.exists():

        for path in run_dir.iterdir():

            name = (
                path.name.lower()
            )

            if (
                "test"
                in name
                and
                "retrieval"
                in name
            ):

                suspicious.append(
                    path
                )

    check(
        len(
            suspicious
        )
        == 0,
        (
            "no Task-4 TEST retrieval "
            "artifact already exists"
        ),
    )

    if suspicious:

        print(
            "Unexpected TEST retrieval artifacts:"
        )

        for path in suspicious:

            print(
                f"  {path}"
            )

    if FAILURES:

        raise RuntimeError(
            "Pre-TEST output protection failed"
        )


# =============================================================================
# 6. VERIFY FROZEN LINEAGE
# =============================================================================

def verify_lineage() -> None:

    banner(
        "2. VERIFY COMPLETE FROZEN LINEAGE"
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
            EXPECTED_MODEL_SELECTION_SHA256,
            "Task-4 model selection",
        ),
        (
            MODEL_SELECTION_LOCK,
            EXPECTED_MODEL_SELECTION_LOCK_SHA256,
            "Task-4 model-selection lock",
        ),
        (
            MULTISEED_CSV,
            EXPECTED_MULTISEED_CSV_SHA256,
            "Task-4 multi-seed validation CSV",
        ),
        (
            MULTISEED_SUMMARY,
            EXPECTED_MULTISEED_SUMMARY_SHA256,
            "Task-4 multi-seed summary",
        ),
        (
            MULTISEED_LOCK,
            EXPECTED_MULTISEED_LOCK_SHA256,
            "Task-4 multi-seed lock",
        ),
        (
            PRETRAINING_LOCK,
            EXPECTED_PRETRAINING_LOCK_SHA256,
            "pretraining lock",
        ),
        (
            CANONICAL_DATASET,
            EXPECTED_CANONICAL_DATASET_SHA256,
            "canonical dataset",
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
        (
            SEED42_CHECKPOINT,
            EXPECTED_SEED42_CHECKPOINT_SHA256,
            "seed-42 selected T4-B checkpoint",
        ),
        (
            SEED1337_CHECKPOINT,
            EXPECTED_SEED1337_CHECKPOINT_SHA256,
            "seed-1337 selected T4-B checkpoint",
        ),
        (
            SEED2026_CHECKPOINT,
            EXPECTED_SEED2026_CHECKPOINT_SHA256,
            "seed-2026 selected T4-B checkpoint",
        ),
        (
            SEED42_SUMMARY,
            EXPECTED_SEED42_SUMMARY_SHA256,
            "seed-42 T4-B summary",
        ),
        (
            SEED1337_SUMMARY,
            EXPECTED_SEED1337_SUMMARY_SHA256,
            "seed-1337 T4-B summary",
        ),
        (
            SEED2026_SUMMARY,
            EXPECTED_SEED2026_SUMMARY_SHA256,
            "seed-2026 T4-B summary",
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
            "Frozen lineage verification failed"
        )


# =============================================================================
# 7. VERIFY SELECTION / MULTI-SEED FREEZE
# =============================================================================

def verify_selection_state() -> None:

    banner(
        "3. VERIFY FINAL MODEL-SELECTION STATE"
    )

    selection = load_json(
        MODEL_SELECTION
    )

    multiseed = load_json(
        MULTISEED_SUMMARY
    )

    multiseed_lock = load_json(
        MULTISEED_LOCK
    )

    check(
        selection[
            "selected_variant"
        ]
        == SELECTED_VARIANT,
        (
            "selected Task-4 architecture "
            "is frozen as T4-B"
        ),
    )

    check(
        selection[
            "test_used"
        ]
        is False,
        (
            "architecture selection used "
            "zero TEST data"
        ),
    )

    check(
        multiseed[
            "selected_variant"
        ]
        == SELECTED_VARIANT,
        (
            "multi-seed summary uses "
            "selected T4-B only"
        ),
    )

    check(
        multiseed[
            "seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "multi-seed summary uses exactly "
            "42/1337/2026"
        ),
    )

    check(
        multiseed[
            "architecture_reselection_performed"
        ]
        is False,
        (
            "multi-seed stage did not reopen "
            "architecture selection"
        ),
    )

    check(
        multiseed[
            "test_access"
        ]
        is False,
        (
            "multi-seed training records "
            "zero TEST access"
        ),
    )

    check(
        multiseed[
            "test_retrieval"
        ]
        is False,
        (
            "multi-seed training records "
            "zero TEST retrieval"
        ),
    )

    check(
        multiseed_lock[
            "selected_variant"
        ]
        == SELECTED_VARIANT,
        (
            "multi-seed lock freezes T4-B"
        ),
    )

    check(
        multiseed_lock[
            "seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "multi-seed lock freezes all "
            "three final seeds"
        ),
    )

    check(
        multiseed_lock[
            "test_access"
        ]
        is False,
        (
            "multi-seed lock records "
            "TEST unused"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Final model-selection state invalid"
        )


# =============================================================================
# 8. VERIFY THREE SELECTED CHECKPOINTS
# =============================================================================

def verify_checkpoints() -> list[dict[str, Any]]:

    banner(
        "4. VERIFY FINAL THREE T4-B CHECKPOINTS"
    )

    definitions = (
        (
            42,
            SEED42_CHECKPOINT,
            EXPECTED_SEED42_CHECKPOINT_SHA256,
        ),
        (
            1337,
            SEED1337_CHECKPOINT,
            EXPECTED_SEED1337_CHECKPOINT_SHA256,
        ),
        (
            2026,
            SEED2026_CHECKPOINT,
            EXPECTED_SEED2026_CHECKPOINT_SHA256,
        ),
    )

    records = []

    reference_state_keys = None

    for (
        seed,
        path,
        digest,
    ) in definitions:

        print()
        print(
            f"Checking seed {seed}..."
        )

        checkpoint = load_checkpoint(
            path
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
                "records correct seed"
            ),
        )

        for key in REQUIRED_DELTA_KEYS:

            check(
                key
                in checkpoint,
                (
                    f"seed {seed} checkpoint "
                    f"contains {key}"
                ),
            )

        if FAILURES:

            raise RuntimeError(
                f"seed {seed} checkpoint "
                "schema invalid"
            )

        for key in REQUIRED_DELTA_KEYS:

            validate_state_dict(
                checkpoint[
                    key
                ],
                (
                    f"seed{seed}."
                    f"{key}"
                ),
            )

        text_projection = (
            checkpoint[
                "text_projection_state_dict"
            ][
                "weight"
            ]
        )

        graph_projection = (
            checkpoint[
                "graph_projection_state_dict"
            ][
                "weight"
            ]
        )

        check(
            tuple(
                text_projection.shape
            )
            ==
            (
                SHARED_DIM,
                BERT_DIM,
            ),
            (
                f"seed {seed} text projection "
                "= (256,768)"
            ),
        )

        check(
            tuple(
                graph_projection.shape
            )
            ==
            (
                SHARED_DIM,
                GAT_DIM,
            ),
            (
                f"seed {seed} graph projection "
                "= (256,128)"
            ),
        )

        state_keys = {
            key:
                tuple(
                    checkpoint[
                        key
                    ].keys()
                )

            for key in REQUIRED_DELTA_KEYS
        }

        if reference_state_keys is None:

            reference_state_keys = (
                state_keys
            )

        else:

            check(
                state_keys
                ==
                reference_state_keys,
                (
                    f"seed {seed} delta-state "
                    "schema matches seed 42"
                ),
            )

        records.append(
            {
                "seed":
                    seed,

                "checkpoint":
                    relative(
                        path
                    ),

                "checkpoint_sha256":
                    digest,

                "best_epoch":
                    int(
                        checkpoint[
                            "best_epoch"
                        ]
                    ),
            }
        )

    if FAILURES:

        raise RuntimeError(
            "Final checkpoint verification failed"
        )

    return records


# =============================================================================
# 9. FREEZE QUALITATIVE / HUMAN QUERY INDICES
# =============================================================================

def freeze_query_indices() -> list[int]:

    banner(
        "5. FREEZE QUALITATIVE QUERY INDICES"
    )

    rng = np.random.default_rng(
        QUALITATIVE_QUERY_SEED
    )

    indices = (
        rng.choice(
            EXPECTED_TEST_PAIRS,
            size=QUALITATIVE_QUERY_COUNT,
            replace=False,
        )
        .astype(
            np.int64
        )
        .tolist()
    )

    # Sort for deterministic reporting/order only.
    indices = sorted(
        int(
            index
        )
        for index
        in indices
    )

    check(
        len(
            indices
        )
        == QUALITATIVE_QUERY_COUNT,
        (
            "exactly 10 qualitative "
            "TEST row indices frozen"
        ),
    )

    check(
        len(
            set(
                indices
            )
        )
        == QUALITATIVE_QUERY_COUNT,
        (
            "qualitative query indices "
            "are unique"
        ),
    )

    check(
        all(
            0
            <= index
            < EXPECTED_TEST_PAIRS

            for index
            in indices
        ),
        (
            "all qualitative row indices "
            "are in 0..513"
        ),
    )

    print()
    print(
        "Frozen TEST row indices:"
    )

    print(
        "  "
        + ", ".join(
            str(
                index
            )
            for index
            in indices
        )
    )

    print()
    print(
        "No TEST caption, track ID, result, "
        "or retrieval rank was read to select them."
    )

    return indices


# =============================================================================
# 10. BUILD PRE-TEST PROTOCOL
# =============================================================================

def build_protocol(
    checkpoint_records: list[dict[str, Any]],
    qualitative_indices: list[int],
) -> dict[str, Any]:

    banner(
        "6. BUILD FINAL PRE-TEST PROTOCOL"
    )

    random_baseline = {
        "R@1":
            1.0
            / EXPECTED_TEST_PAIRS,

        "R@5":
            5.0
            / EXPECTED_TEST_PAIRS,

        "R@10":
            10.0
            / EXPECTED_TEST_PAIRS,
    }

    protocol = {
        "artifact_type":
            "task4_final_pretest_protocol",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "frozen_before_first_task4_test_retrieval",

        # ---------------------------------------------------------------------
        # Final selected model
        # ---------------------------------------------------------------------

        "selected_configuration": {
            "variant":
                "T4-B",

            "architecture_selection":
                "already_frozen_by_script69",

            "architecture_reselection_after_test":
                False,

            "final_seeds":
                [
                    42,
                    1337,
                    2026,
                ],

            "checkpoints":
                checkpoint_records,

            "best_seed_selection":
                False,

            "test_based_seed_selection":
                False,

            "seed_ensemble":
                False,
        },

        # ---------------------------------------------------------------------
        # TEST population
        # ---------------------------------------------------------------------

        "test_population": {
            "expected_pairs":
                EXPECTED_TEST_PAIRS,

            "candidate_pool_per_direction":
                EXPECTED_TEST_PAIRS,

            "caption_to_audio_candidates":
                (
                    "all 514 frozen MusicCaps TEST "
                    "audio graphs"
                ),

            "audio_to_caption_candidates":
                (
                    "all 514 frozen MusicCaps TEST "
                    "captions"
                ),

            "same_candidate_pool_for_all_seeds":
                True,

            "positive_pair":
                (
                    "row-aligned graph/caption pair "
                    "for the same frozen track_id"
                ),

            "script72_must_verify_exact_track_id_alignment":
                True,

            "duplicate_or_missing_track_id":
                "abort",

            "pool_reduction_after_observing_test":
                False,
        },

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        "frozen_test_inputs": {
            "canonical_dataset":
                {
                    "path":
                        relative(
                            CANONICAL_DATASET
                        ),

                    "sha256":
                        EXPECTED_CANONICAL_DATASET_SHA256,
                },

            "bert_token_cache":
                {
                    "path":
                        relative(
                            TOKEN_CACHE
                        ),

                    "sha256":
                        EXPECTED_TOKEN_CACHE_SHA256,
                },

            "bert_token_manifest":
                {
                    "path":
                        relative(
                            TOKEN_MANIFEST
                        ),

                    "sha256":
                        EXPECTED_TOKEN_MANIFEST_SHA256,
                },

            "graph_manifest":
                {
                    "path":
                        relative(
                            GRAPH_MANIFEST
                        ),

                    "sha256":
                        EXPECTED_GRAPH_MANIFEST_SHA256,
                },

            "T1_B":
                {
                    "path":
                        relative(
                            T1_CHECKPOINT
                        ),

                    "sha256":
                        EXPECTED_T1_SHA256,
                },

            "A4":
                {
                    "path":
                        relative(
                            A4_CHECKPOINT
                        ),

                    "sha256":
                        EXPECTED_A4_SHA256,
                },
        },

        # ---------------------------------------------------------------------
        # Exact inference pathway
        # ---------------------------------------------------------------------

        "test_inference": {
            "live_bert":
                True,

            "live_gat_g2":
                True,

            "bert_max_length":
                MAX_LENGTH,

            "graph_representation":
                "A4 GAT using G2 edges",

            "shared_embedding_dimension":
                SHARED_DIM,

            "l2_normalization":
                True,

            "similarity":
                "cosine via normalized dot product",

            "temperature_used_for_retrieval":
                False,

            "batch_size":
                INFERENCE_BATCH_SIZE,

            "shuffle":
                False,

            "drop_last":
                False,

            "model_mode":
                "eval",

            "torch_mode":
                "inference_mode",

            "dtype":
                "float32",

            "AMP":
                False,

            "TF32":
                False,

            "cudnn_deterministic":
                True,

            "cudnn_benchmark":
                False,

            "embedding_generation_device":
                "CUDA",

            "retrieval_similarity_device":
                "CPU",

            "similarity_matrix_shape":
                [
                    EXPECTED_TEST_PAIRS,
                    EXPECTED_TEST_PAIRS,
                ],

            "similarity_matrix_orientation":
                (
                    "rows=audio, columns=caption"
                ),

            "single_forward_pass_per_seed":
                True,
        },

        # ---------------------------------------------------------------------
        # Ranking / metrics
        # ---------------------------------------------------------------------

        "retrieval_metrics": {
            "rank_definition":
                (
                    "1 + number of candidate similarities "
                    "strictly greater than the positive-pair similarity"
                ),

            "tie_behavior":
                (
                    "equal-similarity candidates are not counted "
                    "as outranking the positive"
                ),

            "audio_to_caption":
                [
                    "R@1",
                    "R@5",
                    "R@10",
                ],

            "caption_to_audio":
                [
                    "R@1",
                    "R@5",
                    "R@10",
                ],

            "mR":
                (
                    "arithmetic mean of the six "
                    "bidirectional recall values"
                ),

            "mean_rank":
                "diagnostic_secondary",

            "primary_reporting":
                (
                    "all six recalls plus mR"
                ),

            "random_baseline":
                random_baseline,
        },

        # ---------------------------------------------------------------------
        # Multi-seed reporting
        # ---------------------------------------------------------------------

        "final_reporting": {
            "per_seed_metrics":
                True,

            "aggregate_seeds":
                [
                    42,
                    1337,
                    2026,
                ],

            "aggregate_mean":
                True,

            "aggregate_standard_deviation":
                True,

            "standard_deviation_definition":
                (
                    "population standard deviation "
                    "over the complete predetermined "
                    "three-seed set; ddof=0"
                ),

            "report_best_seed":
                False,

            "select_best_seed":
                False,

            "ensemble_embeddings":
                False,

            "ensemble_similarity_matrices":
                False,
        },

        # ---------------------------------------------------------------------
        # TEST output caching
        # ---------------------------------------------------------------------

        "script72_outputs": {
            "per_seed_embedding_cache":
                (
                    "save normalized 514x256 text "
                    "and 514x256 graph embeddings "
                    "for each final seed"
                ),

            "per_seed_similarity_cache":
                (
                    "save 514x514 audio-by-caption "
                    "cosine similarity matrix"
                ),

            "per_seed_rank_cache":
                (
                    "save both retrieval-direction ranks"
                ),

            "test_manifest":
                (
                    "save row_index + track_id alignment "
                    "used for final retrieval"
                ),

            "per_seed_metrics":
                True,

            "aggregate_metrics":
                True,

            "artifact_lock":
                True,

            "purpose":
                (
                    "later qualitative/human analysis must "
                    "reuse the frozen final TEST retrieval "
                    "rather than rerun/select retrieval"
                ),
        },

        # ---------------------------------------------------------------------
        # Script72 forbidden inputs/actions
        # ---------------------------------------------------------------------

        "script72_forbidden": {
            "test_labels":
                True,

            "tag_targets":
                True,

            "threshold_tuning":
                True,

            "checkpoint_selection":
                True,

            "architecture_selection":
                True,

            "temperature_tuning":
                True,

            "prompt_engineering":
                True,

            "seed_selection":
                True,

            "candidate_pool_filtering":
                True,

            "post_result_reranking":
                True,
        },

        # ---------------------------------------------------------------------
        # Qualitative retrieval
        # ---------------------------------------------------------------------

        "qualitative_retrieval": {
            "query_count":
                QUALITATIVE_QUERY_COUNT,

            "query_selection_seed":
                QUALITATIVE_QUERY_SEED,

            "query_selection_algorithm":
                (
                    "numpy.default_rng(42).choice("
                    "514,size=10,replace=False), "
                    "then sort row indices"
                ),

            "frozen_test_row_indices":
                qualitative_indices,

            "model_seed":
                QUALITATIVE_MODEL_SEED,

            "model_seed_reason":
                (
                    "seed 42 fixed before TEST because it "
                    "was the original architecture-selection seed; "
                    "not chosen using TEST quality"
                ),

            "direction":
                "caption_to_audio",

            "top_k":
                3,

            "hand_picking":
                False,

            "reuse_script72_similarity":
                True,
        },

        # ---------------------------------------------------------------------
        # Human evaluation
        # ---------------------------------------------------------------------

        "human_evaluation": {
            "same_queries_as_qualitative":
                True,

            "frozen_test_row_indices":
                qualitative_indices,

            "model_seed":
                HUMAN_EVAL_MODEL_SEED,

            "retrieval_rank_used":
                1,

            "listeners_minimum":
                MIN_HUMAN_LISTENERS,

            "rating_scale":
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],

            "report":
                "mean_and_standard_deviation",

            "reuse_script72_retrieval":
                True,
        },

        # ---------------------------------------------------------------------
        # Zero-shot tags
        # ---------------------------------------------------------------------

        "zero_shot_tags": {
            "executed_after_final_retrieval":
                True,

            "contrastive_training_uses_tag_labels":
                False,

            "final_seeds":
                [
                    42,
                    1337,
                    2026,
                ],

            "no_best_seed_selection":
                True,

            "canonical_label_text":
                (
                    "exact frozen canonical label phrase; "
                    "no prompt engineering"
                ),

            "primary_metric":
                "Macro Average Precision",

            "primary_metric_threshold_free":
                True,

            "if_F1_reported":
                (
                    "one global threshold must be selected "
                    "using VAL only before TEST labels are opened"
                ),
        },

        # ---------------------------------------------------------------------
        # Post-TEST discipline
        # ---------------------------------------------------------------------

        "post_test_rules": {
            "no_retraining":
                True,

            "no_checkpoint_replacement":
                True,

            "no_seed_replacement":
                True,

            "no_metric_definition_change":
                True,

            "no_candidate_pool_change":
                True,

            "no_temperature_change":
                True,

            "no_embedding_dimension_change":
                True,

            "no_graph_rule_change":
                True,

            "no_query_reselection":
                True,

            "technical_failure_policy":
                (
                    "if Script72 aborts before a complete "
                    "accepted result, preserve the failed "
                    "attempt and rerun only the unchanged "
                    "locked protocol"
                ),
        },

        # ---------------------------------------------------------------------
        # Provenance
        # ---------------------------------------------------------------------

        "parent_hashes": {
            "design":
                EXPECTED_DESIGN_SHA256,

            "design_lock":
                EXPECTED_DESIGN_LOCK_SHA256,

            "model_selection":
                EXPECTED_MODEL_SELECTION_SHA256,

            "model_selection_lock":
                EXPECTED_MODEL_SELECTION_LOCK_SHA256,

            "multiseed_csv":
                EXPECTED_MULTISEED_CSV_SHA256,

            "multiseed_summary":
                EXPECTED_MULTISEED_SUMMARY_SHA256,

            "multiseed_lock":
                EXPECTED_MULTISEED_LOCK_SHA256,

            "pretraining_lock":
                EXPECTED_PRETRAINING_LOCK_SHA256,
        },

        "test_retrieval_observed_before_freeze":
            False,

        "test_metrics_observed_before_freeze":
            False,

        "test_labels_opened_before_freeze":
            False,

        "next_script":
            72,
    }

    passed(
        "complete final TEST protocol constructed"
    )

    return protocol


# =============================================================================
# 11. WRITE PROTOCOL / LOCK
# =============================================================================

def write_protocol_and_lock(
    protocol: dict[str, Any],
) -> tuple[
    str,
    str,
]:

    banner(
        "7. WRITE PRE-TEST PROTOCOL / LOCK"
    )

    write_json_atomic(
        OUTPUT_PATH,
        protocol,
    )

    protocol_sha = sha256_file(
        OUTPUT_PATH
    )

    check(
        OUTPUT_PATH.is_file(),
        (
            "final pre-TEST protocol written"
        ),
    )

    print(
        "Protocol:"
    )

    print(
        f"  {relative(OUTPUT_PATH)}"
    )

    print(
        f"SHA256:"
    )

    print(
        f"  {protocol_sha}"
    )

    lock = {
        "lock_type":
            "task4_final_pretest_protocol_lock",

        "version":
            1,

        "created_utc":
            protocol[
                "created_utc"
            ],

        "status":
            (
                "frozen_before_first_task4_"
                "test_retrieval"
            ),

        "protocol":
            relative(
                OUTPUT_PATH
            ),

        "protocol_sha256":
            protocol_sha,

        "selected_variant":
            "T4-B",

        "final_seeds":
            [
                42,
                1337,
                2026,
            ],

        "expected_test_pairs":
            EXPECTED_TEST_PAIRS,

        "selected_checkpoints": {
            "42": {
                "path":
                    relative(
                        SEED42_CHECKPOINT
                    ),

                "sha256":
                    EXPECTED_SEED42_CHECKPOINT_SHA256,
            },

            "1337": {
                "path":
                    relative(
                        SEED1337_CHECKPOINT
                    ),

                "sha256":
                    EXPECTED_SEED1337_CHECKPOINT_SHA256,
            },

            "2026": {
                "path":
                    relative(
                        SEED2026_CHECKPOINT
                    ),

                "sha256":
                    EXPECTED_SEED2026_CHECKPOINT_SHA256,
            },
        },

        "test_retrieval_observed":
            False,

        "test_metrics_observed":
            False,

        "test_labels_opened":
            False,

        "script71":
            relative(
                Path(
                    __file__
                ).resolve()
            ),

        "script71_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),

        "parent_hashes":
            protocol[
                "parent_hashes"
            ],
    }

    write_json_atomic(
        LOCK_PATH,
        lock,
    )

    lock_sha = sha256_file(
        LOCK_PATH
    )

    check(
        LOCK_PATH.is_file(),
        (
            "final pre-TEST protocol lock "
            "written"
        ),
    )

    print()
    print(
        "Lock:"
    )

    print(
        f"  {relative(LOCK_PATH)}"
    )

    print(
        f"SHA256:"
    )

    print(
        f"  {lock_sha}"
    )

    return (
        protocol_sha,
        lock_sha,
    )


# =============================================================================
# 12. POST-WRITE VALIDATION
# =============================================================================

def post_write_validation(
    protocol_sha: str,
) -> None:

    banner(
        "8. POST-WRITE VALIDATION"
    )

    protocol = load_json(
        OUTPUT_PATH
    )

    lock = load_json(
        LOCK_PATH
    )

    check(
        protocol[
            "selected_configuration"
        ][
            "variant"
        ]
        == "T4-B",
        (
            "written protocol freezes T4-B"
        ),
    )

    check(
        protocol[
            "selected_configuration"
        ][
            "final_seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "written protocol freezes all "
            "three final seeds"
        ),
    )

    check(
        protocol[
            "test_population"
        ][
            "expected_pairs"
        ]
        == EXPECTED_TEST_PAIRS,
        (
            "written protocol freezes "
            "514 TEST pairs"
        ),
    )

    check(
        protocol[
            "final_reporting"
        ][
            "select_best_seed"
        ]
        is False,
        (
            "written protocol forbids "
            "best-seed selection"
        ),
    )

    check(
        protocol[
            "script72_forbidden"
        ][
            "test_labels"
        ]
        is True,
        (
            "Script72 is explicitly forbidden "
            "from reading TEST labels"
        ),
    )

    check(
        len(
            protocol[
                "qualitative_retrieval"
            ][
                "frozen_test_row_indices"
            ]
        )
        == 10,
        (
            "10 qualitative TEST query "
            "indices are frozen"
        ),
    )

    check(
        lock[
            "protocol_sha256"
        ]
        == protocol_sha,
        (
            "pre-TEST lock matches "
            "protocol SHA256"
        ),
    )

    check(
        lock[
            "test_retrieval_observed"
        ]
        is False,
        (
            "lock records zero TEST "
            "retrieval before freeze"
        ),
    )

    check(
        lock[
            "test_metrics_observed"
        ]
        is False,
        (
            "lock records zero TEST "
            "metrics before freeze"
        ),
    )

    check(
        lock[
            "test_labels_opened"
        ]
        is False,
        (
            "lock records zero TEST "
            "label access before freeze"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Pre-TEST post-write validation failed"
        )


# =============================================================================
# 13. EXECUTION-BOUNDARY CONFIRMATION
# =============================================================================

def print_boundary() -> None:

    banner(
        "9. EXECUTION-BOUNDARY CONFIRMATION"
    )

    print(
        "Script 71 instantiated BERT:              NO"
    )

    print(
        "Script 71 instantiated GAT:               NO"
    )

    print(
        "Script 71 ran model inference:            NO"
    )

    print(
        "Script 71 loaded TEST token rows:         NO"
    )

    print(
        "Script 71 opened TEST graph files:        NO"
    )

    print(
        "Script 71 read TEST captions:             NO"
    )

    print(
        "Script 71 read TEST labels:               NO"
    )

    print(
        "Script 71 calculated TEST similarity:     NO"
    )

    print(
        "Script 71 calculated TEST retrieval:      NO"
    )

    print(
        "Script 71 observed TEST metrics:          NO"
    )

    print(
        "Script 71 selected a best seed:           NO"
    )

    print(
        "Script 71 changed model architecture:     NO"
    )

    passed(
        (
            "final Task-4 TEST protocol was "
            "frozen before first TEST retrieval"
        )
    )


# =============================================================================
# 14. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 71 — FINAL TASK-4 PRE-TEST PROTOCOL LOCK"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "Selected configuration: T4-B"
    )

    print(
        "Final seeds:            42 / 1337 / 2026"
    )

    print(
        "Expected TEST pairs:    514"
    )

    print()
    print(
        "TEST inference:         NO"
    )

    print(
        "TEST retrieval:         NO"
    )

    print(
        "TEST labels:            NO"
    )

    try:

        verify_output_protection()

        verify_lineage()

        verify_selection_state()

        checkpoint_records = (
            verify_checkpoints()
        )

        qualitative_indices = (
            freeze_query_indices()
        )

        protocol = build_protocol(
            checkpoint_records,
            qualitative_indices,
        )

        (
            protocol_sha,
            lock_sha,
        ) = write_protocol_and_lock(
            protocol
        )

        post_write_validation(
            protocol_sha
        )

        print_boundary()

    except Exception as exc:

        banner(
            "SCRIPT 71 — ABORTED"
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
            "Do NOT run final Task-4 TEST retrieval."
        )

        return 1

    banner(
        "SCRIPT 71 SUMMARY"
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
        "FINAL TASK-4 PRE-TEST PROTOCOL IS NOW FROZEN."
    )

    print()
    print(
        "Selected model:"
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
        "Final TEST reporting:"
    )

    print(
        "  Audio→Caption R@1/R@5/R@10"
    )

    print(
        "  Caption→Audio R@1/R@5/R@10"
    )

    print(
        "  mR"
    )

    print(
        "  three-seed mean ± population SD"
    )

    print()
    print(
        "Best-seed selection: NO"
    )

    print(
        "Seed ensemble:       NO"
    )

    print(
        "TEST labels in 72:   FORBIDDEN"
    )

    print()
    print(
        "Protocol:"
    )

    print(
        f"  {relative(OUTPUT_PATH)}"
    )

    print(
        f"  SHA256 = {protocol_sha}"
    )

    print()
    print(
        "Lock:"
    )

    print(
        f"  {relative(LOCK_PATH)}"
    )

    print(
        f"  SHA256 = {lock_sha}"
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
        "  Script 72 — ONE-TIME final "
        "Task-4 TEST retrieval."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )