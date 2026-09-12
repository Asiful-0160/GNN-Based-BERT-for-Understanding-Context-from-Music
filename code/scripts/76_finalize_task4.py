#!/usr/bin/env python3
"""
SCRIPT 76 — FINAL TASK-4 RESULTS / FIGURES / AUDIT / ARTIFACT LOCK
==================================================================

FINAL Task-4 consolidation script.

This script DOES NOT:
    - instantiate BERT
    - instantiate GAT
    - run model inference
    - retrain anything
    - reopen raw TEST audio/graphs/tokens
    - recompute retrieval from embeddings
    - tune thresholds
    - select a best seed
    - change any experiment decision

It ONLY consumes already-frozen artifacts from Scripts 65–75.

Outputs
-------
Report-ready tables:
    1. retrieval final table
    2. zero-shot final table
    3. human-evaluation final table
    4. key Task-4 results table
    5. Task3-supervised vs Task4-zero-shot comparison

Report-ready figures:
    1. VAL vs TEST retrieval mR across seeds
    2. final TEST bidirectional Recall@K
    3. zero-shot Caption-vs-Audio Macro AP
    4. human-evaluation per-item ratings

Final artifacts:
    - task4_final_summary.json
    - task4_final_audit.json
    - task4_final_artifact_lock.json

The final lock marks Task 4 COMPLETE.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 0. ROOT / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Critical upstream preprocessing artifacts
# -----------------------------------------------------------------------------

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

TARGETS = (
    ROOT
    / "data/processed/labels/"
      "musiccaps_targets.parquet"
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


# -----------------------------------------------------------------------------
# Task-4 design / selection
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Final selected checkpoints
# -----------------------------------------------------------------------------

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
# Script 70 — final validation multi-seed
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Script 71 — pre-TEST lock
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


# -----------------------------------------------------------------------------
# Script 72 — final TEST retrieval
# -----------------------------------------------------------------------------

TEST_MANIFEST = (
    ROOT
    / "results/runs/task4/"
      "task4_test_retrieval_manifest.csv"
)

TEST_RETRIEVAL_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_test_retrieval_per_seed.csv"
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


# -----------------------------------------------------------------------------
# Script 73 — zero-shot
# -----------------------------------------------------------------------------

ZERO_SHOT_VAL_FREEZE = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_val_freeze.json"
)

ZERO_SHOT_MODE_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_test_per_seed_mode.csv"
)

ZERO_SHOT_LABEL_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_test_per_label.csv"
)

ZERO_SHOT_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_test_summary.json"
)

ZERO_SHOT_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_test_lock.json"
)


# -----------------------------------------------------------------------------
# Script 74 — qualitative
# -----------------------------------------------------------------------------

QUALITATIVE_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3.csv"
)

QUALITATIVE_JSON = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3.json"
)

QUALITATIVE_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3_lock.json"
)


# -----------------------------------------------------------------------------
# Script 75 — human evaluation
# -----------------------------------------------------------------------------

HUMAN_PACKAGE = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_package.json"
)

HUMAN_PACKAGE_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_package_lock.json"
)

HUMAN_PER_ITEM = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_per_item.csv"
)

HUMAN_RESULT = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_result.json"
)

HUMAN_RESULT_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_result_lock.json"
)


# -----------------------------------------------------------------------------
# Script 76 outputs
# -----------------------------------------------------------------------------

FINAL_DIR = (
    ROOT
    / "results/runs/task4/final"
)

TABLE_DIR = (
    FINAL_DIR
    / "tables"
)

FIGURE_DIR = (
    FINAL_DIR
    / "figures"
)


RETRIEVAL_TABLE = (
    TABLE_DIR
    / "task4_retrieval_final.csv"
)

ZERO_SHOT_TABLE = (
    TABLE_DIR
    / "task4_zero_shot_final.csv"
)

HUMAN_TABLE = (
    TABLE_DIR
    / "task4_human_eval_final.csv"
)

KEY_RESULTS_TABLE = (
    TABLE_DIR
    / "task4_key_results.csv"
)

TASK3_COMPARISON_TABLE = (
    TABLE_DIR
    / "task3_supervised_vs_task4_zeroshot.csv"
)


FIG_VAL_TEST = (
    FIGURE_DIR
    / "task4_val_vs_test_mR.png"
)

FIG_RECALL = (
    FIGURE_DIR
    / "task4_test_recall_at_k.png"
)

FIG_ZERO_SHOT = (
    FIGURE_DIR
    / "task4_zero_shot_macro_ap.png"
)

FIG_HUMAN = (
    FIGURE_DIR
    / "task4_human_ratings.png"
)


FINAL_SUMMARY = (
    FINAL_DIR
    / "task4_final_summary.json"
)

FINAL_AUDIT = (
    FINAL_DIR
    / "task4_final_audit.json"
)

FINAL_LOCK = (
    FINAL_DIR
    / "task4_final_artifact_lock.json"
)


# =============================================================================
# 1. FROZEN EXPECTED HASHES
# =============================================================================

EXPECTED_HASHES = {

    # -------------------------------------------------------------------------
    # Upstream preprocessing/model lineage
    # -------------------------------------------------------------------------

    "pretraining_lock":
        "caffa90d9de726a6a28fb15b2ee0e28f"
        "4abac8f0766c90d5b0e027a6c917b4e8",

    "canonical_dataset":
        "704e057ba3807886b35000ff23d533d4"
        "759c6a65fdf5d7a8912e618ba523c4d3",

    "token_cache":
        "0df45ab266884abbc2f3f7f96a5ba789"
        "bbf92bf8da49ac4cb57a1e4eb476188f",

    "token_manifest":
        "4be874ab26375ce63bb0f4b55706d178b"
        "fac8a951fee96fd75d2d49b28a390b8",

    "graph_manifest":
        "f092b7ca5f409f3e4fd34c6e3b0bce6a"
        "d0492bf90eccc929cadc1aad1da9e158",

    "targets":
        "a6f80bb42b0b48b2c47dac5f27ad2fd6"
        "6f8893d83bd5d4e552d356cee3547056",

    "T1_checkpoint":
        "3b675acbacb7191d514e3cc513669442"
        "71e0d5a1b4d98f767c0ccab163f4e275",

    "A4_checkpoint":
        "c4e3084f6f7f6030caf157623663d311"
        "daf74ea0a24f6b02e78e5513078fae1e",

    # -------------------------------------------------------------------------
    # Design / selection
    # -------------------------------------------------------------------------

    "design":
        "61ed1263476e631374790c9669ab4aa96"
        "e3b5cb295436d4e273da66d4da556db",

    "design_lock":
        "53f50205e712833ec4efb506b9b6d911"
        "8dcf46f51b72c5907f33cd61ddca612c",

    "model_selection":
        "831e1535cfe6f45a646f1732ecf268db"
        "02c5214aa48404fd07fb3ae8ff5e41c5",

    "model_selection_lock":
        "a1aa9d2420a97c86b0f3709227ff22d1"
        "ee3ccf111010733b2bf705dee5eb6bb6",

    # -------------------------------------------------------------------------
    # Selected checkpoints
    # -------------------------------------------------------------------------

    "checkpoint_42":
        "2be12c7e6fdc851b2b1efae0a7bed2c4"
        "95cbefb5ae90a4f491c54a72ec8e9ace",

    "checkpoint_1337":
        "824a98d52327d513f67cfa21cf702a459"
        "7f51a91c0b17cea1ea7d10628b4b918",

    "checkpoint_2026":
        "a328d029b46ef14a4199cff0604f190f8"
        "06cf953c7c4ad75a0f884210bb743b9",

    # -------------------------------------------------------------------------
    # Script 70
    # -------------------------------------------------------------------------

    "multiseed_csv":
        "f3b09fe49470fb351fcdf9b0fe5f7293"
        "0ddeaeecc9b9e327ad49af599b190b47",

    "multiseed_summary":
        "0bd0bb281448706cd86a124295be8ae68"
        "2d5b32b74d7e517742caec486959297",

    "multiseed_lock":
        "d6ab94b262903c33f28c8eb57c2de2e6"
        "c1e7fd974ea232186eef8ee19a27c674",

    # -------------------------------------------------------------------------
    # Script 71
    # -------------------------------------------------------------------------

    "pretest_protocol":
        "f1f4651c9bffab2295ec8ef44d17f208"
        "4430386585fb2e9fd489d0bc2af93f7c",

    "pretest_lock":
        "dddbd32013dd7e9912e0df0fa72d0352"
        "49abae8f4c83f6313b57a99931f00836",

    # -------------------------------------------------------------------------
    # Script 72
    # -------------------------------------------------------------------------

    "test_manifest":
        "e3e726814fdd890af0951fd1c4726a4d"
        "e5f506180f4ae6982db51e89e1d39431",

    "test_retrieval_csv":
        "6e8e97838898a66e9df8486ea3891257"
        "b9bebe592456d26558d95a7816bf9764",

    "test_retrieval_summary":
        "77e08ce11af9547319515137197ae122c"
        "83224d1a445619c6daef799a722af90",

    "test_retrieval_lock":
        "45a3adc75581b5ddf0f28fe08ecb70c0"
        "0ddb3bd6d3c665f82fe4e4b556d40504",

    # -------------------------------------------------------------------------
    # Script 73
    # -------------------------------------------------------------------------

    "zero_shot_val_freeze":
        "fffa1a96a7b0f9fd684083f44e9bf90d"
        "fa326eafe07923f4ba2697b58ca05aad",

    "zero_shot_mode_csv":
        "683c4468068b983e31bf7239fdab225df"
        "413d0f28446ff83c2af461aed0e5bbe",

    "zero_shot_label_csv":
        "ed8df061d979fd4686c918d212550db34"
        "cb24c99b2bc5379ecedae7a4427d60d",

    "zero_shot_summary":
        "2a288a2921267f617466a8bc4507300c"
        "eb6461a9c1e96bcf41dbfcb68a2319c8",

    "zero_shot_lock":
        "2552be17a05f7349449b9dc6a7f0d1f"
        "8f4382658ab6ebf2f9fcc36e8990d8611",

    # -------------------------------------------------------------------------
    # Script 74
    # -------------------------------------------------------------------------

    "qualitative_csv":
        "e8cb952c96fa1be6bf6ac44959e1e23a"
        "1934216bb38c28ec965c5ab33aacc027",

    "qualitative_json":
        "f3dfd08a0cadab3edd5818f358c49705"
        "87f89077ea18d370fa4a04ba635ef69c",

    "qualitative_lock":
        "8c2c898914674e3bcda0bbfda1d007c"
        "8460af38c7d9eaa19b54712c23d36ae76",

    # -------------------------------------------------------------------------
    # Script 75
    # -------------------------------------------------------------------------

    "human_package":
        "0968fdd35ff5b05d411e29c9b764a401"
        "16adf1788ea5aa58bc56ec14222defc0",

    "human_package_lock":
        "8e172b134eaa9cb9f65ae0473cc6a744"
        "8e144fdaba73c5b8b3f80112b35fa673",

    "human_per_item":
        "d272d3f72f858a2c30a4f0381297b880"
        "add6226edc167130e2ddcb0ec539ec14",

    "human_result":
        "a52db6f229ce400e200526bf4a0a1a958"
        "f2c783d047dd457130da335adad7c34",

    "human_result_lock":
        "8a8884397d07c3c8da001a281212e420"
        "e6ba8d174c113e120b9656d487872b4f",
}


# =============================================================================
# 2. FINAL SCIENTIFIC CONSTANTS
# =============================================================================

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

N_TEST = 514

EXPECTED_QUALITATIVE_QUERIES = 10

EXPECTED_QUALITATIVE_ROWS = 30

MIN_HUMAN_LISTENERS = 5


# Frozen Task-3 selected supervised comparator.
#
# This is NOT recomputed or used for selection in Script 76.
# It is used only to satisfy the final Task-4 zero-shot-vs-supervised
# reporting requirement.
TASK3_SUPERVISED_MODEL = (
    "Task3 Early-Fusion supervised"
)

TASK3_SUPERVISED_TEST_MACRO_AP = (
    0.83191649
)


RETRIEVAL_METRICS = (
    "audio_to_caption_R@1",
    "audio_to_caption_R@5",
    "audio_to_caption_R@10",
    "caption_to_audio_R@1",
    "caption_to_audio_R@5",
    "caption_to_audio_R@10",
)


# =============================================================================
# 3. REPORTING / AUDIT STATE
# =============================================================================

FAILURES: list[str] = []
WARNINGS: list[str] = []

AUDIT_CHECKS: list[
    dict[str, Any]
] = []


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
    category: str = "integrity",
) -> bool:

    condition = bool(
        condition
    )

    AUDIT_CHECKS.append(
        {
            "category":
                category,

            "check":
                message,

            "pass":
                condition,
        }
    )

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

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


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
    frame: pd.DataFrame,
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

    frame.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
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
        "hash_lineage",
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
            "accepted frozen identity"
        ),
        "hash_lineage",
    )


def find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str:

    normalized = {
        str(
            column
        ).strip().lower():
            column

        for column
        in frame.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in normalized:

            return str(
                normalized[
                    key
                ]
            )

    raise RuntimeError(
        "Could not resolve required column. "
        f"Candidates={candidates}; "
        f"columns={list(frame.columns)}"
    )


def close_enough(
    a: float,
    b: float,
    atol: float = 1e-10,
) -> bool:

    return bool(
        math.isclose(
            float(
                a
            ),
            float(
                b
            ),
            rel_tol=0.0,
            abs_tol=atol,
        )
    )


def save_figure_atomic(
    path: Path,
) -> None:

    temporary = (
        path.parent
        / (
            path.stem
            + ".__tmp__"
            + path.suffix
        )
    )

    if temporary.exists():

        raise RuntimeError(
            f"Temporary figure exists: "
            f"{temporary}"
        )

    plt.savefig(
        temporary,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. FINAL OUTPUT PROTECTION"
    )

    check(
        not FINAL_DIR.exists(),
        (
            "final Task-4 output directory "
            "does not already exist"
        ),
        "output_protection",
    )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite a final "
            "Task-4 result"
        )


# =============================================================================
# 6. VERIFY ENTIRE FROZEN LINEAGE
# =============================================================================

def verify_frozen_lineage() -> None:

    banner(
        "2. VERIFY COMPLETE FROZEN TASK-4 LINEAGE"
    )

    artifacts = (

        # Upstream
        (
            PRETRAINING_LOCK,
            EXPECTED_HASHES[
                "pretraining_lock"
            ],
            "pretraining lock",
        ),
        (
            CANONICAL_DATASET,
            EXPECTED_HASHES[
                "canonical_dataset"
            ],
            "canonical dataset",
        ),
        (
            TOKEN_CACHE,
            EXPECTED_HASHES[
                "token_cache"
            ],
            "BERT token cache",
        ),
        (
            TOKEN_MANIFEST,
            EXPECTED_HASHES[
                "token_manifest"
            ],
            "BERT token manifest",
        ),
        (
            GRAPH_MANIFEST,
            EXPECTED_HASHES[
                "graph_manifest"
            ],
            "graph manifest",
        ),
        (
            TARGETS,
            EXPECTED_HASHES[
                "targets"
            ],
            "frozen 20-label targets",
        ),
        (
            T1_CHECKPOINT,
            EXPECTED_HASHES[
                "T1_checkpoint"
            ],
            "selected T1-B checkpoint",
        ),
        (
            A4_CHECKPOINT,
            EXPECTED_HASHES[
                "A4_checkpoint"
            ],
            "selected A4 checkpoint",
        ),

        # Design / selection
        (
            DESIGN,
            EXPECTED_HASHES[
                "design"
            ],
            "Task-4 design",
        ),
        (
            DESIGN_LOCK,
            EXPECTED_HASHES[
                "design_lock"
            ],
            "Task-4 design lock",
        ),
        (
            MODEL_SELECTION,
            EXPECTED_HASHES[
                "model_selection"
            ],
            "Task-4 model selection",
        ),
        (
            MODEL_SELECTION_LOCK,
            EXPECTED_HASHES[
                "model_selection_lock"
            ],
            "Task-4 model-selection lock",
        ),

        # Script 70
        (
            MULTISEED_CSV,
            EXPECTED_HASHES[
                "multiseed_csv"
            ],
            "multi-seed validation CSV",
        ),
        (
            MULTISEED_SUMMARY,
            EXPECTED_HASHES[
                "multiseed_summary"
            ],
            "multi-seed validation summary",
        ),
        (
            MULTISEED_LOCK,
            EXPECTED_HASHES[
                "multiseed_lock"
            ],
            "multi-seed validation lock",
        ),

        # Script 71
        (
            PRETEST_PROTOCOL,
            EXPECTED_HASHES[
                "pretest_protocol"
            ],
            "pre-TEST protocol",
        ),
        (
            PRETEST_LOCK,
            EXPECTED_HASHES[
                "pretest_lock"
            ],
            "pre-TEST protocol lock",
        ),

        # Script 72
        (
            TEST_MANIFEST,
            EXPECTED_HASHES[
                "test_manifest"
            ],
            "final TEST retrieval manifest",
        ),
        (
            TEST_RETRIEVAL_CSV,
            EXPECTED_HASHES[
                "test_retrieval_csv"
            ],
            "final TEST retrieval per-seed CSV",
        ),
        (
            TEST_RETRIEVAL_SUMMARY,
            EXPECTED_HASHES[
                "test_retrieval_summary"
            ],
            "final TEST retrieval summary",
        ),
        (
            TEST_RETRIEVAL_LOCK,
            EXPECTED_HASHES[
                "test_retrieval_lock"
            ],
            "final TEST retrieval lock",
        ),

        # Script 73
        (
            ZERO_SHOT_VAL_FREEZE,
            EXPECTED_HASHES[
                "zero_shot_val_freeze"
            ],
            "zero-shot validation freeze",
        ),
        (
            ZERO_SHOT_MODE_CSV,
            EXPECTED_HASHES[
                "zero_shot_mode_csv"
            ],
            "zero-shot per-seed/mode CSV",
        ),
        (
            ZERO_SHOT_LABEL_CSV,
            EXPECTED_HASHES[
                "zero_shot_label_csv"
            ],
            "zero-shot per-label CSV",
        ),
        (
            ZERO_SHOT_SUMMARY,
            EXPECTED_HASHES[
                "zero_shot_summary"
            ],
            "zero-shot TEST summary",
        ),
        (
            ZERO_SHOT_LOCK,
            EXPECTED_HASHES[
                "zero_shot_lock"
            ],
            "zero-shot TEST lock",
        ),

        # Script 74
        (
            QUALITATIVE_CSV,
            EXPECTED_HASHES[
                "qualitative_csv"
            ],
            "qualitative retrieval CSV",
        ),
        (
            QUALITATIVE_JSON,
            EXPECTED_HASHES[
                "qualitative_json"
            ],
            "qualitative retrieval JSON",
        ),
        (
            QUALITATIVE_LOCK,
            EXPECTED_HASHES[
                "qualitative_lock"
            ],
            "qualitative retrieval lock",
        ),

        # Script 75
        (
            HUMAN_PACKAGE,
            EXPECTED_HASHES[
                "human_package"
            ],
            "human-evaluation package",
        ),
        (
            HUMAN_PACKAGE_LOCK,
            EXPECTED_HASHES[
                "human_package_lock"
            ],
            "human-evaluation package lock",
        ),
        (
            HUMAN_PER_ITEM,
            EXPECTED_HASHES[
                "human_per_item"
            ],
            "human-evaluation per-item CSV",
        ),
        (
            HUMAN_RESULT,
            EXPECTED_HASHES[
                "human_result"
            ],
            "human-evaluation result",
        ),
        (
            HUMAN_RESULT_LOCK,
            EXPECTED_HASHES[
                "human_result_lock"
            ],
            "human-evaluation result lock",
        ),
    )

    for path, digest, label in artifacts:

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
            EXPECTED_HASHES[
                f"checkpoint_{seed}"
            ],
            (
                f"selected T4-B "
                f"seed-{seed} checkpoint"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen Task-4 lineage verification failed"
        )


# =============================================================================
# 7. LOAD ALL FROZEN RESULTS
# =============================================================================

def load_frozen_results() -> dict[str, Any]:

    banner(
        "3. LOAD FROZEN RESULT ARTIFACTS"
    )

    multiseed = pd.read_csv(
        MULTISEED_CSV
    )

    test_retrieval = pd.read_csv(
        TEST_RETRIEVAL_CSV
    )

    zero_shot = pd.read_csv(
        ZERO_SHOT_MODE_CSV
    )

    zero_label = pd.read_csv(
        ZERO_SHOT_LABEL_CSV
    )

    qualitative = pd.read_csv(
        QUALITATIVE_CSV
    )

    human_items = pd.read_csv(
        HUMAN_PER_ITEM
    )

    protocol = load_json(
        PRETEST_PROTOCOL
    )

    selection = load_json(
        MODEL_SELECTION
    )

    multi_summary = load_json(
        MULTISEED_SUMMARY
    )

    retrieval_summary = load_json(
        TEST_RETRIEVAL_SUMMARY
    )

    zero_summary = load_json(
        ZERO_SHOT_SUMMARY
    )

    qualitative_json = load_json(
        QUALITATIVE_JSON
    )

    human_result = load_json(
        HUMAN_RESULT
    )

    check(
        len(
            test_retrieval
        )
        == 3,
        (
            "final TEST retrieval CSV "
            "contains exactly 3 seeds"
        ),
        "result_schema",
    )

    check(
        len(
            zero_shot
        )
        == 6,
        (
            "zero-shot CSV contains "
            "3 seeds × 2 modes"
        ),
        "result_schema",
    )

    check(
        len(
            qualitative
        )
        ==
        EXPECTED_QUALITATIVE_ROWS,
        (
            "qualitative CSV contains "
            "exactly 30 Top-3 rows"
        ),
        "result_schema",
    )

    check(
        len(
            human_items
        )
        ==
        EXPECTED_QUALITATIVE_QUERIES,
        (
            "human per-item CSV contains "
            "exactly 10 items"
        ),
        "result_schema",
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen result schema validation failed"
        )

    return {
        "multiseed":
            multiseed,

        "test_retrieval":
            test_retrieval,

        "zero_shot":
            zero_shot,

        "zero_label":
            zero_label,

        "qualitative":
            qualitative,

        "human_items":
            human_items,

        "protocol":
            protocol,

        "selection":
            selection,

        "multi_summary":
            multi_summary,

        "retrieval_summary":
            retrieval_summary,

        "zero_summary":
            zero_summary,

        "qualitative_json":
            qualitative_json,

        "human_result":
            human_result,
    }


# =============================================================================
# 8. FINAL METHODOLOGY AUDIT
# =============================================================================

def audit_methodology(
    data: dict[str, Any],
) -> None:

    banner(
        "4. FINAL TASK-4 METHODOLOGY AUDIT"
    )

    protocol = (
        data[
            "protocol"
        ]
    )

    selection = (
        data[
            "selection"
        ]
    )

    retrieval = (
        data[
            "retrieval_summary"
        ]
    )

    zero = (
        data[
            "zero_summary"
        ]
    )

    qualitative = (
        data[
            "qualitative_json"
        ]
    )

    human = (
        data[
            "human_result"
        ]
    )

    # -------------------------------------------------------------------------
    # Model selection discipline
    # -------------------------------------------------------------------------

    check(
        selection[
            "selected_variant"
        ]
        == "T4-B",
        (
            "final selected Task-4 "
            "architecture remains T4-B"
        ),
        "model_selection",
    )

    check(
        selection[
            "test_used"
        ]
        is False,
        (
            "Task-4 architecture selection "
            "used validation only"
        ),
        "model_selection",
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
            "final seed set remained "
            "42/1337/2026"
        ),
        "model_selection",
    )

    check(
        protocol[
            "selected_configuration"
        ][
            "best_seed_selection"
        ]
        is False,
        (
            "best-seed selection was "
            "forbidden before TEST"
        ),
        "model_selection",
    )

    # -------------------------------------------------------------------------
    # TEST retrieval discipline
    # -------------------------------------------------------------------------

    check(
        retrieval[
            "seeds"
        ]
        ==
        [
            42,
            1337,
            2026,
        ],
        (
            "final TEST retrieval evaluated "
            "all predetermined seeds"
        ),
        "test_discipline",
    )

    check(
        retrieval[
            "test_pairs"
        ]
        ==
        N_TEST,
        (
            "final TEST candidate pool "
            "contained exactly 514 pairs"
        ),
        "test_discipline",
    )

    check(
        retrieval[
            "best_seed_selection"
        ]
        is False,
        (
            "no TEST-based seed selection "
            "was performed"
        ),
        "test_discipline",
    )

    check(
        retrieval[
            "seed_ensemble"
        ]
        is False,
        (
            "no seed ensemble was introduced"
        ),
        "test_discipline",
    )

    check(
        retrieval[
            "test_labels_loaded"
        ]
        is False,
        (
            "retrieval evaluation loaded "
            "zero TEST tag labels"
        ),
        "test_discipline",
    )

    check(
        retrieval[
            "temperature_used_for_retrieval"
        ]
        is False,
        (
            "retrieval used normalized cosine "
            "without post-hoc temperature"
        ),
        "test_discipline",
    )

    # -------------------------------------------------------------------------
    # Zero-shot discipline
    # -------------------------------------------------------------------------

    check(
        zero[
            "primary_zero_shot_mode"
        ]
        == "caption",
        (
            "primary zero-shot mode remained "
            "Caption→Label"
        ),
        "zero_shot",
    )

    check(
        zero[
            "primary_metric"
        ]
        ==
        "Macro Average Precision",
        (
            "zero-shot primary metric remained "
            "Macro Average Precision"
        ),
        "zero_shot",
    )

    check(
        zero[
            "prompt_engineering"
        ]
        is False,
        (
            "zero-shot label prompts were "
            "not engineered on TEST"
        ),
        "zero_shot",
    )

    check(
        zero[
            "best_seed_selection"
        ]
        is False,
        (
            "zero-shot stage performed "
            "no best-seed selection"
        ),
        "zero_shot",
    )

    check(
        zero[
            "f1_threshold_policy"
        ][
            "test_threshold_tuning"
        ]
        is False,
        (
            "zero-shot F1 thresholds were "
            "not tuned on TEST"
        ),
        "zero_shot",
    )

    check(
        zero[
            "contrastive_training_used_tag_labels"
        ]
        is False,
        (
            "contrastive Task-4 training "
            "used no tag supervision"
        ),
        "zero_shot",
    )

    # -------------------------------------------------------------------------
    # Qualitative discipline
    # -------------------------------------------------------------------------

    check(
        qualitative[
            "query_count"
        ]
        ==
        EXPECTED_QUALITATIVE_QUERIES,
        (
            "qualitative evaluation used "
            "exactly 10 frozen queries"
        ),
        "qualitative",
    )

    check(
        qualitative[
            "model_seed"
        ]
        == 42,
        (
            "qualitative model seed remained "
            "pre-frozen seed 42"
        ),
        "qualitative",
    )

    check(
        qualitative[
            "hand_picking"
        ]
        is False,
        (
            "qualitative examples were "
            "not hand-picked"
        ),
        "qualitative",
    )

    check(
        qualitative[
            "retrieval_rerun"
        ]
        is False,
        (
            "qualitative retrieval reused "
            "the frozen Script-72 result"
        ),
        "qualitative",
    )

    check(
        qualitative[
            "model_inference_rerun"
        ]
        is False,
        (
            "qualitative stage performed "
            "no model re-inference"
        ),
        "qualitative",
    )

    # -------------------------------------------------------------------------
    # Human-evaluation discipline
    # -------------------------------------------------------------------------

    human_stats = (
        human[
            "statistics"
        ]
    )

    check(
        human_stats[
            "listeners"
        ]
        >= MIN_HUMAN_LISTENERS,
        (
            "human evaluation met the "
            "minimum 5-listener requirement"
        ),
        "human_evaluation",
    )

    check(
        human_stats[
            "items"
        ]
        ==
        EXPECTED_QUALITATIVE_QUERIES,
        (
            "human evaluation used the same "
            "10 frozen queries"
        ),
        "human_evaluation",
    )

    check(
        human_stats[
            "total_ratings"
        ]
        ==
        (
            human_stats[
                "listeners"
            ]
            *
            EXPECTED_QUALITATIVE_QUERIES
        ),
        (
            "human rating matrix is complete"
        ),
        "human_evaluation",
    )

    check(
        human[
            "query_hand_picking"
        ]
        is False,
        (
            "human-evaluation queries "
            "were not reselected"
        ),
        "human_evaluation",
    )

    check(
        human[
            "retrieval_rerun"
        ]
        is False,
        (
            "human evaluation reused "
            "frozen Top-1 retrievals"
        ),
        "human_evaluation",
    )

    check(
        human[
            "model_inference_rerun"
        ]
        is False,
        (
            "human evaluation performed "
            "no model re-inference"
        ),
        "human_evaluation",
    )

    if FAILURES:

        raise RuntimeError(
            "Final Task-4 methodology audit failed"
        )


# =============================================================================
# 9. EXTRACT / RECOMPUTE FINAL NUMBERS
# =============================================================================

def calculate_final_results(
    data: dict[str, Any],
) -> dict[str, Any]:

    banner(
        "5. RECOMPUTE FINAL AGGREGATE STATISTICS"
    )

    # -------------------------------------------------------------------------
    # Script 70 validation mR
    # -------------------------------------------------------------------------

    multiseed = (
        data[
            "multiseed"
        ].copy()
    )

    val_seed_col = find_column(
        multiseed,
        (
            "seed",
        ),
    )

    val_mr_col = find_column(
        multiseed,
        (
            "mR",
            "val_mR",
            "best_val_mR",
            "validation_mR",
        ),
    )

    multiseed[
        val_seed_col
    ] = pd.to_numeric(
        multiseed[
            val_seed_col
        ]
    ).astype(
        int
    )

    val_by_seed = {
        int(
            row[
                val_seed_col
            ]
        ):
            float(
                row[
                    val_mr_col
                ]
            )

        for _, row
        in multiseed.iterrows()
    }

    check(
        set(
            val_by_seed
        )
        ==
        set(
            FINAL_SEEDS
        ),
        (
            "validation table contains "
            "exactly seeds 42/1337/2026"
        ),
        "metric_arithmetic",
    )

    val_values = np.asarray(
        [
            val_by_seed[
                seed
            ]

            for seed
            in FINAL_SEEDS
        ],
        dtype=np.float64,
    )

    val_mean = float(
        np.mean(
            val_values
        )
    )

    val_std = float(
        np.std(
            val_values,
            ddof=0,
        )
    )

    # -------------------------------------------------------------------------
    # Script 72 TEST retrieval
    # -------------------------------------------------------------------------

    test = (
        data[
            "test_retrieval"
        ].copy()
    )

    test[
        "seed"
    ] = pd.to_numeric(
        test[
            "seed"
        ]
    ).astype(
        int
    )

    check(
        set(
            test[
                "seed"
            ].tolist()
        )
        ==
        set(
            FINAL_SEEDS
        ),
        (
            "TEST retrieval table contains "
            "exactly final 3 seeds"
        ),
        "metric_arithmetic",
    )

    test = (
        test
        .set_index(
            "seed"
        )
        .loc[
            list(
                FINAL_SEEDS
            )
        ]
        .reset_index()
    )

    test_mr_values = (
        test[
            "mR"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    test_mr_mean = float(
        np.mean(
            test_mr_values
        )
    )

    test_mr_std = float(
        np.std(
            test_mr_values,
            ddof=0,
        )
    )

    frozen_test_aggregate = (
        data[
            "retrieval_summary"
        ][
            "aggregate_three_seed"
        ][
            "mR"
        ]
    )

    check(
        close_enough(
            test_mr_mean,
            frozen_test_aggregate[
                "mean"
            ],
            atol=1e-12,
        ),
        (
            "recomputed three-seed TEST mR "
            "mean matches frozen summary"
        ),
        "metric_arithmetic",
    )

    check(
        close_enough(
            test_mr_std,
            frozen_test_aggregate[
                "std_population"
            ],
            atol=1e-12,
        ),
        (
            "recomputed three-seed TEST mR "
            "population SD matches frozen summary"
        ),
        "metric_arithmetic",
    )

    # -------------------------------------------------------------------------
    # Retrieval recalls
    # -------------------------------------------------------------------------

    retrieval_aggregate = {}

    for metric in RETRIEVAL_METRICS:

        values = (
            test[
                metric
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        retrieval_aggregate[
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
        }

    random_baseline = (
        data[
            "retrieval_summary"
        ][
            "random_baseline"
        ]
    )

    random_mr = float(
        np.mean(
            [
                random_baseline[
                    "R@1"
                ],
                random_baseline[
                    "R@5"
                ],
                random_baseline[
                    "R@10"
                ],
            ],
            dtype=np.float64,
        )
    )

    retrieval_multiple_over_random = (
        test_mr_mean
        / random_mr
    )

    # -------------------------------------------------------------------------
    # Zero-shot
    # -------------------------------------------------------------------------

    zero = (
        data[
            "zero_shot"
        ].copy()
    )

    zero[
        "seed"
    ] = pd.to_numeric(
        zero[
            "seed"
        ]
    ).astype(
        int
    )

    caption_rows = (
        zero.loc[
            zero[
                "mode"
            ]
            ==
            "caption"
        ]
        .copy()
    )

    audio_rows = (
        zero.loc[
            zero[
                "mode"
            ]
            ==
            "audio"
        ]
        .copy()
    )

    check(
        set(
            caption_rows[
                "seed"
            ]
        )
        ==
        set(
            FINAL_SEEDS
        ),
        (
            "caption zero-shot results cover "
            "all 3 predetermined seeds"
        ),
        "metric_arithmetic",
    )

    check(
        set(
            audio_rows[
                "seed"
            ]
        )
        ==
        set(
            FINAL_SEEDS
        ),
        (
            "audio zero-shot results cover "
            "all 3 predetermined seeds"
        ),
        "metric_arithmetic",
    )

    caption_values = (
        caption_rows[
            "test_macro_ap"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    audio_values = (
        audio_rows[
            "test_macro_ap"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    caption_macro_ap_mean = float(
        np.mean(
            caption_values
        )
    )

    caption_macro_ap_std = float(
        np.std(
            caption_values,
            ddof=0,
        )
    )

    audio_macro_ap_mean = float(
        np.mean(
            audio_values
        )
    )

    audio_macro_ap_std = float(
        np.std(
            audio_values,
            ddof=0,
        )
    )

    frozen_caption = (
        data[
            "zero_summary"
        ][
            "aggregate_three_seed"
        ][
            "caption"
        ][
            "test_macro_ap"
        ]
    )

    frozen_audio = (
        data[
            "zero_summary"
        ][
            "aggregate_three_seed"
        ][
            "audio"
        ][
            "test_macro_ap"
        ]
    )

    check(
        close_enough(
            caption_macro_ap_mean,
            frozen_caption[
                "mean"
            ],
            atol=1e-12,
        ),
        (
            "recomputed caption zero-shot "
            "Macro AP mean matches frozen summary"
        ),
        "metric_arithmetic",
    )

    check(
        close_enough(
            audio_macro_ap_mean,
            frozen_audio[
                "mean"
            ],
            atol=1e-12,
        ),
        (
            "recomputed audio zero-shot "
            "Macro AP mean matches frozen summary"
        ),
        "metric_arithmetic",
    )

    # -------------------------------------------------------------------------
    # Human
    # -------------------------------------------------------------------------

    human_stats = (
        data[
            "human_result"
        ][
            "statistics"
        ]
    )

    human_mean = float(
        human_stats[
            "pooled_rating"
        ][
            "mean"
        ]
    )

    human_sd = float(
        human_stats[
            "pooled_rating"
        ][
            "std_sample"
        ]
    )

    # -------------------------------------------------------------------------
    # Task-3 supervised comparison
    # -------------------------------------------------------------------------

    task3_gap = (
        caption_macro_ap_mean
        -
        TASK3_SUPERVISED_TEST_MACRO_AP
    )

    task4_fraction_of_task3 = (
        caption_macro_ap_mean
        /
        TASK3_SUPERVISED_TEST_MACRO_AP
    )

    if FAILURES:

        raise RuntimeError(
            "Final metric arithmetic validation failed"
        )

    return {
        "validation_retrieval": {
            "mR_by_seed":
                {
                    str(
                        seed
                    ):
                        val_by_seed[
                            seed
                        ]

                    for seed
                    in FINAL_SEEDS
                },

            "mR_mean":
                val_mean,

            "mR_std_population":
                val_std,
        },

        "test_retrieval": {
            "mR_by_seed":
                {
                    str(
                        int(
                            row[
                                "seed"
                            ]
                        )
                    ):
                        float(
                            row[
                                "mR"
                            ]
                        )

                    for _, row
                    in test.iterrows()
                },

            "mR_mean":
                test_mr_mean,

            "mR_std_population":
                test_mr_std,

            "recall_aggregate":
                retrieval_aggregate,

            "random_mR":
                random_mr,

            "mR_multiple_over_random":
                float(
                    retrieval_multiple_over_random
                ),

            "absolute_generalization_gap":
                float(
                    test_mr_mean
                    -
                    val_mean
                ),

            "test_to_validation_ratio":
                float(
                    test_mr_mean
                    /
                    val_mean
                ),
        },

        "zero_shot": {
            "caption_macro_ap_mean":
                caption_macro_ap_mean,

            "caption_macro_ap_std_population":
                caption_macro_ap_std,

            "audio_macro_ap_mean":
                audio_macro_ap_mean,

            "audio_macro_ap_std_population":
                audio_macro_ap_std,

            "caption_minus_audio_macro_ap":
                float(
                    caption_macro_ap_mean
                    -
                    audio_macro_ap_mean
                ),

            "caption_to_audio_macro_ap_ratio":
                float(
                    caption_macro_ap_mean
                    /
                    audio_macro_ap_mean
                ),
        },

        "task3_supervised_comparison": {
            "task3_model":
                TASK3_SUPERVISED_MODEL,

            "task3_supervised_test_macro_ap":
                TASK3_SUPERVISED_TEST_MACRO_AP,

            "task4_caption_zero_shot_test_macro_ap_mean":
                caption_macro_ap_mean,

            "task4_minus_task3":
                float(
                    task3_gap
                ),

            "task4_fraction_of_task3":
                float(
                    task4_fraction_of_task3
                ),

            "task3_recomputed_in_script76":
                False,

            "task3_reselected_in_script76":
                False,
        },

        "human_evaluation": {
            "listeners":
                int(
                    human_stats[
                        "listeners"
                    ]
                ),

            "total_ratings":
                int(
                    human_stats[
                        "total_ratings"
                    ]
                ),

            "pooled_mean":
                human_mean,

            "pooled_sd_sample":
                human_sd,

            "listener_mean":
                float(
                    human_stats[
                        "listener_mean_distribution"
                    ][
                        "mean"
                    ]
                ),

            "listener_mean_sd_sample":
                float(
                    human_stats[
                        "listener_mean_distribution"
                    ][
                        "std_sample"
                    ]
                ),
        },
    }


# =============================================================================
# 10. CREATE REPORT TABLES
# =============================================================================

def create_tables(
    data: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Path]:

    banner(
        "6. CREATE FINAL REPORT TABLES"
    )

    # -------------------------------------------------------------------------
    # Retrieval table
    # -------------------------------------------------------------------------

    test = (
        data[
            "test_retrieval"
        ].copy()
    )

    test[
        "seed"
    ] = pd.to_numeric(
        test[
            "seed"
        ]
    ).astype(
        int
    )

    val_by_seed = {
        int(
            seed
        ):
            float(
                value
            )

        for seed, value
        in results[
            "validation_retrieval"
        ][
            "mR_by_seed"
        ].items()
    }

    rows = []

    for seed in FINAL_SEEDS:

        source = (
            test.loc[
                test[
                    "seed"
                ]
                ==
                seed
            ]
            .iloc[
                0
            ]
        )

        row = {
            "seed":
                seed,

            "VAL_mR":
                val_by_seed[
                    seed
                ],

            "TEST_Audio_to_Caption_R@1":
                float(
                    source[
                        "audio_to_caption_R@1"
                    ]
                ),

            "TEST_Audio_to_Caption_R@5":
                float(
                    source[
                        "audio_to_caption_R@5"
                    ]
                ),

            "TEST_Audio_to_Caption_R@10":
                float(
                    source[
                        "audio_to_caption_R@10"
                    ]
                ),

            "TEST_Caption_to_Audio_R@1":
                float(
                    source[
                        "caption_to_audio_R@1"
                    ]
                ),

            "TEST_Caption_to_Audio_R@5":
                float(
                    source[
                        "caption_to_audio_R@5"
                    ]
                ),

            "TEST_Caption_to_Audio_R@10":
                float(
                    source[
                        "caption_to_audio_R@10"
                    ]
                ),

            "TEST_mR":
                float(
                    source[
                        "mR"
                    ]
                ),
        }

        rows.append(
            row
        )

    retrieval_frame = (
        pd.DataFrame(
            rows
        )
    )

    numeric_columns = [
        column
        for column
        in retrieval_frame.columns
        if column
        !=
        "seed"
    ]

    mean_row = {
        "seed":
            "MEAN"
    }

    std_row = {
        "seed":
            "SD_population"
    }

    for column in numeric_columns:

        values = (
            retrieval_frame[
                column
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        mean_row[
            column
        ] = float(
            np.mean(
                values
            )
        )

        std_row[
            column
        ] = float(
            np.std(
                values,
                ddof=0,
            )
        )

    retrieval_frame = pd.concat(
        [
            retrieval_frame,
            pd.DataFrame(
                [
                    mean_row,
                    std_row,
                ]
            ),
        ],
        ignore_index=True,
    )

    write_csv_atomic(
        RETRIEVAL_TABLE,
        retrieval_frame,
    )

    passed(
        "final retrieval table written"
    )

    # -------------------------------------------------------------------------
    # Zero-shot table
    # -------------------------------------------------------------------------

    zero = (
        data[
            "zero_shot"
        ].copy()
    )

    zero = zero[
        [
            "seed",
            "mode",
            "val_selected_global_threshold",
            "test_macro_ap",
            "test_micro_ap",
            "test_macro_f1",
            "test_micro_f1",
        ]
    ].copy()

    write_csv_atomic(
        ZERO_SHOT_TABLE,
        zero,
    )

    passed(
        "final zero-shot table written"
    )

    # -------------------------------------------------------------------------
    # Human table
    # -------------------------------------------------------------------------

    human = (
        data[
            "human_items"
        ].copy()
    )

    write_csv_atomic(
        HUMAN_TABLE,
        human,
    )

    passed(
        "final human-evaluation table written"
    )

    # -------------------------------------------------------------------------
    # Key results table
    # -------------------------------------------------------------------------

    key_rows = [
        {
            "result":
                "Task4 VAL retrieval mR",

            "mean":
                results[
                    "validation_retrieval"
                ][
                    "mR_mean"
                ],

            "spread":
                results[
                    "validation_retrieval"
                ][
                    "mR_std_population"
                ],

            "spread_type":
                "population SD",

            "scale":
                "0-1",
        },
        {
            "result":
                "Task4 TEST retrieval mR",

            "mean":
                results[
                    "test_retrieval"
                ][
                    "mR_mean"
                ],

            "spread":
                results[
                    "test_retrieval"
                ][
                    "mR_std_population"
                ],

            "spread_type":
                "population SD",

            "scale":
                "0-1",
        },
        {
            "result":
                "Task4 zero-shot Caption→Label Macro AP",

            "mean":
                results[
                    "zero_shot"
                ][
                    "caption_macro_ap_mean"
                ],

            "spread":
                results[
                    "zero_shot"
                ][
                    "caption_macro_ap_std_population"
                ],

            "spread_type":
                "population SD",

            "scale":
                "0-1",
        },
        {
            "result":
                "Task4 zero-shot Audio→Label Macro AP",

            "mean":
                results[
                    "zero_shot"
                ][
                    "audio_macro_ap_mean"
                ],

            "spread":
                results[
                    "zero_shot"
                ][
                    "audio_macro_ap_std_population"
                ],

            "spread_type":
                "population SD",

            "scale":
                "0-1",
        },
        {
            "result":
                "Human caption-audio match rating",

            "mean":
                results[
                    "human_evaluation"
                ][
                    "pooled_mean"
                ],

            "spread":
                results[
                    "human_evaluation"
                ][
                    "pooled_sd_sample"
                ],

            "spread_type":
                "sample SD",

            "scale":
                "1-5",
        },
    ]

    key_frame = pd.DataFrame(
        key_rows
    )

    write_csv_atomic(
        KEY_RESULTS_TABLE,
        key_frame,
    )

    passed(
        "key Task-4 results table written"
    )

    # -------------------------------------------------------------------------
    # Task3 supervised vs Task4 zero-shot
    # -------------------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "system":
                    TASK3_SUPERVISED_MODEL,

                "evaluation":
                    "supervised",

                "TEST_Macro_AP":
                    TASK3_SUPERVISED_TEST_MACRO_AP,

                "recomputed_in_script76":
                    False,
            },
            {
                "system":
                    (
                        "Task4 T4-B "
                        "Caption→Label zero-shot"
                    ),

                "evaluation":
                    "zero-shot",

                "TEST_Macro_AP":
                    results[
                        "zero_shot"
                    ][
                        "caption_macro_ap_mean"
                    ],

                "recomputed_in_script76":
                    False,
            },
        ]
    )

    write_csv_atomic(
        TASK3_COMPARISON_TABLE,
        comparison,
    )

    passed(
        "Task3 supervised vs Task4 "
        "zero-shot comparison table written"
    )

    return {
        "retrieval_table":
            RETRIEVAL_TABLE,

        "zero_shot_table":
            ZERO_SHOT_TABLE,

        "human_table":
            HUMAN_TABLE,

        "key_results_table":
            KEY_RESULTS_TABLE,

        "task3_comparison_table":
            TASK3_COMPARISON_TABLE,
    }


# =============================================================================
# 11. CREATE REPORT FIGURES
# =============================================================================

def create_figures(
    data: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Path]:

    banner(
        "7. CREATE FINAL REPORT FIGURES"
    )

    # -------------------------------------------------------------------------
    # Figure 1 — VAL vs TEST mR
    # -------------------------------------------------------------------------

    seeds = np.asarray(
        FINAL_SEEDS,
        dtype=np.int64,
    )

    val = np.asarray(
        [
            results[
                "validation_retrieval"
            ][
                "mR_by_seed"
            ][
                str(
                    seed
                )
            ]

            for seed
            in FINAL_SEEDS
        ],
        dtype=np.float64,
    )

    test = np.asarray(
        [
            results[
                "test_retrieval"
            ][
                "mR_by_seed"
            ][
                str(
                    seed
                )
            ]

            for seed
            in FINAL_SEEDS
        ],
        dtype=np.float64,
    )

    x = np.arange(
        len(
            seeds
        )
    )

    width = 0.36

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.bar(
        x
        -
        width
        /
        2,
        val,
        width,
        label="Validation",
    )

    plt.bar(
        x
        +
        width
        /
        2,
        test,
        width,
        label="TEST",
    )

    plt.xticks(
        x,
        [
            str(
                seed
            )
            for seed
            in seeds
        ],
    )

    plt.xlabel(
        "Seed"
    )

    plt.ylabel(
        "Mean Recall (mR)"
    )

    plt.title(
        "Task 4 Retrieval: Validation vs TEST"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure_atomic(
        FIG_VAL_TEST
    )

    passed(
        "VAL-vs-TEST retrieval figure written"
    )

    # -------------------------------------------------------------------------
    # Figure 2 — TEST Recall@K
    # -------------------------------------------------------------------------

    metric_labels = [
        "A→C R@1",
        "A→C R@5",
        "A→C R@10",
        "C→A R@1",
        "C→A R@5",
        "C→A R@10",
    ]

    means = np.asarray(
        [
            results[
                "test_retrieval"
            ][
                "recall_aggregate"
            ][
                metric
            ][
                "mean"
            ]

            for metric
            in RETRIEVAL_METRICS
        ],
        dtype=np.float64,
    )

    stds = np.asarray(
        [
            results[
                "test_retrieval"
            ][
                "recall_aggregate"
            ][
                metric
            ][
                "std_population"
            ]

            for metric
            in RETRIEVAL_METRICS
        ],
        dtype=np.float64,
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        np.arange(
            len(
                means
            )
        ),
        means,
        yerr=stds,
        capsize=4,
    )

    plt.xticks(
        np.arange(
            len(
                means
            )
        ),
        metric_labels,
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        (
            "Task 4 Final TEST Retrieval "
            "(mean ± population SD, 3 seeds)"
        )
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure_atomic(
        FIG_RECALL
    )

    passed(
        "final TEST Recall@K figure written"
    )

    # -------------------------------------------------------------------------
    # Figure 3 — zero-shot Macro AP
    # -------------------------------------------------------------------------

    zero_means = [
        results[
            "zero_shot"
        ][
            "caption_macro_ap_mean"
        ],
        results[
            "zero_shot"
        ][
            "audio_macro_ap_mean"
        ],
    ]

    zero_stds = [
        results[
            "zero_shot"
        ][
            "caption_macro_ap_std_population"
        ],
        results[
            "zero_shot"
        ][
            "audio_macro_ap_std_population"
        ],
    ]

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.bar(
        [
            0,
            1,
        ],
        zero_means,
        yerr=zero_stds,
        capsize=5,
    )

    plt.xticks(
        [
            0,
            1,
        ],
        [
            "Caption → Label",
            "Audio → Label",
        ],
    )

    plt.ylabel(
        "Macro Average Precision"
    )

    plt.title(
        (
            "Task 4 Zero-Shot Tag Prediction "
            "(mean ± population SD)"
        )
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure_atomic(
        FIG_ZERO_SHOT
    )

    passed(
        "zero-shot Macro AP figure written"
    )

    # -------------------------------------------------------------------------
    # Figure 4 — human ratings
    # -------------------------------------------------------------------------

    human = (
        data[
            "human_items"
        ].copy()
    )

    human = (
        human.sort_values(
            "item_id"
        )
        .reset_index(
            drop=True
        )
    )

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        np.arange(
            len(
                human
            )
        ),
        human[
            "mean_rating"
        ].to_numpy(
            dtype=np.float64
        ),
        yerr=human[
            "std_sample"
        ].to_numpy(
            dtype=np.float64
        ),
        capsize=4,
    )

    plt.xticks(
        np.arange(
            len(
                human
            )
        ),
        human[
            "item_id"
        ].tolist(),
    )

    plt.ylim(
        0,
        5.5,
    )

    plt.xlabel(
        "Frozen Query"
    )

    plt.ylabel(
        "Human Match Rating (1–5)"
    )

    plt.title(
        "Task 4 Human Evaluation per Frozen Query"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_figure_atomic(
        FIG_HUMAN
    )

    passed(
        "human-evaluation figure written"
    )

    return {
        "val_vs_test":
            FIG_VAL_TEST,

        "test_recall":
            FIG_RECALL,

        "zero_shot":
            FIG_ZERO_SHOT,

        "human":
            FIG_HUMAN,
    }


# =============================================================================
# 12. WRITE FINAL AUDIT
# =============================================================================

def write_final_audit(
    results: dict[str, Any],
) -> str:

    banner(
        "8. WRITE FINAL TASK-4 AUDIT"
    )

    audit = {
        "artifact_type":
            "task4_final_methodology_audit",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            (
                "task4_complete_"
                "all_required_stages_passed"
            ),

        "failures":
            len(
                FAILURES
            ),

        "warnings":
            len(
                WARNINGS
            ),

        "checks":
            AUDIT_CHECKS,

        "final_scientific_state": {
            "selected_variant":
                "T4-B",

            "final_seeds":
                [
                    42,
                    1337,
                    2026,
                ],

            "architecture_selected_on":
                "validation_only",

            "best_seed_selected":
                False,

            "seed_ensemble":
                False,

            "test_retraining":
                False,

            "test_reselection":
                False,

            "test_reranking":
                False,

            "zero_shot_prompt_engineering":
                False,

            "zero_shot_test_threshold_tuning":
                False,

            "qualitative_hand_picking":
                False,

            "qualitative_retrieval_rerun":
                False,

            "human_query_reselection":
                False,

            "human_retrieval_rerun":
                False,
        },

        "final_results":
            results,

        "known_project_level_limitations": [
            (
                "MusicCaps caption/aspect annotations "
                "are semantically related; supervised "
                "tag classification therefore has a "
                "possible lexical/semantic shortcut risk."
            ),
            (
                "Artist/recording near-duplicate leakage "
                "across splits has not been independently "
                "verified."
            ),
            (
                "381 unavailable MusicCaps clips were "
                "excluded; resulting acquisition-selection "
                "bias was not separately quantified."
            ),
            (
                "Human evaluation uses the frozen 10-query "
                "subset and five listeners, so it is a "
                "small qualitative validation rather than "
                "a population-level listening study."
            ),
        ],

        "script76": {
            "path":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),
        },
    }

    write_json_atomic(
        FINAL_AUDIT,
        audit,
    )

    digest = sha256_file(
        FINAL_AUDIT
    )

    passed(
        "final methodology audit written"
    )

    return digest


# =============================================================================
# 13. WRITE FINAL SUMMARY
# =============================================================================

def write_final_summary(
    results: dict[str, Any],
    tables: dict[str, Path],
    figures: dict[str, Path],
    audit_sha: str,
) -> str:

    banner(
        "9. WRITE FINAL TASK-4 SUMMARY"
    )

    summary = {
        "artifact_type":
            "task4_final_results_summary",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "TASK4_COMPLETE",

        "selected_variant":
            "T4-B",

        "final_seeds":
            [
                42,
                1337,
                2026,
            ],

        "retrieval": {
            "validation_mR":
                {
                    "mean":
                        results[
                            "validation_retrieval"
                        ][
                            "mR_mean"
                        ],

                    "std_population":
                        results[
                            "validation_retrieval"
                        ][
                            "mR_std_population"
                        ],
                },

            "test_mR":
                {
                    "mean":
                        results[
                            "test_retrieval"
                        ][
                            "mR_mean"
                        ],

                    "std_population":
                        results[
                            "test_retrieval"
                        ][
                            "mR_std_population"
                        ],
                },

            "test_recall":
                results[
                    "test_retrieval"
                ][
                    "recall_aggregate"
                ],

            "random_mR":
                results[
                    "test_retrieval"
                ][
                    "random_mR"
                ],

            "multiple_over_random_mR":
                results[
                    "test_retrieval"
                ][
                    "mR_multiple_over_random"
                ],
        },

        "zero_shot": {
            "primary":
                "Caption→Label",

            "primary_metric":
                "Macro Average Precision",

            "caption_macro_ap":
                {
                    "mean":
                        results[
                            "zero_shot"
                        ][
                            "caption_macro_ap_mean"
                        ],

                    "std_population":
                        results[
                            "zero_shot"
                        ][
                            "caption_macro_ap_std_population"
                        ],
                },

            "audio_macro_ap":
                {
                    "mean":
                        results[
                            "zero_shot"
                        ][
                            "audio_macro_ap_mean"
                        ],

                    "std_population":
                        results[
                            "zero_shot"
                        ][
                            "audio_macro_ap_std_population"
                        ],
                },

            "prompt_engineering":
                False,

            "test_threshold_tuning":
                False,
        },

        "supervised_comparison":
            results[
                "task3_supervised_comparison"
            ],

        "qualitative": {
            "queries":
                10,

            "top_k":
                3,

            "model_seed":
                42,

            "query_selection":
                "pre-TEST frozen",

            "hand_picking":
                False,
        },

        "human_evaluation":
            results[
                "human_evaluation"
            ],

        "reporting_interpretation": {
            "retrieval":
                (
                    "The contrastive T4-B model retrieves "
                    "paired cross-modal items substantially "
                    "above the 514-item random baseline, "
                    "while TEST performance is lower than "
                    "validation performance."
                ),

            "zero_shot":
                (
                    "Caption embeddings transfer more "
                    "strongly to canonical text-label "
                    "prototypes than audio embeddings, "
                    "but zero-shot Macro AP remains far "
                    "below the frozen supervised Task-3 "
                    "classifier."
                ),

            "human":
                (
                    "The mean human score is approximately "
                    "moderate match, with substantial "
                    "listener/item-level variability."
                ),
        },

        "tables": {
            name: {
                "path":
                    relative(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }

            for name, path
            in tables.items()
        },

        "figures": {
            name: {
                "path":
                    relative(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }

            for name, path
            in figures.items()
        },

        "audit": {
            "path":
                relative(
                    FINAL_AUDIT
                ),

            "sha256":
                audit_sha,
        },

        "next_action":
            (
                "Use frozen Tasks 1–4 artifacts "
                "for final report/demo. "
                "Do not modify experimental results."
            ),
    }

    write_json_atomic(
        FINAL_SUMMARY,
        summary,
    )

    digest = sha256_file(
        FINAL_SUMMARY
    )

    passed(
        "final Task-4 summary written"
    )

    return digest


# =============================================================================
# 14. FINAL ARTIFACT LOCK
# =============================================================================

def write_final_lock(
    tables: dict[str, Path],
    figures: dict[str, Path],
    summary_sha: str,
    audit_sha: str,
) -> str:

    banner(
        "10. WRITE TERMINAL TASK-4 ARTIFACT LOCK"
    )

    source_artifacts = {
        "design":
            DESIGN,

        "design_lock":
            DESIGN_LOCK,

        "model_selection":
            MODEL_SELECTION,

        "model_selection_lock":
            MODEL_SELECTION_LOCK,

        "multiseed_csv":
            MULTISEED_CSV,

        "multiseed_summary":
            MULTISEED_SUMMARY,

        "multiseed_lock":
            MULTISEED_LOCK,

        "pretest_protocol":
            PRETEST_PROTOCOL,

        "pretest_lock":
            PRETEST_LOCK,

        "test_manifest":
            TEST_MANIFEST,

        "test_retrieval_csv":
            TEST_RETRIEVAL_CSV,

        "test_retrieval_summary":
            TEST_RETRIEVAL_SUMMARY,

        "test_retrieval_lock":
            TEST_RETRIEVAL_LOCK,

        "zero_shot_val_freeze":
            ZERO_SHOT_VAL_FREEZE,

        "zero_shot_mode_csv":
            ZERO_SHOT_MODE_CSV,

        "zero_shot_label_csv":
            ZERO_SHOT_LABEL_CSV,

        "zero_shot_summary":
            ZERO_SHOT_SUMMARY,

        "zero_shot_lock":
            ZERO_SHOT_LOCK,

        "qualitative_csv":
            QUALITATIVE_CSV,

        "qualitative_json":
            QUALITATIVE_JSON,

        "qualitative_lock":
            QUALITATIVE_LOCK,

        "human_package":
            HUMAN_PACKAGE,

        "human_package_lock":
            HUMAN_PACKAGE_LOCK,

        "human_per_item":
            HUMAN_PER_ITEM,

        "human_result":
            HUMAN_RESULT,

        "human_result_lock":
            HUMAN_RESULT_LOCK,

        "checkpoint_42":
            CHECKPOINTS[
                42
            ],

        "checkpoint_1337":
            CHECKPOINTS[
                1337
            ],

        "checkpoint_2026":
            CHECKPOINTS[
                2026
            ],
    }

    final_outputs = {
        "summary":
            FINAL_SUMMARY,

        "audit":
            FINAL_AUDIT,

        **{
            f"table_{name}":
                path

            for name, path
            in tables.items()
        },

        **{
            f"figure_{name}":
                path

            for name, path
            in figures.items()
        },
    }

    lock = {
        "lock_type":
            "task4_terminal_artifact_lock",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "TASK4_FINAL_LOCKED",

        "selected_variant":
            "T4-B",

        "final_seeds":
            [
                42,
                1337,
                2026,
            ],

        "architecture_reselection_allowed":
            False,

        "checkpoint_replacement_allowed":
            False,

        "seed_replacement_allowed":
            False,

        "test_retraining_allowed":
            False,

        "test_metric_redefinition_allowed":
            False,

        "source_artifacts": {
            name: {
                "path":
                    relative(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }

            for name, path
            in source_artifacts.items()
        },

        "final_outputs": {
            name: {
                "path":
                    relative(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }

            for name, path
            in final_outputs.items()
        },

        "final_summary_sha256":
            summary_sha,

        "final_audit_sha256":
            audit_sha,

        "script76": {
            "path":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),
        },
    }

    write_json_atomic(
        FINAL_LOCK,
        lock,
    )

    digest = sha256_file(
        FINAL_LOCK
    )

    check(
        load_json(
            FINAL_LOCK
        )[
            "final_summary_sha256"
        ]
        ==
        sha256_file(
            FINAL_SUMMARY
        ),
        (
            "terminal Task-4 lock matches "
            "final summary"
        ),
        "final_lock",
    )

    check(
        load_json(
            FINAL_LOCK
        )[
            "final_audit_sha256"
        ]
        ==
        sha256_file(
            FINAL_AUDIT
        ),
        (
            "terminal Task-4 lock matches "
            "final audit"
        ),
        "final_lock",
    )

    return digest


# =============================================================================
# 15. FINAL OUTPUT VERIFICATION
# =============================================================================

def verify_final_outputs(
    tables: dict[str, Path],
    figures: dict[str, Path],
) -> None:

    banner(
        "11. VERIFY FINAL OUTPUT SET"
    )

    required = [
        FINAL_SUMMARY,
        FINAL_AUDIT,
        FINAL_LOCK,
        *tables.values(),
        *figures.values(),
    ]

    for path in required:

        check(
            path.is_file(),
            (
                f"final output exists: "
                f"{relative(path)}"
            ),
            "final_outputs",
        )

        if path.is_file():

            check(
                path.stat().st_size
                >
                0,
                (
                    f"final output non-empty: "
                    f"{path.name}"
                ),
                "final_outputs",
            )

    if FAILURES:

        raise RuntimeError(
            "Final Task-4 output verification failed"
        )


# =============================================================================
# 16. FINAL DISCIPLINE CONFIRMATION
# =============================================================================

def print_final_boundary() -> None:

    banner(
        "12. FINAL EXPERIMENTAL DISCIPLINE CONFIRMATION"
    )

    print(
        "Script 76 instantiated BERT:               NO"
    )

    print(
        "Script 76 instantiated GAT:                NO"
    )

    print(
        "Script 76 ran model inference:             NO"
    )

    print(
        "Script 76 retrained any model:             NO"
    )

    print(
        "Script 76 reopened raw TEST inputs:        NO"
    )

    print(
        "Script 76 changed model selection:         NO"
    )

    print(
        "Script 76 selected a best seed:            NO"
    )

    print(
        "Script 76 changed retrieval metrics:       NO"
    )

    print(
        "Script 76 changed zero-shot thresholds:    NO"
    )

    print(
        "Script 76 changed qualitative queries:     NO"
    )

    print(
        "Script 76 changed human-eval queries:      NO"
    )

    print(
        "Task-3 supervised model rerun:             NO"
    )

    passed(
        (
            "Script 76 is a read-only scientific "
            "consolidation of frozen experiments"
        )
    )


# =============================================================================
# 17. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 76 — FINAL TASK-4 "
        "RESULTS / FIGURES / AUDIT / LOCK"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "Task-4 selected configuration:"
    )

    print(
        "  T4-B"
    )

    print()
    print(
        "Final seeds:"
    )

    print(
        "  42 / 1337 / 2026"
    )

    print()
    print(
        "Model inference:       NO"
    )

    print(
        "Retraining:            NO"
    )

    print(
        "TEST reopening:        NO"
    )

    print(
        "Best-seed selection:   NO"
    )

    try:

        verify_output_protection()

        verify_frozen_lineage()

        data = (
            load_frozen_results()
        )

        audit_methodology(
            data
        )

        results = (
            calculate_final_results(
                data
            )
        )

        # Create directories only after
        # all frozen source checks have passed.
        TABLE_DIR.mkdir(
            parents=True,
            exist_ok=False,
        )

        FIGURE_DIR.mkdir(
            parents=True,
            exist_ok=False,
        )

        tables = create_tables(
            data,
            results,
        )

        figures = create_figures(
            data,
            results,
        )

        audit_sha = write_final_audit(
            results
        )

        summary_sha = (
            write_final_summary(
                results,
                tables,
                figures,
                audit_sha,
            )
        )

        lock_sha = write_final_lock(
            tables,
            figures,
            summary_sha,
            audit_sha,
        )

        verify_final_outputs(
            tables,
            figures,
        )

        print_final_boundary()

    except Exception as exc:

        banner(
            "SCRIPT 76 — ABORTED"
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
            "Do NOT alter any frozen Tasks 1–4 "
            "experimental result."
        )

        print()
        print(
            "If this is an implementation-only "
            "failure, preserve/archive any partial "
            "Script-76 outputs before rerunning."
        )

        return 1

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    banner(
        "SCRIPT 76 SUMMARY"
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
        "TASK 4 IS COMPLETE AND FINAL-LOCKED."
    )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL RETRIEVAL"
    )

    print(
        "============================================================"
    )

    print(
        "Validation mR:"
    )

    print(
        f"  "
        f"{results['validation_retrieval']['mR_mean']:.8f}"
        f" ± "
        f"{results['validation_retrieval']['mR_std_population']:.8f}"
    )

    print(
        "TEST mR:"
    )

    print(
        f"  "
        f"{results['test_retrieval']['mR_mean']:.8f}"
        f" ± "
        f"{results['test_retrieval']['mR_std_population']:.8f}"
    )

    print(
        "Random mR:"
    )

    print(
        f"  "
        f"{results['test_retrieval']['random_mR']:.8f}"
    )

    print(
        "TEST / random:"
    )

    print(
        f"  "
        f"{results['test_retrieval']['mR_multiple_over_random']:.2f}×"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL ZERO-SHOT TAG PREDICTION"
    )

    print(
        "============================================================"
    )

    print(
        "Caption→Label Macro AP:"
    )

    print(
        f"  "
        f"{results['zero_shot']['caption_macro_ap_mean']:.8f}"
        f" ± "
        f"{results['zero_shot']['caption_macro_ap_std_population']:.8f}"
    )

    print(
        "Audio→Label Macro AP:"
    )

    print(
        f"  "
        f"{results['zero_shot']['audio_macro_ap_mean']:.8f}"
        f" ± "
        f"{results['zero_shot']['audio_macro_ap_std_population']:.8f}"
    )

    print()
    print(
        "Task-3 supervised TEST Macro AP:"
    )

    print(
        f"  "
        f"{TASK3_SUPERVISED_TEST_MACRO_AP:.8f}"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL HUMAN EVALUATION"
    )

    print(
        "============================================================"
    )

    print(
        "Listeners:"
    )

    print(
        f"  "
        f"{results['human_evaluation']['listeners']}"
    )

    print(
        "Ratings:"
    )

    print(
        f"  "
        f"{results['human_evaluation']['total_ratings']}"
    )

    print(
        "Overall match rating:"
    )

    print(
        f"  "
        f"{results['human_evaluation']['pooled_mean']:.4f}"
        f" ± "
        f"{results['human_evaluation']['pooled_sd_sample']:.4f}"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL ARTIFACTS"
    )

    print(
        "============================================================"
    )

    print(
        "Final directory:"
    )

    print(
        f"  {relative(FINAL_DIR)}"
    )

    print()
    print(
        "Final summary SHA256:"
    )

    print(
        f"  {summary_sha}"
    )

    print(
        "Final audit SHA256:"
    )

    print(
        f"  {audit_sha}"
    )

    print(
        "FINAL TERMINAL LOCK SHA256:"
    )

    print(
        f"  {lock_sha}"
    )

    print()
    print(
        "All Task-4 experimental stages "
        "64–76 are now complete."
    )

    print()
    print(
        "NO further experimental tuning "
        "or TEST-driven modification is allowed."
    )

    print()
    print(
        "Next project phase:"
    )

    print(
        "  Final report / presentation / demo "
        "using the frozen Tasks 1–4 results."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )