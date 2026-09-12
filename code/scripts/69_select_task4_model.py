#!/usr/bin/env python3
"""
SCRIPT 69 — FREEZE TASK-4 MODEL SELECTION

Purpose
-------
Select the final Task-4 architecture using ONLY the already completed
formal seed-42 VALIDATION results.

Candidates
----------
T4-A:
    Frozen T1-B + frozen A4/G2
    Train projection heads only

T4-B:
    T4-A initialization
    + BERT layers 10-11
    + GAT3
    + projection heads

Frozen Script-65 rule
---------------------
Primary metric:
    full-pool validation mR

mR:
    arithmetic mean of:
        Audio -> Caption R@1
        Audio -> Caption R@5
        Audio -> Caption R@10
        Caption -> Audio R@1
        Caption -> Audio R@5
        Caption -> Audio R@10

Selection:
    higher seed-42 VAL mR wins

Exact tie:
    choose simpler T4-A

IMPORTANT
---------
- NO training
- NO model construction
- NO forward pass
- NO retrieval recomputation
- NO TEST examples
- NO TEST labels
- NO TEST metrics
- NO TEST model selection

This script only reads frozen formal validation artifacts and writes
the Task-4 model-selection freeze.
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


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DESIGN_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_contrastive_design.json"
)

DESIGN_LOCK_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_contrastive_design_lock.json"
)

T4A_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-A_seed42_best.pt"
)

T4A_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-A_seed42_summary.json"
)

T4B_CHECKPOINT = (
    ROOT
    / "models/task4_candidates/"
      "task4_T4-B_seed42_best.pt"
)

T4B_HISTORY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_seed42_history.csv"
)

T4B_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_seed42_summary.json"
)

OUTPUT_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_model_selection.json"
)

LOCK_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_model_selection_lock.json"
)


# =============================================================================
# 1. ACCEPTED HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "61ed1263476e631374790c9669ab4aa96"
    "e3b5cb295436d4e273da66d4da556db"
)

EXPECTED_DESIGN_LOCK_SHA256 = (
    "53f50205e712833ec4efb506b9b6d911"
    "8dcf46f51b72c5907f33cd61ddca612c"
)

EXPECTED_T4A_CHECKPOINT_SHA256 = (
    "fc3e85fbe6e98d19dee5cb2891fce954"
    "704e09fc49e494a589193d9a0a09ab9f"
)

EXPECTED_T4A_SUMMARY_SHA256 = (
    "f82844e55472a574b4c500dd267435890"
    "e7c55f2d6fb0ca853ce0e8ebed13806"
)

EXPECTED_T4B_CHECKPOINT_SHA256 = (
    "2be12c7e6fdc851b2b1efae0a7bed2c4"
    "95cbefb5ae90a4f491c54a72ec8e9ace"
)

EXPECTED_T4B_HISTORY_SHA256 = (
    "f83b9b3a0d0913357679f15976a132ac"
    "5c0cf231cd0f00a5cc21e0cfd6c5e0e1"
)

EXPECTED_T4B_SUMMARY_SHA256 = (
    "d4300c9f5089ca3efa9626bb60fc10e1"
    "23336604808e99eba3cc7cfd0839e6aa"
)


# =============================================================================
# 2. CONTRACT
# =============================================================================

SEED = 42

EXPECTED_VAL = 514

METRIC_KEYS = (
    "audio_to_caption_R@1",
    "audio_to_caption_R@5",
    "audio_to_caption_R@10",
    "caption_to_audio_R@1",
    "caption_to_audio_R@5",
    "caption_to_audio_R@10",
)

PRIMARY_METRIC = "mR"


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

    print(f"PASS  {message}")


def fail(
    message: str,
) -> None:

    print(f"FAIL  {message}")
    FAILURES.append(message)


def warning(
    message: str,
) -> None:

    print(f"WARN  {message}")
    WARNINGS.append(message)


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
# 4. HELPERS
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

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


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        value = json.load(f)

    if not isinstance(
        value,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not a JSON object"
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
            f"Temporary file already exists: {temp}"
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

        f.write("\n")

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temp,
        path,
    )


def relative(
    path: Path,
) -> str:

    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
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
        f"{label} SHA256 matches accepted identity",
    )


def validate_metric_block(
    metrics: dict[str, Any],
    variant: str,
) -> float:

    for key in (
        *METRIC_KEYS,
        PRIMARY_METRIC,
    ):

        check(
            key in metrics,
            (
                f"{variant} metrics contain "
                f"{key}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            f"{variant} metric schema invalid"
        )

    values = []

    for key in METRIC_KEYS:

        value = float(
            metrics[
                key
            ]
        )

        check(
            math.isfinite(
                value
            ),
            (
                f"{variant} {key} "
                "is finite"
            ),
        )

        check(
            0.0 <= value <= 1.0,
            (
                f"{variant} {key} "
                "is within [0,1]"
            ),
        )

        values.append(
            value
        )

    reported_mr = float(
        metrics[
            PRIMARY_METRIC
        ]
    )

    recomputed_mr = float(
        np.mean(
            values,
            dtype=np.float64,
        )
    )

    print(
        f"{variant} reported mR  = "
        f"{reported_mr:.10f}"
    )

    print(
        f"{variant} computed mR  = "
        f"{recomputed_mr:.10f}"
    )

    check(
        np.isclose(
            reported_mr,
            recomputed_mr,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            f"{variant} mR exactly matches "
            "the arithmetic mean of the six recalls"
        ),
    )

    return reported_mr


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. TASK-4 MODEL-SELECTION OUTPUT PROTECTION"
    )

    check(
        not OUTPUT_PATH.exists(),
        (
            "Task-4 model-selection artifact "
            "does not already exist"
        ),
    )

    check(
        not LOCK_PATH.exists(),
        (
            "Task-4 model-selection lock "
            "does not already exist"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite frozen "
            "Task-4 model selection"
        )


# =============================================================================
# 6. VERIFY INPUT ARTIFACTS
# =============================================================================

def verify_inputs() -> None:

    banner(
        "2. VERIFY FORMAL VALIDATION ARTIFACTS"
    )

    artifacts = (
        (
            DESIGN_PATH,
            EXPECTED_DESIGN_SHA256,
            "Task-4 design",
        ),
        (
            DESIGN_LOCK_PATH,
            EXPECTED_DESIGN_LOCK_SHA256,
            "Task-4 design lock",
        ),
        (
            T4A_CHECKPOINT,
            EXPECTED_T4A_CHECKPOINT_SHA256,
            "T4-A checkpoint",
        ),
        (
            T4A_SUMMARY,
            EXPECTED_T4A_SUMMARY_SHA256,
            "T4-A summary",
        ),
        (
            T4B_CHECKPOINT,
            EXPECTED_T4B_CHECKPOINT_SHA256,
            "T4-B checkpoint",
        ),
        (
            T4B_HISTORY,
            EXPECTED_T4B_HISTORY_SHA256,
            "T4-B history",
        ),
        (
            T4B_SUMMARY,
            EXPECTED_T4B_SUMMARY_SHA256,
            "T4-B summary",
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
            "Formal Task-4 artifact identity "
            "verification failed"
        )


# =============================================================================
# 7. VERIFY FROZEN SELECTION RULE
# =============================================================================

def verify_design() -> dict[str, Any]:

    banner(
        "3. VERIFY FROZEN SELECTION RULE"
    )

    design = load_json(
        DESIGN_PATH
    )

    validation = design[
        "validation_retrieval"
    ]

    check(
        validation[
            "primary_model_selection_metric"
        ]
        == "mR",
        (
            "frozen primary model-selection "
            "metric is mR"
        ),
    )

    check(
        validation[
            "model_selection"
        ]
        ==
        (
            "select T4-A or T4-B using highest seed-42 "
            "VAL mR only"
        ),
        (
            "frozen model-selection rule uses "
            "seed-42 VAL mR only"
        ),
    )

    check(
        validation[
            "exact_model_tie_rule"
        ]
        ==
        (
            "if VAL mR is exactly tied, choose simpler T4-A"
        ),
        (
            "frozen exact-tie rule selects "
            "simpler T4-A"
        ),
    )

    check(
        validation[
            "test_used_for_selection"
        ]
        is False,
        (
            "frozen design forbids TEST "
            "for model selection"
        ),
    )

    check(
        design[
            "dataset"
        ][
            "test_during_development"
        ]
        is False,
        (
            "TEST remains forbidden "
            "during Task-4 development"
        ),
    )

    check(
        design[
            "randomness"
        ][
            "formal_model_selection_seed"
        ]
        == SEED,
        (
            "formal model-selection seed "
            "remains 42"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen model-selection rule changed"
        )

    return design


# =============================================================================
# 8. LOAD / VALIDATE FORMAL RESULTS
# =============================================================================

def load_formal_results():

    banner(
        "4. LOAD FORMAL SEED-42 VALIDATION RESULTS"
    )

    t4a = load_json(
        T4A_SUMMARY
    )

    t4b = load_json(
        T4B_SUMMARY
    )

    # -------------------------------------------------------------------------
    # T4-A
    # -------------------------------------------------------------------------

    check(
        t4a.get(
            "variant"
        )
        == "T4-A",
        "T4-A summary variant is exact",
    )

    check(
        t4a.get(
            "status"
        )
        == "formal_validation_result",
        (
            "T4-A is an accepted formal "
            "validation result"
        ),
    )

    check(
        int(
            t4a.get(
                "seed",
                -1,
            )
        )
        == SEED,
        "T4-A seed = 42",
    )

    check(
        int(
            t4a.get(
                "val_examples",
                -1,
            )
        )
        == EXPECTED_VAL,
        "T4-A VAL examples = 514",
    )

    check(
        int(
            t4a.get(
                "test_examples",
                -1,
            )
        )
        == 0,
        "T4-A TEST examples = 0",
    )

    check(
        t4a[
            "primary_selection_metric"
        ]
        == "VAL mR",
        (
            "T4-A checkpoint was selected "
            "by VAL mR"
        ),
    )

    check(
        t4a[
            "checkpoint_sha256"
        ]
        ==
        EXPECTED_T4A_CHECKPOINT_SHA256,
        (
            "T4-A summary references exact "
            "accepted checkpoint"
        ),
    )

    check(
        t4a[
            "boundaries"
        ][
            "test_access"
        ]
        is False,
        "T4-A records no TEST access",
    )

    check(
        t4a[
            "boundaries"
        ][
            "test_retrieval"
        ]
        is False,
        "T4-A records no TEST retrieval",
    )

    # -------------------------------------------------------------------------
    # T4-B
    # -------------------------------------------------------------------------

    check(
        t4b.get(
            "variant"
        )
        == "T4-B",
        "T4-B summary variant is exact",
    )

    check(
        t4b.get(
            "status"
        )
        == "formal_validation_result",
        (
            "T4-B is an accepted formal "
            "validation result"
        ),
    )

    check(
        int(
            t4b.get(
                "seed",
                -1,
            )
        )
        == SEED,
        "T4-B seed = 42",
    )

    check(
        int(
            t4b.get(
                "val_examples",
                -1,
            )
        )
        == EXPECTED_VAL,
        "T4-B VAL examples = 514",
    )

    check(
        int(
            t4b.get(
                "test_examples",
                -1,
            )
        )
        == 0,
        "T4-B TEST examples = 0",
    )

    check(
        t4b[
            "primary_selection_metric"
        ]
        == "VAL mR",
        (
            "T4-B checkpoint was selected "
            "by VAL mR"
        ),
    )

    check(
        t4b[
            "checkpoint_sha256"
        ]
        ==
        EXPECTED_T4B_CHECKPOINT_SHA256,
        (
            "T4-B summary references exact "
            "accepted checkpoint"
        ),
    )

    check(
        t4b[
            "boundaries"
        ][
            "label_supervision"
        ]
        is False,
        (
            "T4-B records no Task-4 "
            "label supervision"
        ),
    )

    check(
        t4b[
            "boundaries"
        ][
            "test_access"
        ]
        is False,
        "T4-B records no TEST access",
    )

    check(
        t4b[
            "boundaries"
        ][
            "test_retrieval"
        ]
        is False,
        "T4-B records no TEST retrieval",
    )

    if FAILURES:

        raise RuntimeError(
            "Formal Task-4 summary validation failed"
        )

    t4a_metrics = (
        t4a[
            "best_val_metrics"
        ]
    )

    t4b_metrics = (
        t4b[
            "best_val_metrics"
        ]
    )

    t4a_mr = validate_metric_block(
        t4a_metrics,
        "T4-A",
    )

    t4b_mr = validate_metric_block(
        t4b_metrics,
        "T4-B",
    )

    if FAILURES:

        raise RuntimeError(
            "Task-4 metric validation failed"
        )

    return (
        t4a,
        t4b,
        t4a_metrics,
        t4b_metrics,
        t4a_mr,
        t4b_mr,
    )


# =============================================================================
# 9. APPLY PREDECLARED MODEL-SELECTION RULE
# =============================================================================

def select_model(
    t4a_mr: float,
    t4b_mr: float,
) -> tuple[
    str,
    str,
    float,
]:

    banner(
        "5. APPLY VALIDATION-ONLY MODEL-SELECTION RULE"
    )

    print(
        f"T4-A seed-42 VAL mR = "
        f"{t4a_mr:.10f}"
    )

    print(
        f"T4-B seed-42 VAL mR = "
        f"{t4b_mr:.10f}"
    )

    delta = (
        t4b_mr
        - t4a_mr
    )

    print(
        f"T4-B minus T4-A mR = "
        f"{delta:+.10f}"
    )

    # Frozen rule:
    # higher mR wins.
    #
    # Exact tie:
    # simpler T4-A wins.
    if t4b_mr > t4a_mr:

        winner = "T4-B"

        reason = (
            "T4-B has the higher formal seed-42 "
            "validation mR."
        )

    else:

        winner = "T4-A"

        if t4a_mr == t4b_mr:

            reason = (
                "Validation mR is exactly tied; "
                "the predeclared tie rule selects "
                "the simpler T4-A."
            )

        else:

            reason = (
                "T4-A has the higher formal seed-42 "
                "validation mR."
            )

    winner_mr = (
        t4b_mr
        if winner == "T4-B"
        else t4a_mr
    )

    print()
    print(
        f"SELECTED TASK-4 VARIANT: "
        f"{winner}"
    )

    print(
        f"Selection reason:"
    )

    print(
        f"  {reason}"
    )

    passed(
        "Task-4 winner selected using VAL only"
    )

    return (
        winner,
        reason,
        winner_mr,
    )


# =============================================================================
# 10. WRITE MODEL-SELECTION FREEZE
# =============================================================================

def write_selection(
    design: dict[str, Any],
    t4a: dict[str, Any],
    t4b: dict[str, Any],
    t4a_metrics: dict[str, Any],
    t4b_metrics: dict[str, Any],
    t4a_mr: float,
    t4b_mr: float,
    winner: str,
    reason: str,
    winner_mr: float,
) -> tuple[
    str,
    str,
]:

    banner(
        "6. WRITE TASK-4 MODEL-SELECTION FREEZE"
    )

    created_utc = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    selected_checkpoint = (
        T4B_CHECKPOINT
        if winner == "T4-B"
        else T4A_CHECKPOINT
    )

    selected_checkpoint_sha = (
        EXPECTED_T4B_CHECKPOINT_SHA256
        if winner == "T4-B"
        else EXPECTED_T4A_CHECKPOINT_SHA256
    )

    selection = {
        "artifact_type":
            "task4_model_selection",

        "version":
            1,

        "created_utc":
            created_utc,

        "selection_status":
            "frozen_validation_only",

        "selection_seed":
            SEED,

        "primary_metric":
            "VAL mR",

        "selection_rule":
            (
                "higher seed-42 full-pool VAL mR wins; "
                "exact tie selects simpler T4-A"
            ),

        "test_used":
            False,

        "test_examples_evaluated":
            0,

        "test_retrieval_performed":
            False,

        "candidates": {
            "T4-A": {
                "best_epoch":
                    int(
                        t4a[
                            "best_epoch"
                        ]
                    ),

                "val_metrics":
                    t4a_metrics,

                "val_mR":
                    float(
                        t4a_mr
                    ),

                "checkpoint":
                    relative(
                        T4A_CHECKPOINT
                    ),

                "checkpoint_sha256":
                    EXPECTED_T4A_CHECKPOINT_SHA256,

                "summary":
                    relative(
                        T4A_SUMMARY
                    ),

                "summary_sha256":
                    EXPECTED_T4A_SUMMARY_SHA256,
            },

            "T4-B": {
                "best_epoch":
                    int(
                        t4b[
                            "best_epoch"
                        ]
                    ),

                "val_metrics":
                    t4b_metrics,

                "val_mR":
                    float(
                        t4b_mr
                    ),

                "checkpoint":
                    relative(
                        T4B_CHECKPOINT
                    ),

                "checkpoint_sha256":
                    EXPECTED_T4B_CHECKPOINT_SHA256,

                "summary":
                    relative(
                        T4B_SUMMARY
                    ),

                "summary_sha256":
                    EXPECTED_T4B_SUMMARY_SHA256,
            },
        },

        "comparison": {
            "T4B_minus_T4A_val_mR":
                float(
                    t4b_mr
                    - t4a_mr
                ),
        },

        "selected_variant":
            winner,

        "selected_val_mR":
            float(
                winner_mr
            ),

        "selection_reason":
            reason,

        "selected_seed42_checkpoint":
            relative(
                selected_checkpoint
            ),

        "selected_seed42_checkpoint_sha256":
            selected_checkpoint_sha,

        "final_multiseed_policy": (
            design[
                "final_multiseed_policy"
            ]
        ),

        "next_stage": {
            "script":
                70,

            "action":
                (
                    "run final selected Task-4 "
                    "configuration under frozen "
                    "multi-seed policy"
                ),

            "seeds":
                [
                    42,
                    1337,
                    2026,
                ],

            "test_access":
                False,
        },

        "provenance": {
            "script69":
                relative(
                    Path(
                        __file__
                    ).resolve()
                ),

            "script69_sha256":
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                ),

            "design_sha256":
                EXPECTED_DESIGN_SHA256,

            "design_lock_sha256":
                EXPECTED_DESIGN_LOCK_SHA256,

            "T4A_checkpoint_sha256":
                EXPECTED_T4A_CHECKPOINT_SHA256,

            "T4A_summary_sha256":
                EXPECTED_T4A_SUMMARY_SHA256,

            "T4B_checkpoint_sha256":
                EXPECTED_T4B_CHECKPOINT_SHA256,

            "T4B_history_sha256":
                EXPECTED_T4B_HISTORY_SHA256,

            "T4B_summary_sha256":
                EXPECTED_T4B_SUMMARY_SHA256,
        },
    }

    write_json_atomic(
        OUTPUT_PATH,
        selection,
    )

    check(
        OUTPUT_PATH.is_file(),
        "Task-4 model-selection artifact written",
    )

    selection_sha = sha256_file(
        OUTPUT_PATH
    )

    print(
        f"Selection artifact:"
    )

    print(
        f"  {relative(OUTPUT_PATH)}"
    )

    print(
        f"SHA256:"
    )

    print(
        f"  {selection_sha}"
    )

    # -------------------------------------------------------------------------
    # Lock
    # -------------------------------------------------------------------------

    lock = {
        "lock_type":
            "task4_model_selection_lock",

        "version":
            1,

        "created_utc":
            created_utc,

        "selection_path":
            relative(
                OUTPUT_PATH
            ),

        "selection_sha256":
            selection_sha,

        "selected_variant":
            winner,

        "selected_val_mR":
            float(
                winner_mr
            ),

        "selection_basis":
            "seed42_validation_only",

        "test_used":
            False,

        "test_examples_evaluated":
            0,

        "test_retrieval_performed":
            False,

        "parent_hashes": {
            "design":
                EXPECTED_DESIGN_SHA256,

            "design_lock":
                EXPECTED_DESIGN_LOCK_SHA256,

            "T4A_checkpoint":
                EXPECTED_T4A_CHECKPOINT_SHA256,

            "T4A_summary":
                EXPECTED_T4A_SUMMARY_SHA256,

            "T4B_checkpoint":
                EXPECTED_T4B_CHECKPOINT_SHA256,

            "T4B_summary":
                EXPECTED_T4B_SUMMARY_SHA256,
        },

        "script69_sha256":
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

    check(
        LOCK_PATH.is_file(),
        (
            "Task-4 model-selection lock "
            "written"
        ),
    )

    lock_sha = sha256_file(
        LOCK_PATH
    )

    print()
    print(
        "Selection lock:"
    )

    print(
        f"  {relative(LOCK_PATH)}"
    )

    print(
        "SHA256:"
    )

    print(
        f"  {lock_sha}"
    )

    return (
        selection_sha,
        lock_sha,
    )


# =============================================================================
# 11. POST-WRITE VALIDATION
# =============================================================================

def validate_written_selection(
    expected_winner: str,
    expected_winner_mr: float,
) -> None:

    banner(
        "7. POST-WRITE VALIDATION"
    )

    selection = load_json(
        OUTPUT_PATH
    )

    lock = load_json(
        LOCK_PATH
    )

    check(
        selection[
            "selected_variant"
        ]
        == expected_winner,
        (
            "written selection contains "
            "the computed winner"
        ),
    )

    check(
        np.isclose(
            float(
                selection[
                    "selected_val_mR"
                ]
            ),
            expected_winner_mr,
            atol=1e-12,
            rtol=0.0,
        ),
        (
            "written selection contains "
            "the exact winner VAL mR"
        ),
    )

    check(
        selection[
            "test_used"
        ]
        is False,
        (
            "selection artifact records "
            "TEST unused"
        ),
    )

    check(
        lock[
            "selection_sha256"
        ]
        ==
        sha256_file(
            OUTPUT_PATH
        ),
        (
            "selection lock matches "
            "selection artifact"
        ),
    )

    check(
        lock[
            "selected_variant"
        ]
        == expected_winner,
        (
            "selection lock records "
            "the same winner"
        ),
    )

    check(
        lock[
            "test_used"
        ]
        is False,
        (
            "selection lock records "
            "TEST unused"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Written Task-4 model selection "
            "failed validation"
        )


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 69 — FREEZE TASK-4 "
        "VALIDATION-ONLY MODEL SELECTION"
    )

    print(
        f"Project root: "
        f"{ROOT}"
    )

    print()
    print(
        "Candidate models:"
    )
    print(
        "  T4-A"
    )
    print(
        "  T4-B"
    )

    print()
    print(
        "Selection data:      VALIDATION ONLY"
    )

    print(
        "Selection seed:      42"
    )

    print(
        "Primary metric:      mR"
    )

    print(
        "TEST access:         NO"
    )

    print(
        "Model inference:     NO"
    )

    print(
        "Training:            NO"
    )

    try:

        verify_output_protection()

        verify_inputs()

        design = verify_design()

        (
            t4a,
            t4b,
            t4a_metrics,
            t4b_metrics,
            t4a_mr,
            t4b_mr,
        ) = load_formal_results()

        (
            winner,
            reason,
            winner_mr,
        ) = select_model(
            t4a_mr,
            t4b_mr,
        )

        (
            selection_sha,
            lock_sha,
        ) = write_selection(
            design=design,
            t4a=t4a,
            t4b=t4b,
            t4a_metrics=t4a_metrics,
            t4b_metrics=t4b_metrics,
            t4a_mr=t4a_mr,
            t4b_mr=t4b_mr,
            winner=winner,
            reason=reason,
            winner_mr=winner_mr,
        )

        validate_written_selection(
            expected_winner=winner,
            expected_winner_mr=winner_mr,
        )

    except Exception as exc:

        banner(
            "SCRIPT 69 — ABORTED"
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
            "Do NOT proceed to Script 70."
        )

        return 1

    banner(
        "SCRIPT 69 SUMMARY"
    )

    print(
        f"Failures: "
        f"{len(FAILURES)}"
    )

    print(
        f"Warnings: "
        f"{len(WARNINGS)}"
    )

    print()

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "TASK-4 MODEL SELECTION IS NOW FROZEN."
    )

    print()
    print(
        f"T4-A seed-42 VAL mR:"
    )

    print(
        f"  {t4a_mr:.10f}"
    )

    print(
        f"T4-B seed-42 VAL mR:"
    )

    print(
        f"  {t4b_mr:.10f}"
    )

    print()
    print(
        f"SELECTED VARIANT:"
    )

    print(
        f"  {winner}"
    )

    print()
    print(
        "Selection artifact:"
    )

    print(
        f"  {relative(OUTPUT_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{selection_sha}"
    )

    print()
    print(
        "Selection lock:"
    )

    print(
        f"  {relative(LOCK_PATH)}"
    )

    print(
        f"  SHA256 = "
        f"{lock_sha}"
    )

    print()
    print(
        "TEST examples evaluated: 0"
    )

    print(
        "TEST retrieval performed: 0"
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
        "  Script 70 — final selected "
        "configuration multi-seed runs."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )