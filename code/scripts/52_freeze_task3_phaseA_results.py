#!/usr/bin/env python3
"""
Script 52 — Freeze Task-3 Phase-A Results + Phase-B Decisions
=============================================================

Purpose
-------
Freeze the completed formal Task-3 Phase-A results before any Phase-B
selective unfreezing.

This script:

1. Verifies the frozen Script-49 Task-3 design.
2. Verifies accepted Script-50 M3 artifacts.
3. Verifies accepted Script-51 M4 artifacts.
4. Verifies checkpoint/summary consistency.
5. Re-applies the PRE-FROZEN per-variant Phase-B gate.
6. Freezes which variants require Phase B.

NO:
    - model construction
    - forward/inference
    - training
    - threshold tuning
    - TEST access
    - TEST metrics

Intentional output
------------------
results/runs/task3/task3_phaseA_results_freeze.json
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

TASK3_DIR = (
    ROOT
    / "results/runs/task3"
)

DESIGN = (
    TASK3_DIR
    / "task3_phaseA_design.json"
)


# -----------------------------------------------------------------------------
# M3 — accepted Script 50
# -----------------------------------------------------------------------------

M3_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M3_phaseA_seed42_20260910T130250Z_best.pt"
)

M3_EPOCHS = (
    TASK3_DIR
    / "task3_M3_phaseA_seed42_20260910T130250Z_epochs.csv"
)

M3_SUMMARY = (
    TASK3_DIR
    / "task3_M3_phaseA_seed42_20260910T130250Z_summary.json"
)


# -----------------------------------------------------------------------------
# M4 — accepted Script 51
# -----------------------------------------------------------------------------

M4_CHECKPOINT = (
    ROOT
    / "models/task3_candidates/"
    "task3_M4_phaseA_seed42_20260910T131031Z_best.pt"
)

M4_EPOCHS = (
    TASK3_DIR
    / "task3_M4_phaseA_seed42_20260910T131031Z_epochs.csv"
)

M4_SUMMARY = (
    TASK3_DIR
    / "task3_M4_phaseA_seed42_20260910T131031Z_summary.json"
)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUTPUT = (
    TASK3_DIR
    / "task3_phaseA_results_freeze.json"
)

OUTPUT_TEMP = Path(
    str(OUTPUT)
    + ".__tmp__"
)


# =============================================================================
# 1. ACCEPTED HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "0a9d50413b65aeed58d02d6a2dff70d99363ae25fee115d30bc9749f54e1c9ef"
)


# Script 50 — M3
EXPECTED_M3_CHECKPOINT_SHA256 = (
    "f23a0a5f8bb604672de640eca5cbfbb189e99febd32cab02f082a1de9a1fa5b8"
)

EXPECTED_M3_EPOCHS_SHA256 = (
    "0f5e96d9a58450b7e8b5e3c1f9deaa2b70c7690fc0fb33f026786916cb18cf8e"
)

EXPECTED_M3_SUMMARY_SHA256 = (
    "85ecacdcedd391cb0eae6231239075e88d017b7141123ae0fa6a3261b82e7412"
)


# Script 51 — M4
EXPECTED_M4_CHECKPOINT_SHA256 = (
    "ea210e30a932c124ee07b6a5fc90d068e0201152301998ed1eeb67559dd2e112"
)

EXPECTED_M4_EPOCHS_SHA256 = (
    "b3c7e3b83815246074adc7d33aa8263f1d8bb6a25d60462b9045abcfa9885637"
)

EXPECTED_M4_SUMMARY_SHA256 = (
    "86e34cb983b23e56565d496668f2585cb62727507fdb49612f849d973d0ad1af"
)


# =============================================================================
# 2. FROZEN RESULTS
# =============================================================================

M1_VAL_MACRO_AP = (
    0.8405750022914198
)

M2_VAL_MACRO_AP = (
    0.18752693895679443
)

EXPECTED_M3_VAL_MACRO_AP = (
    0.830729168310
)

EXPECTED_M4_VAL_MACRO_AP = (
    0.822898704197
)

EXPECTED_M3_BEST_EPOCH = 5

EXPECTED_M4_BEST_EPOCH = 8


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

    print(
        f"PASS  {message}"
    )


def failed(message: str) -> None:

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

    failed(
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


def load_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


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

    if not isinstance(
        obj,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not a "
            "checkpoint dictionary"
        )

    return obj


def write_json_atomic(
    path: Path,
    obj: Any,
) -> None:

    if OUTPUT_TEMP.exists():

        raise RuntimeError(
            "Temporary Phase-A result freeze "
            f"already exists: {OUTPUT_TEMP}"
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


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. OUTPUT PROTECTION"
    )

    check(
        not OUTPUT.exists(),
        (
            "Phase-A result-freeze artifact "
            "does not already exist"
        ),
    )

    check(
        not OUTPUT_TEMP.exists(),
        (
            "temporary Phase-A result-freeze "
            "artifact does not already exist"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite frozen "
            "Task-3 Phase-A result decision"
        )


# =============================================================================
# 6. VERIFY FILE IDENTITIES
# =============================================================================

def verify_file_identities() -> None:

    banner(
        "2. ACCEPTED PHASE-A ARTIFACT IDENTITIES"
    )

    expected = {
        DESIGN:
            EXPECTED_DESIGN_SHA256,

        M3_CHECKPOINT:
            EXPECTED_M3_CHECKPOINT_SHA256,

        M3_EPOCHS:
            EXPECTED_M3_EPOCHS_SHA256,

        M3_SUMMARY:
            EXPECTED_M3_SUMMARY_SHA256,

        M4_CHECKPOINT:
            EXPECTED_M4_CHECKPOINT_SHA256,

        M4_EPOCHS:
            EXPECTED_M4_EPOCHS_SHA256,

        M4_SUMMARY:
            EXPECTED_M4_SUMMARY_SHA256,
    }

    for path in expected:

        check(
            path.is_file(),
            (
                "artifact exists: "
                f"{path.relative_to(ROOT)}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Required Phase-A artifact missing"
        )

    for path, expected_sha in (
        expected.items()
    ):

        observed = sha256_file(
            path
        )

        print(
            f"{path.name}:"
        )

        print(
            f"  observed = {observed}"
        )

        print(
            f"  expected = {expected_sha}"
        )

        check(
            observed
            == expected_sha,
            (
                f"{path.name} matches "
                "accepted SHA256"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Accepted Phase-A artifact "
            "identity check failed"
        )


# =============================================================================
# 7. VERIFY FROZEN DESIGN / GATE
# =============================================================================

def verify_design() -> dict[str, Any]:

    banner(
        "3. FROZEN PHASE-B GATE"
    )

    design = load_json(
        DESIGN
    )

    check(
        design.get(
            "artifact_type"
        )
        ==
        "task3_phaseA_design_lock",
        (
            "Task-3 design artifact "
            "type is correct"
        ),
    )

    check(
        design.get(
            "design_frozen_before_any_fusion_result"
        )
        is True,
        (
            "Phase-B gate was frozen "
            "before fusion results"
        ),
    )

    gate = design.get(
        "phase_b_gate",
        {},
    )

    check(
        gate.get(
            "application"
        )
        == "per_variant",
        (
            "Phase-B gate applies "
            "independently per variant"
        ),
    )

    check(
        gate.get(
            "strict"
        )
        is True,
        (
            "Phase-B gate uses strict "
            "greater-than comparison"
        ),
    )

    expected_reference = max(
        M1_VAL_MACRO_AP,
        M2_VAL_MACRO_AP,
    )

    observed_reference = float(
        gate.get(
            "effective_reference_to_beat",
            float("nan"),
        )
    )

    check(
        np.isclose(
            observed_reference,
            expected_reference,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "Phase-B effective reference "
            "remains frozen M1 score"
        ),
    )

    check(
        design.get(
            "test_examples_used"
        )
        == 0,
        (
            "design records zero TEST use"
        ),
    )

    check(
        design.get(
            "test_target_rows_loaded"
        )
        == 0,
        (
            "design records zero TEST "
            "target rows"
        ),
    )

    check(
        design.get(
            "threshold_tuning_used"
        )
        is False,
        (
            "design records no "
            "threshold tuning"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen Phase-B gate validation failed"
        )

    return design


# =============================================================================
# 8. VERIFY M3 RESULT
# =============================================================================

def verify_m3() -> dict[str, Any]:

    banner(
        "4. VERIFY M3 PHASE-A RESULT"
    )

    summary = load_json(
        M3_SUMMARY
    )

    checkpoint = load_checkpoint(
        M3_CHECKPOINT
    )

    check(
        summary.get(
            "task"
        )
        == "task3",
        "M3 summary task = task3",
    )

    check(
        summary.get(
            "phase"
        )
        == "A",
        "M3 summary phase = A",
    )

    check(
        summary.get(
            "variant"
        )
        == "M3",
        "M3 summary variant = M3",
    )

    check(
        summary.get(
            "formal_run"
        )
        is True,
        "M3 is a formal run",
    )

    check(
        int(
            summary.get(
                "seed",
                -1,
            )
        )
        == 42,
        "M3 seed = 42",
    )

    check(
        int(
            summary.get(
                "best_epoch",
                -1,
            )
        )
        == EXPECTED_M3_BEST_EPOCH,
        "M3 best epoch = 5",
    )

    summary_ap = float(
        summary[
            "best_val_auc_pr_macro_mean_ap"
        ]
    )

    checkpoint_ap = float(
        checkpoint[
            "val_auc_pr_macro_mean_ap"
        ]
    )

    print(
        f"M3 summary VAL macro AP: "
        f"{summary_ap:.12f}"
    )

    print(
        f"M3 checkpoint VAL macro AP: "
        f"{checkpoint_ap:.12f}"
    )

    check(
        np.isclose(
            summary_ap,
            EXPECTED_M3_VAL_MACRO_AP,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M3 summary VAL macro AP "
            "matches accepted result"
        ),
    )

    check(
        np.isclose(
            checkpoint_ap,
            summary_ap,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M3 checkpoint and summary "
            "VAL macro AP agree"
        ),
    )

    check(
        int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )
        == EXPECTED_M3_BEST_EPOCH,
        (
            "M3 checkpoint is the "
            "accepted best epoch"
        ),
    )

    check(
        checkpoint.get(
            "task3_phaseA_design_sha256"
        )
        ==
        EXPECTED_DESIGN_SHA256,
        (
            "M3 checkpoint references "
            "frozen Script-49 design"
        ),
    )

    check(
        summary.get(
            "test_split_used"
        )
        is False,
        (
            "M3 summary records TEST unused"
        ),
    )

    check(
        int(
            summary.get(
                "test_target_rows_loaded",
                -1,
            )
        )
        == 0,
        (
            "M3 summary records zero "
            "TEST target rows"
        ),
    )

    check(
        int(
            summary.get(
                "test_examples_evaluated",
                -1,
            )
        )
        == 0,
        (
            "M3 summary records zero "
            "TEST inference"
        ),
    )

    check(
        summary.get(
            "threshold_tuning_used"
        )
        is False,
        (
            "M3 summary records no "
            "threshold tuning"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "M3 Phase-A validation failed"
        )

    return summary


# =============================================================================
# 9. VERIFY M4 RESULT
# =============================================================================

def verify_m4() -> dict[str, Any]:

    banner(
        "5. VERIFY M4 PHASE-A RESULT"
    )

    summary = load_json(
        M4_SUMMARY
    )

    checkpoint = load_checkpoint(
        M4_CHECKPOINT
    )

    check(
        summary.get(
            "task"
        )
        == "task3",
        "M4 summary task = task3",
    )

    check(
        summary.get(
            "phase"
        )
        == "A",
        "M4 summary phase = A",
    )

    check(
        summary.get(
            "variant"
        )
        == "M4",
        "M4 summary variant = M4",
    )

    check(
        summary.get(
            "formal_run"
        )
        is True,
        "M4 is a formal run",
    )

    check(
        int(
            summary.get(
                "seed",
                -1,
            )
        )
        == 42,
        "M4 seed = 42",
    )

    check(
        int(
            summary.get(
                "best_epoch",
                -1,
            )
        )
        == EXPECTED_M4_BEST_EPOCH,
        "M4 best epoch = 8",
    )

    summary_ap = float(
        summary[
            "best_val_auc_pr_macro_mean_ap"
        ]
    )

    checkpoint_ap = float(
        checkpoint[
            "val_auc_pr_macro_mean_ap"
        ]
    )

    print(
        f"M4 summary VAL macro AP: "
        f"{summary_ap:.12f}"
    )

    print(
        f"M4 checkpoint VAL macro AP: "
        f"{checkpoint_ap:.12f}"
    )

    check(
        np.isclose(
            summary_ap,
            EXPECTED_M4_VAL_MACRO_AP,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M4 summary VAL macro AP "
            "matches accepted result"
        ),
    )

    check(
        np.isclose(
            checkpoint_ap,
            summary_ap,
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "M4 checkpoint and summary "
            "VAL macro AP agree"
        ),
    )

    check(
        int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )
        == EXPECTED_M4_BEST_EPOCH,
        (
            "M4 checkpoint is the "
            "accepted best epoch"
        ),
    )

    check(
        checkpoint.get(
            "task3_phaseA_design_sha256"
        )
        ==
        EXPECTED_DESIGN_SHA256,
        (
            "M4 checkpoint references "
            "frozen Script-49 design"
        ),
    )

    check(
        checkpoint.get(
            "accepted_m3_phaseA_checkpoint_sha256"
        )
        ==
        EXPECTED_M3_CHECKPOINT_SHA256,
        (
            "M4 checkpoint records accepted "
            "M3 dependency"
        ),
    )

    check(
        summary.get(
            "test_split_used"
        )
        is False,
        (
            "M4 summary records TEST unused"
        ),
    )

    check(
        int(
            summary.get(
                "test_target_rows_loaded",
                -1,
            )
        )
        == 0,
        (
            "M4 summary records zero "
            "TEST target rows"
        ),
    )

    check(
        int(
            summary.get(
                "test_examples_evaluated",
                -1,
            )
        )
        == 0,
        (
            "M4 summary records zero "
            "TEST inference"
        ),
    )

    check(
        summary.get(
            "threshold_tuning_used"
        )
        is False,
        (
            "M4 summary records no "
            "threshold tuning"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "M4 Phase-A validation failed"
        )

    return summary


# =============================================================================
# 10. APPLY FROZEN GATE
# =============================================================================

def determine_phase_b(
    m3_summary: dict[str, Any],
    m4_summary: dict[str, Any],
) -> dict[str, Any]:

    banner(
        "6. APPLY PRE-FROZEN PHASE-B GATE"
    )

    m3_ap = float(
        m3_summary[
            "best_val_auc_pr_macro_mean_ap"
        ]
    )

    m4_ap = float(
        m4_summary[
            "best_val_auc_pr_macro_mean_ap"
        ]
    )

    # Frozen rule:
    #
    # Skip Phase B only if that variant
    # STRICTLY beats BOTH M1 and M2.

    m3_beats_m1 = (
        m3_ap
        >
        M1_VAL_MACRO_AP
    )

    m3_beats_m2 = (
        m3_ap
        >
        M2_VAL_MACRO_AP
    )

    m4_beats_m1 = (
        m4_ap
        >
        M1_VAL_MACRO_AP
    )

    m4_beats_m2 = (
        m4_ap
        >
        M2_VAL_MACRO_AP
    )

    m3_required = not (
        m3_beats_m1
        and
        m3_beats_m2
    )

    m4_required = not (
        m4_beats_m1
        and
        m4_beats_m2
    )

    print(
        f"M1 BERT-only: "
        f"{M1_VAL_MACRO_AP:.8f}"
    )

    print(
        f"M2 GAT/G2-only: "
        f"{M2_VAL_MACRO_AP:.8f}"
    )

    print(
        f"M3 Phase A: "
        f"{m3_ap:.8f}"
    )

    print(
        f"M4 Phase A: "
        f"{m4_ap:.8f}"
    )

    print()

    print(
        f"M3 > M1: "
        f"{m3_beats_m1}"
    )

    print(
        f"M3 > M2: "
        f"{m3_beats_m2}"
    )

    print(
        f"M3 Phase B required: "
        f"{m3_required}"
    )

    print()

    print(
        f"M4 > M1: "
        f"{m4_beats_m1}"
    )

    print(
        f"M4 > M2: "
        f"{m4_beats_m2}"
    )

    print(
        f"M4 Phase B required: "
        f"{m4_required}"
    )

    check(
        m3_required
        is True,
        (
            "M3 correctly triggers "
            "Phase B"
        ),
    )

    check(
        m4_required
        is True,
        (
            "M4 correctly triggers "
            "Phase B"
        ),
    )

    # Phase-A ranking is descriptive only.
    check(
        m3_ap
        >
        m4_ap,
        (
            "M3 is the higher Phase-A "
            "fusion result"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen Phase-B decision validation failed"
        )

    return {
        "m3_ap":
            m3_ap,

        "m4_ap":
            m4_ap,

        "m3_required":
            bool(
                m3_required
            ),

        "m4_required":
            bool(
                m4_required
            ),

        "m3_beats_m1":
            bool(
                m3_beats_m1
            ),

        "m3_beats_m2":
            bool(
                m3_beats_m2
            ),

        "m4_beats_m1":
            bool(
                m4_beats_m1
            ),

        "m4_beats_m2":
            bool(
                m4_beats_m2
            ),
    }


# =============================================================================
# 11. WRITE FREEZE ARTIFACT
# =============================================================================

def freeze_results(
    decision: dict[str, Any],
) -> str:

    banner(
        "7. FREEZE PHASE-A RESULTS"
    )

    artifact = {
        "artifact_type":
            "task3_phaseA_results_freeze",

        "version":
            1,

        "selection_basis":
            "VALIDATION only",

        "formal_phase_a_complete":
            True,

        "task3_phaseA_design_sha256":
            EXPECTED_DESIGN_SHA256,

        "test_split_used":
            False,

        "test_target_rows_loaded":
            0,

        "test_examples_evaluated":
            0,

        "threshold_tuning_used":
            False,

        "baseline_references": {
            "M1_T1B_BERT_only": {
                "val_macro_ap":
                    M1_VAL_MACRO_AP,
            },

            "M2_A4_GAT_G2_only": {
                "val_macro_ap":
                    M2_VAL_MACRO_AP,
            },
        },

        "phase_a_results": {
            "M3": {
                "model":
                    "EarlyFusion",

                "seed":
                    42,

                "best_epoch":
                    EXPECTED_M3_BEST_EPOCH,

                "val_macro_ap":
                    decision[
                        "m3_ap"
                    ],

                "checkpoint":
                    str(
                        M3_CHECKPOINT
                        .relative_to(
                            ROOT
                        )
                    ),

                "checkpoint_sha256":
                    EXPECTED_M3_CHECKPOINT_SHA256,

                "epoch_log_sha256":
                    EXPECTED_M3_EPOCHS_SHA256,

                "summary_sha256":
                    EXPECTED_M3_SUMMARY_SHA256,
            },

            "M4": {
                "model":
                    "CrossAttention",

                "seed":
                    42,

                "best_epoch":
                    EXPECTED_M4_BEST_EPOCH,

                "val_macro_ap":
                    decision[
                        "m4_ap"
                    ],

                "checkpoint":
                    str(
                        M4_CHECKPOINT
                        .relative_to(
                            ROOT
                        )
                    ),

                "checkpoint_sha256":
                    EXPECTED_M4_CHECKPOINT_SHA256,

                "epoch_log_sha256":
                    EXPECTED_M4_EPOCHS_SHA256,

                "summary_sha256":
                    EXPECTED_M4_SUMMARY_SHA256,
            },
        },

        "phase_a_fusion_ranking": [
            {
                "rank":
                    1,

                "variant":
                    "M3",

                "val_macro_ap":
                    decision[
                        "m3_ap"
                    ],
            },

            {
                "rank":
                    2,

                "variant":
                    "M4",

                "val_macro_ap":
                    decision[
                        "m4_ap"
                    ],
            },
        ],

        "phase_b_gate": {
            "application":
                "per_variant",

            "rule":
                (
                    "Skip Phase B only if the "
                    "variant's Phase-A validation "
                    "macro AP strictly exceeds "
                    "both M1 and M2."
                ),

            "M3": {
                "beats_M1":
                    decision[
                        "m3_beats_m1"
                    ],

                "beats_M2":
                    decision[
                        "m3_beats_m2"
                    ],

                "phase_b_required":
                    decision[
                        "m3_required"
                    ],
            },

            "M4": {
                "beats_M1":
                    decision[
                        "m4_beats_m1"
                    ],

                "beats_M2":
                    decision[
                        "m4_beats_m2"
                    ],

                "phase_b_required":
                    decision[
                        "m4_required"
                    ],
            },

            "required_variants": [
                "M3",
                "M4",
            ],
        },

        "next_stage":
            (
                "Task-3 Phase-B selective-"
                "unfreezing preflight"
            ),
    }

    write_json_atomic(
        OUTPUT,
        artifact,
    )

    check(
        OUTPUT.is_file(),
        (
            "Task-3 Phase-A result "
            "freeze artifact created"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Phase-A result freeze failed"
        )

    digest = sha256_file(
        OUTPUT
    )

    print()
    print(
        "Phase-A results freeze SHA256:"
    )

    print(
        f"  {digest}"
    )

    return digest


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 52 — FREEZE TASK-3 PHASE-A RESULTS"
    )

    print(
        f"Project root: "
        f"{ROOT}"
    )

    print()

    print(
        "Model construction:      NO"
    )

    print(
        "Training:                NO"
    )

    print(
        "Forward/inference:       NO"
    )

    print(
        "Threshold tuning:        NO"
    )

    print(
        "TEST access:             NO"
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "  Freeze accepted M3/M4 "
        "Phase-A outcomes and apply "
        "the already-frozen Phase-B gate."
    )

    try:

        verify_output_protection()

        verify_file_identities()

        verify_design()

        m3_summary = verify_m3()

        m4_summary = verify_m4()

        decision = determine_phase_b(
            m3_summary=
                m3_summary,

            m4_summary=
                m4_summary,
        )

        freeze_sha = freeze_results(
            decision
        )

    except Exception as exc:

        banner(
            "SCRIPT 52 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        if FAILURES:

            print()

            print(
                f"Failed checks: "
                f"{len(FAILURES)}"
            )

            for message in FAILURES:

                print(
                    f"  - {message}"
                )

        print()

        if OUTPUT_TEMP.exists():

            print(
                "Temporary output exists:"
            )

            print(
                f"  {OUTPUT_TEMP}"
            )

        print()

        print(
            "Do NOT begin Phase B."
        )

        return 1

    banner(
        "SCRIPT 52 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Frozen Phase-A results:"
    )

    print(
        f"  M1 BERT-only = "
        f"{M1_VAL_MACRO_AP:.8f}"
    )

    print(
        f"  M2 GAT/G2    = "
        f"{M2_VAL_MACRO_AP:.8f}"
    )

    print(
        f"  M3 Phase A   = "
        f"{decision['m3_ap']:.8f}"
    )

    print(
        f"  M4 Phase A   = "
        f"{decision['m4_ap']:.8f}"
    )

    print()

    print(
        "Phase-A fusion ranking:"
    )

    print(
        "  1. M3 Early Fusion"
    )

    print(
        "  2. M4 Cross-Attention"
    )

    print()

    print(
        "Frozen Phase-B decisions:"
    )

    print(
        "  M3 Phase B required = True"
    )

    print(
        "  M4 Phase B required = True"
    )

    print()

    print(
        "TEST examples evaluated: 0"
    )

    print(
        "TEST target rows loaded:  0"
    )

    print(
        "Threshold tuning:          0"
    )

    print()

    print(
        "Result-freeze artifact:"
    )

    print(
        f"  {OUTPUT}"
    )

    print()

    print(
        "Result-freeze SHA256:"
    )

    print(
        f"  {freeze_sha}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  Phase-B selective-"
        "unfreezing preflight."
    )

    print()

    print(
        "STOP HERE."
    )

    print(
        "Review Script 52 output before "
        "constructing any Phase-B model."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )