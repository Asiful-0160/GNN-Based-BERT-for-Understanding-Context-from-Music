#!/usr/bin/env python3
"""
Script 55 — Freeze Task-3 Validation-Only Model Selection
==========================================================

Purpose
-------
Use the completed seed-42 validation results to select:

1. Best M3 phase
2. Best M4 phase
3. Overall Task-3 fusion winner

Selection metric:
    VAL macro mean per-label Average Precision

No model inference.
No training.
No threshold tuning.
No TEST labels or TEST metrics.

Expected outcome from existing formal results:
    M3 winner -> Phase B
    M4 winner -> Phase B
    Task-3 winner -> M3 Phase B
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = ROOT / "results/runs/task3"
MODEL_DIR = ROOT / "models/task3_candidates"

DESIGN = (
    RUN_DIR / "task3_phaseA_design.json"
)

PHASEA_FREEZE = (
    RUN_DIR / "task3_phaseA_results_freeze.json"
)


# M3 Phase A
M3A_CHECKPOINT = (
    MODEL_DIR
    / "task3_M3_phaseA_seed42_20260910T130250Z_best.pt"
)

M3A_SUMMARY = (
    RUN_DIR
    / "task3_M3_phaseA_seed42_20260910T130250Z_summary.json"
)


# M4 Phase A
M4A_CHECKPOINT = (
    MODEL_DIR
    / "task3_M4_phaseA_seed42_20260910T131031Z_best.pt"
)

M4A_SUMMARY = (
    RUN_DIR
    / "task3_M4_phaseA_seed42_20260910T131031Z_summary.json"
)


# M3 Phase B
M3B_CHECKPOINT = (
    MODEL_DIR
    / "task3_M3_phaseB_seed42_20260910T133125Z_best.pt"
)

M3B_SUMMARY = (
    RUN_DIR
    / "task3_M3_phaseB_seed42_20260910T133125Z_summary.json"
)


# M4 Phase B
M4B_CHECKPOINT = (
    MODEL_DIR
    / "task3_M4_phaseB_seed42_20260910T133743Z_best.pt"
)

M4B_SUMMARY = (
    RUN_DIR
    / "task3_M4_phaseB_seed42_20260910T133743Z_summary.json"
)


OUTPUT = (
    RUN_DIR / "task3_model_selection.json"
)

OUTPUT_TEMP = Path(
    str(OUTPUT) + ".__tmp__"
)


# =============================================================================
# ACCEPTED HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "0a9d50413b65aeed58d02d6a2dff70d99363ae25fee115d30bc9749f54e1c9ef"
)

EXPECTED_PHASEA_FREEZE_SHA256 = (
    "f063288047d3d04ce60713180a4b5c5746d8c740080ea78b3cf899702db987be"
)


# M3 Phase A
EXPECTED_M3A_CHECKPOINT_SHA256 = (
    "f23a0a5f8bb604672de640eca5cbfbb189e99febd32cab02f082a1de9a1fa5b8"
)

EXPECTED_M3A_SUMMARY_SHA256 = (
    "85ecacdcedd391cb0eae6231239075e88d017b7141123ae0fa6a3261b82e7412"
)


# M4 Phase A
EXPECTED_M4A_CHECKPOINT_SHA256 = (
    "ea210e30a932c124ee07b6a5fc90d068e0201152301998ed1eeb67559dd2e112"
)

EXPECTED_M4A_SUMMARY_SHA256 = (
    "86e34cb983b23e56565d496668f2585cb62727507fdb49612f849d973d0ad1af"
)


# M3 Phase B
EXPECTED_M3B_CHECKPOINT_SHA256 = (
    "e853c64b22a20f9000ecb3260fd45f8c09a60379eee04294606193e17353dd89"
)

EXPECTED_M3B_SUMMARY_SHA256 = (
    "8fae01740e3bb9ae3e77a107fe4a50e47c11d260b8c514d23e5bb8775ad55dbd"
)


# M4 Phase B
EXPECTED_M4B_CHECKPOINT_SHA256 = (
    "7a27a270074ef4e50143aec94b8b5d7ada001de0c0708c18b3aea406d159c165"
)

EXPECTED_M4B_SUMMARY_SHA256 = (
    "84956691106206327bc9e42798de33e1ab5fa8125d4921f65e9363c6682a6d80"
)


# =============================================================================
# BASELINE REFERENCES
# =============================================================================

M1_BERT_VAL_MACRO_AP = (
    0.8405750022914198
)

M2_GAT_VAL_MACRO_AP = (
    0.18752693895679443
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        value = json.load(f)

    if not isinstance(value, dict):

        raise RuntimeError(
            f"{path.name} is not a JSON object"
        )

    return value


def write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:

    if OUTPUT_TEMP.exists():

        raise RuntimeError(
            f"Temporary output already exists: "
            f"{OUTPUT_TEMP}"
        )

    with OUTPUT_TEMP.open(
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

    os.replace(
        OUTPUT_TEMP,
        path,
    )


def banner(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:

        raise RuntimeError(
            message
        )

    print(
        f"PASS  {message}"
    )


# =============================================================================
# ARTIFACT VALIDATION
# =============================================================================

def verify_artifact(
    path: Path,
    expected_sha: str,
) -> None:

    require(
        path.is_file(),
        f"artifact exists: {path.relative_to(ROOT)}",
    )

    observed = sha256_file(
        path
    )

    require(
        observed == expected_sha,
        f"{path.name} SHA256 matches accepted identity",
    )


def verify_summary(
    path: Path,
    *,
    variant: str,
    phase: str,
    checkpoint_sha: str,
) -> dict[str, Any]:

    summary = load_json(
        path
    )

    require(
        summary.get("task") == "task3",
        f"{variant}-{phase} task = task3",
    )

    require(
        summary.get("variant") == variant,
        f"{variant}-{phase} variant is correct",
    )

    require(
        summary.get("phase") == phase,
        f"{variant}-{phase} phase is correct",
    )

    require(
        summary.get("formal_run") is True,
        f"{variant}-{phase} is a formal run",
    )

    require(
        int(summary.get("seed", -1)) == 42,
        f"{variant}-{phase} seed = 42",
    )

    require(
        summary.get("checkpoint_sha256")
        == checkpoint_sha,
        (
            f"{variant}-{phase} summary references "
            "accepted checkpoint"
        ),
    )

    require(
        summary.get("test_split_used") is False,
        f"{variant}-{phase} records TEST unused",
    )

    require(
        int(
            summary.get(
                "test_target_rows_loaded",
                -1,
            )
        )
        == 0,
        (
            f"{variant}-{phase} records zero "
            "TEST target rows"
        ),
    )

    require(
        int(
            summary.get(
                "test_examples_evaluated",
                -1,
            )
        )
        == 0,
        (
            f"{variant}-{phase} records zero "
            "TEST inference"
        ),
    )

    require(
        summary.get(
            "threshold_tuning_used"
        )
        is False,
        (
            f"{variant}-{phase} records "
            "no threshold tuning"
        ),
    )

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 55 — TASK-3 VALIDATION-ONLY MODEL SELECTION"
    )

    print(
        "Selection basis: VALIDATION only"
    )

    print(
        "Training performed: 0"
    )

    print(
        "TEST inference:     0"
    )

    print(
        "TEST labels:        0"
    )

    print(
        "Threshold tuning:   0"
    )

    try:

        # ---------------------------------------------------------------------
        # Protect the final selection.
        # ---------------------------------------------------------------------

        banner(
            "1. OUTPUT PROTECTION"
        )

        require(
            not OUTPUT.exists(),
            (
                "Task-3 model-selection artifact "
                "does not already exist"
            ),
        )

        require(
            not OUTPUT_TEMP.exists(),
            (
                "temporary Task-3 selection "
                "artifact does not exist"
            ),
        )

        # ---------------------------------------------------------------------
        # Accepted dependencies.
        # ---------------------------------------------------------------------

        banner(
            "2. ACCEPTED FORMAL ARTIFACTS"
        )

        verify_artifact(
            DESIGN,
            EXPECTED_DESIGN_SHA256,
        )

        verify_artifact(
            PHASEA_FREEZE,
            EXPECTED_PHASEA_FREEZE_SHA256,
        )

        verify_artifact(
            M3A_CHECKPOINT,
            EXPECTED_M3A_CHECKPOINT_SHA256,
        )

        verify_artifact(
            M3A_SUMMARY,
            EXPECTED_M3A_SUMMARY_SHA256,
        )

        verify_artifact(
            M4A_CHECKPOINT,
            EXPECTED_M4A_CHECKPOINT_SHA256,
        )

        verify_artifact(
            M4A_SUMMARY,
            EXPECTED_M4A_SUMMARY_SHA256,
        )

        verify_artifact(
            M3B_CHECKPOINT,
            EXPECTED_M3B_CHECKPOINT_SHA256,
        )

        verify_artifact(
            M3B_SUMMARY,
            EXPECTED_M3B_SUMMARY_SHA256,
        )

        verify_artifact(
            M4B_CHECKPOINT,
            EXPECTED_M4B_CHECKPOINT_SHA256,
        )

        verify_artifact(
            M4B_SUMMARY,
            EXPECTED_M4B_SUMMARY_SHA256,
        )

        # ---------------------------------------------------------------------
        # Read formal results.
        # ---------------------------------------------------------------------

        banner(
            "3. LOAD VALIDATION RESULTS"
        )

        m3a = verify_summary(
            M3A_SUMMARY,
            variant="M3",
            phase="A",
            checkpoint_sha=
                EXPECTED_M3A_CHECKPOINT_SHA256,
        )

        m4a = verify_summary(
            M4A_SUMMARY,
            variant="M4",
            phase="A",
            checkpoint_sha=
                EXPECTED_M4A_CHECKPOINT_SHA256,
        )

        m3b = verify_summary(
            M3B_SUMMARY,
            variant="M3",
            phase="B",
            checkpoint_sha=
                EXPECTED_M3B_CHECKPOINT_SHA256,
        )

        m4b = verify_summary(
            M4B_SUMMARY,
            variant="M4",
            phase="B",
            checkpoint_sha=
                EXPECTED_M4B_CHECKPOINT_SHA256,
        )

        candidates = {
            "M3_phaseA": {
                "variant": "M3",
                "phase": "A",
                "model": "EarlyFusion",
                "val_macro_ap":
                    float(
                        m3a[
                            "best_val_auc_pr_macro_mean_ap"
                        ]
                    ),
                "checkpoint":
                    str(
                        M3A_CHECKPOINT.relative_to(ROOT)
                    ),
                "checkpoint_sha256":
                    EXPECTED_M3A_CHECKPOINT_SHA256,
            },

            "M3_phaseB": {
                "variant": "M3",
                "phase": "B",
                "model": "EarlyFusionJoint",
                "val_macro_ap":
                    float(
                        m3b[
                            "best_val_auc_pr_macro_mean_ap"
                        ]
                    ),
                "checkpoint":
                    str(
                        M3B_CHECKPOINT.relative_to(ROOT)
                    ),
                "checkpoint_sha256":
                    EXPECTED_M3B_CHECKPOINT_SHA256,
            },

            "M4_phaseA": {
                "variant": "M4",
                "phase": "A",
                "model": "CrossAttention",
                "val_macro_ap":
                    float(
                        m4a[
                            "best_val_auc_pr_macro_mean_ap"
                        ]
                    ),
                "checkpoint":
                    str(
                        M4A_CHECKPOINT.relative_to(ROOT)
                    ),
                "checkpoint_sha256":
                    EXPECTED_M4A_CHECKPOINT_SHA256,
            },

            "M4_phaseB": {
                "variant": "M4",
                "phase": "B",
                "model": "CrossAttentionJoint",
                "val_macro_ap":
                    float(
                        m4b[
                            "best_val_auc_pr_macro_mean_ap"
                        ]
                    ),
                "checkpoint":
                    str(
                        M4B_CHECKPOINT.relative_to(ROOT)
                    ),
                "checkpoint_sha256":
                    EXPECTED_M4B_CHECKPOINT_SHA256,
            },
        }

        for name, result in candidates.items():

            print(
                f"{name:12s} "
                f"{result['val_macro_ap']:.12f}"
            )

        # ---------------------------------------------------------------------
        # Select best phase independently for M3 and M4.
        # ---------------------------------------------------------------------

        banner(
            "4. VARIANT SELECTION"
        )

        m3_winner_name = max(
            (
                "M3_phaseA",
                "M3_phaseB",
            ),
            key=lambda key:
                candidates[key][
                    "val_macro_ap"
                ],
        )

        m4_winner_name = max(
            (
                "M4_phaseA",
                "M4_phaseB",
            ),
            key=lambda key:
                candidates[key][
                    "val_macro_ap"
                ],
        )

        require(
            m3_winner_name
            == "M3_phaseB",
            (
                "M3 Phase B is the "
                "validation-selected M3 configuration"
            ),
        )

        require(
            m4_winner_name
            == "M4_phaseB",
            (
                "M4 Phase B is the "
                "validation-selected M4 configuration"
            ),
        )

        m3_winner = candidates[
            m3_winner_name
        ]

        m4_winner = candidates[
            m4_winner_name
        ]

        # ---------------------------------------------------------------------
        # Overall Task-3 fusion selection.
        # ---------------------------------------------------------------------

        banner(
            "5. TASK-3 FUSION SELECTION"
        )

        fusion_winners = {
            m3_winner_name:
                m3_winner,

            m4_winner_name:
                m4_winner,
        }

        overall_name = max(
            fusion_winners,
            key=lambda key:
                fusion_winners[key][
                    "val_macro_ap"
                ],
        )

        overall = fusion_winners[
            overall_name
        ]

        require(
            overall_name
            == "M3_phaseB",
            (
                "overall Task-3 winner "
                "is M3 Phase B"
            ),
        )

        best_fusion_ap = float(
            overall[
                "val_macro_ap"
            ]
        )

        beats_m1 = (
            best_fusion_ap
            >
            M1_BERT_VAL_MACRO_AP
        )

        beats_m2 = (
            best_fusion_ap
            >
            M2_GAT_VAL_MACRO_AP
        )

        print()
        print(
            f"Selected fusion VAL AP: "
            f"{best_fusion_ap:.12f}"
        )

        print(
            f"M1 BERT VAL AP:          "
            f"{M1_BERT_VAL_MACRO_AP:.12f}"
        )

        print(
            f"M2 GAT VAL AP:           "
            f"{M2_GAT_VAL_MACRO_AP:.12f}"
        )

        print()

        print(
            f"Selected fusion > M1: "
            f"{beats_m1}"
        )

        print(
            f"Selected fusion > M2: "
            f"{beats_m2}"
        )

        # ---------------------------------------------------------------------
        # Freeze selection.
        # ---------------------------------------------------------------------

        banner(
            "6. FREEZE TASK-3 SELECTION"
        )

        ranking = sorted(
            candidates.items(),
            key=lambda item:
                item[1]["val_macro_ap"],
            reverse=True,
        )

        artifact = {
            "artifact_type":
                "task3_model_selection",

            "version":
                1,

            "selection_seed":
                42,

            "selection_basis":
                "VALIDATION only",

            "selection_metric":
                (
                    "macro mean per-label "
                    "Average Precision"
                ),

            "test_examples_evaluated":
                0,

            "test_target_rows_loaded":
                0,

            "threshold_tuning_used":
                False,

            "dependencies": {
                "task3_phaseA_design_sha256":
                    EXPECTED_DESIGN_SHA256,

                "task3_phaseA_results_freeze_sha256":
                    EXPECTED_PHASEA_FREEZE_SHA256,
            },

            "baseline_references": {
                "M1_T1B_BERT_only_val_macro_ap":
                    M1_BERT_VAL_MACRO_AP,

                "M2_A4_GAT_G2_val_macro_ap":
                    M2_GAT_VAL_MACRO_AP,
            },

            "candidates":
                candidates,

            "ranking": [
                {
                    "rank":
                        rank,

                    "candidate":
                        name,

                    "val_macro_ap":
                        result[
                            "val_macro_ap"
                        ],
                }

                for rank, (
                    name,
                    result,
                )
                in enumerate(
                    ranking,
                    start=1,
                )
            ],

            "selected_within_variant": {
                "M3":
                    m3_winner_name,

                "M4":
                    m4_winner_name,
            },

            "selected_task3_fusion":
                overall_name,

            "selected_checkpoint":
                overall[
                    "checkpoint"
                ],

            "selected_checkpoint_sha256":
                overall[
                    "checkpoint_sha256"
                ],

            "selected_val_macro_ap":
                best_fusion_ap,

            "comparison_to_baselines": {
                "beats_M1_BERT_on_seed42_validation":
                    beats_m1,

                "delta_vs_M1":
                    (
                        best_fusion_ap
                        - M1_BERT_VAL_MACRO_AP
                    ),

                "beats_M2_GNN_on_seed42_validation":
                    beats_m2,

                "delta_vs_M2":
                    (
                        best_fusion_ap
                        - M2_GAT_VAL_MACRO_AP
                    ),
            },

            "interpretation_status":
                (
                    "single-seed model selection only; "
                    "do not make final superiority claim "
                    "before planned multi-seed and TEST evaluation"
                ),

            "next_stage":
                (
                    "post-selection validation protocol "
                    "before final TEST"
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
            "SCRIPT 55 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "Do NOT proceed to threshold tuning, "
            "multi-seed reruns, or TEST."
        )

        return 1

    banner(
        "SCRIPT 55 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "Validation-only Task-3 ranking:"
    )

    for rank, (
        name,
        result,
    ) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"  {rank}. "
            f"{name:12s} "
            f"{result['val_macro_ap']:.8f}"
        )

    print()

    print(
        "Selected within M3:"
    )
    print(
        f"  {m3_winner_name}"
    )

    print()

    print(
        "Selected within M4:"
    )
    print(
        f"  {m4_winner_name}"
    )

    print()

    print(
        "Selected Task-3 fusion:"
    )
    print(
        f"  {overall_name}"
    )

    print(
        f"  VAL macro AP = "
        f"{best_fusion_ap:.8f}"
    )

    print()

    print(
        f"M1 BERT-only = "
        f"{M1_BERT_VAL_MACRO_AP:.8f}"
    )

    print(
        f"Delta vs M1   = "
        f"{best_fusion_ap - M1_BERT_VAL_MACRO_AP:+.8f}"
    )

    print()

    print(
        "Seed-42 conclusion:"
    )

    print(
        "  Fusion beats BERT: "
        f"{beats_m1}"
    )

    print(
        "  Fusion beats GNN:  "
        f"{beats_m2}"
    )

    print()

    print(
        "TEST inference:      0"
    )

    print(
        "TEST labels:         0"
    )

    print(
        "Threshold tuning:    0"
    )

    print()

    print(
        "Selection artifact:"
    )

    print(
        f"  {OUTPUT.relative_to(ROOT)}"
    )

    print()

    print(
        "Selection SHA256:"
    )

    print(
        f"  {output_sha}"
    )

    print()

    print(
        "STOP HERE."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())