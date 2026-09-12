#!/usr/bin/env python3
"""
Script 61 — Post-hoc Statistical / Error / Fusion Analysis
===========================================================

Uses ONLY the frozen outputs saved by Script 60.

NO:
    model loading
    model inference
    training
    threshold tuning
    model selection

Primary comparison:
    Early Fusion M3 Phase B
        vs
    BERT T1-B

Both have final runs for:
    seed 42
    seed 1337
    seed 2026

Analyses
--------
1. Verify frozen Script-60 outputs.
2. Reproduce final BERT / Early-Fusion point metrics.
3. Report paired seed-wise metric differences.
4. Exploratory paired track-level bootstrap:
       2000 resamples
       seed = 42
       95% percentile interval
   for:
       Macro-F1
       Micro-F1
       macro AUC-PR
5. Per-label fusion improvement/harm analysis.
6. Five lowest-support labels.
7. Multi-label confusion/substitution analysis.
8. Representative TEST case-study candidates using saved
   probabilities only.

Important interpretation
------------------------
The bootstrap is POST-HOC exploratory uncertainty analysis.
It is not a preregistered confirmatory significance test.
No p-value or superiority claim is generated automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

FINAL_TEST_DIR = (
    ROOT
    / "results/runs/final_eval/final_test"
)

EVALUATION_JSON = (
    FINAL_TEST_DIR
    / "final_test_evaluation.json"
)

PREDICTIONS = (
    FINAL_TEST_DIR
    / "final_test_predictions.npz"
)

PER_RUN_METRICS = (
    FINAL_TEST_DIR
    / "final_test_metrics_per_run.csv"
)

PER_LABEL_SUMMARY = (
    FINAL_TEST_DIR
    / "final_test_per_label_summary.csv"
)

COMPARISON_TABLE = (
    FINAL_TEST_DIR
    / "final_comparison_table.csv"
)


OUTPUT_DIR = (
    FINAL_TEST_DIR
    / "posthoc_analysis"
)

TEMP_DIR = (
    FINAL_TEST_DIR
    / "posthoc_analysis.__tmp__"
)


# =============================================================================
# FROZEN SCRIPT-60 IDENTITY
# =============================================================================

EXPECTED_EVALUATION_SHA256 = (
    "8c6db5aed0cd23b7300da3b403089e7048f7f4ffcb64dcb8f3d1f555bd59f35d"
)


# =============================================================================
# FIXED ANALYSIS CONTRACT
# =============================================================================

FINAL_SEEDS = (
    42,
    1337,
    2026,
)

NUM_LABELS = 20
N_TEST = 514

BERT_THRESHOLD = 0.70
M3_THRESHOLD = 0.70

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42

CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5


LABEL_NAMES = [
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
]


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
) -> None:

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
            f"{path.name} is not a JSON object"
        )

    return value


def save_json(
    value: dict[str, Any],
    path: Path,
) -> None:

    with path.open(
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


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    prediction = (
        probability
        >= threshold
    ).astype(
        np.int64
    )

    macro_f1 = float(
        f1_score(
            y_true,
            prediction,
            average="macro",
            zero_division=0,
        )
    )

    micro_f1 = float(
        f1_score(
            y_true,
            prediction,
            average="micro",
            zero_division=0,
        )
    )

    macro_ap = float(
        np.mean(
            average_precision_score(
                y_true,
                probability,
                average=None,
            )
        )
    )

    return {
        "macro_f1":
            macro_f1,

        "micro_f1":
            micro_f1,

        "macro_ap":
            macro_ap,
    }


def per_label_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    prediction = (
        probability
        >= threshold
    ).astype(
        np.int64
    )

    f1 = f1_score(
        y_true,
        prediction,
        average=None,
        zero_division=0,
    ).astype(
        np.float64
    )

    ap = average_precision_score(
        y_true,
        probability,
        average=None,
    ).astype(
        np.float64
    )

    return (
        f1,
        ap,
    )


# =============================================================================
# VERIFY FROZEN SCRIPT-60 OUTPUTS
# =============================================================================

def verify_outputs():

    banner(
        "1. VERIFY FROZEN FINAL TEST OUTPUTS"
    )

    require(
        FINAL_TEST_DIR.is_dir(),
        (
            "Script-60 final TEST "
            "directory exists"
        ),
    )

    require(
        EVALUATION_JSON.is_file(),
        (
            "final_test_evaluation.json exists"
        ),
    )

    require(
        sha256_file(
            EVALUATION_JSON
        )
        ==
        EXPECTED_EVALUATION_SHA256,
        (
            "Script-60 final evaluation "
            "JSON matches frozen SHA256"
        ),
    )

    evaluation = load_json(
        EVALUATION_JSON
    )

    require(
        evaluation.get(
            "artifact_type"
        )
        == "final_test_evaluation",
        (
            "final evaluation artifact "
            "type is correct"
        ),
    )

    require(
        evaluation.get(
            "test_evaluation_status"
        )
        == "FINAL",
        (
            "TEST evaluation status = FINAL"
        ),
    )

    require(
        evaluation.get(
            "threshold_tuning_on_test"
        )
        is False,
        (
            "no TEST threshold tuning "
            "was performed"
        ),
    )

    require(
        evaluation.get(
            "training_after_test_access"
        )
        is False,
        (
            "no training occurred "
            "after TEST access"
        ),
    )

    require(
        evaluation.get(
            "model_selection_after_test_access"
        )
        is False,
        (
            "no model selection occurred "
            "after TEST access"
        ),
    )

    require(
        int(
            evaluation[
                "test_examples"
            ]
        )
        == N_TEST,
        (
            "frozen TEST size = 514"
        ),
    )

    require(
        tuple(
            evaluation[
                "final_seeds"
            ]
        )
        == FINAL_SEEDS,
        (
            "final seeds remain "
            "{42,1337,2026}"
        ),
    )

    # -------------------------------------------------------------------------
    # Verify every Script-60 saved artifact using hashes embedded
    # in the frozen final evaluation JSON.
    # -------------------------------------------------------------------------

    output_hashes = evaluation.get(
        "output_hashes",
        {},
    )

    require(
        isinstance(
            output_hashes,
            dict,
        )
        and len(
            output_hashes
        )
        >= 6,
        (
            "Script-60 output-hash "
            "manifest is present"
        ),
    )

    for filename, expected_sha in (
        output_hashes.items()
    ):

        path = (
            FINAL_TEST_DIR
            / filename
        )

        require(
            path.is_file(),
            (
                f"saved final output exists: "
                f"{filename}"
            ),
        )

        require(
            sha256_file(
                path
            )
            == expected_sha,
            (
                f"{filename} matches "
                "Script-60 SHA256"
            ),
        )

    require(
        not OUTPUT_DIR.exists(),
        (
            "post-hoc output directory "
            "does not already exist"
        ),
    )

    require(
        not TEMP_DIR.exists(),
        (
            "temporary post-hoc directory "
            "does not exist"
        ),
    )

    return evaluation


# =============================================================================
# LOAD SAVED TEST PROBABILITIES ONLY
# =============================================================================

def load_saved_predictions():

    banner(
        "2. LOAD SAVED TEST OUTPUTS ONLY"
    )

    required_probability_keys = {
        "y_true",
        "track_ids",
        "test_support",

        "prob__BERT_T1B__seed42",
        "prob__BERT_T1B__seed1337",
        "prob__BERT_T1B__seed2026",

        "prob__EarlyFusion_M3B__seed42",
        "prob__EarlyFusion_M3B__seed1337",
        "prob__EarlyFusion_M3B__seed2026",
    }

    with np.load(
        PREDICTIONS,
        allow_pickle=False,
    ) as cache:

        available = set(
            cache.files
        )

        require(
            required_probability_keys
            .issubset(
                available
            ),
            (
                "saved probability cache "
                "contains required BERT/M3 arrays"
            ),
        )

        y_true = np.asarray(
            cache[
                "y_true"
            ],
            dtype=np.int64,
        )

        track_ids = np.asarray(
            cache[
                "track_ids"
            ]
        ).astype(str)

        support = np.asarray(
            cache[
                "test_support"
            ],
            dtype=np.int64,
        )

        bert = {
            seed:
                np.asarray(
                    cache[
                        f"prob__BERT_T1B__seed{seed}"
                    ],
                    dtype=np.float64,
                )

            for seed in FINAL_SEEDS
        }

        fusion = {
            seed:
                np.asarray(
                    cache[
                        f"prob__EarlyFusion_M3B__seed{seed}"
                    ],
                    dtype=np.float64,
                )

            for seed in FINAL_SEEDS
        }

    require(
        y_true.shape
        == (
            N_TEST,
            NUM_LABELS,
        ),
        (
            "saved TEST target matrix "
            "= (514,20)"
        ),
    )

    require(
        len(track_ids)
        == N_TEST,
        (
            "saved TEST track IDs = 514"
        ),
    )

    require(
        support.shape
        == (
            NUM_LABELS,
        ),
        (
            "saved TEST support vector "
            "= 20 labels"
        ),
    )

    require(
        np.array_equal(
            support,
            y_true.sum(
                axis=0
            ),
        ),
        (
            "saved support matches "
            "saved TEST targets"
        ),
    )

    for seed in FINAL_SEEDS:

        require(
            bert[
                seed
            ].shape
            == (
                N_TEST,
                NUM_LABELS,
            ),
            (
                f"BERT seed {seed} "
                "probabilities = (514,20)"
            ),
        )

        require(
            fusion[
                seed
            ].shape
            == (
                N_TEST,
                NUM_LABELS,
            ),
            (
                f"M3 seed {seed} "
                "probabilities = (514,20)"
            ),
        )

        require(
            np.isfinite(
                bert[
                    seed
                ]
            ).all(),
            (
                f"BERT seed {seed} "
                "probabilities are finite"
            ),
        )

        require(
            np.isfinite(
                fusion[
                    seed
                ]
            ).all(),
            (
                f"M3 seed {seed} "
                "probabilities are finite"
            ),
        )

    print()
    print(
        "Model inference performed: 0"
    )

    print(
        "Threshold tuning performed: 0"
    )

    return (
        y_true,
        track_ids,
        support,
        bert,
        fusion,
    )


# =============================================================================
# POINT METRICS + SEED DIFFERENCES
# =============================================================================

def analyze_seed_results(
    evaluation,
    y_true,
    bert,
    fusion,
):

    banner(
        "3. REPRODUCE BERT VS EARLY-FUSION FINAL METRICS"
    )

    rows = []

    for seed in FINAL_SEEDS:

        bert_metrics = compute_metrics(
            y_true,
            bert[
                seed
            ],
            BERT_THRESHOLD,
        )

        fusion_metrics = compute_metrics(
            y_true,
            fusion[
                seed
            ],
            M3_THRESHOLD,
        )

        row = {
            "seed":
                seed,

            "bert_macro_f1":
                bert_metrics[
                    "macro_f1"
                ],

            "fusion_macro_f1":
                fusion_metrics[
                    "macro_f1"
                ],

            "delta_macro_f1":
                fusion_metrics[
                    "macro_f1"
                ]
                -
                bert_metrics[
                    "macro_f1"
                ],

            "bert_micro_f1":
                bert_metrics[
                    "micro_f1"
                ],

            "fusion_micro_f1":
                fusion_metrics[
                    "micro_f1"
                ],

            "delta_micro_f1":
                fusion_metrics[
                    "micro_f1"
                ]
                -
                bert_metrics[
                    "micro_f1"
                ],

            "bert_macro_ap":
                bert_metrics[
                    "macro_ap"
                ],

            "fusion_macro_ap":
                fusion_metrics[
                    "macro_ap"
                ],

            "delta_macro_ap":
                fusion_metrics[
                    "macro_ap"
                ]
                -
                bert_metrics[
                    "macro_ap"
                ],
        }

        rows.append(
            row
        )

        print()
        print(
            f"Seed {seed}"
        )

        print(
            f"  BERT AUC-PR   = "
            f"{row['bert_macro_ap']:.8f}"
        )

        print(
            f"  Fusion AUC-PR = "
            f"{row['fusion_macro_ap']:.8f}"
        )

        print(
            f"  Delta         = "
            f"{row['delta_macro_ap']:+.8f}"
        )

    frame = pd.DataFrame(
        rows
    )

    summary = {}

    for metric in (
        "macro_f1",
        "micro_f1",
        "macro_ap",
    ):

        values = (
            frame[
                f"delta_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        summary[
            metric
        ] = {
            "mean_paired_seed_delta":
                float(
                    values.mean()
                ),

            "sample_std_paired_seed_delta":
                float(
                    values.std(
                        ddof=1
                    )
                ),

            "seed_deltas":
                [
                    float(x)
                    for x in values
                ],
        }

    # -------------------------------------------------------------------------
    # Check against Script-60 frozen table.
    # -------------------------------------------------------------------------

    comparison = {
        row[
            "model"
        ]:
            row

        for row
        in evaluation[
            "comparison_table"
        ]
    }

    bert_macro_ap = float(
        frame[
            "bert_macro_ap"
        ].mean()
    )

    fusion_macro_ap = float(
        frame[
            "fusion_macro_ap"
        ].mean()
    )

    require(
        np.isclose(
            bert_macro_ap,
            float(
                comparison[
                    "BERT"
                ][
                    "auc_pr_mean"
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "recomputed BERT mean "
            "AUC-PR matches Script 60"
        ),
    )

    require(
        np.isclose(
            fusion_macro_ap,
            float(
                comparison[
                    "Early Fusion"
                ][
                    "auc_pr_mean"
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        ),
        (
            "recomputed Early-Fusion mean "
            "AUC-PR matches Script 60"
        ),
    )

    print()
    print(
        "Three-seed means:"
    )

    print(
        f"  BERT AUC-PR   = "
        f"{bert_macro_ap:.8f}"
    )

    print(
        f"  Fusion AUC-PR = "
        f"{fusion_macro_ap:.8f}"
    )

    print(
        f"  Delta         = "
        f"{fusion_macro_ap - bert_macro_ap:+.8f}"
    )

    return (
        frame,
        summary,
    )


# =============================================================================
# PAIRED TRACK-LEVEL BOOTSTRAP
# =============================================================================

def bootstrap_comparison(
    y_true,
    bert,
    fusion,
):

    banner(
        "4. EXPLORATORY PAIRED TRACK-LEVEL BOOTSTRAP"
    )

    print(
        f"Replicates: {BOOTSTRAP_REPLICATES}"
    )

    print(
        f"Bootstrap RNG seed: {BOOTSTRAP_SEED}"
    )

    print(
        "Unit resampled: TEST track"
    )

    print(
        "Pairing: identical sampled tracks "
        "for BERT and Early Fusion"
    )

    print(
        "Aggregation: compute each seed's metric, "
        "then average across 3 seeds"
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []

    for bootstrap_index in range(
        BOOTSTRAP_REPLICATES
    ):

        indices = rng.integers(
            low=0,
            high=N_TEST,
            size=N_TEST,
        )

        y_sample = y_true[
            indices
        ]

        bert_macro_f1 = []
        bert_micro_f1 = []
        bert_macro_ap = []

        fusion_macro_f1 = []
        fusion_micro_f1 = []
        fusion_macro_ap = []

        for seed in FINAL_SEEDS:

            b = compute_metrics(
                y_sample,
                bert[
                    seed
                ][
                    indices
                ],
                BERT_THRESHOLD,
            )

            f = compute_metrics(
                y_sample,
                fusion[
                    seed
                ][
                    indices
                ],
                M3_THRESHOLD,
            )

            bert_macro_f1.append(
                b[
                    "macro_f1"
                ]
            )

            bert_micro_f1.append(
                b[
                    "micro_f1"
                ]
            )

            bert_macro_ap.append(
                b[
                    "macro_ap"
                ]
            )

            fusion_macro_f1.append(
                f[
                    "macro_f1"
                ]
            )

            fusion_micro_f1.append(
                f[
                    "micro_f1"
                ]
            )

            fusion_macro_ap.append(
                f[
                    "macro_ap"
                ]
            )

        rows.append(
            {
                "bootstrap":
                    bootstrap_index,

                "delta_macro_f1":
                    float(
                        np.mean(
                            fusion_macro_f1
                        )
                        -
                        np.mean(
                            bert_macro_f1
                        )
                    ),

                "delta_micro_f1":
                    float(
                        np.mean(
                            fusion_micro_f1
                        )
                        -
                        np.mean(
                            bert_micro_f1
                        )
                    ),

                "delta_macro_ap":
                    float(
                        np.mean(
                            fusion_macro_ap
                        )
                        -
                        np.mean(
                            bert_macro_ap
                        )
                    ),
            }
        )

        completed = (
            bootstrap_index
            + 1
        )

        if (
            completed == 1
            or completed % 250 == 0
            or completed
            == BOOTSTRAP_REPLICATES
        ):

            print(
                f"  bootstrap "
                f"{completed}/"
                f"{BOOTSTRAP_REPLICATES}"
            )

    frame = pd.DataFrame(
        rows
    )

    summary = {}

    print()
    print(
        "95% percentile intervals "
        "for Fusion - BERT:"
    )

    for metric in (
        "macro_f1",
        "micro_f1",
        "macro_ap",
    ):

        values = (
            frame[
                f"delta_{metric}"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        lower = float(
            np.percentile(
                values,
                CI_LOWER_PERCENTILE,
            )
        )

        upper = float(
            np.percentile(
                values,
                CI_UPPER_PERCENTILE,
            )
        )

        positive_fraction = float(
            np.mean(
                values > 0.0
            )
        )

        includes_zero = bool(
            lower <= 0.0 <= upper
        )

        summary[
            metric
        ] = {
            "ci_lower":
                lower,

            "ci_upper":
                upper,

            "includes_zero":
                includes_zero,

            "bootstrap_fraction_positive":
                positive_fraction,
        }

        print(
            f"  {metric:9s}: "
            f"[{lower:+.8f}, "
            f"{upper:+.8f}]  "
            f"includes_zero="
            f"{includes_zero}"
        )

    return (
        frame,
        summary,
    )


# =============================================================================
# PER-LABEL FUSION ANALYSIS
# =============================================================================

def analyze_labels(
    y_true,
    support,
    bert,
    fusion,
):

    banner(
        "5. PER-LABEL FUSION IMPROVEMENT / HARM"
    )

    bert_f1 = []
    bert_ap = []

    fusion_f1 = []
    fusion_ap = []

    for seed in FINAL_SEEDS:

        b_f1, b_ap = per_label_metrics(
            y_true,
            bert[
                seed
            ],
            BERT_THRESHOLD,
        )

        f_f1, f_ap = per_label_metrics(
            y_true,
            fusion[
                seed
            ],
            M3_THRESHOLD,
        )

        bert_f1.append(
            b_f1
        )

        bert_ap.append(
            b_ap
        )

        fusion_f1.append(
            f_f1
        )

        fusion_ap.append(
            f_ap
        )

    bert_f1 = np.stack(
        bert_f1,
        axis=0,
    )

    bert_ap = np.stack(
        bert_ap,
        axis=0,
    )

    fusion_f1 = np.stack(
        fusion_f1,
        axis=0,
    )

    fusion_ap = np.stack(
        fusion_ap,
        axis=0,
    )

    rows = []

    for index in range(
        NUM_LABELS
    ):

        b_f1_mean = float(
            bert_f1[
                :,
                index
            ].mean()
        )

        f_f1_mean = float(
            fusion_f1[
                :,
                index
            ].mean()
        )

        b_ap_mean = float(
            bert_ap[
                :,
                index
            ].mean()
        )

        f_ap_mean = float(
            fusion_ap[
                :,
                index
            ].mean()
        )

        rows.append(
            {
                "label_index":
                    index,

                "label":
                    LABEL_NAMES[
                        index
                    ],

                "test_support":
                    int(
                        support[
                            index
                        ]
                    ),

                "bert_f1_mean":
                    b_f1_mean,

                "fusion_f1_mean":
                    f_f1_mean,

                "delta_f1":
                    f_f1_mean
                    - b_f1_mean,

                "bert_ap_mean":
                    b_ap_mean,

                "fusion_ap_mean":
                    f_ap_mean,

                "delta_ap":
                    f_ap_mean
                    - b_ap_mean,
            }
        )

    frame = pd.DataFrame(
        rows
    )

    improved_ap = (
        frame.sort_values(
            "delta_ap",
            ascending=False,
        )
        .head(5)
    )

    harmed_ap = (
        frame.sort_values(
            "delta_ap",
            ascending=True,
        )
        .head(5)
    )

    improved_f1 = (
        frame.sort_values(
            "delta_f1",
            ascending=False,
        )
        .head(5)
    )

    harmed_f1 = (
        frame.sort_values(
            "delta_f1",
            ascending=True,
        )
        .head(5)
    )

    rare = (
        frame.sort_values(
            [
                "test_support",
                "label_index",
            ]
        )
        .head(5)
    )

    print()
    print(
        "Top 5 labels improved by fusion "
        "(AUC-PR):"
    )

    for _, row in (
        improved_ap.iterrows()
    ):

        print(
            f"  {row['label']:20s} "
            f"support={int(row['test_support']):3d}  "
            f"delta={row['delta_ap']:+.6f}"
        )

    print()
    print(
        "Top 5 labels harmed by fusion "
        "(AUC-PR):"
    )

    for _, row in (
        harmed_ap.iterrows()
    ):

        print(
            f"  {row['label']:20s} "
            f"support={int(row['test_support']):3d}  "
            f"delta={row['delta_ap']:+.6f}"
        )

    print()
    print(
        "Five lowest-support TEST labels:"
    )

    for _, row in (
        rare.iterrows()
    ):

        print(
            f"  {row['label']:20s} "
            f"support={int(row['test_support']):3d}  "
            f"AP delta={row['delta_ap']:+.6f}  "
            f"F1 delta={row['delta_f1']:+.6f}"
        )

    summary = {
        "top5_improved_auc_pr":
            improved_ap[
                [
                    "label",
                    "test_support",
                    "delta_ap",
                ]
            ].to_dict(
                orient="records"
            ),

        "top5_harmed_auc_pr":
            harmed_ap[
                [
                    "label",
                    "test_support",
                    "delta_ap",
                ]
            ].to_dict(
                orient="records"
            ),

        "top5_improved_f1":
            improved_f1[
                [
                    "label",
                    "test_support",
                    "delta_f1",
                ]
            ].to_dict(
                orient="records"
            ),

        "top5_harmed_f1":
            harmed_f1[
                [
                    "label",
                    "test_support",
                    "delta_f1",
                ]
            ].to_dict(
                orient="records"
            ),

        "five_lowest_support_labels":
            rare[
                [
                    "label",
                    "test_support",
                    "delta_ap",
                    "delta_f1",
                ]
            ].to_dict(
                orient="records"
            ),
    }

    return (
        frame,
        summary,
    )


# =============================================================================
# MULTI-LABEL CONFUSION / SUBSTITUTION ANALYSIS
# =============================================================================

def confusion_matrix(
    y_true,
    probabilities_by_seed,
    threshold,
):

    counts = np.zeros(
        (
            NUM_LABELS,
            NUM_LABELS,
        ),
        dtype=np.int64,
    )

    for seed in FINAL_SEEDS:

        prediction = (
            probabilities_by_seed[
                seed
            ]
            >= threshold
        ).astype(
            np.int64
        )

        false_negative = (
            (
                y_true == 1
            )
            &
            (
                prediction == 0
            )
        ).astype(
            np.int64
        )

        false_positive = (
            (
                y_true == 0
            )
            &
            (
                prediction == 1
            )
        ).astype(
            np.int64
        )

        # Rows:
        #   true label that was missed.
        #
        # Columns:
        #   wrong label predicted instead/on same example.
        counts += (
            false_negative.T
            @ false_positive
        )

    np.fill_diagonal(
        counts,
        0,
    )

    return counts


def analyze_confusions(
    y_true,
    bert,
    fusion,
):

    banner(
        "6. MULTI-LABEL CONFUSION / SUBSTITUTION ANALYSIS"
    )

    bert_counts = confusion_matrix(
        y_true,
        bert,
        BERT_THRESHOLD,
    )

    fusion_counts = confusion_matrix(
        y_true,
        fusion,
        M3_THRESHOLD,
    )

    rows = []

    for true_index in range(
        NUM_LABELS
    ):

        for wrong_index in range(
            NUM_LABELS
        ):

            if (
                true_index
                == wrong_index
            ):
                continue

            rows.append(
                {
                    "missed_true_label_index":
                        true_index,

                    "missed_true_label":
                        LABEL_NAMES[
                            true_index
                        ],

                    "wrong_predicted_label_index":
                        wrong_index,

                    "wrong_predicted_label":
                        LABEL_NAMES[
                            wrong_index
                        ],

                    "bert_count_across_3_seeds":
                        int(
                            bert_counts[
                                true_index,
                                wrong_index,
                            ]
                        ),

                    "fusion_count_across_3_seeds":
                        int(
                            fusion_counts[
                                true_index,
                                wrong_index,
                            ]
                        ),

                    "fusion_minus_bert":
                        int(
                            fusion_counts[
                                true_index,
                                wrong_index,
                            ]
                            -
                            bert_counts[
                                true_index,
                                wrong_index,
                            ]
                        ),
                }
            )

    frame = pd.DataFrame(
        rows
    )

    top_fusion = (
        frame.sort_values(
            [
                "fusion_count_across_3_seeds",
                "bert_count_across_3_seeds",
            ],
            ascending=False,
        )
        .head(10)
    )

    most_reduced = (
        frame.sort_values(
            "fusion_minus_bert",
            ascending=True,
        )
        .head(10)
    )

    most_increased = (
        frame.sort_values(
            "fusion_minus_bert",
            ascending=False,
        )
        .head(10)
    )

    print()
    print(
        "Most frequent Early-Fusion "
        "confusion pairs:"
    )

    for _, row in (
        top_fusion.iterrows()
    ):

        print(
            f"  true/missed: "
            f"{row['missed_true_label']:18s} "
            f"→ wrong: "
            f"{row['wrong_predicted_label']:18s} "
            f"count="
            f"{int(row['fusion_count_across_3_seeds'])}"
        )

    summary = {
        "top10_fusion_confusions":
            top_fusion.to_dict(
                orient="records"
            ),

        "top10_confusions_reduced_by_fusion":
            most_reduced.to_dict(
                orient="records"
            ),

        "top10_confusions_increased_by_fusion":
            most_increased.to_dict(
                orient="records"
            ),
    }

    return (
        frame,
        summary,
    )


# =============================================================================
# CASE-STUDY CANDIDATES — SAVED OUTPUTS ONLY
# =============================================================================

def labels_to_string(
    binary_vector,
) -> str:

    labels = [
        LABEL_NAMES[
            index
        ]

        for index in range(
            NUM_LABELS
        )

        if int(
            binary_vector[
                index
            ]
        )
        == 1
    ]

    return (
        " | ".join(
            labels
        )
        if labels
        else "<none>"
    )


def choose_case_candidates(
    y_true,
    track_ids,
    bert,
    fusion,
):

    banner(
        "7. CASE-STUDY CANDIDATES FROM SAVED OUTPUTS"
    )

    # Use the representative seed-42 final M3 run.
    m3_probability = fusion[
        42
    ]

    bert_probability = bert[
        42
    ]

    m3_prediction = (
        m3_probability
        >= M3_THRESHOLD
    ).astype(
        np.int64
    )

    bert_prediction = (
        bert_probability
        >= BERT_THRESHOLD
    ).astype(
        np.int64
    )

    exact_match = np.all(
        m3_prediction
        == y_true,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Correct high-confidence:
    # exact label-set match and highest average classification confidence.
    # -------------------------------------------------------------------------

    confidence = np.mean(
        np.where(
            m3_prediction == 1,
            m3_probability,
            1.0 - m3_probability,
        ),
        axis=1,
    )

    correct_candidates = np.flatnonzero(
        exact_match
        &
        (
            y_true.sum(
                axis=1
            )
            > 0
        )
    )

    if len(
        correct_candidates
    ) == 0:

        correct_candidates = np.flatnonzero(
            exact_match
        )

    require(
        len(
            correct_candidates
        )
        > 0,
        (
            "at least one exact-match "
            "M3 TEST case exists"
        ),
    )

    correct_index = int(
        correct_candidates[
            np.argmax(
                confidence[
                    correct_candidates
                ]
            )
        ]
    )

    # -------------------------------------------------------------------------
    # High-confidence incorrect:
    # largest confidence attached to any false positive / false negative.
    # -------------------------------------------------------------------------

    fp = (
        (
            m3_prediction == 1
        )
        &
        (
            y_true == 0
        )
    )

    fn = (
        (
            m3_prediction == 0
        )
        &
        (
            y_true == 1
        )
    )

    wrong_confidence = np.maximum(
        np.max(
            np.where(
                fp,
                m3_probability,
                0.0,
            ),
            axis=1,
        ),

        np.max(
            np.where(
                fn,
                1.0 - m3_probability,
                0.0,
            ),
            axis=1,
        ),
    )

    incorrect_candidates = np.flatnonzero(
        ~exact_match
    )

    require(
        len(
            incorrect_candidates
        )
        > 0,
        (
            "at least one incorrect "
            "M3 TEST case exists"
        ),
    )

    incorrect_index = int(
        incorrect_candidates[
            np.argmax(
                wrong_confidence[
                    incorrect_candidates
                ]
            )
        ]
    )

    # -------------------------------------------------------------------------
    # Hard:
    # smallest average of the three closest probability margins
    # to the frozen threshold.
    # -------------------------------------------------------------------------

    margin = np.abs(
        m3_probability
        - M3_THRESHOLD
    )

    three_smallest = np.sort(
        margin,
        axis=1,
    )[
        :,
        :3
    ]

    hardness = np.mean(
        three_smallest,
        axis=1,
    )

    excluded = {
        correct_index,
        incorrect_index,
    }

    candidate_order = np.argsort(
        hardness
    )

    hard_index = None

    for index in candidate_order:

        index = int(
            index
        )

        if index not in excluded:

            hard_index = index
            break

    require(
        hard_index is not None,
        (
            "distinct hard-example "
            "candidate found"
        ),
    )

    selected = [
        (
            "correct_high_confidence",
            correct_index,
        ),

        (
            "incorrect_high_confidence",
            incorrect_index,
        ),

        (
            "hard_near_threshold",
            hard_index,
        ),
    ]

    rows = []

    for case_type, index in selected:

        m3_error_count = int(
            np.sum(
                m3_prediction[
                    index
                ]
                !=
                y_true[
                    index
                ]
            )
        )

        bert_error_count = int(
            np.sum(
                bert_prediction[
                    index
                ]
                !=
                y_true[
                    index
                ]
            )
        )

        rows.append(
            {
                "case_type":
                    case_type,

                "track_id":
                    track_ids[
                        index
                    ],

                "test_row_index":
                    index,

                "true_labels":
                    labels_to_string(
                        y_true[
                            index
                        ]
                    ),

                "m3_predicted_labels":
                    labels_to_string(
                        m3_prediction[
                            index
                        ]
                    ),

                "bert_predicted_labels":
                    labels_to_string(
                        bert_prediction[
                            index
                        ]
                    ),

                "m3_label_error_count":
                    m3_error_count,

                "bert_label_error_count":
                    bert_error_count,

                "m3_average_classification_confidence":
                    float(
                        confidence[
                            index
                        ]
                    ),

                "m3_max_wrong_confidence":
                    float(
                        wrong_confidence[
                            index
                        ]
                    ),

                "m3_three_closest_threshold_margin_mean":
                    float(
                        hardness[
                            index
                        ]
                    ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    print()
    print(
        frame[
            [
                "case_type",
                "track_id",
                "m3_label_error_count",
                "bert_label_error_count",
            ]
        ].to_string(
            index=False
        )
    )

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 61 — POST-HOC STATISTICAL / ERROR / FUSION ANALYSIS"
    )

    print(
        "Source: frozen Script-60 outputs only"
    )

    print(
        "Model inference:   0"
    )

    print(
        "Training:          0"
    )

    print(
        "Threshold tuning:  0"
    )

    print(
        "Model selection:   0"
    )

    try:

        evaluation = (
            verify_outputs()
        )

        (
            y_true,
            track_ids,
            support,
            bert,
            fusion,
        ) = load_saved_predictions()

        (
            seed_frame,
            seed_summary,
        ) = analyze_seed_results(
            evaluation,
            y_true,
            bert,
            fusion,
        )

        (
            bootstrap_frame,
            bootstrap_summary,
        ) = bootstrap_comparison(
            y_true,
            bert,
            fusion,
        )

        (
            label_frame,
            label_summary,
        ) = analyze_labels(
            y_true,
            support,
            bert,
            fusion,
        )

        (
            confusion_frame,
            confusion_summary,
        ) = analyze_confusions(
            y_true,
            bert,
            fusion,
        )

        case_frame = (
            choose_case_candidates(
                y_true,
                track_ids,
                bert,
                fusion,
            )
        )

        # ---------------------------------------------------------------------
        # Write to temporary output directory.
        # ---------------------------------------------------------------------

        TEMP_DIR.mkdir(
            parents=True,
            exist_ok=False,
        )

        seed_path = (
            TEMP_DIR
            / "bert_vs_fusion_seed_differences.csv"
        )

        seed_frame.to_csv(
            seed_path,
            index=False,
        )

        bootstrap_path = (
            TEMP_DIR
            / "bert_vs_fusion_bootstrap.csv"
        )

        bootstrap_frame.to_csv(
            bootstrap_path,
            index=False,
        )

        label_path = (
            TEMP_DIR
            / "fusion_per_label_deltas.csv"
        )

        label_frame.to_csv(
            label_path,
            index=False,
        )

        confusion_path = (
            TEMP_DIR
            / "fusion_confusion_pairs.csv"
        )

        confusion_frame.to_csv(
            confusion_path,
            index=False,
        )

        cases_path = (
            TEMP_DIR
            / "case_study_candidates.csv"
        )

        case_frame.to_csv(
            cases_path,
            index=False,
        )

        # ---------------------------------------------------------------------
        # Point effect values.
        # ---------------------------------------------------------------------

        comparison_lookup = {
            row[
                "model"
            ]:
                row

            for row
            in evaluation[
                "comparison_table"
            ]
        }

        bert_row = (
            comparison_lookup[
                "BERT"
            ]
        )

        fusion_row = (
            comparison_lookup[
                "Early Fusion"
            ]
        )

        point_differences = {
            "macro_f1":
                float(
                    fusion_row[
                        "macro_f1_mean"
                    ]
                    -
                    bert_row[
                        "macro_f1_mean"
                    ]
                ),

            "micro_f1":
                float(
                    fusion_row[
                        "micro_f1_mean"
                    ]
                    -
                    bert_row[
                        "micro_f1_mean"
                    ]
                ),

            "macro_auc_pr":
                float(
                    fusion_row[
                        "auc_pr_mean"
                    ]
                    -
                    bert_row[
                        "auc_pr_mean"
                    ]
                ),
        }

        analysis = {
            "artifact_type":
                "posthoc_statistical_error_fusion_analysis",

            "version":
                1,

            "source_final_evaluation_sha256":
                EXPECTED_EVALUATION_SHA256,

            "analysis_scope":
                (
                    "descriptive and exploratory "
                    "post-hoc analysis of frozen "
                    "Script-60 TEST outputs only"
                ),

            "primary_comparison":
                "Early Fusion M3 Phase B vs BERT T1-B",

            "final_seeds":
                list(
                    FINAL_SEEDS
                ),

            "thresholds": {
                "bert":
                    BERT_THRESHOLD,

                "early_fusion":
                    M3_THRESHOLD,
            },

            "point_differences_fusion_minus_bert":
                point_differences,

            "paired_seed_analysis":
                seed_summary,

            "bootstrap": {
                "method":
                    (
                        "paired nonparametric "
                        "track-level bootstrap"
                    ),

                "replicates":
                    BOOTSTRAP_REPLICATES,

                "rng_seed":
                    BOOTSTRAP_SEED,

                "confidence_interval":
                    "95% percentile",

                "aggregation":
                    (
                        "metric calculated separately "
                        "for each of 3 seeds, then "
                        "averaged; Fusion-BERT difference"
                    ),

                "results":
                    bootstrap_summary,

                "interpretation":
                    (
                        "exploratory post-hoc uncertainty "
                        "analysis; not a preregistered "
                        "confirmatory significance test"
                    ),
            },

            "per_label_analysis":
                label_summary,

            "confusion_analysis":
                confusion_summary,

            "case_study_candidate_rule":
                {
                    "representative_model":
                        "Early Fusion seed42",

                    "correct_high_confidence":
                        (
                            "exact label-set match with "
                            "highest average classification "
                            "confidence"
                        ),

                    "incorrect_high_confidence":
                        (
                            "incorrect example with largest "
                            "confidence attached to a false "
                            "positive or false negative"
                        ),

                    "hard_example":
                        (
                            "distinct example with smallest "
                            "mean distance to threshold among "
                            "its three closest labels"
                        ),
                },

            "model_inference_performed":
                False,

            "threshold_tuning_performed":
                False,

            "training_performed":
                False,

            "model_selection_performed":
                False,

            "outputs": {
                "seed_differences":
                    "bert_vs_fusion_seed_differences.csv",

                "bootstrap":
                    "bert_vs_fusion_bootstrap.csv",

                "per_label_deltas":
                    "fusion_per_label_deltas.csv",

                "confusion_pairs":
                    "fusion_confusion_pairs.csv",

                "case_study_candidates":
                    "case_study_candidates.csv",
            },
        }

        analysis_path = (
            TEMP_DIR
            / "posthoc_analysis.json"
        )

        save_json(
            analysis,
            analysis_path,
        )

        output_hashes = {}

        for path in (
            seed_path,
            bootstrap_path,
            label_path,
            confusion_path,
            cases_path,
            analysis_path,
        ):

            output_hashes[
                path.name
            ] = sha256_file(
                path
            )

        hashes_path = (
            TEMP_DIR
            / "output_hashes.json"
        )

        save_json(
            output_hashes,
            hashes_path,
        )

        require(
            not OUTPUT_DIR.exists(),
            (
                "post-hoc final directory "
                "still unused before commit"
            ),
        )

        os.replace(
            TEMP_DIR,
            OUTPUT_DIR,
        )

        final_analysis_path = (
            OUTPUT_DIR
            / "posthoc_analysis.json"
        )

        final_sha = sha256_file(
            final_analysis_path
        )

    except Exception as exc:

        banner(
            "SCRIPT 61 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print()

        print(
            "No model inference was performed."
        )

        print(
            "Do not modify Script-60 final outputs."
        )

        if TEMP_DIR.exists():

            print(
                "Temporary post-hoc directory retained:"
            )

            print(
                f"  "
                f"{TEMP_DIR.relative_to(ROOT)}"
            )

        return 1

    # =========================================================================
    # SUMMARY
    # =========================================================================

    banner(
        "SCRIPT 61 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()
    print(
        "Primary final effect — Early Fusion minus BERT:"
    )

    print(
        f"  Macro-F1 = "
        f"{point_differences['macro_f1']:+.8f}"
    )

    print(
        f"  Micro-F1 = "
        f"{point_differences['micro_f1']:+.8f}"
    )

    print(
        f"  AUC-PR   = "
        f"{point_differences['macro_auc_pr']:+.8f}"
    )

    print()
    print(
        "Exploratory paired bootstrap 95% CI:"
    )

    for metric in (
        "macro_f1",
        "micro_f1",
        "macro_ap",
    ):

        result = (
            bootstrap_summary[
                metric
            ]
        )

        print(
            f"  {metric:9s}: "
            f"[{result['ci_lower']:+.8f}, "
            f"{result['ci_upper']:+.8f}]  "
            f"includes_zero="
            f"{result['includes_zero']}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  These intervals are exploratory/post-hoc."
    )

    print(
        "  Do not claim statistical superiority "
        "solely from the +0.000583 AUC-PR difference."
    )

    print()
    print(
        "Post-hoc artifacts:"
    )

    print(
        "  results/runs/final_eval/final_test/"
        "posthoc_analysis/"
    )

    print()
    print(
        "Analysis JSON SHA256:"
    )

    print(
        f"  {final_sha}"
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "  Training curves + graph visualization + "
        "case-study presentation using frozen artifacts."
    )

    print()
    print(
        "STOP HERE."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )