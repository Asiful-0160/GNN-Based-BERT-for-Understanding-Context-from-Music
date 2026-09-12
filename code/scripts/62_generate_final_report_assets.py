#!/usr/bin/env python3
"""
Script 62 — Generate Final Report / Presentation Assets
========================================================

Uses only frozen artifacts from completed experiments.

Creates:
    1. T1-B three-seed training curve
    2. A1 CNN three-seed training curve
    3. M3 Phase-A three-seed training curve
    4. M3 Phase-B three-seed training curve
    5. Final TEST Macro-F1 comparison
    6. Final TEST Micro-F1 comparison
    7. Final TEST AUC-PR comparison
    8. Per-label Fusion - BERT AUC-PR delta
    9. Top Early-Fusion confusion pairs
   10. G2 topology visualization for frozen hard case
   11. Detailed BERT vs M3 case-study table
   12. Paper-ready numerical findings summary
   13. Frozen report-asset manifest with SHA256 hashes

NO:
    training
    model loading
    model inference
    threshold tuning
    model selection

TEST predictions are read only from Script 60's saved cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]


MULTISEED_MANIFEST = (
    ROOT
    / "results/runs/final_eval/multiseed/"
    "final_multiseed_training_manifest.json"
)

FINAL_THRESHOLD_LOCK = (
    ROOT
    / "results/runs/final_eval/"
    "final_report_thresholds_validation.json"
)

FINAL_TEST_DIR = (
    ROOT
    / "results/runs/final_eval/final_test"
)

FINAL_TEST_EVALUATION = (
    FINAL_TEST_DIR
    / "final_test_evaluation.json"
)

FINAL_COMPARISON = (
    FINAL_TEST_DIR
    / "final_comparison_table.csv"
)

FINAL_PREDICTIONS = (
    FINAL_TEST_DIR
    / "final_test_predictions.npz"
)

POSTHOC_DIR = (
    FINAL_TEST_DIR
    / "posthoc_analysis"
)

POSTHOC_JSON = (
    POSTHOC_DIR
    / "posthoc_analysis.json"
)

POSTHOC_HASHES = (
    POSTHOC_DIR
    / "output_hashes.json"
)

LABEL_DELTAS = (
    POSTHOC_DIR
    / "fusion_per_label_deltas.csv"
)

CONFUSIONS = (
    POSTHOC_DIR
    / "fusion_confusion_pairs.csv"
)

CASE_CANDIDATES = (
    POSTHOC_DIR
    / "case_study_candidates.csv"
)

GRAPH_MANIFEST = (
    ROOT
    / "data/processed/graph_manifest.parquet"
)


OUTPUT_DIR = (
    ROOT
    / "results/runs/final_eval/"
    "report_assets"
)

TEMP_DIR = (
    ROOT
    / "results/runs/final_eval/"
    "report_assets.__tmp__"
)


# =============================================================================
# 1. FROZEN HASHES
# =============================================================================

EXPECTED_MULTISEED_SHA256 = (
    "ca4955591cb85957d628ac1c7cbcfd5595ad16010fece7362cf79dec8b524446"
)

EXPECTED_THRESHOLD_SHA256 = (
    "20118489eba0b5ee9d0568bfa1d76eb4c4a6a829c50042b83d738b31526b26e9"
)

EXPECTED_FINAL_TEST_SHA256 = (
    "8c6db5aed0cd23b7300da3b403089e7048f7f4ffcb64dcb8f3d1f555bd59f35d"
)

EXPECTED_POSTHOC_SHA256 = (
    "f2b114634c5002ace1df164e0f475c079c604c81f0f5014dd7bf1a196e93fdae"
)

EXPECTED_GRAPH_MANIFEST_SHA256 = (
    "f092b7ca5f409f3e4fd34c6e3b0bce6ad0492bf90eccc929cadc1aad1da9e158"
)


FINAL_SEEDS = (
    42,
    1337,
    2026,
)

N_TEST = 514
NUM_LABELS = 20

BERT_THRESHOLD = 0.70
M3_THRESHOLD = 0.70


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
# 2. HELPERS
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

        value = json.load(f)

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


def save_current_figure(
    path: Path,
) -> None:

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    require(
        path.is_file(),
        f"figure created: {path.name}",
    )


# =============================================================================
# 3. VERIFY FROZEN SOURCES
# =============================================================================

def verify_sources():

    banner(
        "1. VERIFY FROZEN REPORT SOURCES"
    )

    require(
        not OUTPUT_DIR.exists(),
        (
            "report-assets directory "
            "does not already exist"
        ),
    )

    require(
        not TEMP_DIR.exists(),
        (
            "temporary report-assets "
            "directory does not exist"
        ),
    )

    expected = {
        MULTISEED_MANIFEST:
            EXPECTED_MULTISEED_SHA256,

        FINAL_THRESHOLD_LOCK:
            EXPECTED_THRESHOLD_SHA256,

        FINAL_TEST_EVALUATION:
            EXPECTED_FINAL_TEST_SHA256,

        POSTHOC_JSON:
            EXPECTED_POSTHOC_SHA256,

        GRAPH_MANIFEST:
            EXPECTED_GRAPH_MANIFEST_SHA256,
    }

    for path, expected_sha in (
        expected.items()
    ):

        require(
            path.is_file(),
            (
                "artifact exists: "
                f"{path.relative_to(ROOT)}"
            ),
        )

        require(
            sha256_file(path)
            == expected_sha,
            (
                f"{path.name} SHA256 "
                "matches frozen identity"
            ),
        )

    final_evaluation = load_json(
        FINAL_TEST_EVALUATION
    )

    posthoc = load_json(
        POSTHOC_JSON
    )

    require(
        final_evaluation.get(
            "test_evaluation_status"
        )
        == "FINAL",
        (
            "Script-60 TEST evaluation "
            "status remains FINAL"
        ),
    )

    require(
        posthoc.get(
            "model_inference_performed"
        )
        is False,
        (
            "Script-61 post-hoc analysis "
            "performed zero inference"
        ),
    )

    # Verify Script-60 outputs using hashes embedded
    # in final_test_evaluation.json.
    output_hashes = final_evaluation[
        "output_hashes"
    ]

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
                "Script-60 output exists: "
                f"{filename}"
            ),
        )

        require(
            sha256_file(path)
            == expected_sha,
            (
                f"{filename} matches "
                "Script-60 frozen hash"
            ),
        )

    # Verify Script-61 outputs.
    require(
        POSTHOC_HASHES.is_file(),
        "Script-61 output_hashes.json exists",
    )

    posthoc_hashes = load_json(
        POSTHOC_HASHES
    )

    for filename, expected_sha in (
        posthoc_hashes.items()
    ):

        path = (
            POSTHOC_DIR
            / filename
        )

        require(
            path.is_file(),
            (
                "Script-61 output exists: "
                f"{filename}"
            ),
        )

        require(
            sha256_file(path)
            == expected_sha,
            (
                f"{filename} matches "
                "Script-61 frozen hash"
            ),
        )

    print()
    print(
        "Model inference performed: 0"
    )

    print(
        "Training performed:        0"
    )

    print(
        "Threshold tuning:          0"
    )

    return (
        final_evaluation,
        posthoc,
    )


# =============================================================================
# 4. TRAINING CURVES
# =============================================================================

def read_epoch_log(
    seed: int,
    filename: str,
) -> pd.DataFrame:

    path = (
        ROOT
        / "results/runs/final_eval/"
        "multiseed"
        / f"seed{seed}"
        / filename
    )

    require(
        path.is_file(),
        (
            f"training log exists: "
            f"seed{seed}/{filename}"
        ),
    )

    frame = pd.read_csv(
        path
    )

    required_columns = {
        "epoch",
        "val_macro_ap",
    }

    require(
        required_columns.issubset(
            set(
                frame.columns
            )
        ),
        (
            f"{filename} contains "
            "epoch + val_macro_ap"
        ),
    )

    return frame


def plot_training_curve(
    *,
    filename: str,
    title: str,
    output_name: str,
):

    plt.figure(
        figsize=(8.5, 5.5)
    )

    for seed in FINAL_SEEDS:

        frame = read_epoch_log(
            seed,
            filename,
        )

        plt.plot(
            frame[
                "epoch"
            ],
            frame[
                "val_macro_ap"
            ],
            marker="o",
            label=f"Seed {seed}",
        )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation Macro AUC-PR"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.25,
    )

    save_current_figure(
        TEMP_DIR
        / output_name
    )


def generate_training_figures():

    banner(
        "2. GENERATE FINAL TRAINING CURVES"
    )

    plot_training_curve(
        filename=
            "task1_T1B_epochs.csv",

        title=
            "T1-B BERT — Validation Macro AUC-PR",

        output_name=
            "figure_training_T1B.png",
    )

    plot_training_curve(
        filename=
            "task2_A1_CNN_epochs.csv",

        title=
            "A1 CNN — Validation Macro AUC-PR",

        output_name=
            "figure_training_A1_CNN.png",
    )

    plot_training_curve(
        filename=
            "task3_M3_phaseA_epochs.csv",

        title=
            "M3 Early Fusion Phase A — Validation Macro AUC-PR",

        output_name=
            "figure_training_M3_phaseA.png",
    )

    plot_training_curve(
        filename=
            "task3_M3_phaseB_epochs.csv",

        title=
            "M3 Early Fusion Phase B — Validation Macro AUC-PR",

        output_name=
            "figure_training_M3_phaseB.png",
    )


# =============================================================================
# 5. FINAL BENCHMARK FIGURES
# =============================================================================

def plot_benchmark_metric(
    frame: pd.DataFrame,
    *,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output_name: str,
):

    labels = (
        frame[
            "model"
        ].astype(str)
        .tolist()
    )

    means = (
        frame[
            mean_column
        ]
        .astype(float)
        .to_numpy()
    )

    stds = (
        frame[
            std_column
        ]
        .astype(float)
        .fillna(0.0)
        .to_numpy()
    )

    x = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(10, 5.8)
    )

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    save_current_figure(
        TEMP_DIR
        / output_name
    )


def generate_benchmark_figures():

    banner(
        "3. GENERATE FINAL TEST BENCHMARK FIGURES"
    )

    comparison = pd.read_csv(
        FINAL_COMPARISON
    )

    required = {
        "model",
        "macro_f1_mean",
        "macro_f1_sample_std",
        "micro_f1_mean",
        "micro_f1_sample_std",
        "auc_pr_mean",
        "auc_pr_sample_std",
    }

    require(
        required.issubset(
            set(
                comparison.columns
            )
        ),
        (
            "final comparison table "
            "contains expected metrics"
        ),
    )

    plot_benchmark_metric(
        comparison,
        mean_column=
            "macro_f1_mean",
        std_column=
            "macro_f1_sample_std",
        ylabel=
            "Macro-F1",
        title=
            "Final TEST Macro-F1",
        output_name=
            "figure_final_macro_f1.png",
    )

    plot_benchmark_metric(
        comparison,
        mean_column=
            "micro_f1_mean",
        std_column=
            "micro_f1_sample_std",
        ylabel=
            "Micro-F1",
        title=
            "Final TEST Micro-F1",
        output_name=
            "figure_final_micro_f1.png",
    )

    plot_benchmark_metric(
        comparison,
        mean_column=
            "auc_pr_mean",
        std_column=
            "auc_pr_sample_std",
        ylabel=
            "Macro AUC-PR",
        title=
            "Final TEST AUC-PR",
        output_name=
            "figure_final_auc_pr.png",
    )

    return comparison


# =============================================================================
# 6. PER-LABEL FUSION EFFECT
# =============================================================================

def generate_label_delta_figure():

    banner(
        "4. GENERATE PER-LABEL FUSION EFFECT FIGURE"
    )

    frame = pd.read_csv(
        LABEL_DELTAS
    )

    require(
        {
            "label",
            "delta_ap",
            "delta_f1",
            "test_support",
        }.issubset(
            set(
                frame.columns
            )
        ),
        (
            "fusion per-label delta "
            "table has expected schema"
        ),
    )

    frame = (
        frame.sort_values(
            "delta_ap",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(frame)
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.barh(
        y,
        frame[
            "delta_ap"
        ],
    )

    plt.yticks(
        y,
        frame[
            "label"
        ],
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )

    plt.xlabel(
        "Early Fusion − BERT AUC-PR"
    )

    plt.ylabel(
        "Label"
    )

    plt.title(
        "Per-Label Effect of Early Fusion"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    save_current_figure(
        TEMP_DIR
        / "figure_fusion_per_label_aucpr_delta.png"
    )

    return frame


# =============================================================================
# 7. CONFUSION FIGURE
# =============================================================================

def generate_confusion_figure():

    banner(
        "5. GENERATE FUSION CONFUSION FIGURE"
    )

    frame = pd.read_csv(
        CONFUSIONS
    )

    required = {
        "missed_true_label",
        "wrong_predicted_label",
        "fusion_count_across_3_seeds",
    }

    require(
        required.issubset(
            set(
                frame.columns
            )
        ),
        (
            "confusion table "
            "has expected schema"
        ),
    )

    top = (
        frame.sort_values(
            "fusion_count_across_3_seeds",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    top[
        "pair"
    ] = (
        top[
            "missed_true_label"
        ]
        + " → "
        + top[
            "wrong_predicted_label"
        ]
    )

    top = top.iloc[
        ::-1
    ]

    y = np.arange(
        len(top)
    )

    plt.figure(
        figsize=(10, 6.5)
    )

    plt.barh(
        y,
        top[
            "fusion_count_across_3_seeds"
        ],
    )

    plt.yticks(
        y,
        top[
            "pair"
        ],
    )

    plt.xlabel(
        "Count across 3 final seeds"
    )

    plt.ylabel(
        "Missed true label → wrong predicted label"
    )

    plt.title(
        "Most Frequent Early-Fusion Confusion Pairs"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    save_current_figure(
        TEMP_DIR
        / "figure_fusion_confusion_pairs.png"
    )

    return top


# =============================================================================
# 8. CASE-STUDY DETAIL TABLE
# =============================================================================

def generate_case_details():

    banner(
        "6. GENERATE FROZEN CASE-STUDY DETAILS"
    )

    cases = pd.read_csv(
        CASE_CANDIDATES
    )

    require(
        len(cases)
        == 3,
        (
            "Script-61 contains exactly "
            "3 frozen case-study candidates"
        ),
    )

    required_case_types = {
        "correct_high_confidence",
        "incorrect_high_confidence",
        "hard_near_threshold",
    }

    require(
        set(
            cases[
                "case_type"
            ]
        )
        == required_case_types,
        (
            "case-study candidate types "
            "match Script-61 selection"
        ),
    )

    with np.load(
        FINAL_PREDICTIONS,
        allow_pickle=False,
    ) as cache:

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

        bert = np.asarray(
            cache[
                "prob__BERT_T1B__seed42"
            ],
            dtype=np.float64,
        )

        fusion = np.asarray(
            cache[
                "prob__EarlyFusion_M3B__seed42"
            ],
            dtype=np.float64,
        )

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

    lookup = {
        track_id:
            index

        for index, track_id
        in enumerate(
            track_ids
        )
    }

    rows = []

    for _, case in (
        cases.iterrows()
    ):

        track_id = str(
            case[
                "track_id"
            ]
        )

        require(
            track_id in lookup,
            (
                f"case track exists in "
                f"saved TEST cache: {track_id}"
            ),
        )

        row_index = lookup[
            track_id
        ]

        bert_prediction = (
            bert[
                row_index
            ]
            >= BERT_THRESHOLD
        ).astype(
            np.int64
        )

        fusion_prediction = (
            fusion[
                row_index
            ]
            >= M3_THRESHOLD
        ).astype(
            np.int64
        )

        for label_index in range(
            NUM_LABELS
        ):

            rows.append(
                {
                    "case_type":
                        case[
                            "case_type"
                        ],

                    "track_id":
                        track_id,

                    "label_index":
                        label_index,

                    "label":
                        LABEL_NAMES[
                            label_index
                        ],

                    "true":
                        int(
                            y_true[
                                row_index,
                                label_index,
                            ]
                        ),

                    "bert_probability":
                        float(
                            bert[
                                row_index,
                                label_index,
                            ]
                        ),

                    "bert_prediction":
                        int(
                            bert_prediction[
                                label_index
                            ]
                        ),

                    "fusion_probability":
                        float(
                            fusion[
                                row_index,
                                label_index,
                            ]
                        ),

                    "fusion_prediction":
                        int(
                            fusion_prediction[
                                label_index
                            ]
                        ),

                    "fusion_minus_bert_probability":
                        float(
                            fusion[
                                row_index,
                                label_index,
                            ]
                            -
                            bert[
                                row_index,
                                label_index,
                            ]
                        ),
                }
            )

    detail = pd.DataFrame(
        rows
    )

    detail_path = (
        TEMP_DIR
        / "case_study_label_details.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    require(
        detail_path.is_file(),
        (
            "case-study detailed "
            "label table created"
        ),
    )

    # Produce compact human-readable case summary.
    summary_rows = []

    for _, case in (
        cases.iterrows()
    ):

        summary_rows.append(
            {
                "case_type":
                    case[
                        "case_type"
                    ],

                "track_id":
                    case[
                        "track_id"
                    ],

                "true_labels":
                    case[
                        "true_labels"
                    ],

                "bert_predicted_labels":
                    case[
                        "bert_predicted_labels"
                    ],

                "m3_predicted_labels":
                    case[
                        "m3_predicted_labels"
                    ],

                "bert_label_error_count":
                    int(
                        case[
                            "bert_label_error_count"
                        ]
                    ),

                "m3_label_error_count":
                    int(
                        case[
                            "m3_label_error_count"
                        ]
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        TEMP_DIR
        / "case_study_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    require(
        summary_path.is_file(),
        (
            "case-study compact "
            "summary created"
        ),
    )

    return (
        cases,
        detail,
    )


# =============================================================================
# 9. FROZEN G2 GRAPH VISUALIZATION
# =============================================================================

def resolve_graph_path(
    value: str,
) -> Path:

    raw = Path(value)

    candidates = []

    if raw.is_absolute():

        candidates.append(
            raw
        )

    else:

        candidates.extend(
            [
                ROOT / raw,

                GRAPH_MANIFEST.parent
                / raw,

                ROOT
                / "data/processed/graphs"
                / raw.name,
            ]
        )

    matches = {}

    for candidate in candidates:

        if candidate.is_file():

            resolved = (
                candidate.resolve()
            )

            matches[
                str(resolved)
            ] = resolved

    if len(matches) != 1:

        raise RuntimeError(
            "cannot uniquely resolve "
            f"graph path: {value}"
        )

    return next(
        iter(
            matches.values()
        )
    )


def undirected_edge_set(
    edge_index: np.ndarray,
):

    edges = set()

    for source, target in (
        edge_index.T.tolist()
    ):

        source = int(source)
        target = int(target)

        if source == target:
            continue

        edges.add(
            tuple(
                sorted(
                    (
                        source,
                        target,
                    )
                )
            )
        )

    return edges


def generate_graph_visualization(
    cases: pd.DataFrame,
):

    banner(
        "7. GENERATE FROZEN G2 GRAPH VISUALIZATION"
    )

    hard_case = (
        cases.loc[
            cases[
                "case_type"
            ]
            == "hard_near_threshold"
        ]
        .iloc[0]
    )

    track_id = str(
        hard_case[
            "track_id"
        ]
    )

    manifest = pd.read_parquet(
        GRAPH_MANIFEST
    ).copy()

    manifest[
        "track_id"
    ] = (
        manifest[
            "track_id"
        ].astype(str)
    )

    lookup = manifest.set_index(
        "track_id",
        drop=False,
    )

    require(
        track_id
        in lookup.index,
        (
            "hard case exists in "
            "graph manifest"
        ),
    )

    row = lookup.loc[
        track_id
    ]

    path = resolve_graph_path(
        str(
            row[
                "graph_path"
            ]
        )
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as graph:

        x = np.asarray(
            graph[
                "x"
            ]
        )

        edge_g1 = np.asarray(
            graph[
                "edge_index_g1"
            ],
            dtype=np.int64,
        )

        edge_g2 = np.asarray(
            graph[
                "edge_index_g2"
            ],
            dtype=np.int64,
        )

    require(
        x.shape
        == (
            9,
            140,
        ),
        (
            "case-study graph has "
            "9 nodes × 140 features"
        ),
    )

    temporal_edges = (
        undirected_edge_set(
            edge_g1
        )
    )

    g2_edges = (
        undirected_edge_set(
            edge_g2
        )
    )

    extra_edges = (
        g2_edges
        -
        temporal_edges
    )

    require(
        temporal_edges.issubset(
            g2_edges
        ),
        (
            "G2 contains all temporal "
            "G1 relationships"
        ),
    )

    edge_rows = []

    for source, target in sorted(
        temporal_edges
    ):

        edge_rows.append(
            {
                "source_node":
                    source,

                "target_node":
                    target,

                "edge_type":
                    "temporal",
            }
        )

    for source, target in sorted(
        extra_edges
    ):

        edge_rows.append(
            {
                "source_node":
                    source,

                "target_node":
                    target,

                "edge_type":
                    "nonlocal_similarity",
            }
        )

    edge_frame = pd.DataFrame(
        edge_rows
    )

    edge_path = (
        TEMP_DIR
        / "g2_case_graph_edges.csv"
    )

    edge_frame.to_csv(
        edge_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plot topology.
    # Nodes remain in temporal order 0..8.
    # Temporal edges are straight.
    # Added non-local G2 relations are drawn as arcs.
    # -------------------------------------------------------------------------

    node_x = np.arange(
        9,
        dtype=np.float64,
    )

    node_y = np.zeros(
        9,
        dtype=np.float64,
    )

    plt.figure(
        figsize=(10, 4.8)
    )

    for source, target in sorted(
        temporal_edges
    ):

        plt.plot(
            [
                node_x[source],
                node_x[target],
            ],
            [
                0.0,
                0.0,
            ],
            linewidth=2,
        )

    for source, target in sorted(
        extra_edges
    ):

        left = float(source)
        right = float(target)

        xs = np.linspace(
            left,
            right,
            100,
        )

        midpoint = (
            left + right
        ) / 2.0

        half_span = max(
            (
                right - left
            ) / 2.0,
            1e-6,
        )

        normalized = (
            xs - midpoint
        ) / half_span

        height = (
            0.20
            * (
                right - left
            )
        )

        ys = (
            height
            * (
                1.0
                -
                normalized ** 2
            )
        )

        plt.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=1.5,
        )

    plt.scatter(
        node_x,
        node_y,
        s=350,
        zorder=5,
    )

    for node in range(
        9
    ):

        plt.text(
            node_x[node],
            node_y[node],
            str(node),
            ha="center",
            va="center",
            zorder=6,
        )

    plt.xlabel(
        "Temporal segment node"
    )

    plt.ylabel(
        "Graph layout"
    )

    plt.title(
        (
            "Frozen G2 Topology — "
            f"{track_id}\n"
            "Solid: temporal edges | "
            "Dashed: added non-local similarity edges"
        )
    )

    plt.yticks([])

    plt.grid(
        axis="x",
        alpha=0.20,
    )

    save_current_figure(
        TEMP_DIR
        / "figure_G2_hard_case_graph.png"
    )

    graph_summary = {
        "track_id":
            track_id,

        "nodes":
            9,

        "node_feature_dimension":
            140,

        "temporal_undirected_edges":
            len(
                temporal_edges
            ),

        "added_nonlocal_undirected_edges":
            len(
                extra_edges
            ),

        "total_G2_undirected_edges":
            len(
                g2_edges
            ),

        "graph_path":
            str(
                path.relative_to(
                    ROOT
                )
            ),
    }

    save_json(
        graph_summary,
        TEMP_DIR
        / "g2_case_graph_summary.json",
    )

    print()
    print(
        f"Graph case: {track_id}"
    )

    print(
        f"Temporal undirected edges: "
        f"{len(temporal_edges)}"
    )

    print(
        f"Added non-local edges: "
        f"{len(extra_edges)}"
    )

    return graph_summary


# =============================================================================
# 10. PAPER-READY FINDINGS
# =============================================================================

def generate_findings(
    final_evaluation,
    posthoc,
    comparison,
    label_frame,
):

    banner(
        "8. GENERATE PAPER-READY NUMERICAL FINDINGS"
    )

    lookup = (
        comparison.set_index(
            "model"
        )
    )

    bert = lookup.loc[
        "BERT"
    ]

    fusion = lookup.loc[
        "Early Fusion"
    ]

    cnn = lookup.loc[
        "B2 CNN"
    ]

    gat = lookup.loc[
        "GAT"
    ]

    bootstrap = (
        posthoc[
            "bootstrap"
        ][
            "results"
        ]
    )

    best_improved = (
        label_frame.sort_values(
            "delta_ap",
            ascending=False,
        )
        .iloc[0]
    )

    most_harmed = (
        label_frame.sort_values(
            "delta_ap",
            ascending=True,
        )
        .iloc[0]
    )

    lines = [
        "FINAL RESEARCH FINDINGS",
        "=======================",
        "",
        "Primary research question:",
        (
            "Does GNN+BERT fusion improve "
            "multi-label music-context understanding "
            "over either modality alone?"
        ),
        "",
        "Overall TEST performance:",
        (
            f"BERT mean AUC-PR = "
            f"{float(bert['auc_pr_mean']):.8f}"
        ),
        (
            f"Early Fusion mean AUC-PR = "
            f"{float(fusion['auc_pr_mean']):.8f}"
        ),
        (
            "Early Fusion - BERT AUC-PR = "
            f"{float(fusion['auc_pr_mean'] - bert['auc_pr_mean']):+.8f}"
        ),
        "",
        (
            f"BERT mean Macro-F1 = "
            f"{float(bert['macro_f1_mean']):.8f}"
        ),
        (
            f"Early Fusion mean Macro-F1 = "
            f"{float(fusion['macro_f1_mean']):.8f}"
        ),
        (
            "Early Fusion - BERT Macro-F1 = "
            f"{float(fusion['macro_f1_mean'] - bert['macro_f1_mean']):+.8f}"
        ),
        "",
        (
            f"BERT mean Micro-F1 = "
            f"{float(bert['micro_f1_mean']):.8f}"
        ),
        (
            f"Early Fusion mean Micro-F1 = "
            f"{float(fusion['micro_f1_mean']):.8f}"
        ),
        (
            "Early Fusion - BERT Micro-F1 = "
            f"{float(fusion['micro_f1_mean'] - bert['micro_f1_mean']):+.8f}"
        ),
        "",
        "Exploratory paired bootstrap:",
        (
            "AUC-PR 95% CI = "
            f"[{bootstrap['macro_ap']['ci_lower']:+.8f}, "
            f"{bootstrap['macro_ap']['ci_upper']:+.8f}]"
        ),
        (
            "Macro-F1 95% CI = "
            f"[{bootstrap['macro_f1']['ci_lower']:+.8f}, "
            f"{bootstrap['macro_f1']['ci_upper']:+.8f}]"
        ),
        (
            "Micro-F1 95% CI = "
            f"[{bootstrap['micro_f1']['ci_lower']:+.8f}, "
            f"{bootstrap['micro_f1']['ci_upper']:+.8f}]"
        ),
        "",
        "Interpretation:",
        (
            "All exploratory confidence intervals include zero. "
            "Therefore the experiment does not provide evidence "
            "of a reliable overall performance advantage for "
            "Early Fusion over BERT."
        ),
        (
            "Early Fusion and BERT should be described as "
            "essentially comparable overall, with a very small "
            "mean advantage for Early Fusion."
        ),
        "",
        "Audio-only comparison:",
        (
            f"CNN mean AUC-PR = "
            f"{float(cnn['auc_pr_mean']):.8f}"
        ),
        (
            f"GAT seed-42 AUC-PR = "
            f"{float(gat['auc_pr_mean']):.8f}"
        ),
        (
            "Early Fusion clearly outperformed the "
            "audio-only baselines."
        ),
        "",
        "Largest label-specific effects:",
        (
            f"Best AUC-PR gain: "
            f"{best_improved['label']} "
            f"({float(best_improved['delta_ap']):+.6f})"
        ),
        (
            f"Largest AUC-PR reduction: "
            f"{most_harmed['label']} "
            f"({float(most_harmed['delta_ap']):+.6f})"
        ),
        "",
        "Reporting caution:",
        (
            "The bootstrap analysis is exploratory and post-hoc. "
            "Do not describe the tiny overall Fusion-BERT "
            "difference as statistically significant."
        ),
    ]

    path = (
        TEMP_DIR
        / "paper_ready_findings.txt"
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    require(
        path.is_file(),
        (
            "paper-ready findings "
            "summary created"
        ),
    )


# =============================================================================
# 11. FINAL ASSET MANIFEST
# =============================================================================

def create_manifest(
    graph_summary,
):

    banner(
        "9. FREEZE REPORT-ASSET MANIFEST"
    )

    output_hashes = {}

    for path in sorted(
        TEMP_DIR.iterdir()
    ):

        if (
            path.is_file()
            and
            path.name
            != "report_assets_manifest.json"
        ):

            output_hashes[
                path.name
            ] = sha256_file(
                path
            )

    manifest = {
        "artifact_type":
            "final_report_assets",

        "version":
            1,

        "source_artifacts": {
            "final_multiseed_manifest_sha256":
                EXPECTED_MULTISEED_SHA256,

            "final_threshold_lock_sha256":
                EXPECTED_THRESHOLD_SHA256,

            "final_test_evaluation_sha256":
                EXPECTED_FINAL_TEST_SHA256,

            "posthoc_analysis_sha256":
                EXPECTED_POSTHOC_SHA256,

            "graph_manifest_sha256":
                EXPECTED_GRAPH_MANIFEST_SHA256,
        },

        "analysis_policy": {
            "model_inference":
                False,

            "training":
                False,

            "threshold_tuning":
                False,

            "model_selection":
                False,

            "test_data_source":
                (
                    "saved Script-60 "
                    "probability cache only"
                ),
        },

        "figures": [
            "figure_training_T1B.png",
            "figure_training_A1_CNN.png",
            "figure_training_M3_phaseA.png",
            "figure_training_M3_phaseB.png",
            "figure_final_macro_f1.png",
            "figure_final_micro_f1.png",
            "figure_final_auc_pr.png",
            "figure_fusion_per_label_aucpr_delta.png",
            "figure_fusion_confusion_pairs.png",
            "figure_G2_hard_case_graph.png",
        ],

        "case_study_outputs": [
            "case_study_summary.csv",
            "case_study_label_details.csv",
        ],

        "graph_visualization":
            graph_summary,

        "research_interpretation":
            (
                "Early Fusion has a very small "
                "mean advantage over BERT, but "
                "exploratory paired-bootstrap "
                "intervals include zero for all "
                "three primary metrics; no reliable "
                "overall superiority claim."
            ),

        "output_hashes":
            output_hashes,
    }

    path = (
        TEMP_DIR
        / "report_assets_manifest.json"
    )

    save_json(
        manifest,
        path,
    )

    require(
        path.is_file(),
        (
            "report-assets manifest created"
        ),
    )


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 62 — GENERATE FINAL REPORT ASSETS"
    )

    print(
        "Training:         NO"
    )

    print(
        "Model inference:  NO"
    )

    print(
        "Threshold tuning: NO"
    )

    print(
        "Model selection:  NO"
    )

    try:

        (
            final_evaluation,
            posthoc,
        ) = verify_sources()

        TEMP_DIR.mkdir(
            parents=True,
            exist_ok=False,
        )

        generate_training_figures()

        comparison = (
            generate_benchmark_figures()
        )

        label_frame = (
            generate_label_delta_figure()
        )

        generate_confusion_figure()

        (
            cases,
            _,
        ) = generate_case_details()

        graph_summary = (
            generate_graph_visualization(
                cases
            )
        )

        generate_findings(
            final_evaluation,
            posthoc,
            comparison,
            label_frame,
        )

        create_manifest(
            graph_summary
        )

        require(
            not OUTPUT_DIR.exists(),
            (
                "final report-assets "
                "directory unused before commit"
            ),
        )

        os.replace(
            TEMP_DIR,
            OUTPUT_DIR,
        )

        manifest_path = (
            OUTPUT_DIR
            / "report_assets_manifest.json"
        )

        manifest_sha = sha256_file(
            manifest_path
        )

    except Exception as exc:

        banner(
            "SCRIPT 62 — ABORTED"
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
            "No frozen experimental "
            "artifact was modified."
        )

        if TEMP_DIR.exists():

            print()
            print(
                "Temporary directory retained:"
            )

            print(
                f"  "
                f"{TEMP_DIR.relative_to(ROOT)}"
            )

        return 1

    banner(
        "SCRIPT 62 SUMMARY"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Final figures and presentation "
        "assets generated."
    )

    print()

    print(
        "Output directory:"
    )

    print(
        "  results/runs/final_eval/report_assets/"
    )

    print()

    print(
        "Key figures:"
    )

    print(
        "  figure_training_T1B.png"
    )

    print(
        "  figure_training_A1_CNN.png"
    )

    print(
        "  figure_training_M3_phaseA.png"
    )

    print(
        "  figure_training_M3_phaseB.png"
    )

    print(
        "  figure_final_macro_f1.png"
    )

    print(
        "  figure_final_micro_f1.png"
    )

    print(
        "  figure_final_auc_pr.png"
    )

    print(
        "  figure_fusion_per_label_aucpr_delta.png"
    )

    print(
        "  figure_fusion_confusion_pairs.png"
    )

    print(
        "  figure_G2_hard_case_graph.png"
    )

    print()

    print(
        "Case-study assets:"
    )

    print(
        "  case_study_summary.csv"
    )

    print(
        "  case_study_label_details.csv"
    )

    print()

    print(
        "Research summary:"
    )

    print(
        "  paper_ready_findings.txt"
    )

    print()

    print(
        "Manifest SHA256:"
    )

    print(
        f"  {manifest_sha}"
    )

    print()

    print(
        "Model inference performed: 0"
    )

    print(
        "Training performed:        0"
    )

    print(
        "Threshold tuning:          0"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  Results / Discussion / Conclusion "
        "writing from frozen final artifacts."
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