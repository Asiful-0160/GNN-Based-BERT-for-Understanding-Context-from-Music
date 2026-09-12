#!/usr/bin/env python3

"""
Script 42 — Freeze Task-2 graph-topology selection

Purpose
-------
Select the graph topology to be used by A4 GAT based only on the
already-completed formal GraphSAGE validation experiments:

    A2 = GraphSAGE / G1
    A3 = GraphSAGE / G2

Selection rule:
    highest validation macro mean per-label AP

This script:
- does NOT train a model
- does NOT load TRAIN/VAL/TEST examples
- does NOT tune thresholds
- does NOT evaluate TEST
- does NOT modify candidate checkpoints
- writes one immutable selection record

A4 must consume the selected graph from this record rather than
choosing a topology independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRIALS_PATH = (
    ROOT
    / "results/csv/hyperparameter_trials.csv"
)

PRETRAINING_LOCK_PATH = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

OUTPUT_PATH = (
    ROOT
    / "results/runs/task2/task2_graph_selection.json"
)


# ============================================================
# FROZEN IDENTITIES
# ============================================================

EXPECTED_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_A2_SHA256 = (
    "cb8dbb3976e5ed059289a30c900d84b2c0cc7cd4dc161eba7bd0dd8e65d292cf"
)

EXPECTED_A3_SHA256 = (
    "abac9ffed7907177b215332d9758e4cb86482f856b5492f2d73352412d84d8ea"
)

SEED = 42

SELECTION_METRIC = (
    "val_auc_pr_macro_mean_ap"
)

TOLERANCE = 1.0e-12


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(block)

    return digest.hexdigest()


def relative(
    path: Path,
) -> str:

    try:
        return str(
            path.relative_to(ROOT)
        )

    except ValueError:
        return str(path)


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_candidate(
    row: pd.Series,
    expected_variant: str,
    expected_graph: str,
    expected_sha: str,
):

    require(
        str(row["variant"])
        == expected_variant,
        "Variant mismatch.",
    )

    require(
        str(row["model"])
        == "GraphSAGE",
        "Expected GraphSAGE candidate.",
    )

    require(
        str(row["graph"])
        == expected_graph,
        "Graph type mismatch.",
    )

    require(
        int(row["seed"])
        == SEED,
        "Candidate seed mismatch.",
    )

    require(
        str(row["status"])
        == "COMPLETE",
        "Candidate not COMPLETE.",
    )

    require(
        bool(row["test_split_used"])
        is False,
        (
            "Candidate indicates "
            "TEST usage."
        ),
    )

    metric = float(
        row["best_val_auc_pr_macro"]
    )

    require(
        math.isfinite(metric),
        "Candidate metric is non-finite.",
    )

    checkpoint_path = (
        ROOT
        / str(
            row["checkpoint_path"]
        )
    )

    require(
        checkpoint_path.is_file(),
        (
            "Candidate checkpoint "
            f"missing: {checkpoint_path}"
        ),
    )

    actual_sha = sha256_file(
        checkpoint_path
    )

    require(
        actual_sha == expected_sha,
        (
            f"{expected_variant} checkpoint "
            "SHA256 changed."
        ),
    )

    # If trial CSV contains the checkpoint hash,
    # verify it too.
    if (
        "checkpoint_sha256"
        in row.index
        and pd.notna(
            row["checkpoint_sha256"]
        )
    ):

        require(
            str(
                row["checkpoint_sha256"]
            )
            == actual_sha,
            (
                "Trial CSV checkpoint SHA "
                "does not match physical file."
            ),
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        checkpoint["task"]
        == "task2",
        "Checkpoint task mismatch.",
    )

    require(
        checkpoint["variant"]
        == expected_variant,
        "Checkpoint variant mismatch.",
    )

    require(
        checkpoint["model"]
        == "GraphSAGE",
        "Checkpoint model mismatch.",
    )

    require(
        checkpoint["graph"]
        == expected_graph,
        "Checkpoint graph mismatch.",
    )

    require(
        int(
            checkpoint["seed"]
        )
        == SEED,
        "Checkpoint seed mismatch.",
    )

    require(
        checkpoint[
            "pretraining_lock_sha256"
        ]
        == EXPECTED_LOCK_SHA256,
        (
            "Checkpoint uses different "
            "frozen artifacts."
        ),
    )

    checkpoint_metric = float(
        checkpoint[
            "val_auc_pr_macro_mean_ap"
        ]
    )

    require(
        abs(
            checkpoint_metric
            - metric
        )
        <= TOLERANCE,
        (
            "Checkpoint metric differs "
            "from trial CSV."
        ),
    )

    return {
        "variant":
            expected_variant,

        "model":
            "GraphSAGE",

        "graph":
            expected_graph,

        "seed":
            SEED,

        "best_epoch":
            int(
                row[
                    "best_epoch"
                ]
            ),

        "best_val_auc_pr_macro":
            metric,

        "checkpoint_path":
            relative(
                checkpoint_path
            ),

        "checkpoint_sha256":
            actual_sha,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "SCRIPT 42 — SELECT TASK-2 "
        "GRAPH TOPOLOGY"
    )
    print("=" * 78)

    print()
    print(
        "Selection uses VALIDATION only."
    )

    print(
        "No dataset examples are loaded."
    )

    print(
        "TEST is not evaluated."
    )

    print(
        "No threshold tuning is performed."
    )


    # ========================================================
    # 1. IMMUTABILITY
    # ========================================================

    section(
        "1. OUTPUT PROTECTION"
    )

    require(
        not OUTPUT_PATH.exists(),
        (
            "Task-2 graph selection "
            "already exists. "
            "Refusing overwrite."
        ),
    )

    print(
        "Selection artifact does not "
        "already exist: PASS"
    )


    # ========================================================
    # 2. FROZEN LOCK
    # ========================================================

    section(
        "2. PRETRAINING LOCK"
    )

    require(
        PRETRAINING_LOCK_PATH.is_file(),
        (
            "Pretraining lock "
            "is missing."
        ),
    )

    actual_lock_sha = sha256_file(
        PRETRAINING_LOCK_PATH
    )

    print(
        "Expected SHA256:",
        EXPECTED_LOCK_SHA256,
    )

    print(
        "Actual SHA256:  ",
        actual_lock_sha,
    )

    require(
        actual_lock_sha
        == EXPECTED_LOCK_SHA256,
        "Pretraining lock changed.",
    )

    print(
        "Pretraining lock: PASS"
    )


    # ========================================================
    # 3. TRIAL TABLE
    # ========================================================

    section(
        "3. FORMAL A2 / A3 CANDIDATES"
    )

    require(
        TRIALS_PATH.is_file(),
        "Trial CSV missing.",
    )

    trials = pd.read_csv(
        TRIALS_PATH
    )

    required_columns = {
        "run_id",
        "status",
        "task",
        "variant",
        "model",
        "graph",
        "seed",
        "selection_metric",
        "best_epoch",
        "best_val_auc_pr_macro",
        "checkpoint_path",
        "pretraining_lock_sha256",
        "test_split_used",
    }

    require(
        required_columns.issubset(
            trials.columns
        ),
        (
            "Trial CSV does not contain "
            "all required selection fields."
        ),
    )

    candidates = trials.loc[
        (trials["task"] == "task2")
        & (
            trials["variant"]
            .isin(
                [
                    "A2",
                    "A3",
                ]
            )
        )
        & (trials["model"] == "GraphSAGE")
        & (trials["seed"] == SEED)
        & (trials["status"] == "COMPLETE")
    ].copy()

    require(
        len(candidates) == 2,
        (
            "Expected exactly two "
            "completed GraphSAGE "
            "candidates: A2 and A3."
        ),
    )

    require(
        set(
            candidates[
                "variant"
            ]
        )
        == {
            "A2",
            "A3",
        },
        (
            "A2/A3 candidate set "
            "is incomplete."
        ),
    )

    require(
        candidates[
            "selection_metric"
        ].nunique()
        == 1,
        (
            "A2 and A3 used different "
            "selection metrics."
        ),
    )

    require(
        candidates[
            "selection_metric"
        ].iloc[0]
        == SELECTION_METRIC,
        (
            "Unexpected A2/A3 "
            "selection metric."
        ),
    )

    require(
        (
            candidates[
                "pretraining_lock_sha256"
            ]
            == EXPECTED_LOCK_SHA256
        ).all(),
        (
            "A2/A3 do not reference "
            "the same frozen lock."
        ),
    )

    require(
        (
            candidates[
                "test_split_used"
            ]
            == False
        ).all(),
        (
            "At least one candidate "
            "indicates TEST usage."
        ),
    )


    # ========================================================
    # 4. CHECKPOINT VALIDATION
    # ========================================================

    section(
        "4. CHECKPOINT VALIDATION"
    )

    a2_row = candidates.loc[
        candidates[
            "variant"
        ]
        == "A2"
    ].iloc[0]

    a3_row = candidates.loc[
        candidates[
            "variant"
        ]
        == "A3"
    ].iloc[0]

    a2 = validate_candidate(
        row=a2_row,
        expected_variant="A2",
        expected_graph="G1",
        expected_sha=EXPECTED_A2_SHA256,
    )

    print(
        "A2 checkpoint: PASS"
    )

    a3 = validate_candidate(
        row=a3_row,
        expected_variant="A3",
        expected_graph="G2",
        expected_sha=EXPECTED_A3_SHA256,
    )

    print(
        "A3 checkpoint: PASS"
    )


    # ========================================================
    # 5. PARITY CHECK
    # ========================================================

    section(
        "5. A2 / A3 CONTROLLED-COMPARISON PARITY"
    )

    a2_checkpoint = torch.load(
        ROOT / a2[
            "checkpoint_path"
        ],
        map_location="cpu",
        weights_only=False,
    )

    a3_checkpoint = torch.load(
        ROOT / a3[
            "checkpoint_path"
        ],
        map_location="cpu",
        weights_only=False,
    )

    require(
        a2_checkpoint[
            "architecture"
        ]
        == {
            key:
                a3_checkpoint[
                    "architecture"
                ][key]
            for key
            in a2_checkpoint[
                "architecture"
            ]
        },
        (
            "A2/A3 architecture "
            "metadata differs."
        ),
    )

    require(
        a2_checkpoint[
            "training"
        ]
        == a3_checkpoint[
            "training"
        ],
        (
            "A2/A3 training "
            "metadata differs."
        ),
    )

    require(
        a2_checkpoint[
            "pos_weight"
        ]
        == a3_checkpoint[
            "pos_weight"
        ],
        (
            "A2/A3 class weights differ."
        ),
    )

    print(
        "Architecture parity: PASS"
    )

    print(
        "Training-config parity: PASS"
    )

    print(
        "Class-weight parity: PASS"
    )

    print()
    print(
        "Controlled variable:"
    )

    print(
        "A2 = G1 temporal topology"
    )

    print(
        "A3 = G2 temporal + "
        "similarity topology"
    )


    # ========================================================
    # 6. VALIDATION SELECTION
    # ========================================================

    section(
        "6. VALIDATION-ONLY SELECTION"
    )

    validated = [
        a2,
        a3,
    ]

    validated = sorted(
        validated,
        key=lambda item:
            item[
                "best_val_auc_pr_macro"
            ],
        reverse=True,
    )

    winner = validated[0]
    runner_up = validated[1]

    require(
        abs(
            winner[
                "best_val_auc_pr_macro"
            ]
            - runner_up[
                "best_val_auc_pr_macro"
            ]
        )
        > TOLERANCE,
        (
            "A2/A3 validation scores "
            "are tied; graph selection "
            "requires an explicit "
            "tie-break rule."
        ),
    )

    delta = (
        winner[
            "best_val_auc_pr_macro"
        ]
        - runner_up[
            "best_val_auc_pr_macro"
        ]
    )

    print(
        "A2 G1 VAL macro AP:",
        f"{a2['best_val_auc_pr_macro']:.8f}",
    )

    print(
        "A3 G2 VAL macro AP:",
        f"{a3['best_val_auc_pr_macro']:.8f}",
    )

    print()

    print(
        "Winner:",
        winner["variant"],
        winner["graph"],
    )

    print(
        "Absolute VAL macro AP gain:",
        f"{delta:+.8f}",
    )

    require(
        winner[
            "variant"
        ]
        == "A3"
        and winner[
            "graph"
        ]
        == "G2",
        (
            "Observed winner differs "
            "from expected current result."
        ),
    )


    # ========================================================
    # 7. FREEZE SELECTION RECORD
    # ========================================================

    section(
        "7. FREEZE GRAPH SELECTION"
    )

    result = {
        "status":
            "FROZEN",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "task":
            "task2",

        "selection_stage":
            "graph_topology_for_A4_GAT",

        "selection_basis":
            "validation_only",

        "selection_metric":
            SELECTION_METRIC,

        "seed":
            SEED,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "pretraining_lock_sha256":
            actual_lock_sha,

        "candidates": [
            a2,
            a3,
        ],

        "winner":
            winner,

        "runner_up":
            runner_up,

        "absolute_val_macro_ap_gain":
            delta,

        "a4_graph":
            winner[
                "graph"
            ],

        "notes": [
            (
                "A4 GAT must use this "
                "selected graph topology."
            ),

            (
                "The graph decision was "
                "made before any formal "
                "A4 GAT result."
            ),

            (
                "Single-seed validation "
                "selection does not imply "
                "statistical significance."
            ),

            (
                "TEST examples and labels "
                "were not used."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "Saved:",
        relative(
            OUTPUT_PATH
        ),
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 78)
    print(
        "TASK-2 GRAPH SELECTION: FROZEN"
    )
    print("=" * 78)

    print(
        "Selected graph for A4 GAT:",
        winner["graph"],
    )

    print(
        "Winning experiment:",
        winner["variant"],
    )

    print(
        "Winning VAL macro AP:",
        f"{winner['best_val_auc_pr_macro']:.8f}",
    )

    print(
        "Runner-up VAL macro AP:",
        f"{runner_up['best_val_auc_pr_macro']:.8f}",
    )

    print(
        "Difference:",
        f"{delta:+.8f}",
    )

    print()
    print(
        "TEST examples evaluated: 0"
    )

    print(
        "Threshold tuning performed: NO"
    )

    print()
    print(
        "A4 GAT must use G2."
    )


if __name__ == "__main__":
    main()