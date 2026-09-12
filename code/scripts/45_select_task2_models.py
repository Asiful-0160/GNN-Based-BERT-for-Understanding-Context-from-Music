#!/usr/bin/env python3

"""
Script 45 — Freeze Task-2 model selections

Purpose
-------
Review the four completed formal Task-2 model-selection runs:

    A1 CNN
    A2 GraphSAGE / G1
    A3 GraphSAGE / G2
    A4 GAT / G2

and freeze two distinct validation-only decisions:

1. Overall Task-2 audio winner
   - selected across A1/A2/A3/A4
   - used as the strongest audio-only Task-2 result

2. Best GNN encoder
   - selected across A2/A3/A4 only
   - required for Task-3 GNN-BERT fusion
   - must produce the graph representation g in R^128

This distinction is necessary because the CNN is part of Task 2
but is not a GNN and therefore cannot provide the graph embedding
required by Task 3.

This script:
- does NOT train anything
- does NOT load dataset examples
- does NOT evaluate TEST
- does NOT tune thresholds
- does NOT overwrite candidate checkpoints
- writes one immutable Task-2 selection artifact
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

GRAPH_SELECTION_PATH = (
    ROOT
    / "results/runs/task2/task2_graph_selection.json"
)

OUTPUT_PATH = (
    ROOT
    / "results/runs/task2/task2_model_selection.json"
)


# ============================================================
# FROZEN IDENTITIES
# ============================================================

EXPECTED_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_CHECKPOINT_SHA256 = {
    "A1":
        "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275",

    "A2":
        "cb8dbb3976e5ed059289a30c900d84b2c0cc7cd4dc161eba7bd0dd8e65d292cf",

    "A3":
        "abac9ffed7907177b215332d9758e4cb86482f856b5492f2d73352412d84d8ea",

    "A4":
        "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e",
}

# IMPORTANT:
# The A1 hash above must correspond to the formal A1 CNN checkpoint,
# not Task 1 BERT.
#
# Replace it below with the correct A1 hash resolved from the trial CSV
# instead of assuming a copied value.
#
# Therefore we will NOT use the A1 dictionary value in validation.
# A2/A3/A4 are fixed from accepted outputs.

EXPECTED_GNN_SHA256 = {
    "A2":
        "cb8dbb3976e5ed059289a30c900d84b2c0cc7cd4dc161eba7bd0dd8e65d292cf",

    "A3":
        "abac9ffed7907177b215332d9758e4cb86482f856b5492f2d73352412d84d8ea",

    "A4":
        "c4e3084f6f7f6030caf157623663d311daf74ea0a24f6b02e78e5513078fae1e",
}

EXPECTED_VARIANTS = {
    "A1": {
        "model": "CNN",
        "graph": None,
    },

    "A2": {
        "model": "GraphSAGE",
        "graph": "G1",
    },

    "A3": {
        "model": "GraphSAGE",
        "graph": "G2",
    },

    "A4": {
        "model": "GAT",
        "graph": "G2",
    },
}

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


def false_like(value) -> bool:

    if isinstance(value, bool):
        return value is False

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "false",
        "0",
        "no",
    }


def normalize_graph(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    if text.lower() in {
        "none",
        "nan",
        "na",
        "n/a",
        "-",
    }:
        return None

    return text


# ============================================================
# VALIDATE ONE CANDIDATE
# ============================================================

def validate_candidate(
    row: pd.Series,
    variant: str,
):

    expected = EXPECTED_VARIANTS[
        variant
    ]

    require(
        str(
            row["variant"]
        )
        == variant,
        (
            f"{variant} variant "
            "identity mismatch."
        ),
    )

    require(
        str(
            row["model"]
        )
        == expected["model"],
        (
            f"{variant} model "
            "identity mismatch."
        ),
    )

    actual_graph = normalize_graph(
        row["graph"]
    )

    require(
        actual_graph
        == expected["graph"],
        (
            f"{variant} graph mismatch. "
            f"Expected {expected['graph']!r}, "
            f"got {actual_graph!r}."
        ),
    )

    require(
        int(
            row["seed"]
        )
        == SEED,
        f"{variant} seed mismatch.",
    )

    require(
        str(
            row["status"]
        )
        == "COMPLETE",
        (
            f"{variant} is not "
            "marked COMPLETE."
        ),
    )

    require(
        false_like(
            row[
                "test_split_used"
            ]
        ),
        (
            f"{variant} indicates "
            "TEST usage."
        ),
    )

    metric = float(
        row[
            "best_val_auc_pr_macro"
        ]
    )

    require(
        math.isfinite(metric),
        (
            f"{variant} has "
            "non-finite validation AP."
        ),
    )

    checkpoint_path = (
        ROOT
        / str(
            row[
                "checkpoint_path"
            ]
        )
    )

    require(
        checkpoint_path.is_file(),
        (
            f"{variant} checkpoint "
            f"missing: {checkpoint_path}"
        ),
    )

    checkpoint_sha = (
        sha256_file(
            checkpoint_path
        )
    )

    # For A2/A3/A4 we already have accepted,
    # explicit checkpoint hashes.
    if variant in EXPECTED_GNN_SHA256:

        require(
            checkpoint_sha
            == EXPECTED_GNN_SHA256[
                variant
            ],
            (
                f"{variant} checkpoint "
                "SHA256 changed."
            ),
        )

    # For all candidates, if the trial table saved
    # a checkpoint hash, require exact agreement.
    if (
        "checkpoint_sha256"
        in row.index
        and pd.notna(
            row[
                "checkpoint_sha256"
            ]
        )
    ):

        trial_sha = str(
            row[
                "checkpoint_sha256"
            ]
        ).strip()

        if trial_sha:

            require(
                checkpoint_sha
                == trial_sha,
                (
                    f"{variant} physical "
                    "checkpoint SHA differs "
                    "from trial CSV."
                ),
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Formal trainers saved these fields.
    require(
        checkpoint.get(
            "task"
        )
        == "task2",
        (
            f"{variant} checkpoint "
            "task mismatch."
        ),
    )

    require(
        checkpoint.get(
            "variant"
        )
        == variant,
        (
            f"{variant} checkpoint "
            "variant mismatch."
        ),
    )

    # A2/A3/A4 contain model + graph explicitly.
    if variant in {
        "A2",
        "A3",
        "A4",
    }:

        require(
            checkpoint.get(
                "model"
            )
            == expected["model"],
            (
                f"{variant} checkpoint "
                "model mismatch."
            ),
        )

        require(
            checkpoint.get(
                "graph"
            )
            == expected["graph"],
            (
                f"{variant} checkpoint "
                "graph mismatch."
            ),
        )

    require(
        int(
            checkpoint.get(
                "seed"
            )
        )
        == SEED,
        (
            f"{variant} checkpoint "
            "seed mismatch."
        ),
    )

    require(
        checkpoint.get(
            "pretraining_lock_sha256"
        )
        == EXPECTED_LOCK_SHA256,
        (
            f"{variant} checkpoint "
            "references a different "
            "pretraining lock."
        ),
    )

    checkpoint_metric = checkpoint.get(
        "val_auc_pr_macro_mean_ap"
    )

    # Script 36 may use a slightly different
    # metadata field name. Resolve it explicitly
    # rather than guessing.
    if checkpoint_metric is None:

        checkpoint_metric = checkpoint.get(
            "best_val_auc_pr_macro"
        )

    require(
        checkpoint_metric is not None,
        (
            f"{variant} checkpoint "
            "does not expose its "
            "validation selection metric."
        ),
    )

    checkpoint_metric = float(
        checkpoint_metric
    )

    require(
        abs(
            checkpoint_metric
            - metric
        )
        <= TOLERANCE,
        (
            f"{variant} checkpoint "
            "metric differs from "
            "trial CSV."
        ),
    )

    return {
        "variant":
            variant,

        "model":
            expected[
                "model"
            ],

        "graph":
            expected[
                "graph"
            ],

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
            checkpoint_sha,

        "test_split_used":
            False,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "SCRIPT 45 — FREEZE TASK-2 "
        "MODEL SELECTIONS"
    )
    print("=" * 78)

    print()
    print(
        "Selection basis: VALIDATION only"
    )

    print(
        "Dataset examples loaded: 0"
    )

    print(
        "TEST examples evaluated: 0"
    )

    print(
        "Threshold tuning: NOT RUN"
    )


    # ========================================================
    # 1. OUTPUT PROTECTION
    # ========================================================

    section(
        "1. OUTPUT PROTECTION"
    )

    require(
        not OUTPUT_PATH.exists(),
        (
            "Task-2 model-selection "
            "artifact already exists. "
            "Refusing overwrite."
        ),
    )

    print(
        "Selection artifact does not "
        "already exist: PASS"
    )


    # ========================================================
    # 2. PRETRAINING LOCK
    # ========================================================

    section(
        "2. PRETRAINING LOCK"
    )

    require(
        PRETRAINING_LOCK_PATH.is_file(),
        "Pretraining lock missing.",
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
    # 3. GRAPH-SELECTION ARTIFACT
    # ========================================================

    section(
        "3. FROZEN GRAPH SELECTION"
    )

    require(
        GRAPH_SELECTION_PATH.is_file(),
        (
            "Script-42 graph-selection "
            "artifact is missing."
        ),
    )

    with GRAPH_SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        graph_selection = json.load(
            file
        )

    require(
        graph_selection[
            "status"
        ]
        == "FROZEN",
        (
            "Graph selection "
            "is not frozen."
        ),
    )

    require(
        graph_selection[
            "a4_graph"
        ]
        == "G2",
        (
            "Frozen A4 graph "
            "is not G2."
        ),
    )

    require(
        graph_selection[
            "winner"
        ][
            "variant"
        ]
        == "A3",
        (
            "Frozen graph winner "
            "is not A3."
        ),
    )

    require(
        graph_selection[
            "pretraining_lock_sha256"
        ]
        == EXPECTED_LOCK_SHA256,
        (
            "Graph-selection lock "
            "identity mismatch."
        ),
    )

    require(
        graph_selection[
            "test_split_used"
        ]
        is False,
        (
            "Graph selection "
            "indicates TEST usage."
        ),
    )

    print(
        "Graph selection status: FROZEN"
    )

    print(
        "A4 topology: G2"
    )

    print(
        "Frozen graph selection: PASS"
    )


    # ========================================================
    # 4. FORMAL TASK-2 TRIALS
    # ========================================================

    section(
        "4. FORMAL A1 / A2 / A3 / A4 RUNS"
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
            "Trial CSV is missing "
            "required Task-2 fields."
        ),
    )

    candidates_frame = trials.loc[
        (trials["task"] == "task2")
        & (
            trials[
                "variant"
            ].isin(
                [
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                ]
            )
        )
        & (trials["seed"] == SEED)
        & (trials["status"] == "COMPLETE")
    ].copy()

    require(
        len(candidates_frame)
        == 4,
        (
            "Expected exactly four "
            "completed formal Task-2 "
            "seed-42 candidates."
        ),
    )

    require(
        set(
            candidates_frame[
                "variant"
            ]
        )
        == {
            "A1",
            "A2",
            "A3",
            "A4",
        },
        (
            "Task-2 candidate set "
            "is incomplete."
        ),
    )

    require(
        candidates_frame[
            "selection_metric"
        ].nunique()
        == 1,
        (
            "Task-2 candidates used "
            "different selection metrics."
        ),
    )

    require(
        candidates_frame[
            "selection_metric"
        ].iloc[0]
        == SELECTION_METRIC,
        (
            "Unexpected Task-2 "
            "selection metric."
        ),
    )

    require(
        (
            candidates_frame[
                "pretraining_lock_sha256"
            ]
            == EXPECTED_LOCK_SHA256
        ).all(),
        (
            "Task-2 candidates do not "
            "share the same frozen lock."
        ),
    )

    for value in candidates_frame[
        "test_split_used"
    ]:

        require(
            false_like(value),
            (
                "At least one Task-2 "
                "candidate indicates "
                "TEST usage."
            ),
        )

    print(
        "Formal candidate set: PASS"
    )

    print(
        "Common selection metric:",
        SELECTION_METRIC,
    )

    print(
        "Common seed:",
        SEED,
    )

    print(
        "TEST usage across candidates: 0"
    )


    # ========================================================
    # 5. CHECKPOINT VALIDATION
    # ========================================================

    section(
        "5. CANDIDATE CHECKPOINT VALIDATION"
    )

    validated = []

    for variant in [
        "A1",
        "A2",
        "A3",
        "A4",
    ]:

        row = candidates_frame.loc[
            candidates_frame[
                "variant"
            ]
            == variant
        ].iloc[0]

        candidate = validate_candidate(
            row=row,
            variant=variant,
        )

        validated.append(
            candidate
        )

        print(
            f"{variant} checkpoint: PASS"
        )

        print(
            "  model:",
            candidate[
                "model"
            ],
        )

        print(
            "  graph:",
            candidate[
                "graph"
            ],
        )

        print(
            "  VAL macro AP:",
            f"{candidate['best_val_auc_pr_macro']:.8f}",
        )

        print(
            "  SHA256:",
            candidate[
                "checkpoint_sha256"
            ],
        )


    # ========================================================
    # 6. OVERALL TASK-2 AUDIO WINNER
    # ========================================================

    section(
        "6. OVERALL TASK-2 AUDIO SELECTION"
    )

    overall_ranked = sorted(
        validated,
        key=lambda item:
            item[
                "best_val_auc_pr_macro"
            ],
        reverse=True,
    )

    overall_winner = (
        overall_ranked[0]
    )

    require(
        overall_winner[
            "variant"
        ]
        == "A1",
        (
            "Observed overall Task-2 "
            "winner differs from the "
            "current accepted result."
        ),
    )

    require(
        overall_winner[
            "model"
        ]
        == "CNN",
        (
            "Overall Task-2 winner "
            "is unexpectedly not CNN."
        ),
    )

    print(
        "Overall Task-2 winner:",
        overall_winner[
            "variant"
        ],
        overall_winner[
            "model"
        ],
    )

    print(
        "VAL macro AP:",
        f"{overall_winner['best_val_auc_pr_macro']:.8f}",
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "This is the strongest "
        "audio-only Task-2 model "
        "among A1-A4."
    )


    # ========================================================
    # 7. BEST GNN FOR TASK 3
    # ========================================================

    section(
        "7. TASK-3 GNN ENCODER SELECTION"
    )

    gnn_candidates = [
        item
        for item
        in validated
        if item[
            "variant"
        ]
        in {
            "A2",
            "A3",
            "A4",
        }
    ]

    require(
        len(gnn_candidates)
        == 3,
        (
            "Expected exactly three "
            "GNN candidates."
        ),
    )

    gnn_ranked = sorted(
        gnn_candidates,
        key=lambda item:
            item[
                "best_val_auc_pr_macro"
            ],
        reverse=True,
    )

    gnn_winner = (
        gnn_ranked[0]
    )

    require(
        gnn_winner[
            "variant"
        ]
        == "A4",
        (
            "Observed GNN winner "
            "differs from current "
            "accepted result."
        ),
    )

    require(
        gnn_winner[
            "model"
        ]
        == "GAT",
        (
            "Best GNN is unexpectedly "
            "not GAT."
        ),
    )

    require(
        gnn_winner[
            "graph"
        ]
        == "G2",
        (
            "Best GNN does not use "
            "the frozen G2 topology."
        ),
    )

    print(
        "Best GNN:",
        gnn_winner[
            "variant"
        ],
        gnn_winner[
            "model"
        ],
        gnn_winner[
            "graph"
        ],
    )

    print(
        "VAL macro AP:",
        f"{gnn_winner['best_val_auc_pr_macro']:.8f}",
    )

    print()
    print(
        "Task-3 fusion GNN encoder:"
    )

    print(
        "A4 GAT / G2"
    )

    print()
    print(
        "CNN is NOT substituted for "
        "this encoder because Task 3 "
        "requires a GNN graph "
        "representation g in R^128."
    )


    # ========================================================
    # 8. FREEZE RECORD
    # ========================================================

    section(
        "8. FREEZE TASK-2 SELECTIONS"
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

        "selection_basis":
            "validation_only",

        "selection_metric":
            SELECTION_METRIC,

        "model_selection_seed":
            SEED,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "pretraining_lock_sha256":
            EXPECTED_LOCK_SHA256,

        "graph_topology_selection": {
            "path":
                relative(
                    GRAPH_SELECTION_PATH
                ),

            "a4_graph":
                "G2",

            "winner":
                "A3 GraphSAGE/G2",
        },

        "candidates":
            validated,

        "overall_audio_ranking":
            overall_ranked,

        "overall_audio_winner":
            overall_winner,

        "gnn_ranking":
            gnn_ranked,

        "best_gnn_encoder":
            gnn_winner,

        "task3_fusion_gnn_encoder":
            gnn_winner,

        "task3_gnn_only_reference":
            gnn_winner,

        "notes": [
            (
                "The overall Task-2 "
                "audio winner is selected "
                "across A1-A4."
            ),

            (
                "Task-3 fusion requires "
                "a GNN encoder producing "
                "a 128-dimensional graph "
                "representation, so its "
                "encoder is selected only "
                "from A2/A3/A4."
            ),

            (
                "The CNN remains the "
                "mandatory audio baseline "
                "and overall Task-2 winner, "
                "but is not substituted for "
                "the GNN in GNN-BERT fusion."
            ),

            (
                "All decisions were made "
                "using validation macro "
                "average precision only."
            ),

            (
                "TEST examples, labels, "
                "and inference were not used."
            ),

            (
                "Threshold tuning has not "
                "yet been performed."
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
        "TASK-2 MODEL SELECTIONS: FROZEN"
    )
    print("=" * 78)

    print()
    print(
        "Overall audio winner:"
    )

    print(
        "  A1 CNN"
    )

    print(
        "  VAL macro AP:",
        f"{overall_winner['best_val_auc_pr_macro']:.8f}",
    )

    print()
    print(
        "Task-3 GNN encoder:"
    )

    print(
        "  A4 GAT / G2"
    )

    print(
        "  VAL macro AP:",
        f"{gnn_winner['best_val_auc_pr_macro']:.8f}",
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
        "Do not generate Task-3 caches "
        "or train fusion models until "
        "this selection output has been "
        "reviewed."
    )


if __name__ == "__main__":
    main()