#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

TRIALS_PATH = (
    ROOT
    / "results/csv/hyperparameter_trials.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results/runs/task1/task1_model_selection.json"
)

EXPECTED_PRETRAINING_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_INITIAL_BERT_SHA256 = (
    "e545aa3917de26e12ec35edd3fe68a19085cee93aa77293c1e1b1aa72ac7ab0b"
)

EXPECTED_SEED = 42

EXPECTED_VARIANTS = {
    "T1-A",
    "T1-B",
}

SELECTION_COLUMN = (
    "best_val_auc_pr_macro"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:

    print()
    print("=" * 72)
    print("SCRIPT 33 — SELECT TASK-1 BERT WINNER")
    print("=" * 72)

    print()
    print("Selection uses VALIDATION only.")
    print("TEST is not loaded or evaluated.")

    require(
        TRIALS_PATH.is_file(),
        "Missing hyperparameter_trials.csv",
    )

    trials = pd.read_csv(TRIALS_PATH)

    required_columns = {
        "run_id",
        "status",
        "task",
        "variant",
        "seed",
        "selection_metric",
        "best_epoch",
        SELECTION_COLUMN,
        "checkpoint_path",
        "pretraining_lock_sha256",
        "initial_pretrained_bert_sha256",
        "test_split_used",
    }

    missing = (
        required_columns
        - set(trials.columns)
    )

    require(
        not missing,
        f"Missing trial columns: {sorted(missing)}",
    )

    task1 = trials.loc[
        (trials["task"] == "task1")
        & (trials["status"] == "COMPLETE")
        & (trials["seed"] == EXPECTED_SEED)
    ].copy()

    require(
        len(task1) == 2,
        (
            "Expected exactly two completed "
            "Task-1 seed-42 model-selection runs."
        ),
    )

    require(
        set(task1["variant"])
        == EXPECTED_VARIANTS,
        (
            "Expected completed variants "
            "T1-A and T1-B only."
        ),
    )

    require(
        task1["variant"].is_unique,
        "Duplicate Task-1 variant found.",
    )

    require(
        (
            task1["pretraining_lock_sha256"]
            == EXPECTED_PRETRAINING_LOCK_SHA256
        ).all(),
        "Pretraining-lock mismatch between Task-1 runs.",
    )

    require(
        (
            task1["initial_pretrained_bert_sha256"]
            == EXPECTED_INITIAL_BERT_SHA256
        ).all(),
        (
            "Task-1 variants did not begin "
            "from the same pretrained BERT."
        ),
    )

    # Pandas may read False as bool or text depending on CSV history.
    test_used = (
        task1["test_split_used"]
        .astype(str)
        .str.lower()
    )

    require(
        (test_used == "false").all(),
        "A Task-1 model-selection run used TEST.",
    )

    require(
        task1[
            SELECTION_COLUMN
        ].notna().all(),
        "Missing validation selection metric.",
    )

    require(
        task1[
            SELECTION_COLUMN
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all(),
        "Invalid validation AP value.",
    )

    print()
    print("=== VALIDATED CANDIDATES ===")

    display_columns = [
        "variant",
        "best_epoch",
        SELECTION_COLUMN,
        "checkpoint_path",
    ]

    print(
        task1[
            display_columns
        ]
        .sort_values("variant")
        .to_string(index=False)
    )

    winner_index = (
        task1[
            SELECTION_COLUMN
        ].idxmax()
    )

    winner = task1.loc[
        winner_index
    ]

    losers = task1.drop(
        index=winner_index
    )

    require(
        len(losers) == 1,
        "Expected exactly one losing variant.",
    )

    runner_up = losers.iloc[0]

    winner_checkpoint = (
        ROOT
        / str(
            winner["checkpoint_path"]
        )
    )

    require(
        winner_checkpoint.is_file(),
        (
            "Winning checkpoint does not exist: "
            f"{winner_checkpoint}"
        ),
    )

    checkpoint_sha256 = sha256_file(
        winner_checkpoint
    )

    checkpoint = torch.load(
        winner_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    require(
        checkpoint["variant"]
        == winner["variant"],
        "Checkpoint variant metadata mismatch.",
    )

    require(
        int(checkpoint["seed"])
        == EXPECTED_SEED,
        "Checkpoint seed mismatch.",
    )

    require(
        int(checkpoint["epoch"])
        == int(winner["best_epoch"]),
        "Checkpoint best-epoch mismatch.",
    )

    require(
        abs(
            float(
                checkpoint[
                    "val_auc_pr_macro_mean_ap"
                ]
            )
            - float(
                winner[
                    SELECTION_COLUMN
                ]
            )
        )
        <= 1.0e-12,
        "Checkpoint validation metric mismatch.",
    )

    require(
        checkpoint[
            "pretraining_lock_sha256"
        ]
        == EXPECTED_PRETRAINING_LOCK_SHA256,
        "Winner checkpoint lock mismatch.",
    )

    require(
        checkpoint[
            "initial_pretrained_bert_sha256"
        ]
        == EXPECTED_INITIAL_BERT_SHA256,
        "Winner checkpoint pretrained-BERT mismatch.",
    )

    result = {
        "status": "FROZEN",
        "task": "task1",
        "selection_stage": "model_selection",
        "selection_data": "validation_only",
        "test_split_used": False,

        "selection_metric": (
            "validation macro mean "
            "per-label average precision"
        ),

        "seed": EXPECTED_SEED,

        "candidates": [
            {
                "variant": row["variant"],
                "best_epoch": int(
                    row["best_epoch"]
                ),
                "best_val_auc_pr_macro": float(
                    row[
                        SELECTION_COLUMN
                    ]
                ),
                "checkpoint_path": str(
                    row[
                        "checkpoint_path"
                    ]
                ),
            }
            for _, row in (
                task1
                .sort_values("variant")
                .iterrows()
            )
        ],

        "winner": {
            "variant": str(
                winner["variant"]
            ),
            "best_epoch": int(
                winner["best_epoch"]
            ),
            "best_val_auc_pr_macro": float(
                winner[
                    SELECTION_COLUMN
                ]
            ),
            "checkpoint_path": str(
                winner[
                    "checkpoint_path"
                ]
            ),
            "checkpoint_sha256": (
                checkpoint_sha256
            ),
        },

        "runner_up": {
            "variant": str(
                runner_up["variant"]
            ),
            "best_epoch": int(
                runner_up["best_epoch"]
            ),
            "best_val_auc_pr_macro": float(
                runner_up[
                    SELECTION_COLUMN
                ]
            ),
        },

        "pretraining_lock_sha256": (
            EXPECTED_PRETRAINING_LOCK_SHA256
        ),

        "initial_pretrained_bert_sha256": (
            EXPECTED_INITIAL_BERT_SHA256
        ),

        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "notes": [
            (
                "Winner selected using validation "
                "macro mean AP only."
            ),
            (
                "No prediction threshold was tuned "
                "during Task-1 model selection."
            ),
            (
                "TEST split was not used."
            ),
            (
                "Candidate checkpoints were not "
                "renamed or modified."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    require(
        not OUTPUT_PATH.exists(),
        (
            "Task-1 selection artifact already exists. "
            "Refusing to overwrite it."
        ),
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TASK-1 MODEL SELECTION: FROZEN")
    print("=" * 72)

    print(
        "Winner:",
        result["winner"]["variant"],
    )

    print(
        "Best epoch:",
        result["winner"]["best_epoch"],
    )

    print(
        "VAL macro AP:",
        f'{result["winner"]["best_val_auc_pr_macro"]:.8f}',
    )

    print(
        "Checkpoint:",
        result["winner"]["checkpoint_path"],
    )

    print(
        "Checkpoint SHA256:",
        result["winner"]["checkpoint_sha256"],
    )

    print(
        "Saved:",
        OUTPUT_PATH.relative_to(ROOT),
    )

    print()
    print("TEST examples evaluated: 0")
    print("Threshold tuning performed: NO")


if __name__ == "__main__":
    main()