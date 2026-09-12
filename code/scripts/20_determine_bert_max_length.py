from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "config.yaml"
)

CANONICAL_PATH = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_canonical.parquet"
)

TRAIN_SPLIT_PATH = (
    ROOT
    / "data"
    / "splits"
    / "train.json"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "bert_length_audit.json"
)

LENGTHS_PATH = (
    ROOT
    / "data"
    / "audit"
    / "bert_train_caption_lengths.csv"
)


EXPECTED_TRAIN = 4112


def round_up_multiple(
    value: int,
    multiple: int,
) -> int:
    return int(
        math.ceil(
            value / multiple
        ) * multiple
    )


def main():

    # ---------------------------------------------
    # Config
    # ---------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    model_name = config[
        "text"
    ][
        "model_name"
    ]

    percentile = float(
        config[
            "text"
        ][
            "percentile"
        ]
    )

    multiple = int(
        config[
            "text"
        ][
            "round_to_multiple"
        ]
    )

    max_allowed = int(
        config[
            "text"
        ][
            "max_allowed_length"
        ]
    )

    # ---------------------------------------------
    # Frozen TRAIN split only
    # ---------------------------------------------

    with TRAIN_SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        train_split = json.load(f)

    train_ids = [
        str(x)
        for x in train_split[
            "track_ids"
        ]
    ]

    if len(train_ids) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} "
            f"training IDs, found "
            f"{len(train_ids)}."
        )

    # ---------------------------------------------
    # Canonical captions
    # ---------------------------------------------

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    train = df.loc[
        df[
            "track_id"
        ]
        .astype(str)
        .isin(
            set(train_ids)
        )
    ].copy()

    if len(train) != EXPECTED_TRAIN:
        raise RuntimeError(
            "Training row-count mismatch."
        )

    if train[
        "caption"
    ].isna().any():
        raise RuntimeError(
            "Missing training caption."
        )

    # ---------------------------------------------
    # BERT tokenizer
    # ---------------------------------------------

    print()
    print(
        f"Loading tokenizer: "
        f"{model_name}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name
        )
    )

    records = []

    print(
        "Tokenizing 4112 TRAIN captions..."
    )

    for _, row in (
        train.iterrows()
    ):

        caption = str(
            row[
                "caption"
            ]
        )

        encoded = tokenizer(
            caption,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        length = len(
            encoded[
                "input_ids"
            ]
        )

        records.append({
            "track_id": str(
                row[
                    "track_id"
                ]
            ),
            "token_length": (
                length
            ),
        })

    lengths_df = pd.DataFrame(
        records
    )

    lengths = (
        lengths_df[
            "token_length"
        ]
        .to_numpy()
    )

    # ---------------------------------------------
    # Determine length from TRAIN only
    # ---------------------------------------------

    percentile_length = int(
        math.ceil(
            np.percentile(
                lengths,
                percentile,
            )
        )
    )

    candidate = (
        round_up_multiple(
            percentile_length,
            multiple,
        )
    )

    candidate = min(
        candidate,
        max_allowed,
    )

    # Explicitly enforce the <=5% truncation rule.
    truncation_rate = float(
        np.mean(
            lengths > candidate
        )
    )

    while (
        truncation_rate > 0.05
        and candidate < max_allowed
    ):

        candidate += multiple

        candidate = min(
            candidate,
            max_allowed,
        )

        truncation_rate = float(
            np.mean(
                lengths > candidate
            )
        )

    truncated_tracks = int(
        np.sum(
            lengths > candidate
        )
    )

    if truncation_rate > 0.05:
        raise RuntimeError(
            "Could not meet <=5% "
            "truncation requirement "
            "within BERT's maximum length."
        )

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    stats = {
        "count": int(
            len(lengths)
        ),

        "min": int(
            lengths.min()
        ),

        "max": int(
            lengths.max()
        ),

        "mean": float(
            lengths.mean()
        ),

        "median": float(
            np.median(
                lengths
            )
        ),

        "p90": float(
            np.percentile(
                lengths,
                90,
            )
        ),

        "p95": float(
            np.percentile(
                lengths,
                95,
            )
        ),

        "p99": float(
            np.percentile(
                lengths,
                99,
            )
        ),
    }

    report = {
        "dataset": (
            "MusicCaps"
        ),

        "split_used": (
            "train_only"
        ),

        "training_tracks": (
            EXPECTED_TRAIN
        ),

        "model_name": (
            model_name
        ),

        "special_tokens_included": (
            True
        ),

        "truncation_during_audit": (
            False
        ),

        "selection_percentile": (
            percentile
        ),

        "round_to_multiple": (
            multiple
        ),

        "max_allowed_length": (
            max_allowed
        ),

        "token_length_statistics": (
            stats
        ),

        "raw_percentile_length": (
            percentile_length
        ),

        "selected_max_length": (
            candidate
        ),

        "tracks_truncated": (
            truncated_tracks
        ),

        "truncation_rate": (
            truncation_rate
        ),

        "truncation_percent": (
            truncation_rate
            * 100.0
        ),
    }

    # ---------------------------------------------
    # Save audit
    # ---------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lengths_df.to_csv(
        LENGTHS_PATH,
        index=False,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------
    # Update config safely
    # ---------------------------------------------

    current_value = config[
        "text"
    ].get(
        "max_length"
    )

    if (
        current_value is not None
        and int(current_value)
        != candidate
    ):
        raise RuntimeError(
            "config.yaml already has "
            f"text.max_length={current_value}, "
            f"but audit derived {candidate}. "
            "Refusing to overwrite."
        )

    config[
        "text"
    ][
        "max_length"
    ] = candidate

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    # ---------------------------------------------
    # Console
    # ---------------------------------------------

    print()
    print(
        "BERT TRAIN-only length audit"
    )

    print("=" * 52)

    print(
        f"Training captions    : "
        f"{EXPECTED_TRAIN}"
    )

    print(
        f"Minimum tokens       : "
        f"{stats['min']}"
    )

    print(
        f"Maximum tokens       : "
        f"{stats['max']}"
    )

    print(
        f"Mean tokens          : "
        f"{stats['mean']:.2f}"
    )

    print(
        f"Median tokens        : "
        f"{stats['median']:.2f}"
    )

    print(
        f"P90                  : "
        f"{stats['p90']:.2f}"
    )

    print(
        f"P95                  : "
        f"{stats['p95']:.2f}"
    )

    print(
        f"P99                  : "
        f"{stats['p99']:.2f}"
    )

    print()
    print(
        f"Raw {percentile:g}th percentile : "
        f"{percentile_length}"
    )

    print(
        f"Selected max_length  : "
        f"{candidate}"
    )

    print(
        f"Tracks truncated     : "
        f"{truncated_tracks}"
    )

    print(
        f"Truncation rate      : "
        f"{truncation_rate * 100:.2f}%"
    )

    print()
    print(
        f"Report : {REPORT_PATH}"
    )

    print(
        f"Lengths: {LENGTHS_PATH}"
    )

    print(
        f"Config : text.max_length="
        f"{candidate}"
    )


if __name__ == "__main__":
    main()