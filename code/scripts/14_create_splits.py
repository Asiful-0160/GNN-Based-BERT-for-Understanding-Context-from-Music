from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

CANONICAL_PATH = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_canonical.parquet"
)

SPLITS_DIR = (
    ROOT
    / "data"
    / "splits"
)

TRAIN_PATH = SPLITS_DIR / "train.json"
VAL_PATH = SPLITS_DIR / "val.json"
TEST_PATH = SPLITS_DIR / "test.json"

REPORT_PATH = (
    SPLITS_DIR
    / "split_report.json"
)


def save_split(
    path: Path,
    name: str,
    ids: list[str],
    seed: int,
):
    payload = {
        "split": name,
        "seed": seed,
        "count": len(ids),
        "track_ids": ids,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():

    # Prevent accidental split regeneration.
    existing = [
        path
        for path in [
            TRAIN_PATH,
            VAL_PATH,
            TEST_PATH,
        ]
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "Split files already exist. "
            "Do NOT regenerate them.\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    seed = int(
        config["dataset"]["split"]["seed"]
    )

    train_fraction = float(
        config["dataset"]["split"][
            "train"
        ]
    )

    val_fraction = float(
        config["dataset"]["split"][
            "val"
        ]
    )

    test_fraction = float(
        config["dataset"]["split"][
            "test"
        ]
    )

    if not np.isclose(
        train_fraction
        + val_fraction
        + test_fraction,
        1.0,
    ):
        raise RuntimeError(
            "Split fractions do not sum to 1."
        )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    if df["track_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate track_id detected."
        )

    if df["source_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate source_id detected."
        )

    total = len(df)

    # Stable base ordering before RNG permutation.
    df = df.sort_values(
        "track_id"
    ).reset_index(drop=True)

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        total
    )

    # 5140 gives exact:
    # train = 4112
    # val   = 514
    # test  = 514

    n_train = int(
        round(
            total * train_fraction
        )
    )

    n_val = int(
        round(
            total * val_fraction
        )
    )

    n_test = (
        total
        - n_train
        - n_val
    )

    train_indices = permutation[
        :n_train
    ]

    val_indices = permutation[
        n_train:
        n_train + n_val
    ]

    test_indices = permutation[
        n_train + n_val:
    ]

    train_ids = (
        df.iloc[
            train_indices
        ]["track_id"]
        .astype(str)
        .tolist()
    )

    val_ids = (
        df.iloc[
            val_indices
        ]["track_id"]
        .astype(str)
        .tolist()
    )

    test_ids = (
        df.iloc[
            test_indices
        ]["track_id"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------
    # Leakage checks
    # --------------------------------------------------

    train_set = set(
        train_ids
    )

    val_set = set(
        val_ids
    )

    test_set = set(
        test_ids
    )

    if train_set & val_set:
        raise RuntimeError(
            "Train/validation overlap detected."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test overlap detected."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Validation/test overlap detected."
        )

    union = (
        train_set
        | val_set
        | test_set
    )

    if len(union) != total:
        raise RuntimeError(
            "Split union does not equal dataset."
        )

    # Source-level check, despite source IDs being
    # unique in this MusicCaps version.
    track_to_source = dict(
        zip(
            df["track_id"].astype(str),
            df["source_id"].astype(str),
        )
    )

    train_sources = {
        track_to_source[x]
        for x in train_ids
    }

    val_sources = {
        track_to_source[x]
        for x in val_ids
    }

    test_sources = {
        track_to_source[x]
        for x in test_ids
    }

    if train_sources & val_sources:
        raise RuntimeError(
            "Train/validation source leakage."
        )

    if train_sources & test_sources:
        raise RuntimeError(
            "Train/test source leakage."
        )

    if val_sources & test_sources:
        raise RuntimeError(
            "Validation/test source leakage."
        )

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_split(
        TRAIN_PATH,
        "train",
        train_ids,
        seed,
    )

    save_split(
        VAL_PATH,
        "val",
        val_ids,
        seed,
    )

    save_split(
        TEST_PATH,
        "test",
        test_ids,
        seed,
    )

    report = {
        "seed": seed,
        "total": total,
        "train": len(
            train_ids
        ),
        "validation": len(
            val_ids
        ),
        "test": len(
            test_ids
        ),
        "train_fraction": (
            len(train_ids) / total
        ),
        "validation_fraction": (
            len(val_ids) / total
        ),
        "test_fraction": (
            len(test_ids) / total
        ),
        "track_overlap": False,
        "source_overlap": False,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("MusicCaps deterministic split")
    print("=" * 42)

    print(
        f"Total      : {total}"
    )

    print(
        f"Train      : "
        f"{len(train_ids)} "
        f"({len(train_ids)/total*100:.2f}%)"
    )

    print(
        f"Validation : "
        f"{len(val_ids)} "
        f"({len(val_ids)/total*100:.2f}%)"
    )

    print(
        f"Test       : "
        f"{len(test_ids)} "
        f"({len(test_ids)/total*100:.2f}%)"
    )

    print(
        f"Seed       : {seed}"
    )

    print(
        "Track overlap: NONE"
    )

    print(
        "Source overlap: NONE"
    )

    print()
    print(
        f"Train: {TRAIN_PATH}"
    )

    print(
        f"Val  : {VAL_PATH}"
    )

    print(
        f"Test : {TEST_PATH}"
    )


if __name__ == "__main__":
    main()