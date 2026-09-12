from __future__ import annotations

import json
import os
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

TRAIN_PATH = (
    ROOT
    / "data"
    / "splits"
    / "train.json"
)

VAL_PATH = (
    ROOT
    / "data"
    / "splits"
    / "val.json"
)

TEST_PATH = (
    ROOT
    / "data"
    / "splits"
    / "test.json"
)

FROZEN_HASH_PATH = (
    ROOT
    / "data"
    / "splits"
    / "frozen_hashes.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "bert_tokens"
)

TOKENS_PATH = (
    OUTPUT_DIR
    / "bert_tokens.npz"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "bert_token_manifest.parquet"
)

MANIFEST_CSV_PATH = (
    OUTPUT_DIR
    / "bert_token_manifest.csv"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "bert_token_cache_report.json"
)


EXPECTED_TOTAL = 5140
EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_TEST = 514


def load_split(
    path: Path,
    expected_count: int,
) -> list[str]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    ids = [
        str(x)
        for x in data["track_ids"]
    ]

    if len(ids) != expected_count:
        raise RuntimeError(
            f"{path.name}: expected "
            f"{expected_count}, found "
            f"{len(ids)}."
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{path.name}: duplicate IDs."
        )

    return ids


def save_npz_atomic(
    path: Path,
    **arrays,
):

    temp_path = path.with_suffix(
        ".tmp.npz"
    )

    np.savez_compressed(
        temp_path,
        **arrays,
    )

    os.replace(
        temp_path,
        path,
    )


def main():

    # --------------------------------------------------
    # Confirm dataset/split/label freeze happened
    # --------------------------------------------------

    if not FROZEN_HASH_PATH.exists():
        raise RuntimeError(
            "frozen_hashes.json missing. "
            "Run Script 19 first."
        )

    with FROZEN_HASH_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        frozen = json.load(f)

    if not frozen.get(
        "frozen",
        False,
    ):
        raise RuntimeError(
            "Dataset is not marked frozen."
        )

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    model_name = str(
        config["text"]["model_name"]
    )

    max_length = config[
        "text"
    ].get(
        "max_length"
    )

    if max_length is None:
        raise RuntimeError(
            "text.max_length is not set. "
            "Run Script 20 first."
        )

    max_length = int(
        max_length
    )

    if max_length != 112:
        raise RuntimeError(
            f"Expected reviewed "
            f"max_length=112, "
            f"found {max_length}."
        )

    # --------------------------------------------------
    # Frozen splits
    # --------------------------------------------------

    train_ids = load_split(
        TRAIN_PATH,
        EXPECTED_TRAIN,
    )

    val_ids = load_split(
        VAL_PATH,
        EXPECTED_VAL,
    )

    test_ids = load_split(
        TEST_PATH,
        EXPECTED_TEST,
    )

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
            "Train/val overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test overlap."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Val/test overlap."
        )

    all_ids = (
        train_set
        | val_set
        | test_set
    )

    if len(all_ids) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Split union does not contain "
            "5140 tracks."
        )

    # --------------------------------------------------
    # Canonical dataset
    # --------------------------------------------------

    df = pd.read_parquet(
        CANONICAL_PATH
    ).copy()

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"found {len(df)}."
        )

    if df[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical track ID."
        )

    if df[
        "caption"
    ].isna().any():
        raise RuntimeError(
            "Missing caption detected."
        )

    if (
        set(
            df["track_id"].astype(str)
        )
        != all_ids
    ):
        raise RuntimeError(
            "Canonical IDs do not exactly "
            "match frozen split IDs."
        )

    # Stable cache ordering.
    df = (
        df.sort_values(
            "track_id"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

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

    captions = (
        df["caption"]
        .astype(str)
        .tolist()
    )

    print(
        f"Tokenizing {len(captions)} captions "
        f"with max_length={max_length}..."
    )

    encoded = tokenizer(
        captions,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )

    input_ids = (
        encoded["input_ids"]
        .astype(
            np.int32,
            copy=False,
        )
    )

    attention_mask = (
        encoded["attention_mask"]
        .astype(
            np.uint8,
            copy=False,
        )
    )

    if "token_type_ids" in encoded:

        token_type_ids = (
            encoded[
                "token_type_ids"
            ]
            .astype(
                np.uint8,
                copy=False,
            )
        )

    else:

        token_type_ids = np.zeros(
            (
                EXPECTED_TOTAL,
                max_length,
            ),
            dtype=np.uint8,
        )

    # --------------------------------------------------
    # Shape checks
    # --------------------------------------------------

    expected_shape = (
        EXPECTED_TOTAL,
        max_length,
    )

    if (
        input_ids.shape
        != expected_shape
    ):
        raise RuntimeError(
            f"input_ids shape "
            f"{input_ids.shape}, expected "
            f"{expected_shape}."
        )

    if (
        attention_mask.shape
        != expected_shape
    ):
        raise RuntimeError(
            "attention_mask shape mismatch."
        )

    if (
        token_type_ids.shape
        != expected_shape
    ):
        raise RuntimeError(
            "token_type_ids shape mismatch."
        )

    if not np.isin(
        attention_mask,
        [0, 1],
    ).all():
        raise RuntimeError(
            "Non-binary attention mask."
        )

    if not np.isin(
        token_type_ids,
        [0, 1],
    ).all():
        raise RuntimeError(
            "Unexpected token_type_ids value."
        )

    # --------------------------------------------------
    # Compute unpadded cached length
    # --------------------------------------------------

    cached_lengths = (
        attention_mask.sum(
            axis=1
        )
        .astype(
            np.int16
        )
    )

    if np.any(
        cached_lengths <= 0
    ):
        raise RuntimeError(
            "Invalid zero-length tokenized caption."
        )

    if np.any(
        cached_lengths
        > max_length
    ):
        raise RuntimeError(
            "Cached token length exceeds "
            "max_length."
        )

    # --------------------------------------------------
    # Split lookup
    # --------------------------------------------------

    split_lookup = {}

    for track_id in train_ids:
        split_lookup[
            track_id
        ] = "train"

    for track_id in val_ids:
        split_lookup[
            track_id
        ] = "val"

    for track_id in test_ids:
        split_lookup[
            track_id
        ] = "test"

    # --------------------------------------------------
    # Manifest
    # --------------------------------------------------

    manifest = pd.DataFrame({
        "cache_index": np.arange(
            EXPECTED_TOTAL,
            dtype=np.int32,
        ),

        "track_id": (
            df["track_id"]
            .astype(str)
            .to_numpy()
        ),

        "split": [
            split_lookup[
                str(track_id)
            ]
            for track_id
            in df["track_id"]
        ],

        "cached_token_length": (
            cached_lengths
        ),
    })

    if manifest[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate token manifest ID."
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_npz_atomic(
        TOKENS_PATH,

        input_ids=input_ids,

        attention_mask=(
            attention_mask
        ),

        token_type_ids=(
            token_type_ids
        ),
    )

    manifest.to_parquet(
        MANIFEST_PATH,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_CSV_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Reload and validate cache
    # --------------------------------------------------

    cache = np.load(
        TOKENS_PATH,
        allow_pickle=False,
    )

    for key in [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]:

        if key not in cache.files:
            raise RuntimeError(
                f"Missing cache array: "
                f"{key}"
            )

    if (
        cache["input_ids"].shape
        != expected_shape
    ):
        raise RuntimeError(
            "Reloaded input_ids "
            "shape mismatch."
        )

    if (
        cache[
            "attention_mask"
        ].shape
        != expected_shape
    ):
        raise RuntimeError(
            "Reloaded attention_mask "
            "shape mismatch."
        )

    if (
        cache[
            "token_type_ids"
        ].shape
        != expected_shape
    ):
        raise RuntimeError(
            "Reloaded token_type_ids "
            "shape mismatch."
        )

    # --------------------------------------------------
    # Split counts
    # --------------------------------------------------

    split_counts = (
        manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        split_counts.get(
            "train",
            0,
        )
        != EXPECTED_TRAIN
    ):
        raise RuntimeError(
            "Train cache count mismatch."
        )

    if (
        split_counts.get(
            "val",
            0,
        )
        != EXPECTED_VAL
    ):
        raise RuntimeError(
            "Val cache count mismatch."
        )

    if (
        split_counts.get(
            "test",
            0,
        )
        != EXPECTED_TEST
    ):
        raise RuntimeError(
            "Test cache count mismatch."
        )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    report = {
        "tracks": EXPECTED_TOTAL,

        "model_name": model_name,

        "max_length": max_length,

        "input_ids_shape": list(
            input_ids.shape
        ),

        "attention_mask_shape": list(
            attention_mask.shape
        ),

        "token_type_ids_shape": list(
            token_type_ids.shape
        ),

        "input_ids_dtype": (
            str(input_ids.dtype)
        ),

        "attention_mask_dtype": (
            str(
                attention_mask.dtype
            )
        ),

        "token_type_ids_dtype": (
            str(
                token_type_ids.dtype
            )
        ),

        "split_counts": {
            str(k): int(v)
            for k, v
            in split_counts.items()
        },

        "cached_length_statistics": {
            "min": int(
                cached_lengths.min()
            ),

            "max": int(
                cached_lengths.max()
            ),

            "mean": float(
                cached_lengths.mean()
            ),

            "median": float(
                np.median(
                    cached_lengths
                )
            ),
        },

        "cache_validation": "PASS",
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "MusicCaps BERT token cache complete"
    )

    print("=" * 58)

    print(
        f"Tracks          : "
        f"{EXPECTED_TOTAL}"
    )

    print(
        f"Model           : "
        f"{model_name}"
    )

    print(
        f"max_length      : "
        f"{max_length}"
    )

    print(
        f"input_ids       : "
        f"{input_ids.shape}"
    )

    print(
        f"attention_mask  : "
        f"{attention_mask.shape}"
    )

    print(
        f"token_type_ids  : "
        f"{token_type_ids.shape}"
    )

    print()

    print(
        f"Train / Val / Test: "
        f"{split_counts['train']} / "
        f"{split_counts['val']} / "
        f"{split_counts['test']}"
    )

    print()

    print(
        f"Cached length min    : "
        f"{cached_lengths.min()}"
    )

    print(
        f"Cached length max    : "
        f"{cached_lengths.max()}"
    )

    print(
        f"Cached length mean   : "
        f"{cached_lengths.mean():.2f}"
    )

    print(
        f"Cached length median : "
        f"{np.median(cached_lengths):.2f}"
    )

    print()

    print(
        "Cache validation: PASS"
    )

    print()

    print(
        f"Tokens  : {TOKENS_PATH}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"Report  : {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()