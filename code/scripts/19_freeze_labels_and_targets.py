from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
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

SYNONYM_PATH = (
    ROOT
    / "data"
    / "splits"
    / "aspect_synonyms.json"
)

VOCAB_PATH = (
    ROOT
    / "data"
    / "splits"
    / "label_vocab.json"
)

HASH_PATH = (
    ROOT
    / "data"
    / "splits"
    / "frozen_hashes.json"
)

TARGET_DIR = (
    ROOT
    / "data"
    / "processed"
    / "labels"
)

TARGET_PARQUET = (
    TARGET_DIR
    / "musiccaps_targets.parquet"
)

TARGET_CSV = (
    TARGET_DIR
    / "musiccaps_targets.csv"
)

SUPPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "label_eda"
    / "final_label_support_by_split.csv"
)


EXPECTED_TOTAL = 5140
EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_TEST = 514
TOP_K = 20


# Explicit reviewed vocabulary expected from Script 17.
EXPECTED_TOP20 = [
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_aspects(value) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    try:
        if pd.isna(value):
            return []
    except (
        TypeError,
        ValueError,
    ):
        pass

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
            ),
        ):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return [text]


def normalize_phrase(
    phrase: str,
) -> str:

    phrase = unicodedata.normalize(
        "NFKC",
        str(phrase),
    )

    phrase = (
        phrase
        .lower()
        .strip()
    )

    phrase = "".join(
        " "
        if unicodedata.category(
            char
        ).startswith("P")
        else char
        for char in phrase
    )

    phrase = re.sub(
        r"\s+",
        " ",
        phrase,
    ).strip()

    return phrase


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
        for x in data[
            "track_ids"
        ]
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


def load_synonyms():

    with SYNONYM_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    mapping = {
        normalize_phrase(k):
        normalize_phrase(v)

        for k, v
        in data["mapping"].items()
    }

    if len(mapping) != 17:
        raise RuntimeError(
            f"Expected 17 reviewed synonym "
            f"mappings, found {len(mapping)}."
        )

    for source, target in (
        mapping.items()
    ):
        if target in mapping:
            raise RuntimeError(
                f"Synonym chain detected: "
                f"{source} -> {target}"
            )

    return data, mapping


def get_merged_aspects(
    value,
    synonyms,
) -> set[str]:

    merged = set()

    for raw_phrase in parse_aspects(
        value
    ):
        phrase = normalize_phrase(
            raw_phrase
        )

        if not phrase:
            continue

        phrase = synonyms.get(
            phrase,
            phrase,
        )

        merged.add(
            phrase
        )

    return merged


def atomic_json_write(
    data,
    path: Path,
):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def main():

    # --------------------------------------------------
    # Prevent accidental re-freezing
    # --------------------------------------------------

    frozen_outputs = [
        VOCAB_PATH,
        HASH_PATH,
        TARGET_PARQUET,
        TARGET_CSV,
    ]

    existing = [
        path
        for path in frozen_outputs
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "Frozen label outputs already exist.\n"
            "Do NOT regenerate them:\n"
            + "\n".join(
                str(x)
                for x in existing
            )
        )

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    min_support = config[
        "labels"
    ][
        "final_min_train_support"
    ]

    if min_support != 30:
        raise RuntimeError(
            "Set labels.final_min_train_support "
            "to 30 in config.yaml before freezing."
        )

    if int(
        config["labels"]["top_k"]
    ) != TOP_K:
        raise RuntimeError(
            "Expected labels.top_k = 20."
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

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if train_set & val_set:
        raise RuntimeError(
            "Train/validation overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test overlap."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Validation/test overlap."
        )

    all_split_ids = (
        train_set
        | val_set
        | test_set
    )

    if len(all_split_ids) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Split union is not 5140."
        )

    # --------------------------------------------------
    # Canonical dataset
    # --------------------------------------------------

    df = pd.read_parquet(
        CANONICAL_PATH
    ).copy()

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} "
            f"canonical rows, found "
            f"{len(df)}."
        )

    if df["track_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical track ID."
        )

    canonical_ids = set(
        df["track_id"].astype(str)
    )

    if canonical_ids != all_split_ids:
        raise RuntimeError(
            "Canonical track IDs do not "
            "exactly match split track IDs."
        )

    synonym_data, synonyms = (
        load_synonyms()
    )

    # --------------------------------------------------
    # Reconstruct TRAIN label support independently
    # --------------------------------------------------

    train = df.loc[
        df["track_id"]
        .astype(str)
        .isin(train_set)
    ].copy()

    train_support = Counter()

    for _, row in train.iterrows():

        aspects = get_merged_aspects(
            row["aspects"],
            synonyms,
        )

        for phrase in aspects:
            train_support[
                phrase
            ] += 1

    eligible = [
        (
            phrase,
            int(support),
        )
        for phrase, support
        in train_support.items()
        if support >= min_support
    ]

    eligible.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    if len(eligible) < TOP_K:
        raise RuntimeError(
            "Fewer than 20 eligible labels."
        )

    top20 = eligible[
        :TOP_K
    ]

    derived_labels = [
        phrase
        for phrase, _
        in top20
    ]

    if derived_labels != EXPECTED_TOP20:
        print()
        print("Expected Top-20:")
        print(EXPECTED_TOP20)

        print()
        print("Derived Top-20:")
        print(derived_labels)

        raise RuntimeError(
            "Top-20 differs from reviewed "
            "vocabulary. Refusing to freeze."
        )

    rank20_support = top20[
        -1
    ][1]

    rank21_phrase, rank21_support = (
        eligible[TOP_K]
    )

    if (
        rank20_support != 183
        or rank21_phrase
        != "punchy snare"
        or rank21_support != 178
    ):
        raise RuntimeError(
            "Rank-20/21 boundary differs "
            "from reviewed results."
        )

    # --------------------------------------------------
    # Create label vocabulary
    # --------------------------------------------------

    label_records = []

    for index, (
        label,
        support,
    ) in enumerate(top20):

        label_records.append({
            "index": index,
            "label": label,
            "train_support": int(
                support
            ),
            "train_support_percent": (
                float(
                    support
                    / EXPECTED_TRAIN
                    * 100.0
                )
            ),
        })

    vocab = {
        "version": 1,

        "dataset": "MusicCaps",

        "construction_split": (
            "train_only"
        ),

        "train_tracks": (
            EXPECTED_TRAIN
        ),

        "normalization": (
            "NFKC + lowercase + "
            "punctuation-to-space + "
            "whitespace collapse"
        ),

        "synonym_map_version": (
            synonym_data.get(
                "version"
            )
        ),

        "final_min_train_support": (
            min_support
        ),

        "eligible_phrase_count": (
            len(eligible)
        ),

        "top_k": TOP_K,

        "rank20_support": int(
            rank20_support
        ),

        "rank21": {
            "label": rank21_phrase,
            "train_support": int(
                rank21_support
            ),
        },

        "labels": label_records,

        "label_to_index": {
            label: index
            for index, label
            in enumerate(
                derived_labels
            )
        },
    }

    # --------------------------------------------------
    # Generate targets for ALL 5140 tracks
    #
    # Vocabulary is already determined entirely from
    # TRAIN above. Validation/test data cannot change it.
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

    label_to_index = vocab[
        "label_to_index"
    ]

    target_rows = []

    for _, row in df.iterrows():

        track_id = str(
            row["track_id"]
        )

        aspects = get_merged_aspects(
            row["aspects"],
            synonyms,
        )

        vector = np.zeros(
            TOP_K,
            dtype=np.int8,
        )

        for phrase in aspects:

            index = label_to_index.get(
                phrase
            )

            if index is not None:
                vector[index] = 1

        output = {
            "track_id": track_id,
            "split": split_lookup[
                track_id
            ],
            "positive_labels": int(
                vector.sum()
            ),
        }

        for i in range(TOP_K):
            output[
                f"y_{i:02d}"
            ] = int(vector[i])

        target_rows.append(
            output
        )

    targets = pd.DataFrame(
        target_rows
    )

    if len(targets) != EXPECTED_TOTAL:
        raise RuntimeError(
            "Target row-count mismatch."
        )

    if targets[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate target track IDs."
        )

    if set(
        targets["track_id"]
    ) != canonical_ids:
        raise RuntimeError(
            "Target IDs do not match "
            "canonical IDs."
        )

    y_columns = [
        f"y_{i:02d}"
        for i in range(
            TOP_K
        )
    ]

    values = targets[
        y_columns
    ].to_numpy()

    if not np.isin(
        values,
        [0, 1],
    ).all():
        raise RuntimeError(
            "Non-binary target detected."
        )

    # --------------------------------------------------
    # Verify TRAIN target supports against vocabulary
    # --------------------------------------------------

    train_targets = targets.loc[
        targets["split"]
        == "train"
    ]

    for label_info in label_records:

        index = label_info[
            "index"
        ]

        expected = label_info[
            "train_support"
        ]

        actual = int(
            train_targets[
                f"y_{index:02d}"
            ].sum()
        )

        if actual != expected:
            raise RuntimeError(
                f"Target support mismatch "
                f"for {label_info['label']}: "
                f"{actual} != {expected}"
            )

    # --------------------------------------------------
    # Final support report
    # --------------------------------------------------

    support_rows = []

    for label_info in (
        label_records
    ):

        index = label_info[
            "index"
        ]

        column = f"y_{index:02d}"

        train_count = int(
            targets.loc[
                targets["split"]
                == "train",
                column,
            ].sum()
        )

        val_count = int(
            targets.loc[
                targets["split"]
                == "val",
                column,
            ].sum()
        )

        test_count = int(
            targets.loc[
                targets["split"]
                == "test",
                column,
            ].sum()
        )

        support_rows.append({
            "index": index,
            "label": label_info[
                "label"
            ],
            "train_support": (
                train_count
            ),
            "val_support": (
                val_count
            ),
            "test_support": (
                test_count
            ),
            "total_support": (
                train_count
                + val_count
                + test_count
            ),
        })

    support_df = pd.DataFrame(
        support_rows
    )

    # --------------------------------------------------
    # Write frozen outputs atomically
    # --------------------------------------------------

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VOCAB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUPPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocab_temp = (
        VOCAB_PATH.with_suffix(
            ".json.tmp"
        )
    )

    vocab_temp.write_text(
        json.dumps(
            vocab,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    target_parquet_temp = (
        TARGET_PARQUET.with_suffix(
            ".tmp.parquet"
        )
    )

    targets.to_parquet(
        target_parquet_temp,
        index=False,
    )

    target_csv_temp = (
        TARGET_CSV.with_suffix(
            ".tmp.csv"
        )
    )

    targets.to_csv(
        target_csv_temp,
        index=False,
    )

    support_temp = (
        SUPPORT_PATH.with_suffix(
            ".tmp.csv"
        )
    )

    support_df.to_csv(
        support_temp,
        index=False,
    )

    os.replace(
        vocab_temp,
        VOCAB_PATH,
    )

    os.replace(
        target_parquet_temp,
        TARGET_PARQUET,
    )

    os.replace(
        target_csv_temp,
        TARGET_CSV,
    )

    os.replace(
        support_temp,
        SUPPORT_PATH,
    )

    # --------------------------------------------------
    # Hash lock
    # --------------------------------------------------

    hashes = {
        "version": 1,

        "dataset": "MusicCaps",

        "dataset_tracks": (
            EXPECTED_TOTAL
        ),

        "split_counts": {
            "train": EXPECTED_TRAIN,
            "val": EXPECTED_VAL,
            "test": EXPECTED_TEST,
        },

        "label_count": TOP_K,

        "final_min_train_support": (
            min_support
        ),

        "hashes": {
            "canonical_dataset": (
                sha256_file(
                    CANONICAL_PATH
                )
            ),

            "train_split": (
                sha256_file(
                    TRAIN_PATH
                )
            ),

            "val_split": (
                sha256_file(
                    VAL_PATH
                )
            ),

            "test_split": (
                sha256_file(
                    TEST_PATH
                )
            ),

            "aspect_synonyms": (
                sha256_file(
                    SYNONYM_PATH
                )
            ),

            "label_vocab": (
                sha256_file(
                    VOCAB_PATH
                )
            ),

            "targets_parquet": (
                sha256_file(
                    TARGET_PARQUET
                )
            ),

            "targets_csv": (
                sha256_file(
                    TARGET_CSV
                )
            ),
        },

        "frozen": True,
    }

    atomic_json_write(
        hashes,
        HASH_PATH,
    )

    # --------------------------------------------------
    # Console
    # --------------------------------------------------

    print()
    print(
        "MusicCaps labels + targets FROZEN"
    )
    print("=" * 58)

    print(
        f"Dataset tracks        : "
        f"{EXPECTED_TOTAL}"
    )

    print(
        f"Train / Val / Test    : "
        f"{EXPECTED_TRAIN} / "
        f"{EXPECTED_VAL} / "
        f"{EXPECTED_TEST}"
    )

    print(
        f"Final minimum support : "
        f"{min_support}"
    )

    print(
        f"Eligible labels       : "
        f"{len(eligible)}"
    )

    print(
        f"Frozen label count    : "
        f"{TOP_K}"
    )

    print()
    print("Frozen vocabulary")
    print("-" * 58)

    print(
        support_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Rank 20 : piano "
        f"({rank20_support})"
    )

    print(
        f"Rank 21 : "
        f"{rank21_phrase} "
        f"({rank21_support})"
    )

    print()
    print("Saved:")
    print(VOCAB_PATH)
    print(TARGET_PARQUET)
    print(TARGET_CSV)
    print(SUPPORT_PATH)
    print(HASH_PATH)

    print()
    print(
        "Split files, synonym map, vocabulary, "
        "and target vectors are now hash-locked."
    )


if __name__ == "__main__":
    main()