from __future__ import annotations

import ast
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

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


TARGETS = [
    # Male vocal family
    "male voice",
    "male vocal",
    "male vocalist",
    "male voice singing",

    # Female vocal family
    "female voice",
    "female vocal",
    "female vocalist",
    "female voice singing",

    # Bass family
    "e bass",
    "bass guitar",
    "bass",

    # Electric guitar family
    "e guitar",
    "e guitars",
    "electric guitar",
]

EXAMPLES_PER_PHRASE = 5


def parse_aspects(value):

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(
            parsed,
            (list, tuple),
        ):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [text] if text else []


def normalize_phrase(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.lower().strip()

    text = "".join(
        " "
        if unicodedata.category(
            ch
        ).startswith("P")
        else ch
        for ch in text
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def main():

    with TRAIN_SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_ids = set(
        str(x)
        for x in split["track_ids"]
    )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    train = df.loc[
        df["track_id"]
        .astype(str)
        .isin(train_ids)
    ].copy()

    if len(train) != 4112:
        raise RuntimeError(
            f"Expected 4112 train tracks, "
            f"found {len(train)}."
        )

    target_set = set(TARGETS)

    matches = {
        target: []
        for target in TARGETS
    }

    for _, row in train.iterrows():

        aspects = {
            normalize_phrase(x)
            for x in parse_aspects(
                row["aspects"]
            )
        }

        for target in (
            aspects & target_set
        ):
            matches[target].append({
                "track_id": str(
                    row["track_id"]
                ),
                "caption": str(
                    row["caption"]
                ),
            })

    print()
    print(
        "TRAIN-ONLY ambiguous synonym review"
    )
    print("=" * 70)

    for target in TARGETS:

        rows = matches[target]

        print()
        print(
            f"[{target}]"
            f"  support={len(rows)}"
        )

        print("-" * 70)

        for i, item in enumerate(
            rows[:EXAMPLES_PER_PHRASE],
            start=1,
        ):
            print(
                f"{i}. {item['caption']}"
            )

        if not rows:
            print(
                "(no training examples)"
            )


if __name__ == "__main__":
    main()