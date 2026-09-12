from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter, defaultdict
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

OUTPUT_DIR = (
    ROOT
    / "data"
    / "audit"
    / "label_eda"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "train_label_eda_summary.json"
)

TOP100_PATH = (
    OUTPUT_DIR
    / "train_top100_aspects.csv"
)

ALL_COUNTS_PATH = (
    OUTPUT_DIR
    / "train_all_aspect_counts.csv"
)

RAW_MAPPING_PATH = (
    OUTPUT_DIR
    / "train_raw_to_normalized_aspects.csv"
)

VARIANTS_PATH = (
    OUTPUT_DIR
    / "train_normalized_phrase_variants.csv"
)

TRACK_STATS_PATH = (
    OUTPUT_DIR
    / "train_track_aspect_counts.csv"
)

SUPPORT_PATH = (
    OUTPUT_DIR
    / "train_support_thresholds.csv"
)


EXPECTED_TRAIN_COUNT = 4112


# ============================================================
# Aspect parsing
# ============================================================

def parse_aspects(value) -> list[str]:
    """
    Convert the MusicCaps aspects field into a list of
    COMPLETE aspect phrases.

    Important:
    We never split an individual aspect phrase into words.
    """

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
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return []

    # Handles CSV-style serialized Python lists.
    try:
        parsed = ast.literal_eval(text)

        if isinstance(
            parsed,
            (
                list,
                tuple,
            ),
        ):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    # Do NOT split a plain string on commas.
    # Doing that could destroy legitimate complete phrases.
    return [text]


# ============================================================
# Phrase normalization
# ============================================================

def normalize_phrase(
    phrase: str,
) -> str:
    """
    Basic normalization only.

    No synonym merging occurs here.

    Operations:
      1. Unicode normalization
      2. lowercase
      3. punctuation -> spaces
      4. collapse whitespace
      5. strip

    Multi-word phrases remain complete labels.
    """

    phrase = unicodedata.normalize(
        "NFKC",
        str(phrase),
    )

    phrase = phrase.lower().strip()

    # Replace Unicode punctuation characters with spaces.
    phrase = "".join(
        " "
        if unicodedata.category(char).startswith("P")
        else char
        for char in phrase
    )

    phrase = re.sub(
        r"\s+",
        " ",
        phrase,
    ).strip()

    return phrase


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset missing:\n"
            f"{CANONICAL_PATH}"
        )

    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Training split missing:\n"
            f"{TRAIN_SPLIT_PATH}"
        )

    # --------------------------------------------------------
    # Load frozen split
    # --------------------------------------------------------

    with TRAIN_SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_ids = [
        str(track_id)
        for track_id
        in split["track_ids"]
    ]

    if len(train_ids) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_COUNT} "
            f"training IDs, found {len(train_ids)}."
        )

    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError(
            "Duplicate IDs in train split."
        )

    # --------------------------------------------------------
    # Load canonical dataset
    # --------------------------------------------------------

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    if df["track_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical track IDs."
        )

    canonical_ids = set(
        df["track_id"].astype(str)
    )

    missing_ids = (
        set(train_ids)
        - canonical_ids
    )

    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} training IDs "
            "are absent from the canonical dataset."
        )

    # Select TRAIN ONLY.
    train = (
        df.loc[
            df["track_id"]
            .astype(str)
            .isin(train_ids)
        ]
        .copy()
    )

    if len(train) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_COUNT} "
            f"training rows, found {len(train)}."
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    raw_occurrence_count = Counter()
    raw_track_support = Counter()

    normalized_occurrence_count = Counter()
    normalized_track_support = Counter()

    normalized_variants = defaultdict(
        Counter
    )

    raw_to_normalized = {}

    track_statistics = []

    empty_normalized_count = 0

    # --------------------------------------------------------
    # Process each TRAIN track
    # --------------------------------------------------------

    for _, row in train.iterrows():

        track_id = str(
            row["track_id"]
        )

        raw_aspects = parse_aspects(
            row["aspects"]
        )

        normalized_aspects = []

        # Track-level sets are important:
        # support = number of tracks containing the phrase,
        # not number of duplicate mentions.
        raw_unique = set()

        normalized_unique = set()

        for raw_phrase in raw_aspects:

            raw_phrase = str(
                raw_phrase
            ).strip()

            if not raw_phrase:
                continue

            normalized = normalize_phrase(
                raw_phrase
            )

            if not normalized:
                empty_normalized_count += 1
                continue

            raw_occurrence_count[
                raw_phrase
            ] += 1

            normalized_occurrence_count[
                normalized
            ] += 1

            raw_unique.add(
                raw_phrase
            )

            normalized_unique.add(
                normalized
            )

            raw_to_normalized[
                raw_phrase
            ] = normalized

            normalized_variants[
                normalized
            ][raw_phrase] += 1

            normalized_aspects.append(
                normalized
            )

        for phrase in raw_unique:
            raw_track_support[
                phrase
            ] += 1

        for phrase in normalized_unique:
            normalized_track_support[
                phrase
            ] += 1

        track_statistics.append({
            "track_id": track_id,
            "raw_aspect_count": (
                len(raw_aspects)
            ),
            "unique_raw_aspects": (
                len(raw_unique)
            ),
            "unique_normalized_aspects": (
                len(normalized_unique)
            ),
        })

    # --------------------------------------------------------
    # Build normalized support table
    # --------------------------------------------------------

    rows = []

    all_phrases = sorted(
        normalized_track_support.keys()
    )

    for phrase in all_phrases:

        variants = (
            normalized_variants[
                phrase
            ]
        )

        sorted_variants = sorted(
            variants.items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        )

        rows.append({
            "normalized_phrase": phrase,

            # THIS is the important label support.
            "track_support": int(
                normalized_track_support[
                    phrase
                ]
            ),

            "occurrence_count": int(
                normalized_occurrence_count[
                    phrase
                ]
            ),

            "num_raw_variants": int(
                len(variants)
            ),

            "most_common_raw_variant": (
                sorted_variants[0][0]
            ),

            "raw_variants": " | ".join(
                variant
                for variant, _
                in sorted_variants
            ),
        })

    counts_df = pd.DataFrame(
        rows
    )

    counts_df = counts_df.sort_values(
        [
            "track_support",
            "normalized_phrase",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    counts_df.insert(
        0,
        "rank",
        range(
            1,
            len(counts_df) + 1,
        ),
    )

    # --------------------------------------------------------
    # Raw -> normalized mapping
    # --------------------------------------------------------

    mapping_rows = []

    for raw_phrase, normalized in sorted(
        raw_to_normalized.items()
    ):

        mapping_rows.append({
            "raw_phrase": raw_phrase,
            "normalized_phrase": (
                normalized
            ),
            "raw_track_support": int(
                raw_track_support[
                    raw_phrase
                ]
            ),
            "raw_occurrence_count": int(
                raw_occurrence_count[
                    raw_phrase
                ]
            ),
        })

    mapping_df = pd.DataFrame(
        mapping_rows
    )

    # --------------------------------------------------------
    # Variant table
    # --------------------------------------------------------

    variant_rows = []

    for normalized, variants in (
        normalized_variants.items()
    ):

        for raw_phrase, count in (
            variants.items()
        ):
            variant_rows.append({
                "normalized_phrase": (
                    normalized
                ),
                "raw_phrase": (
                    raw_phrase
                ),
                "occurrence_count": int(
                    count
                ),
                "normalized_track_support": int(
                    normalized_track_support[
                        normalized
                    ]
                ),
            })

    variants_df = pd.DataFrame(
        variant_rows
    )

    variants_df = variants_df.sort_values(
        [
            "normalized_track_support",
            "normalized_phrase",
            "occurrence_count",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    )

    # --------------------------------------------------------
    # Track-level aspect statistics
    # --------------------------------------------------------

    track_stats_df = pd.DataFrame(
        track_statistics
    )

    # --------------------------------------------------------
    # Support-threshold analysis
    # --------------------------------------------------------

    thresholds = [
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        75,
        100,
        150,
        200,
    ]

    support_rows = []

    total_positive_assignments = int(
        counts_df[
            "track_support"
        ].sum()
    )

    for threshold in thresholds:

        eligible = counts_df.loc[
            counts_df[
                "track_support"
            ] >= threshold
        ]

        covered_assignments = int(
            eligible[
                "track_support"
            ].sum()
        )

        support_rows.append({
            "minimum_track_support": (
                threshold
            ),

            "eligible_phrase_count": int(
                len(eligible)
            ),

            "covered_train_assignments": (
                covered_assignments
            ),

            "assignment_coverage_percent": (
                covered_assignments
                / total_positive_assignments
                * 100.0
                if total_positive_assignments
                else 0.0
            ),
        })

    support_df = pd.DataFrame(
        support_rows
    )

    # --------------------------------------------------------
    # Top-20 provisional statistics
    # --------------------------------------------------------

    provisional_top20 = (
        counts_df.head(20)
    )

    provisional_top20_min_support = (
        int(
            provisional_top20[
                "track_support"
            ].min()
        )
        if len(provisional_top20)
        else 0
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    aspect_counts_per_track = (
        track_stats_df[
            "unique_normalized_aspects"
        ]
    )

    summary = {
        "split_used": "train_only",

        "training_tracks": int(
            len(train)
        ),

        "unique_raw_phrases": int(
            len(raw_track_support)
        ),

        "unique_normalized_phrases": int(
            len(normalized_track_support)
        ),

        "total_normalized_track_assignments": (
            total_positive_assignments
        ),

        "empty_after_normalization": int(
            empty_normalized_count
        ),

        "aspects_per_track": {
            "min": int(
                aspect_counts_per_track.min()
            ),

            "max": int(
                aspect_counts_per_track.max()
            ),

            "mean": float(
                aspect_counts_per_track.mean()
            ),

            "median": float(
                aspect_counts_per_track.median()
            ),

            "p95": float(
                np.percentile(
                    aspect_counts_per_track,
                    95,
                )
            ),
        },

        "phrases_with_support_ge_30": int(
            (
                counts_df[
                    "track_support"
                ] >= 30
            ).sum()
        ),

        "provisional_top20_min_support": (
            provisional_top20_min_support
        ),

        "important": (
            "No synonym merging and no label "
            "vocabulary freeze were performed "
            "by this script."
        ),
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts_df.to_csv(
        ALL_COUNTS_PATH,
        index=False,
    )

    counts_df.head(
        100
    ).to_csv(
        TOP100_PATH,
        index=False,
    )

    mapping_df.to_csv(
        RAW_MAPPING_PATH,
        index=False,
    )

    variants_df.to_csv(
        VARIANTS_PATH,
        index=False,
    )

    track_stats_df.to_csv(
        TRACK_STATS_PATH,
        index=False,
    )

    support_df.to_csv(
        SUPPORT_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("MusicCaps TRAIN-ONLY label EDA")
    print("=" * 52)

    print(
        f"Training tracks             : "
        f"{summary['training_tracks']}"
    )

    print(
        f"Unique raw phrases          : "
        f"{summary['unique_raw_phrases']}"
    )

    print(
        f"Unique normalized phrases   : "
        f"{summary['unique_normalized_phrases']}"
    )

    print(
        f"Phrases with support >= 30  : "
        f"{summary['phrases_with_support_ge_30']}"
    )

    print(
        f"Top-20 minimum support      : "
        f"{summary['provisional_top20_min_support']}"
    )

    print()
    print("Top 30 normalized phrases")
    print("-" * 52)

    print(
        counts_df[
            [
                "rank",
                "normalized_phrase",
                "track_support",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print("Support thresholds")
    print("-" * 52)

    print(
        support_df.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(SUMMARY_PATH)
    print(TOP100_PATH)
    print(ALL_COUNTS_PATH)
    print(RAW_MAPPING_PATH)
    print(VARIANTS_PATH)
    print(TRACK_STATS_PATH)
    print(SUPPORT_PATH)

    print()
    print(
        "NO synonym mapping or label "
        "vocabulary has been frozen yet."
    )


if __name__ == "__main__":
    main()