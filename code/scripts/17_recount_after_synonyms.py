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

SYNONYM_PATH = (
    ROOT
    / "data"
    / "splits"
    / "aspect_synonyms.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "audit"
    / "label_eda"
)

MERGED_COUNTS_PATH = (
    OUTPUT_DIR
    / "train_merged_aspect_counts.csv"
)

TOP100_PATH = (
    OUTPUT_DIR
    / "train_merged_top100.csv"
)

MERGE_IMPACT_PATH = (
    OUTPUT_DIR
    / "synonym_merge_impact.csv"
)

MERGE_GROUP_IMPACT_PATH = (
    OUTPUT_DIR
    / "synonym_merge_group_impact.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "train_merged_label_summary.json"
)


EXPECTED_TRAIN_COUNT = 4112
PROVISIONAL_MIN_SUPPORT = 30
TOP_K = 20


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
                str(item).strip()
                for item in parsed
                if str(item).strip()
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
            character
        ).startswith("P")
        else character

        for character in phrase
    )

    phrase = re.sub(
        r"\s+",
        " ",
        phrase,
    ).strip()

    return phrase


def load_synonyms() -> dict[str, str]:

    with SYNONYM_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    mapping = data.get(
        "mapping",
        {}
    )

    if not isinstance(
        mapping,
        dict,
    ):
        raise RuntimeError(
            "Synonym mapping must be "
            "a JSON object."
        )

    clean_mapping = {}

    for source, target in (
        mapping.items()
    ):

        source_norm = normalize_phrase(
            source
        )

        target_norm = normalize_phrase(
            target
        )

        if not source_norm:
            raise RuntimeError(
                "Empty synonym source."
            )

        if not target_norm:
            raise RuntimeError(
                "Empty synonym target."
            )

        if source_norm == target_norm:
            raise RuntimeError(
                f"Redundant synonym mapping: "
                f"{source_norm}"
            )

        clean_mapping[
            source_norm
        ] = target_norm

    # Prevent chains/cycles.
    for source, target in (
        clean_mapping.items()
    ):

        if target in clean_mapping:
            raise RuntimeError(
                f"Synonym chain detected: "
                f"{source} -> {target} -> "
                f"{clean_mapping[target]}. "
                "Mappings must point directly "
                "to final canonical phrases."
            )

    return clean_mapping


def main() -> None:

    # --------------------------------------------------
    # Load TRAIN split only
    # --------------------------------------------------

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
            f"train IDs, found "
            f"{len(train_ids)}."
        )

    # --------------------------------------------------
    # Canonical dataset
    # --------------------------------------------------

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    train = df.loc[
        df["track_id"]
        .astype(str)
        .isin(
            set(train_ids)
        )
    ].copy()

    if len(train) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(
            "Training-set size mismatch."
        )

    synonyms = load_synonyms()

    # --------------------------------------------------
    # Pre-merge and post-merge support
    # --------------------------------------------------

    before_support = Counter()
    after_support = Counter()

    # Track IDs containing each normalized phrase.
    # This lets us calculate actual set overlap rather
    # than infer overlap from aggregate support counts.
    phrase_track_ids = defaultdict(set)

    # alias_tracks = Counter()

    for _, row in train.iterrows():

        track_id = str(row["track_id"])

        raw_aspects = parse_aspects(
            row["aspects"]
        )

        before_track = set()
        after_track = set()

        aliases_seen = set()

        for raw_phrase in raw_aspects:

            phrase = normalize_phrase(
                raw_phrase
            )

            if not phrase:
                continue

            before_track.add(
                phrase
            )

            canonical = synonyms.get(
                phrase,
                phrase,
            )

            after_track.add(
                canonical
            )

            if phrase in synonyms:
                aliases_seen.add(
                    phrase
                )

        for phrase in before_track:
            before_support[
                phrase
            ] += 1

            phrase_track_ids[
                phrase
            ].add(
                track_id
            )

        for phrase in after_track:
            after_support[
                phrase
            ] += 1

        # for alias in aliases_seen:
        #     alias_tracks[
        #         alias
        #     ] += 1

    # --------------------------------------------------
    # Final merged count table
    # --------------------------------------------------

    rows = []

    for phrase, support in (
        after_support.items()
    ):

        rows.append({
            "merged_phrase": phrase,
            "track_support": int(
                support
            ),
            "support_percent": (
                support
                / EXPECTED_TRAIN_COUNT
                * 100.0
            ),
        })

    counts = pd.DataFrame(
        rows
    )

    counts = counts.sort_values(
        [
            "track_support",
            "merged_phrase",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    counts.insert(
        0,
        "rank",
        range(
            1,
            len(counts) + 1,
        ),
    )

    # --------------------------------------------------
    # Merge impact — alias level
    # --------------------------------------------------

    impact_rows = []

    canonical_to_aliases = defaultdict(list)

    for alias, canonical in synonyms.items():
        canonical_to_aliases[
            canonical
        ].append(alias)

        alias_ids = phrase_track_ids.get(
            alias,
            set(),
        )

        canonical_ids = phrase_track_ids.get(
            canonical,
            set(),
        )

        overlap = (
            alias_ids
            & canonical_ids
        )

        newly_contributed = (
            alias_ids
            - canonical_ids
        )

        impact_rows.append({
            "alias": alias,

            "canonical_phrase": canonical,

            "alias_support_before": (
                len(alias_ids)
            ),

            "canonical_support_before": (
                len(canonical_ids)
            ),

            "alias_canonical_overlap": (
                len(overlap)
            ),

            "new_tracks_vs_original_canonical": (
                len(newly_contributed)
            ),

            "canonical_support_after_all_merges": (
                int(
                    after_support.get(
                        canonical,
                        0,
                    )
                )
            ),
        })

    impact = pd.DataFrame(
        impact_rows
    )

    # --------------------------------------------------
    # Merge impact — canonical group level
    # --------------------------------------------------

    group_rows = []

    for canonical, aliases in sorted(
        canonical_to_aliases.items()
    ):
        canonical_ids = set(
            phrase_track_ids.get(
                canonical,
                set(),
            )
        )

        merged_ids = set(
            canonical_ids
        )

        total_alias_support = 0

        for alias in aliases:
            alias_ids = set(
                phrase_track_ids.get(
                    alias,
                    set(),
                )
            )

            total_alias_support += len(
                alias_ids
            )

            merged_ids.update(
                alias_ids
            )

        expected_after = len(
            merged_ids
        )

        actual_after = int(
            after_support.get(
                canonical,
                0,
            )
        )

        if expected_after != actual_after:
            raise RuntimeError(
                f"Merge support mismatch for "
                f"{canonical}: expected "
                f"{expected_after}, got "
                f"{actual_after}."
            )

        group_rows.append({
            "canonical_phrase": canonical,

            "aliases": " | ".join(
                sorted(aliases)
            ),

            "canonical_support_before": (
                len(canonical_ids)
            ),

            "total_alias_support_before": (
                total_alias_support
            ),

            "canonical_support_after": (
                actual_after
            ),

            "net_support_gain": (
                actual_after
                - len(canonical_ids)
            ),
        })

    group_impact = pd.DataFrame(
        group_rows
    )

    # --------------------------------------------------
    # Support cutoff check
    # --------------------------------------------------

    eligible = counts.loc[
        counts["track_support"]
        >= PROVISIONAL_MIN_SUPPORT
    ]

    if len(eligible) < TOP_K:
        raise RuntimeError(
            f"Only {len(eligible)} labels have "
            f"support >= "
            f"{PROVISIONAL_MIN_SUPPORT}. "
            f"Cannot construct Top-{TOP_K}."
        )

    top20 = counts.head(
        TOP_K
    )

    top20_min_support = int(
        top20[
            "track_support"
        ].min()
    )

    rank20 = top20.iloc[
        -1
    ]

    rank21 = (
        counts.iloc[TOP_K]
        if len(counts) > TOP_K
        else None
    )

    summary = {
        "training_tracks": (
            EXPECTED_TRAIN_COUNT
        ),

        "synonym_mappings": int(
            len(synonyms)
        ),

        "unique_phrases_before_merge": int(
            len(before_support)
        ),

        "unique_phrases_after_merge": int(
            len(after_support)
        ),

        "provisional_min_support": (
            PROVISIONAL_MIN_SUPPORT
        ),

        "eligible_phrases_support_ge_30": int(
            len(eligible)
        ),

        "top_k": TOP_K,

        "top20_min_support": (
            top20_min_support
        ),

        "rank20": {
            "phrase": str(
                rank20[
                    "merged_phrase"
                ]
            ),

            "support": int(
                rank20[
                    "track_support"
                ]
            ),
        },

        "rank21": (
            {
                "phrase": str(
                    rank21[
                        "merged_phrase"
                    ]
                ),

                "support": int(
                    rank21[
                        "track_support"
                    ]
                ),
            }

            if rank21 is not None

            else None
        ),

        "label_vocab_frozen": False,
    }

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts.to_csv(
        MERGED_COUNTS_PATH,
        index=False,
    )

    counts.head(
        100
    ).to_csv(
        TOP100_PATH,
        index=False,
    )

    impact.to_csv(
        MERGE_IMPACT_PATH,
        index=False,
    )

    group_impact.to_csv(
        MERGE_GROUP_IMPACT_PATH,
        index=False,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------
    # Console
    # --------------------------------------------------

    print()
    print(
        "MusicCaps synonym-merged TRAIN recount"
    )
    print("=" * 58)

    print(
        f"Train tracks                  : "
        f"{EXPECTED_TRAIN_COUNT}"
    )

    print(
        f"Reviewed synonym mappings     : "
        f"{len(synonyms)}"
    )

    print(
        f"Unique phrases before merge   : "
        f"{len(before_support)}"
    )

    print(
        f"Unique phrases after merge    : "
        f"{len(after_support)}"
    )

    print(
        f"Support >= 30 after merge     : "
        f"{len(eligible)}"
    )

    print(
        f"Top-20 minimum support        : "
        f"{top20_min_support}"
    )

    print()
    print("Merge impact")
    print("-" * 58)

    print(
        impact.to_string(
            index=False
        )
    )

    print()
    print("Canonical-group merge impact")
    print("-" * 58)

    print(
        group_impact.to_string(
            index=False
        )
    )

    print()
    print("Top 30 after synonym merging")
    print("-" * 58)

    print(
        counts[
            [
                "rank",
                "merged_phrase",
                "track_support",
                "support_percent",
            ]
        ]
        .head(30)
        .to_string(
            index=False,
            formatters={
                "support_percent":
                    lambda x:
                    f"{x:.2f}%"
            },
        )
    )

    print()
    print(
        f"Rank 20: "
        f"{summary['rank20']['phrase']} "
        f"({summary['rank20']['support']})"
    )

    if summary["rank21"]:
        print(
            f"Rank 21: "
            f"{summary['rank21']['phrase']} "
            f"({summary['rank21']['support']})"
        )

    print()
    print("Saved:")
    print(MERGED_COUNTS_PATH)
    print(TOP100_PATH)
    print(MERGE_IMPACT_PATH)
    print(MERGE_GROUP_IMPACT_PATH)
    print(SUMMARY_PATH)

    print()
    print(
        "NO label vocabulary has been "
        "frozen yet."
    )


if __name__ == "__main__":
    main()