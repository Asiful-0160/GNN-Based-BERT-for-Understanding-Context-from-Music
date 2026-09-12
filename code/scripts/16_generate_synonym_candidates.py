from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COUNTS_PATH = (
    ROOT
    / "data"
    / "audit"
    / "label_eda"
    / "train_all_aspect_counts.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "label_eda"
    / "synonym_candidates.csv"
)


# Only inspect reasonably supported phrases.
MIN_SUPPORT = 30

# Lexical similarity threshold.
MIN_SIMILARITY = 0.72


def token_set(text: str) -> set[str]:
    return set(
        text.lower().split()
    )


def jaccard(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    union = a_tokens | b_tokens

    if not union:
        return 0.0

    return (
        len(a_tokens & b_tokens)
        / len(union)
    )


def sequence_similarity(
    a: str,
    b: str,
) -> float:
    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def relation_hint(
    phrase_a: str,
    phrase_b: str,
) -> str:

    a_tokens = token_set(phrase_a)
    b_tokens = token_set(phrase_b)

    if a_tokens == b_tokens:
        return "same_tokens"

    if (
        a_tokens.issubset(b_tokens)
        or b_tokens.issubset(a_tokens)
    ):
        return "token_subset"

    if jaccard(
        phrase_a,
        phrase_b,
    ) >= 0.5:
        return "high_token_overlap"

    return "lexical_similarity"


def main() -> None:

    df = pd.read_csv(
        COUNTS_PATH
    )

    required = [
        "normalized_phrase",
        "track_support",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df = df.loc[
        df["track_support"]
        >= MIN_SUPPORT
    ].copy()

    df = df.sort_values(
        [
            "track_support",
            "normalized_phrase",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    rows = []

    for i in range(len(df)):

        phrase_a = str(
            df.at[
                i,
                "normalized_phrase",
            ]
        )

        support_a = int(
            df.at[
                i,
                "track_support",
            ]
        )

        for j in range(
            i + 1,
            len(df),
        ):

            phrase_b = str(
                df.at[
                    j,
                    "normalized_phrase",
                ]
            )

            support_b = int(
                df.at[
                    j,
                    "track_support",
                ]
            )

            seq = sequence_similarity(
                phrase_a,
                phrase_b,
            )

            jac = jaccard(
                phrase_a,
                phrase_b,
            )

            subset = (
                token_set(phrase_a)
                .issubset(
                    token_set(phrase_b)
                )
                or
                token_set(phrase_b)
                .issubset(
                    token_set(phrase_a)
                )
            )

            candidate = (
                seq >= MIN_SIMILARITY
                or jac >= 0.5
                or subset
            )

            if not candidate:
                continue

            rows.append({
                "phrase_a": phrase_a,
                "support_a": support_a,

                "phrase_b": phrase_b,
                "support_b": support_b,

                "sequence_similarity": (
                    round(seq, 4)
                ),

                "token_jaccard": (
                    round(jac, 4)
                ),

                "relation_hint": (
                    relation_hint(
                        phrase_a,
                        phrase_b,
                    )
                ),

                # IMPORTANT:
                # Nothing is merged automatically.
                "decision": "",
                "canonical_phrase": "",
                "review_note": "",
            })

    candidates = pd.DataFrame(
        rows
    )

    if len(candidates) > 0:

        candidates[
            "max_support"
        ] = candidates[
            [
                "support_a",
                "support_b",
            ]
        ].max(
            axis=1
        )

        candidates = (
            candidates
            .sort_values(
                [
                    "max_support",
                    "token_jaccard",
                    "sequence_similarity",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .drop(
                columns=[
                    "max_support",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Train-only synonym candidate generation"
    )
    print("=" * 52)

    print(
        f"Eligible phrases (support >= {MIN_SUPPORT}): "
        f"{len(df)}"
    )

    print(
        f"Candidate pairs generated               : "
        f"{len(candidates)}"
    )

    print()
    print("Top candidate pairs")
    print("-" * 52)

    if len(candidates):

        print(
            candidates[
                [
                    "phrase_a",
                    "support_a",
                    "phrase_b",
                    "support_b",
                    "sequence_similarity",
                    "token_jaccard",
                    "relation_hint",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    else:
        print(
            "No candidates generated."
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print(
        "IMPORTANT: No phrases were merged."
    )


if __name__ == "__main__":
    main()