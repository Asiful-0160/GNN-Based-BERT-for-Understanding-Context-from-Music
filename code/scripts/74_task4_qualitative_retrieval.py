#!/usr/bin/env python3
"""
SCRIPT 74 — FROZEN TASK-4 QUALITATIVE CAPTION -> AUDIO RETRIEVAL
================================================================

Purpose
-------
Produce the required 10 qualitative Task-4 caption -> Top-3 audio
retrieval examples.

NO retrieval is rerun.

The script reuses:
    - 10 TEST query row indices frozen by Script 71
    - fixed qualitative model seed = 42
    - frozen Script-72 TEST manifest
    - frozen Script-72 seed-42 similarity matrix
    - frozen Script-72 seed-42 rank cache

For each frozen query:
    - show query caption
    - show ground-truth track_id
    - show ground-truth retrieval rank
    - show Top-3 retrieved audio track_ids
    - show cosine similarities
    - mark whether each retrieved item is the true paired audio

No cherry-picking.
No seed selection.
No model inference.
No reranking.
No TEST labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRETEST_PROTOCOL = (
    ROOT
    / "results/runs/task4/"
      "task4_pretest_protocol.json"
)

PRETEST_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_pretest_protocol_lock.json"
)

TEST_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_test_retrieval_summary.json"
)

TEST_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_T4-B_test_retrieval_lock.json"
)

TEST_MANIFEST = (
    ROOT
    / "results/runs/task4/"
      "task4_test_retrieval_manifest.csv"
)

ZERO_SHOT_SUMMARY = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_test_summary.json"
)

ZERO_SHOT_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_zero_shot_test_lock.json"
)

CANONICAL_DATASET = (
    ROOT
    / "data/interim/"
      "musiccaps_canonical.parquet"
)

SEED42_SIMILARITY = (
    ROOT
    / "data/processed/task4_test_retrieval/"
      "seed42_audio_by_caption_similarity.npy"
)

SEED42_RANKS = (
    ROOT
    / "data/processed/task4_test_retrieval/"
      "seed42_retrieval_ranks.npz"
)

OUTPUT_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3.csv"
)

OUTPUT_JSON = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3.json"
)

LOCK_PATH = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3_lock.json"
)


# =============================================================================
# 1. ACCEPTED HASHES
# =============================================================================

EXPECTED_PRETEST_PROTOCOL_SHA256 = (
    "f1f4651c9bffab2295ec8ef44d17f208"
    "4430386585fb2e9fd489d0bc2af93f7c"
)

EXPECTED_PRETEST_LOCK_SHA256 = (
    "dddbd32013dd7e9912e0df0fa72d0352"
    "49abae8f4c83f6313b57a99931f00836"
)

EXPECTED_TEST_SUMMARY_SHA256 = (
    "77e08ce11af9547319515137197ae122c"
    "83224d1a445619c6daef799a722af90"
)

EXPECTED_TEST_LOCK_SHA256 = (
    "45a3adc75581b5ddf0f28fe08ecb70c0"
    "0ddb3bd6d3c665f82fe4e4b556d40504"
)

EXPECTED_TEST_MANIFEST_SHA256 = (
    "e3e726814fdd890af0951fd1c4726a4d"
    "e5f506180f4ae6982db51e89e1d39431"
)

EXPECTED_ZERO_SHOT_SUMMARY_SHA256 = (
    "2a288a2921267f617466a8bc4507300c"
    "eb6461a9c1e96bcf41dbfcb68a2319c8"
)

EXPECTED_ZERO_SHOT_LOCK_SHA256 = (
    "2552be17a05f7349449b9dc6a7f0d1f"
    "8f4382658ab6ebf2f9fcc36e8990d8611"
)

EXPECTED_CANONICAL_SHA256 = (
    "704e057ba3807886b35000ff23d533d4"
    "759c6a65fdf5d7a8912e618ba523c4d3"
)


# =============================================================================
# 2. FROZEN CONTRACT
# =============================================================================

N_TEST = 514

QUERY_COUNT = 10

QUALITATIVE_SEED = 42

TOP_K = 3

DIRECTION = "caption_to_audio"


# =============================================================================
# 3. REPORTING
# =============================================================================

FAILURES: list[str] = []
WARNINGS: list[str] = []


def banner(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(
    message: str,
) -> None:

    print(
        f"PASS  {message}"
    )


def fail(
    message: str,
) -> None:

    print(
        f"FAIL  {message}"
    )

    FAILURES.append(
        message
    )


def check(
    condition: bool,
    message: str,
) -> bool:

    if condition:

        passed(
            message
        )

        return True

    fail(
        message
    )

    return False


# =============================================================================
# 4. HELPERS
# =============================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        value = json.load(
            f
        )

    if not isinstance(
        value,
        dict,
    ):

        raise RuntimeError(
            f"{path.name} is not "
            "a JSON dictionary"
        )

    return value


def relative(
    path: Path,
) -> str:

    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    )


def verify_sha(
    path: Path,
    expected: str,
    label: str,
) -> None:

    check(
        path.is_file(),
        f"{label} exists",
    )

    if not path.is_file():
        return

    observed = sha256_file(
        path
    )

    print(
        f"{label}:\n"
        f"  observed = {observed}\n"
        f"  expected = {expected}"
    )

    check(
        observed
        == expected,
        (
            f"{label} SHA256 matches "
            "frozen identity"
        ),
    )


def write_json_atomic(
    path: Path,
    value: Any,
) -> None:

    temporary = Path(
        str(
            path
        )
        + ".__tmp__"
    )

    if temporary.exists():

        raise RuntimeError(
            f"Temporary file exists: "
            f"{temporary}"
        )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            value,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

        f.write(
            "\n"
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:

        raise RuntimeError(
            "Cannot write empty CSV"
        )

    temporary = Path(
        str(
            path
        )
        + ".__tmp__"
    )

    if temporary.exists():

        raise RuntimeError(
            f"Temporary file exists: "
            f"{temporary}"
        )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# 5. OUTPUT PROTECTION
# =============================================================================

def verify_output_protection() -> None:

    banner(
        "1. QUALITATIVE OUTPUT PROTECTION"
    )

    for path in (
        OUTPUT_CSV,
        OUTPUT_JSON,
        LOCK_PATH,
    ):

        check(
            not path.exists(),
            (
                f"output unused: "
                f"{path.name}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite existing "
            "qualitative Task-4 output"
        )


# =============================================================================
# 6. VERIFY LINEAGE
# =============================================================================

def verify_lineage() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    banner(
        "2. VERIFY FROZEN QUALITATIVE LINEAGE"
    )

    artifacts = (
        (
            PRETEST_PROTOCOL,
            EXPECTED_PRETEST_PROTOCOL_SHA256,
            "pre-TEST protocol",
        ),
        (
            PRETEST_LOCK,
            EXPECTED_PRETEST_LOCK_SHA256,
            "pre-TEST lock",
        ),
        (
            TEST_SUMMARY,
            EXPECTED_TEST_SUMMARY_SHA256,
            "final TEST retrieval summary",
        ),
        (
            TEST_LOCK,
            EXPECTED_TEST_LOCK_SHA256,
            "final TEST retrieval lock",
        ),
        (
            TEST_MANIFEST,
            EXPECTED_TEST_MANIFEST_SHA256,
            "final TEST retrieval manifest",
        ),
        (
            ZERO_SHOT_SUMMARY,
            EXPECTED_ZERO_SHOT_SUMMARY_SHA256,
            "zero-shot summary",
        ),
        (
            ZERO_SHOT_LOCK,
            EXPECTED_ZERO_SHOT_LOCK_SHA256,
            "zero-shot lock",
        ),
        (
            CANONICAL_DATASET,
            EXPECTED_CANONICAL_SHA256,
            "canonical dataset",
        ),
    )

    for path, digest, label in artifacts:

        verify_sha(
            path,
            digest,
            label,
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen qualitative lineage failed"
        )

    protocol = load_json(
        PRETEST_PROTOCOL
    )

    test_lock = load_json(
        TEST_LOCK
    )

    qualitative = (
        protocol[
            "qualitative_retrieval"
        ]
    )

    check(
        qualitative[
            "query_count"
        ]
        == QUERY_COUNT,
        (
            "qualitative query count "
            "remains 10"
        ),
    )

    check(
        qualitative[
            "model_seed"
        ]
        == QUALITATIVE_SEED,
        (
            "qualitative model seed "
            "remains 42"
        ),
    )

    check(
        qualitative[
            "direction"
        ]
        == DIRECTION,
        (
            "qualitative direction remains "
            "caption_to_audio"
        ),
    )

    check(
        qualitative[
            "top_k"
        ]
        == TOP_K,
        (
            "qualitative Top-K remains 3"
        ),
    )

    check(
        qualitative[
            "hand_picking"
        ]
        is False,
        (
            "qualitative hand-picking "
            "remains forbidden"
        ),
    )

    check(
        qualitative[
            "reuse_script72_similarity"
        ]
        is True,
        (
            "qualitative analysis must "
            "reuse Script-72 similarity"
        ),
    )

    query_indices = (
        qualitative[
            "frozen_test_row_indices"
        ]
    )

    check(
        len(
            query_indices
        )
        == QUERY_COUNT,
        (
            "exactly 10 frozen query "
            "indices present"
        ),
    )

    check(
        len(
            set(
                query_indices
            )
        )
        == QUERY_COUNT,
        (
            "frozen query indices unique"
        ),
    )

    check(
        all(
            isinstance(
                index,
                int,
            )
            and
            0
            <= index
            < N_TEST

            for index
            in query_indices
        ),
        (
            "all query indices in 0..513"
        ),
    )

    seed_hashes = (
        test_lock[
            "seed_artifact_hashes"
        ][
            str(
                QUALITATIVE_SEED
            )
        ]
    )

    check(
        "similarity"
        in seed_hashes,
        (
            "Script-72 lock contains "
            "seed-42 similarity hash"
        ),
    )

    check(
        "ranks"
        in seed_hashes,
        (
            "Script-72 lock contains "
            "seed-42 rank hash"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Qualitative protocol validation failed"
        )

    return (
        protocol,
        test_lock,
    )


# =============================================================================
# 7. VERIFY SCRIPT-72 SEED-42 RETRIEVAL CACHE
# =============================================================================

def load_retrieval_cache(
    test_lock: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    banner(
        "3. LOAD FROZEN SCRIPT-72 "
        "SEED-42 RETRIEVAL CACHE"
    )

    seed_hashes = (
        test_lock[
            "seed_artifact_hashes"
        ][
            "42"
        ]
    )

    verify_sha(
        SEED42_SIMILARITY,
        seed_hashes[
            "similarity"
        ],
        "seed-42 frozen similarity matrix",
    )

    verify_sha(
        SEED42_RANKS,
        seed_hashes[
            "ranks"
        ],
        "seed-42 frozen retrieval ranks",
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen seed-42 retrieval "
            "cache verification failed"
        )

    similarity = np.load(
        SEED42_SIMILARITY,
        allow_pickle=False,
    )

    check(
        similarity.shape
        ==
        (
            N_TEST,
            N_TEST,
        ),
        (
            "seed-42 similarity "
            "= (514,514)"
        ),
    )

    check(
        similarity.dtype
        ==
        np.float32,
        (
            "seed-42 similarity "
            "dtype = float32"
        ),
    )

    check(
        np.isfinite(
            similarity
        ).all(),
        (
            "seed-42 similarity finite"
        ),
    )

    with np.load(
        SEED42_RANKS,
        allow_pickle=False,
    ) as ranks:

        check(
            set(
                ranks.files
            )
            ==
            {
                "audio_to_caption_rank",
                "caption_to_audio_rank",
            },
            (
                "seed-42 rank cache "
                "schema exact"
            ),
        )

        caption_to_audio_rank = (
            np.asarray(
                ranks[
                    "caption_to_audio_rank"
                ],
                dtype=np.int64,
            )
        )

    check(
        caption_to_audio_rank.shape
        ==
        (
            N_TEST,
        ),
        (
            "caption→audio ranks = (514,)"
        ),
    )

    check(
        np.all(
            (
                caption_to_audio_rank
                >= 1
            )
            &
            (
                caption_to_audio_rank
                <= N_TEST
            )
        ),
        (
            "all caption→audio ranks "
            "within 1..514"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen retrieval cache invalid"
        )

    return (
        similarity,
        caption_to_audio_rank,
    )


# =============================================================================
# 8. LOAD FROZEN TEST MANIFEST
# =============================================================================

def load_test_manifest() -> pd.DataFrame:

    banner(
        "4. LOAD SCRIPT-72 TEST MANIFEST"
    )

    frame = pd.read_csv(
        TEST_MANIFEST
    )

    check(
        list(
            frame.columns
        )
        ==
        [
            "row_index",
            "track_id",
        ],
        (
            "TEST manifest schema exact"
        ),
    )

    check(
        len(
            frame
        )
        ==
        N_TEST,
        (
            "TEST manifest = 514 rows"
        ),
    )

    check(
        frame[
            "row_index"
        ].tolist()
        ==
        list(
            range(
                N_TEST
            )
        ),
        (
            "TEST manifest row order "
            "= 0..513"
        ),
    )

    check(
        frame[
            "track_id"
        ].astype(str).nunique()
        ==
        N_TEST,
        (
            "TEST manifest track IDs unique"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "TEST manifest validation failed"
        )

    frame[
        "track_id"
    ] = (
        frame[
            "track_id"
        ].astype(str)
    )

    return frame


# =============================================================================
# 9. LOAD CANONICAL CAPTIONS
# =============================================================================

def load_captions(
    test_manifest: pd.DataFrame,
) -> dict[str, str]:

    banner(
        "5. LOAD CANONICAL MUSICCAPS CAPTIONS"
    )

    schema = (
        pq.ParquetFile(
            CANONICAL_DATASET
        )
        .schema
        .names
    )

    print(
        "Canonical columns:"
    )

    print(
        "  "
        + ", ".join(
            schema
        )
    )

    check(
        "track_id"
        in schema,
        (
            "canonical dataset contains "
            "track_id"
        ),
    )

    check(
        "caption"
        in schema,
        (
            "canonical dataset contains "
            "caption"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Canonical caption schema "
            "does not match expected MusicCaps schema"
        )

    columns = [
        "track_id",
        "caption",
    ]

    if (
        "split"
        in schema
    ):

        columns.append(
            "split"
        )

    frame = (
        pd.read_parquet(
            CANONICAL_DATASET,
            columns=columns,
        )
        .copy()
    )

    frame[
        "track_id"
    ] = (
        frame[
            "track_id"
        ].astype(str)
    )

    check(
        frame[
            "track_id"
        ].nunique()
        ==
        len(
            frame
        ),
        (
            "canonical track IDs unique"
        ),
    )

    if (
        "split"
        in frame.columns
    ):

        frame[
            "split"
        ] = (
            frame[
                "split"
            ]
            .astype(str)
            .str.lower()
        )

        frame = (
            frame.loc[
                frame[
                    "split"
                ]
                ==
                "test"
            ]
            .copy()
        )

        check(
            len(
                frame
            )
            ==
            N_TEST,
            (
                "canonical TEST rows = 514"
            ),
        )

    test_ids = set(
        test_manifest[
            "track_id"
        ]
    )

    caption_ids = set(
        frame[
            "track_id"
        ]
    )

    check(
        test_ids.issubset(
            caption_ids
        ),
        (
            "every Script-72 TEST track "
            "has a canonical caption"
        ),
    )

    check(
        frame[
            "caption"
        ].notna().all(),
        (
            "canonical captions non-null"
        ),
    )

    lookup = {
        str(
            row.track_id
        ):
            str(
                row.caption
            )

        for row
        in frame.itertuples(
            index=False
        )
    }

    if FAILURES:

        raise RuntimeError(
            "Canonical caption alignment failed"
        )

    return lookup


# =============================================================================
# 10. BUILD 10 FROZEN QUALITATIVE CASES
# =============================================================================

def build_cases(
    protocol: dict[str, Any],
    test_manifest: pd.DataFrame,
    caption_lookup: dict[str, str],
    similarity: np.ndarray,
    cached_c2a_rank: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    banner(
        "6. BUILD 10 FROZEN CAPTION→TOP-3 CASES"
    )

    query_indices = (
        protocol[
            "qualitative_retrieval"
        ][
            "frozen_test_row_indices"
        ]
    )

    track_ids = (
        test_manifest[
            "track_id"
        ].tolist()
    )

    long_rows = []
    cases = []

    all_audio_rows = np.arange(
        N_TEST,
        dtype=np.int64,
    )

    for query_number, query_index in enumerate(
        query_indices,
        start=1,
    ):

        query_index = int(
            query_index
        )

        query_track_id = (
            track_ids[
                query_index
            ]
        )

        caption = (
            caption_lookup[
                query_track_id
            ]
        )

        # ---------------------------------------------------------------------
        # Similarity orientation frozen by Script 71/72:
        #
        # rows    = audio
        # columns = caption
        #
        # Therefore caption query j uses similarity[:, j].
        # ---------------------------------------------------------------------

        scores = (
            similarity[
                :,
                query_index
            ]
        )

        positive_score = float(
            scores[
                query_index
            ]
        )

        recomputed_gt_rank = int(
            1
            +
            np.sum(
                scores
                >
                positive_score
            )
        )

        cached_gt_rank = int(
            cached_c2a_rank[
                query_index
            ]
        )

        check(
            recomputed_gt_rank
            ==
            cached_gt_rank,
            (
                f"query {query_number} "
                "ground-truth rank exactly "
                "matches Script-72 cache"
            ),
        )

        # ---------------------------------------------------------------------
        # Top-3 deterministic ordering:
        #
        # primary:
        #     similarity descending
        #
        # exact similarity tie:
        #     audio row_index ascending
        #
        # np.lexsort uses its LAST key as the primary key.
        # ---------------------------------------------------------------------

        order = np.lexsort(
            (
                all_audio_rows,
                -scores.astype(
                    np.float64
                ),
            )
        )

        top_rows = (
            order[
                :TOP_K
            ]
        )

        retrieved = []

        for retrieval_rank, audio_row in enumerate(
            top_rows,
            start=1,
        ):

            audio_row = int(
                audio_row
            )

            retrieved_track_id = (
                track_ids[
                    audio_row
                ]
            )

            is_ground_truth = (
                audio_row
                ==
                query_index
            )

            item = {
                "rank":
                    retrieval_rank,

                "audio_row_index":
                    audio_row,

                "track_id":
                    retrieved_track_id,

                "similarity":
                    float(
                        scores[
                            audio_row
                        ]
                    ),

                "is_ground_truth":
                    bool(
                        is_ground_truth
                    ),
            }

            retrieved.append(
                item
            )

            long_rows.append(
                {
                    "query_number":
                        query_number,

                    "query_row_index":
                        query_index,

                    "query_track_id":
                        query_track_id,

                    "query_caption":
                        caption,

                    "ground_truth_rank":
                        cached_gt_rank,

                    "retrieval_rank":
                        retrieval_rank,

                    "retrieved_audio_row_index":
                        audio_row,

                    "retrieved_audio_track_id":
                        retrieved_track_id,

                    "cosine_similarity":
                        float(
                            scores[
                                audio_row
                            ]
                        ),

                    "is_ground_truth":
                        bool(
                            is_ground_truth
                        ),
                }
            )

        check(
            (
                any(
                    item[
                        "is_ground_truth"
                    ]

                    for item
                    in retrieved
                )
            )
            ==
            (
                cached_gt_rank
                <= TOP_K
            ),
            (
                f"query {query_number} "
                "Top-3 GT membership agrees "
                "with frozen GT rank"
            ),
        )

        cases.append(
            {
                "query_number":
                    query_number,

                "query_row_index":
                    query_index,

                "query_track_id":
                    query_track_id,

                "query_caption":
                    caption,

                "ground_truth_audio_row_index":
                    query_index,

                "ground_truth_audio_track_id":
                    query_track_id,

                "ground_truth_similarity":
                    positive_score,

                "ground_truth_rank":
                    cached_gt_rank,

                "top3":
                    retrieved,
            }
        )

        print()
        print(
            f"Query {query_number:02d} "
            f"| row={query_index} "
            f"| GT rank={cached_gt_rank}"
        )

        for item in retrieved:

            marker = (
                "  [GT]"
                if item[
                    "is_ground_truth"
                ]
                else
                ""
            )

            print(
                f"  #{item['rank']} "
                f"{item['track_id']} "
                f"sim={item['similarity']:.6f}"
                f"{marker}"
            )

    check(
        len(
            cases
        )
        ==
        QUERY_COUNT,
        (
            "exactly 10 qualitative "
            "cases constructed"
        ),
    )

    check(
        len(
            long_rows
        )
        ==
        QUERY_COUNT
        * TOP_K,
        (
            "exactly 30 Top-3 retrieval "
            "rows constructed"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Qualitative case construction failed"
        )

    return (
        cases,
        long_rows,
    )


# =============================================================================
# 11. WRITE QUALITATIVE ARTIFACTS
# =============================================================================

def write_outputs(
    protocol: dict[str, Any],
    test_lock: dict[str, Any],
    cases: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
) -> tuple[
    str,
    str,
    str,
]:

    banner(
        "7. WRITE QUALITATIVE RESULT / LOCK"
    )

    write_csv_atomic(
        OUTPUT_CSV,
        long_rows,
    )

    csv_sha = sha256_file(
        OUTPUT_CSV
    )

    qualitative = (
        protocol[
            "qualitative_retrieval"
        ]
    )

    seed_hashes = (
        test_lock[
            "seed_artifact_hashes"
        ][
            "42"
        ]
    )

    result = {
        "artifact_type":
            "task4_qualitative_caption_to_audio_top3",

        "version":
            1,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "frozen_qualitative_retrieval_complete",

        "query_count":
            QUERY_COUNT,

        "top_k":
            TOP_K,

        "direction":
            DIRECTION,

        "model_seed":
            QUALITATIVE_SEED,

        "query_selection_seed":
            qualitative[
                "query_selection_seed"
            ],

        "query_selection_algorithm":
            qualitative[
                "query_selection_algorithm"
            ],

        "frozen_test_row_indices":
            qualitative[
                "frozen_test_row_indices"
            ],

        "hand_picking":
            False,

        "model_inference_rerun":
            False,

        "retrieval_rerun":
            False,

        "similarity_source":
            {
                "path":
                    relative(
                        SEED42_SIMILARITY
                    ),

                "sha256":
                    seed_hashes[
                        "similarity"
                    ],

                "orientation":
                    "rows=audio, columns=caption",
            },

        "rank_source":
            {
                "path":
                    relative(
                        SEED42_RANKS
                    ),

                "sha256":
                    seed_hashes[
                        "ranks"
                    ],

                "rank_rule":
                    (
                        "1 + number of candidate "
                        "similarities strictly greater "
                        "than positive similarity"
                    ),
            },

        "top3_ordering":
            (
                "cosine similarity descending; "
                "exact tie -> lower audio row_index first"
            ),

        "cases":
            cases,

        "csv":
            {
                "path":
                    relative(
                        OUTPUT_CSV
                    ),

                "sha256":
                    csv_sha,
            },

        "provenance":
            {
                "script74":
                    relative(
                        Path(
                            __file__
                        ).resolve()
                    ),

                "script74_sha256":
                    sha256_file(
                        Path(
                            __file__
                        ).resolve()
                    ),

                "pretest_protocol_sha256":
                    EXPECTED_PRETEST_PROTOCOL_SHA256,

                "test_retrieval_summary_sha256":
                    EXPECTED_TEST_SUMMARY_SHA256,

                "test_retrieval_lock_sha256":
                    EXPECTED_TEST_LOCK_SHA256,

                "zero_shot_summary_sha256":
                    EXPECTED_ZERO_SHOT_SUMMARY_SHA256,
            },

        "next_script":
            75,
    }

    write_json_atomic(
        OUTPUT_JSON,
        result,
    )

    json_sha = sha256_file(
        OUTPUT_JSON
    )

    lock = {
        "lock_type":
            "task4_qualitative_caption_to_audio_top3_lock",

        "version":
            1,

        "created_utc":
            result[
                "created_utc"
            ],

        "result":
            relative(
                OUTPUT_JSON
            ),

        "result_sha256":
            json_sha,

        "csv":
            relative(
                OUTPUT_CSV
            ),

        "csv_sha256":
            csv_sha,

        "model_seed":
            QUALITATIVE_SEED,

        "query_indices":
            qualitative[
                "frozen_test_row_indices"
            ],

        "similarity_sha256":
            seed_hashes[
                "similarity"
            ],

        "ranks_sha256":
            seed_hashes[
                "ranks"
            ],

        "hand_picking":
            False,

        "retrieval_rerun":
            False,

        "script74_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        LOCK_PATH,
        lock,
    )

    lock_sha = sha256_file(
        LOCK_PATH
    )

    check(
        load_json(
            LOCK_PATH
        )[
            "result_sha256"
        ]
        ==
        sha256_file(
            OUTPUT_JSON
        ),
        (
            "qualitative lock matches "
            "qualitative JSON"
        ),
    )

    return (
        csv_sha,
        json_sha,
        lock_sha,
    )


# =============================================================================
# 12. FINAL DISCIPLINE CHECK
# =============================================================================

def print_boundary() -> None:

    banner(
        "8. QUALITATIVE DISCIPLINE CONFIRMATION"
    )

    print(
        "Queries hand-picked:                 0"
    )

    print(
        "Query indices changed after TEST:    0"
    )

    print(
        "Best seed selected:                  0"
    )

    print(
        "Model inference rerun:               0"
    )

    print(
        "Retrieval recomputed from model:     0"
    )

    print(
        "Similarity matrix reused:            YES"
    )

    print(
        "Ground-truth ranks reused/verified:  YES"
    )

    print(
        "TEST tag labels loaded:              0"
    )

    passed(
        (
            "10 qualitative cases follow "
            "the frozen Script-71 protocol"
        )
    )


# =============================================================================
# 13. MAIN
# =============================================================================

def main() -> int:

    banner(
        "SCRIPT 74 — FROZEN TASK-4 "
        "QUALITATIVE CAPTION→TOP-3 RETRIEVAL"
    )

    print(
        f"Project root: {ROOT}"
    )

    print()
    print(
        "Queries:     10 frozen TEST rows"
    )

    print(
        "Direction:   Caption → Audio"
    )

    print(
        "Top-K:       3"
    )

    print(
        "Model seed:  42"
    )

    print()
    print(
        "Model inference: NO"
    )

    print(
        "Retrieval rerun: NO"
    )

    try:

        verify_output_protection()

        (
            protocol,
            test_lock,
        ) = verify_lineage()

        (
            similarity,
            caption_to_audio_rank,
        ) = load_retrieval_cache(
            test_lock
        )

        test_manifest = (
            load_test_manifest()
        )

        caption_lookup = (
            load_captions(
                test_manifest
            )
        )

        (
            cases,
            long_rows,
        ) = build_cases(
            protocol,
            test_manifest,
            caption_lookup,
            similarity,
            caption_to_audio_rank,
        )

        (
            csv_sha,
            json_sha,
            lock_sha,
        ) = write_outputs(
            protocol,
            test_lock,
            cases,
            long_rows,
        )

        print_boundary()

    except Exception as exc:

        banner(
            "SCRIPT 74 — ABORTED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        if FAILURES:

            print()
            print(
                "Failed checks:"
            )

            for message in FAILURES:

                print(
                    f"  - {message}"
                )

        print()
        print(
            "Do NOT change the frozen query "
            "indices, seed, Top-K, or retrieval."
        )

        print()
        print(
            "Do NOT proceed to Script 75."
        )

        return 1

    banner(
        "SCRIPT 74 SUMMARY"
    )

    print(
        f"Failures: {len(FAILURES)}"
    )

    print(
        f"Warnings: {len(WARNINGS)}"
    )

    print()
    print(
        "RESULT: PASS"
    )

    print()
    print(
        "10 FROZEN QUALITATIVE "
        "CAPTION→TOP-3 CASES COMPLETE."
    )

    print()
    print(
        "Model seed: 42"
    )

    print(
        "Query selection: PRE-TEST frozen"
    )

    print(
        "Hand-picking: NO"
    )

    print(
        "Retrieval rerun: NO"
    )

    print()
    print(
        "CSV:"
    )

    print(
        f"  {relative(OUTPUT_CSV)}"
    )

    print(
        f"  SHA256 = {csv_sha}"
    )

    print()
    print(
        "JSON:"
    )

    print(
        f"  {relative(OUTPUT_JSON)}"
    )

    print(
        f"  SHA256 = {json_sha}"
    )

    print()
    print(
        "Lock:"
    )

    print(
        f"  {relative(LOCK_PATH)}"
    )

    print(
        f"  SHA256 = {lock_sha}"
    )

    print()
    print(
        "STOP HERE."
    )

    print()
    print(
        "Next:"
    )

    print(
        "  Script 75 — human evaluation "
        "package for the same 10 frozen queries."
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )