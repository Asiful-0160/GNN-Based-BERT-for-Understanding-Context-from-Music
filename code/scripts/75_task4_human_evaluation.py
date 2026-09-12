#!/usr/bin/env python3
"""
SCRIPT 75 — TASK-4 HUMAN RETRIEVAL EVALUATION
==============================================

Modes
-----
PREPARE:
    python scripts/75_task4_human_evaluation.py --prepare

FINALIZE:
    python scripts/75_task4_human_evaluation.py --finalize

Purpose
-------
Human-evaluate the SAME ten frozen qualitative Caption -> Top-1
retrievals created by Script 74.

Requirements
------------
- exact Script-71 frozen ten queries
- exact Script-74 seed-42 Top-1 retrieval
- no query reselection
- no retrieval rerun
- no model inference
- no TEST tag labels
- at least 5 independent listeners
- each listener rates all 10 pairs
- rating scale: 1–5

Rating scale
------------
1 = No match
2 = Weak match
3 = Moderate match
4 = Good match
5 = Excellent match

Human-facing package is blinded:
    listeners see only:
        caption
        retrieved Top-1 audio

They do NOT see:
    track_id
    similarity
    retrieval rank
    ground-truth identity
    model seed
    whether retrieval is correct

Reporting
---------
Per-item:
    mean rating
    sample SD

Overall:
    pooled mean rating
    pooled sample SD

Also:
    mean of listener means
    sample SD of listener means

The SD rule is fixed here BEFORE ratings are collected:
    sample standard deviation, ddof=1
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# 0. ROOT / PATHS
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

QUALITATIVE_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3.csv"
)

QUALITATIVE_JSON = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3.json"
)

QUALITATIVE_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_qualitative_caption_to_audio_top3_lock.json"
)

CANONICAL_DATASET = (
    ROOT
    / "data/interim/"
      "musiccaps_canonical.parquet"
)


# -----------------------------------------------------------------------------
# Human-evaluation package
# -----------------------------------------------------------------------------

PACKAGE_DIR = (
    ROOT
    / "results/runs/task4/"
      "human_eval_package"
)

PACKAGE_AUDIO_DIR = (
    PACKAGE_DIR
    / "audio"
)

PACKAGE_HTML = (
    PACKAGE_DIR
    / "index.html"
)

PACKAGE_ITEMS_CSV = (
    PACKAGE_DIR
    / "human_eval_items_audit.csv"
)

PACKAGE_INSTRUCTIONS = (
    PACKAGE_DIR
    / "README.txt"
)

PACKAGE_MANIFEST = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_package.json"
)

PACKAGE_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_package_lock.json"
)


# -----------------------------------------------------------------------------
# Listener responses
# -----------------------------------------------------------------------------

RESPONSES_DIR = (
    ROOT
    / "results/runs/task4/"
      "human_eval_responses"
)


# -----------------------------------------------------------------------------
# Final human-evaluation outputs
# -----------------------------------------------------------------------------

PER_ITEM_CSV = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_per_item.csv"
)

RESULT_JSON = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_result.json"
)

RESULT_LOCK = (
    ROOT
    / "results/runs/task4/"
      "task4_human_eval_result_lock.json"
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

EXPECTED_QUALITATIVE_CSV_SHA256 = (
    "e8cb952c96fa1be6bf6ac44959e1e23a"
    "1934216bb38c28ec965c5ab33aacc027"
)

EXPECTED_QUALITATIVE_JSON_SHA256 = (
    "f3dfd08a0cadab3edd5818f358c49705"
    "87f89077ea18d370fa4a04ba635ef69c"
)

EXPECTED_QUALITATIVE_LOCK_SHA256 = (
    "8c2c898914674e3bcda0bbfda1d007c"
    "8460af38c7d9eaa19b54712c23d36ae76"
)

EXPECTED_CANONICAL_SHA256 = (
    "704e057ba3807886b35000ff23d533d4"
    "759c6a65fdf5d7a8912e618ba523c4d3"
)


# =============================================================================
# 2. FROZEN HUMAN-EVAL CONTRACT
# =============================================================================

QUERY_COUNT = 10

TOP_K_HUMAN = 1

MODEL_SEED = 42

MIN_LISTENERS = 5

RATING_MIN = 1
RATING_MAX = 5

STD_DDOF = 1

SUPPORTED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
}


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


def warning(
    message: str,
) -> None:

    print(
        f"WARN  {message}"
    )

    WARNINGS.append(
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


def sha256_text(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def relative(
    path: Path,
) -> str:

    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    )


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


def write_text_atomic(
    path: Path,
    text: str,
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

        f.write(
            text
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
    frame: pd.DataFrame,
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

    frame.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# 5. VERIFY COMPLETE LINEAGE
# =============================================================================

def verify_lineage() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:

    banner(
        "1. VERIFY FROZEN HUMAN-EVAL LINEAGE"
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
            QUALITATIVE_CSV,
            EXPECTED_QUALITATIVE_CSV_SHA256,
            "qualitative CSV",
        ),
        (
            QUALITATIVE_JSON,
            EXPECTED_QUALITATIVE_JSON_SHA256,
            "qualitative JSON",
        ),
        (
            QUALITATIVE_LOCK,
            EXPECTED_QUALITATIVE_LOCK_SHA256,
            "qualitative lock",
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
            "Human-evaluation lineage "
            "verification failed"
        )

    protocol = load_json(
        PRETEST_PROTOCOL
    )

    qualitative = load_json(
        QUALITATIVE_JSON
    )

    human_protocol = (
        protocol[
            "human_evaluation"
        ]
    )

    check(
        human_protocol[
            "same_queries_as_qualitative"
        ]
        is True,
        (
            "human evaluation uses same "
            "10 qualitative queries"
        ),
    )

    check(
        human_protocol[
            "model_seed"
        ]
        == MODEL_SEED,
        (
            "human evaluation model "
            "seed remains 42"
        ),
    )

    check(
        human_protocol[
            "retrieval_rank_used"
        ]
        == TOP_K_HUMAN,
        (
            "human evaluation uses "
            "Top-1 retrieval"
        ),
    )

    check(
        human_protocol[
            "listeners_minimum"
        ]
        == MIN_LISTENERS,
        (
            "minimum listeners remains 5"
        ),
    )

    check(
        human_protocol[
            "rating_scale"
        ]
        ==
        [
            1,
            2,
            3,
            4,
            5,
        ],
        (
            "human rating scale remains 1–5"
        ),
    )

    check(
        qualitative[
            "model_seed"
        ]
        == MODEL_SEED,
        (
            "qualitative result model "
            "seed = 42"
        ),
    )

    check(
        qualitative[
            "query_count"
        ]
        == QUERY_COUNT,
        (
            "qualitative result contains "
            "10 frozen queries"
        ),
    )

    check(
        qualitative[
            "retrieval_rerun"
        ]
        is False,
        (
            "qualitative retrieval "
            "was not rerun"
        ),
    )

    check(
        human_protocol[
            "frozen_test_row_indices"
        ]
        ==
        qualitative[
            "frozen_test_row_indices"
        ],
        (
            "human-eval query indices exactly "
            "match qualitative frozen indices"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Human-evaluation protocol changed"
        )

    return (
        protocol,
        qualitative,
    )


# =============================================================================
# 6. RESOLVE SOURCE AUDIO
# =============================================================================

def build_audio_index() -> dict[str, list[Path]]:

    search_root = (
        ROOT
        / "data/raw/musiccaps"
    )

    check(
        search_root.is_dir(),
        (
            "MusicCaps raw-data directory exists"
        ),
    )

    if not search_root.is_dir():

        raise RuntimeError(
            "Cannot locate MusicCaps audio root"
        )

    index: dict[
        str,
        list[Path],
    ] = {}

    for path in search_root.rglob(
        "*"
    ):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in SUPPORTED_AUDIO_SUFFIXES
        ):

            continue

        # Ignore temporary acquisition files.
        lower_parts = {
            part.lower()
            for part in path.parts
        }

        if (
            "tmp"
            in lower_parts
        ):

            continue

        index.setdefault(
            path.stem,
            [],
        ).append(
            path
        )

    return index


def resolve_audio(
    track_id: str,
    audio_index: dict[str, list[Path]],
) -> Path:

    candidates = list(
        audio_index.get(
            track_id,
            [],
        )
    )

    # Prefer non-audit files.
    non_audit = [
        path

        for path
        in candidates

        if "audit"
        not in {
            part.lower()
            for part
            in path.parts
        }
    ]

    if non_audit:

        candidates = (
            non_audit
        )

    if not candidates:

        raise RuntimeError(
            "No source audio found for "
            f"{track_id}"
        )

    if len(
        candidates
    ) == 1:

        return candidates[
            0
        ]

    # Multiple copies are acceptable only if
    # their bytes are exactly identical.
    hashes = {
        sha256_file(
            path
        )

        for path
        in candidates
    }

    if len(
        hashes
    ) != 1:

        print()
        print(
            f"Conflicting audio copies "
            f"for {track_id}:"
        )

        for path in candidates:

            print(
                f"  {path}"
            )

        raise RuntimeError(
            "Ambiguous non-identical "
            "source audio copies"
        )

    # Identical duplicates:
    # deterministic lexicographic choice.
    return sorted(
        candidates,
        key=lambda path:
            str(
                path.resolve()
            ),
    )[0]


# =============================================================================
# 7. PREPARE PACKAGE
# =============================================================================

def prepare_package() -> int:

    banner(
        "SCRIPT 75A — PREPARE BLINDED "
        "HUMAN-EVALUATION PACKAGE"
    )

    verify_lineage()

    banner(
        "2. PACKAGE OUTPUT PROTECTION"
    )

    for path in (
        PACKAGE_DIR,
        PACKAGE_MANIFEST,
        PACKAGE_LOCK,
    ):

        check(
            not path.exists(),
            (
                f"package output unused: "
                f"{path.name}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite existing "
            "human-evaluation package"
        )

    qualitative = load_json(
        QUALITATIVE_JSON
    )

    cases = qualitative[
        "cases"
    ]

    check(
        len(
            cases
        )
        == QUERY_COUNT,
        (
            "exactly 10 frozen cases loaded"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Frozen case count invalid"
        )

    # -------------------------------------------------------------------------
    # Create package directories.
    # -------------------------------------------------------------------------

    PACKAGE_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    RESPONSES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    banner(
        "3. RESOLVE / COPY TOP-1 RETRIEVED AUDIO"
    )

    audio_index = (
        build_audio_index()
    )

    audit_rows = []
    browser_items = []
    package_items = []

    for case in cases:

        query_number = int(
            case[
                "query_number"
            ]
        )

        item_id = (
            f"Q{query_number:02d}"
        )

        caption = str(
            case[
                "query_caption"
            ]
        )

        top1 = (
            case[
                "top3"
            ][
                0
            ]
        )

        retrieved_track_id = str(
            top1[
                "track_id"
            ]
        )

        source_audio = (
            resolve_audio(
                retrieved_track_id,
                audio_index,
            )
        )

        suffix = (
            source_audio
            .suffix
            .lower()
        )

        destination_name = (
            f"{item_id}{suffix}"
        )

        destination = (
            PACKAGE_AUDIO_DIR
            / destination_name
        )

        shutil.copyfile(
            source_audio,
            destination,
        )

        source_sha = sha256_file(
            source_audio
        )

        copied_sha = sha256_file(
            destination
        )

        check(
            source_sha
            == copied_sha,
            (
                f"{item_id} copied audio "
                "matches source SHA256"
            ),
        )

        audit_rows.append(
            {
                "item_id":
                    item_id,

                "query_number":
                    query_number,

                "query_row_index":
                    int(
                        case[
                            "query_row_index"
                        ]
                    ),

                "query_track_id":
                    str(
                        case[
                            "query_track_id"
                        ]
                    ),

                "retrieved_top1_track_id":
                    retrieved_track_id,

                "ground_truth_rank":
                    int(
                        case[
                            "ground_truth_rank"
                        ]
                    ),

                "retrieved_similarity":
                    float(
                        top1[
                            "similarity"
                        ]
                    ),

                "retrieved_is_ground_truth":
                    bool(
                        top1[
                            "is_ground_truth"
                        ]
                    ),

                "packaged_audio":
                    f"audio/{destination_name}",

                "audio_sha256":
                    copied_sha,
            }
        )

        browser_items.append(
            {
                "item_id":
                    item_id,

                "caption":
                    caption,

                "audio":
                    f"audio/{destination_name}",
            }
        )

        package_items.append(
            {
                "item_id":
                    item_id,

                "query_row_index":
                    int(
                        case[
                            "query_row_index"
                        ]
                    ),

                "caption_sha256":
                    sha256_text(
                        caption
                    ),

                "retrieved_track_id":
                    retrieved_track_id,

                "audio_file":
                    f"audio/{destination_name}",

                "audio_sha256":
                    copied_sha,
            }
        )

    if FAILURES:

        raise RuntimeError(
            "Audio packaging failed"
        )

    audit_frame = pd.DataFrame(
        audit_rows
    )

    write_csv_atomic(
        PACKAGE_ITEMS_CSV,
        audit_frame,
    )

    # -------------------------------------------------------------------------
    # Human instructions.
    # -------------------------------------------------------------------------

    instructions = """TASK-4 HUMAN EVALUATION

You are rating how well a retrieved music clip matches a written caption.

For every item:
1. Read the caption.
2. Listen to the complete retrieved clip.
3. Give ONE rating from 1 to 5.

Scale:
1 = No match
2 = Weak match
3 = Moderate match
4 = Good match
5 = Excellent match

Judge the overall semantic/contextual match between the caption
and the audio. There are no right or wrong personal answers.

Please rate all 10 items independently.

Do not discuss your ratings with other listeners before completing them.

Use a pseudonymous listener ID such as:
L01
L02
L03

Do not enter your real name or other personal information.

Open index.html in a browser.

After rating all 10 items, click:
DOWNLOAD RATINGS CSV

Send the downloaded CSV back to the researcher.
"""

    write_text_atomic(
        PACKAGE_INSTRUCTIONS,
        instructions,
    )

    # -------------------------------------------------------------------------
    # Blinded local HTML evaluator.
    # -------------------------------------------------------------------------

    browser_json = json.dumps(
        browser_items,
        ensure_ascii=False,
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music–Caption Match Evaluation</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: 32px auto;
    padding: 0 20px 60px;
    line-height: 1.5;
}}
h1 {{
    margin-bottom: 8px;
}}
.instructions {{
    background: #f3f3f3;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 24px;
}}
.item {{
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 18px;
    margin: 20px 0;
}}
.caption {{
    font-size: 1.05rem;
    margin-bottom: 14px;
}}
audio {{
    width: 100%;
    margin-bottom: 14px;
}}
.rating {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}}
button {{
    padding: 12px 18px;
    font-size: 1rem;
    cursor: pointer;
}}
input[type="text"] {{
    padding: 8px;
    font-size: 1rem;
}}
#error {{
    color: #a00;
    font-weight: bold;
    margin: 12px 0;
}}
</style>
</head>

<body>

<h1>Music–Caption Match Evaluation</h1>

<div class="instructions">
<p>
For each item, read the caption, listen to the complete audio clip,
and rate how well the music matches the caption.
</p>

<p>
<strong>1</strong> = No match<br>
<strong>2</strong> = Weak match<br>
<strong>3</strong> = Moderate match<br>
<strong>4</strong> = Good match<br>
<strong>5</strong> = Excellent match
</p>

<p>
Please complete all 10 ratings independently.
Do not discuss your ratings with another listener before finishing.
</p>
</div>

<p>
<label>
<strong>Listener ID:</strong>
<input id="listenerId"
       type="text"
       placeholder="Example: L01"
       autocomplete="off">
</label>
</p>

<div id="items"></div>

<div id="error"></div>

<button id="downloadButton">
DOWNLOAD RATINGS CSV
</button>

<script>
const items = {browser_json};

const container =
    document.getElementById("items");

for (const item of items) {{

    const section =
        document.createElement("section");

    section.className =
        "item";

    const heading =
        document.createElement("h2");

    heading.textContent =
        item.item_id;

    section.appendChild(
        heading
    );

    const caption =
        document.createElement("div");

    caption.className =
        "caption";

    caption.textContent =
        item.caption;

    section.appendChild(
        caption
    );

    const audio =
        document.createElement("audio");

    audio.controls =
        true;

    audio.preload =
        "metadata";

    audio.src =
        item.audio;

    section.appendChild(
        audio
    );

    const label =
        document.createElement("div");

    label.textContent =
        "Your rating:";

    section.appendChild(
        label
    );

    const rating =
        document.createElement("div");

    rating.className =
        "rating";

    for (
        let score = 1;
        score <= 5;
        score++
    ) {{

        const wrapper =
            document.createElement("label");

        const input =
            document.createElement("input");

        input.type =
            "radio";

        input.name =
            "rating_" + item.item_id;

        input.value =
            String(score);

        wrapper.appendChild(
            input
        );

        wrapper.appendChild(
            document.createTextNode(
                " " + score
            )
        );

        rating.appendChild(
            wrapper
        );
    }}

    section.appendChild(
        rating
    );

    container.appendChild(
        section
    );
}}


function csvEscape(value) {{

    const text =
        String(value);

    return '"' +
        text.replaceAll(
            '"',
            '""'
        ) +
        '"';
}}


document
.getElementById(
    "downloadButton"
)
.addEventListener(
    "click",
    () => {{

        const error =
            document.getElementById(
                "error"
            );

        error.textContent =
            "";

        const listenerId =
            document
            .getElementById(
                "listenerId"
            )
            .value
            .trim();

        if (!listenerId) {{

            error.textContent =
                "Please enter a listener ID.";

            return;
        }}

        const rows = [
            [
                "listener_id",
                "item_id",
                "rating"
            ]
        ];

        for (const item of items) {{

            const selected =
                document.querySelector(
                    'input[name="rating_' +
                    item.item_id +
                    '"]:checked'
                );

            if (!selected) {{

                error.textContent =
                    "Please rate all 10 items.";

                return;
            }}

            rows.push(
                [
                    listenerId,
                    item.item_id,
                    selected.value
                ]
            );
        }}

        const csv =
            rows
            .map(
                row =>
                    row
                    .map(
                        csvEscape
                    )
                    .join(",")
            )
            .join("\\n");

        const blob =
            new Blob(
                [csv],
                {{
                    type:
                        "text/csv;charset=utf-8"
                }}
            );

        const url =
            URL.createObjectURL(
                blob
            );

        const safeListener =
            listenerId.replace(
                /[^A-Za-z0-9_-]/g,
                "_"
            );

        const link =
            document.createElement(
                "a"
            );

        link.href =
            url;

        link.download =
            "task4_human_eval_" +
            safeListener +
            ".csv";

        document.body.appendChild(
            link
        );

        link.click();

        link.remove();

        URL.revokeObjectURL(
            url
        );
    }}
);
</script>

</body>
</html>
"""

    write_text_atomic(
        PACKAGE_HTML,
        html_text,
    )

    # -------------------------------------------------------------------------
    # Package manifest.
    # -------------------------------------------------------------------------

    created_utc = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    package_manifest = {
        "artifact_type":
            "task4_human_evaluation_package",

        "version":
            1,

        "created_utc":
            created_utc,

        "status":
            "prepared_before_human_ratings",

        "selected_model_seed":
            MODEL_SEED,

        "query_count":
            QUERY_COUNT,

        "retrieval_rank_used":
            1,

        "listeners_required_minimum":
            MIN_LISTENERS,

        "rating_scale": {
            "1":
                "No match",

            "2":
                "Weak match",

            "3":
                "Moderate match",

            "4":
                "Good match",

            "5":
                "Excellent match",
        },

        "statistics_protocol": {
            "per_item_mean":
                True,

            "per_item_standard_deviation":
                True,

            "pooled_mean":
                True,

            "pooled_standard_deviation":
                True,

            "listener_mean_analysis":
                True,

            "standard_deviation":
                "sample SD",

            "ddof":
                STD_DDOF,
        },

        "blinding": {
            "human_sees_caption":
                True,

            "human_sees_top1_audio":
                True,

            "human_sees_track_id":
                False,

            "human_sees_similarity":
                False,

            "human_sees_ground_truth_rank":
                False,

            "human_sees_correctness":
                False,

            "human_sees_model_seed":
                False,
        },

        "items":
            package_items,

        "package_files": {
            "html":
                relative(
                    PACKAGE_HTML
                ),

            "html_sha256":
                sha256_file(
                    PACKAGE_HTML
                ),

            "instructions":
                relative(
                    PACKAGE_INSTRUCTIONS
                ),

            "instructions_sha256":
                sha256_file(
                    PACKAGE_INSTRUCTIONS
                ),

            "audit_csv":
                relative(
                    PACKAGE_ITEMS_CSV
                ),

            "audit_csv_sha256":
                sha256_file(
                    PACKAGE_ITEMS_CSV
                ),
        },

        "response_directory":
            relative(
                RESPONSES_DIR
            ),

        "query_hand_picking":
            False,

        "retrieval_rerun":
            False,

        "test_tag_labels_loaded":
            False,

        "parent_hashes": {
            "pretest_protocol":
                EXPECTED_PRETEST_PROTOCOL_SHA256,

            "test_retrieval_summary":
                EXPECTED_TEST_SUMMARY_SHA256,

            "qualitative_csv":
                EXPECTED_QUALITATIVE_CSV_SHA256,

            "qualitative_json":
                EXPECTED_QUALITATIVE_JSON_SHA256,

            "qualitative_lock":
                EXPECTED_QUALITATIVE_LOCK_SHA256,
        },

        "script75_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        PACKAGE_MANIFEST,
        package_manifest,
    )

    package_manifest_sha = (
        sha256_file(
            PACKAGE_MANIFEST
        )
    )

    package_lock = {
        "lock_type":
            "task4_human_evaluation_package_lock",

        "version":
            1,

        "created_utc":
            created_utc,

        "status":
            "frozen_before_human_ratings",

        "package_manifest":
            relative(
                PACKAGE_MANIFEST
            ),

        "package_manifest_sha256":
            package_manifest_sha,

        "query_count":
            QUERY_COUNT,

        "model_seed":
            MODEL_SEED,

        "minimum_listeners":
            MIN_LISTENERS,

        "rating_scale":
            [
                1,
                2,
                3,
                4,
                5,
            ],

        "standard_deviation_ddof":
            STD_DDOF,

        "retrieval_rerun":
            False,

        "script75_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        PACKAGE_LOCK,
        package_lock,
    )

    package_lock_sha = (
        sha256_file(
            PACKAGE_LOCK
        )
    )

    banner(
        "SCRIPT 75A SUMMARY"
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
        "BLINDED HUMAN-EVALUATION "
        "PACKAGE PREPARED."
    )

    print()
    print(
        "Package:"
    )

    print(
        f"  {relative(PACKAGE_DIR)}"
    )

    print()
    print(
        "Open:"
    )

    print(
        f"  {relative(PACKAGE_HTML)}"
    )

    print()
    print(
        "Package manifest SHA256:"
    )

    print(
        f"  {package_manifest_sha}"
    )

    print(
        "Package lock SHA256:"
    )

    print(
        f"  {package_lock_sha}"
    )

    print()
    print(
        "Human listeners required:"
    )

    print(
        f"  >= {MIN_LISTENERS}"
    )

    print()
    print(
        "Each listener must rate:"
    )

    print(
        "  all 10 items"
    )

    print()
    print(
        "Save returned listener CSV files in:"
    )

    print(
        f"  {relative(RESPONSES_DIR)}"
    )

    print()
    print(
        "DO NOT proceed to Script 76 yet."
    )

    print()
    print(
        "After collecting >=5 complete "
        "listener CSVs, run:"
    )

    print()
    print(
        "  python scripts/"
        "75_task4_human_evaluation.py "
        "--finalize"
    )

    return 0


# =============================================================================
# 8. VERIFY PACKAGE BEFORE FINALIZATION
# =============================================================================

def verify_package() -> dict[str, Any]:

    banner(
        "1. VERIFY FROZEN HUMAN-EVAL PACKAGE"
    )

    check(
        PACKAGE_MANIFEST.is_file(),
        (
            "human-eval package manifest exists"
        ),
    )

    check(
        PACKAGE_LOCK.is_file(),
        (
            "human-eval package lock exists"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Human-eval package missing"
        )

    manifest = load_json(
        PACKAGE_MANIFEST
    )

    lock = load_json(
        PACKAGE_LOCK
    )

    manifest_sha = sha256_file(
        PACKAGE_MANIFEST
    )

    check(
        lock[
            "package_manifest_sha256"
        ]
        ==
        manifest_sha,
        (
            "package lock matches "
            "package manifest"
        ),
    )

    check(
        manifest[
            "status"
        ]
        ==
        "prepared_before_human_ratings",
        (
            "package was frozen before ratings"
        ),
    )

    check(
        manifest[
            "query_count"
        ]
        ==
        QUERY_COUNT,
        (
            "package contains 10 items"
        ),
    )

    check(
        manifest[
            "listeners_required_minimum"
        ]
        ==
        MIN_LISTENERS,
        (
            "minimum listener count "
            "remains 5"
        ),
    )

    check(
        manifest[
            "statistics_protocol"
        ][
            "ddof"
        ]
        ==
        STD_DDOF,
        (
            "sample-SD rule remains ddof=1"
        ),
    )

    package_files = (
        manifest[
            "package_files"
        ]
    )

    verify_sha(
        PACKAGE_HTML,
        package_files[
            "html_sha256"
        ],
        "human-eval HTML",
    )

    verify_sha(
        PACKAGE_INSTRUCTIONS,
        package_files[
            "instructions_sha256"
        ],
        "human-eval instructions",
    )

    verify_sha(
        PACKAGE_ITEMS_CSV,
        package_files[
            "audit_csv_sha256"
        ],
        "human-eval audit CSV",
    )

    for item in manifest[
        "items"
    ]:

        audio_path = (
            PACKAGE_DIR
            / item[
                "audio_file"
            ]
        )

        verify_sha(
            audio_path,
            item[
                "audio_sha256"
            ],
            (
                f"{item['item_id']} "
                "packaged audio"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Frozen human-eval package "
            "verification failed"
        )

    return manifest


# =============================================================================
# 9. LOAD / VALIDATE LISTENER RESPONSES
# =============================================================================

def load_responses(
    manifest: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:

    banner(
        "2. LOAD HUMAN LISTENER RESPONSES"
    )

    check(
        RESPONSES_DIR.is_dir(),
        (
            "human response directory exists"
        ),
    )

    if not RESPONSES_DIR.is_dir():

        raise RuntimeError(
            "Human response directory missing"
        )

    response_files = sorted(
        RESPONSES_DIR.glob(
            "*.csv"
        )
    )

    print(
        f"Response CSV files found: "
        f"{len(response_files)}"
    )

    check(
        len(
            response_files
        )
        >= MIN_LISTENERS,
        (
            "at least 5 listener "
            "response files present"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Insufficient human listeners"
        )

    expected_items = [
        item[
            "item_id"
        ]

        for item
        in manifest[
            "items"
        ]
    ]

    all_frames = []

    listener_records = []

    listener_ids_seen = set()

    for path in response_files:

        frame = pd.read_csv(
            path
        )

        required = {
            "listener_id",
            "item_id",
            "rating",
        }

        check(
            required.issubset(
                frame.columns
            ),
            (
                f"{path.name} contains "
                "required columns"
            ),
        )

        if not required.issubset(
            frame.columns
        ):

            continue

        frame = (
            frame[
                [
                    "listener_id",
                    "item_id",
                    "rating",
                ]
            ]
            .copy()
        )

        check(
            len(
                frame
            )
            ==
            QUERY_COUNT,
            (
                f"{path.name} contains "
                "exactly 10 ratings"
            ),
        )

        listener_ids = (
            frame[
                "listener_id"
            ]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        check(
            len(
                listener_ids
            )
            == 1,
            (
                f"{path.name} contains "
                "exactly one listener ID"
            ),
        )

        if len(
            listener_ids
        ) != 1:

            continue

        listener_id = (
            listener_ids[
                0
            ]
        )

        check(
            len(
                listener_id
            )
            > 0,
            (
                f"{path.name} listener ID "
                "is non-empty"
            ),
        )

        check(
            listener_id
            not in listener_ids_seen,
            (
                f"listener ID {listener_id} "
                "is unique"
            ),
        )

        listener_ids_seen.add(
            listener_id
        )

        frame[
            "item_id"
        ] = (
            frame[
                "item_id"
            ]
            .astype(str)
            .str.strip()
        )

        check(
            frame[
                "item_id"
            ].nunique()
            ==
            QUERY_COUNT,
            (
                f"{listener_id} rated "
                "10 unique items"
            ),
        )

        check(
            set(
                frame[
                    "item_id"
                ]
            )
            ==
            set(
                expected_items
            ),
            (
                f"{listener_id} rated "
                "exact frozen item set"
            ),
        )

        numeric_rating = (
            pd.to_numeric(
                frame[
                    "rating"
                ],
                errors="coerce",
            )
        )

        check(
            numeric_rating.notna().all(),
            (
                f"{listener_id} ratings "
                "are numeric"
            ),
        )

        if not numeric_rating.notna().all():

            continue

        ratings_np = (
            numeric_rating
            .to_numpy(
                dtype=np.float64
            )
        )

        check(
            np.all(
                np.equal(
                    ratings_np,
                    np.floor(
                        ratings_np
                    ),
                )
            ),
            (
                f"{listener_id} ratings "
                "are integers"
            ),
        )

        check(
            np.all(
                (
                    ratings_np
                    >= RATING_MIN
                )
                &
                (
                    ratings_np
                    <= RATING_MAX
                )
            ),
            (
                f"{listener_id} ratings "
                "are within 1–5"
            ),
        )

        frame[
            "rating"
        ] = (
            numeric_rating
            .astype(
                np.int64
            )
        )

        # Restore canonical item order.
        frame[
            "item_id"
        ] = pd.Categorical(
            frame[
                "item_id"
            ],
            categories=expected_items,
            ordered=True,
        )

        frame = (
            frame
            .sort_values(
                "item_id"
            )
            .reset_index(
                drop=True
            )
        )

        frame[
            "item_id"
        ] = (
            frame[
                "item_id"
            ]
            .astype(
                str
            )
        )

        all_frames.append(
            frame
        )

        listener_records.append(
            {
                "listener_id_sha256":
                    sha256_text(
                        listener_id
                    ),

                "response_file":
                    relative(
                        path
                    ),

                "response_file_sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    if FAILURES:

        raise RuntimeError(
            "Human response validation failed"
        )

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    n_listeners = len(
        listener_ids_seen
    )

    check(
        n_listeners
        >= MIN_LISTENERS,
        (
            "validated unique listeners >= 5"
        ),
    )

    check(
        len(
            combined
        )
        ==
        (
            n_listeners
            * QUERY_COUNT
        ),
        (
            "complete listener × item "
            "rating matrix"
        ),
    )

    if FAILURES:

        raise RuntimeError(
            "Human response matrix incomplete"
        )

    print()
    print(
        f"Validated listeners: "
        f"{n_listeners}"
    )

    print(
        f"Validated ratings:   "
        f"{len(combined)}"
    )

    return (
        combined,
        listener_records,
    )


# =============================================================================
# 10. FINAL HUMAN STATISTICS
# =============================================================================

def calculate_statistics(
    manifest: dict[str, Any],
    ratings: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    banner(
        "3. CALCULATE HUMAN-EVALUATION STATISTICS"
    )

    item_lookup = {
        item[
            "item_id"
        ]:
            item

        for item
        in manifest[
            "items"
        ]
    }

    rows = []

    for item_id in [
        item[
            "item_id"
        ]

        for item
        in manifest[
            "items"
        ]
    ]:

        values = (
            ratings.loc[
                ratings[
                    "item_id"
                ]
                ==
                item_id,
                "rating",
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        item = (
            item_lookup[
                item_id
            ]
        )

        rows.append(
            {
                "item_id":
                    item_id,

                "query_row_index":
                    int(
                        item[
                            "query_row_index"
                        ]
                    ),

                "n_listeners":
                    len(
                        values
                    ),

                "mean_rating":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "std_sample":
                    float(
                        np.std(
                            values,
                            ddof=STD_DDOF,
                        )
                    ),

                "min_rating":
                    int(
                        np.min(
                            values
                        )
                    ),

                "max_rating":
                    int(
                        np.max(
                            values
                        )
                    ),
            }
        )

    per_item = pd.DataFrame(
        rows
    )

    all_ratings = (
        ratings[
            "rating"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    listener_means = (
        ratings
        .groupby(
            "listener_id",
            sort=True,
        )[
            "rating"
        ]
        .mean()
        .to_numpy(
            dtype=np.float64
        )
    )

    statistics = {
        "listeners":
            int(
                ratings[
                    "listener_id"
                ].nunique()
            ),

        "items":
            QUERY_COUNT,

        "total_ratings":
            int(
                len(
                    ratings
                )
            ),

        "pooled_rating": {
            "mean":
                float(
                    np.mean(
                        all_ratings
                    )
                ),

            "std_sample":
                float(
                    np.std(
                        all_ratings,
                        ddof=STD_DDOF,
                    )
                ),
        },

        "listener_mean_distribution": {
            "mean":
                float(
                    np.mean(
                        listener_means
                    )
                ),

            "std_sample":
                float(
                    np.std(
                        listener_means,
                        ddof=STD_DDOF,
                    )
                ),
        },

        "per_item_mean_distribution": {
            "mean":
                float(
                    per_item[
                        "mean_rating"
                    ].mean()
                ),

            "std_sample":
                float(
                    per_item[
                        "mean_rating"
                    ].std(
                        ddof=STD_DDOF
                    )
                ),
        },
    }

    print()
    print(
        "Pooled human match rating:"
    )

    print(
        f"  mean = "
        f"{statistics['pooled_rating']['mean']:.4f}"
    )

    print(
        f"  SD   = "
        f"{statistics['pooled_rating']['std_sample']:.4f}"
    )

    print()

    print(
        "Listener-mean distribution:"
    )

    print(
        f"  mean = "
        f"{statistics['listener_mean_distribution']['mean']:.4f}"
    )

    print(
        f"  SD   = "
        f"{statistics['listener_mean_distribution']['std_sample']:.4f}"
    )

    return (
        per_item,
        statistics,
    )


# =============================================================================
# 11. FINALIZE
# =============================================================================

def finalize() -> int:

    banner(
        "SCRIPT 75B — FINALIZE "
        "TASK-4 HUMAN EVALUATION"
    )

    verify_lineage()

    manifest = (
        verify_package()
    )

    banner(
        "1A. FINAL RESULT OUTPUT PROTECTION"
    )

    for path in (
        PER_ITEM_CSV,
        RESULT_JSON,
        RESULT_LOCK,
    ):

        check(
            not path.exists(),
            (
                f"final output unused: "
                f"{path.name}"
            ),
        )

    if FAILURES:

        raise RuntimeError(
            "Refusing to overwrite final "
            "human-evaluation result"
        )

    (
        ratings,
        listener_records,
    ) = load_responses(
        manifest
    )

    (
        per_item,
        statistics,
    ) = calculate_statistics(
        manifest,
        ratings,
    )

    write_csv_atomic(
        PER_ITEM_CSV,
        per_item,
    )

    per_item_sha = (
        sha256_file(
            PER_ITEM_CSV
        )
    )

    created_utc = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    result = {
        "artifact_type":
            "task4_human_evaluation_result",

        "version":
            1,

        "created_utc":
            created_utc,

        "status":
            "human_evaluation_complete",

        "selected_model_seed":
            MODEL_SEED,

        "query_count":
            QUERY_COUNT,

        "retrieval_rank_used":
            1,

        "rating_scale":
            [
                1,
                2,
                3,
                4,
                5,
            ],

        "rating_scale_meaning": {
            "1":
                "No match",

            "2":
                "Weak match",

            "3":
                "Moderate match",

            "4":
                "Good match",

            "5":
                "Excellent match",
        },

        "minimum_listener_requirement":
            MIN_LISTENERS,

        "statistics_protocol": {
            "standard_deviation":
                "sample SD",

            "ddof":
                STD_DDOF,
        },

        "statistics":
            statistics,

        "per_item_csv": {
            "path":
                relative(
                    PER_ITEM_CSV
                ),

            "sha256":
                per_item_sha,
        },

        "listener_response_provenance":
            listener_records,

        "listeners_pseudonymous":
            True,

        "query_hand_picking":
            False,

        "retrieval_rerun":
            False,

        "model_inference_rerun":
            False,

        "test_tag_labels_loaded":
            False,

        "package_manifest": {
            "path":
                relative(
                    PACKAGE_MANIFEST
                ),

            "sha256":
                sha256_file(
                    PACKAGE_MANIFEST
                ),
        },

        "package_lock": {
            "path":
                relative(
                    PACKAGE_LOCK
                ),

            "sha256":
                sha256_file(
                    PACKAGE_LOCK
                ),
        },

        "parent_hashes": {
            "qualitative_json":
                EXPECTED_QUALITATIVE_JSON_SHA256,

            "qualitative_lock":
                EXPECTED_QUALITATIVE_LOCK_SHA256,
        },

        "script75_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),

        "next_script":
            76,
    }

    write_json_atomic(
        RESULT_JSON,
        result,
    )

    result_sha = (
        sha256_file(
            RESULT_JSON
        )
    )

    lock = {
        "lock_type":
            "task4_human_evaluation_result_lock",

        "version":
            1,

        "created_utc":
            created_utc,

        "status":
            "human_evaluation_frozen",

        "result":
            relative(
                RESULT_JSON
            ),

        "result_sha256":
            result_sha,

        "per_item_csv_sha256":
            per_item_sha,

        "listeners":
            statistics[
                "listeners"
            ],

        "total_ratings":
            statistics[
                "total_ratings"
            ],

        "query_count":
            QUERY_COUNT,

        "model_seed":
            MODEL_SEED,

        "ddof":
            STD_DDOF,

        "retrieval_rerun":
            False,

        "query_reselection":
            False,

        "script75_sha256":
            sha256_file(
                Path(
                    __file__
                ).resolve()
            ),
    }

    write_json_atomic(
        RESULT_LOCK,
        lock,
    )

    lock_sha = (
        sha256_file(
            RESULT_LOCK
        )
    )

    check(
        load_json(
            RESULT_LOCK
        )[
            "result_sha256"
        ]
        ==
        sha256_file(
            RESULT_JSON
        ),
        (
            "human-eval result lock "
            "matches result JSON"
        ),
    )

    banner(
        "4. HUMAN-EVALUATION DISCIPLINE CONFIRMATION"
    )

    print(
        "Frozen queries changed:              0"
    )

    print(
        "Retrieved Top-1 changed:             0"
    )

    print(
        "Model inference rerun:               0"
    )

    print(
        "Retrieval rerun:                     0"
    )

    print(
        "Listener minimum satisfied:          YES"
    )

    print(
        "Every listener rated all 10 items:   YES"
    )

    print(
        "Ratings outside 1–5:                 0"
    )

    print(
        "TEST tag labels loaded:              0"
    )

    passed(
        (
            "Task-4 human evaluation "
            "followed the frozen protocol"
        )
    )

    banner(
        "SCRIPT 75 SUMMARY"
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
        "TASK-4 HUMAN EVALUATION COMPLETE."
    )

    print()
    print(
        f"Listeners: "
        f"{statistics['listeners']}"
    )

    print(
        f"Ratings:   "
        f"{statistics['total_ratings']}"
    )

    print()
    print(
        "Overall human match rating:"
    )

    print(
        f"  mean = "
        f"{statistics['pooled_rating']['mean']:.4f}"
    )

    print(
        f"  SD   = "
        f"{statistics['pooled_rating']['std_sample']:.4f}"
    )

    print()
    print(
        "Per-item CSV SHA256:"
    )

    print(
        f"  {per_item_sha}"
    )

    print(
        "Result JSON SHA256:"
    )

    print(
        f"  {result_sha}"
    )

    print(
        "Result lock SHA256:"
    )

    print(
        f"  {lock_sha}"
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
        "  Script 76 — final Task-4 "
        "figures, tables, audit and artifact lock."
    )

    return 0


# =============================================================================
# 12. CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Task-4 frozen human evaluation"
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--prepare",
        action="store_true",
        help=(
            "Prepare blinded human-evaluation package"
        ),
    )

    mode.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Validate collected ratings and freeze final result"
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    try:

        if args.prepare:

            return prepare_package()

        if args.finalize:

            return finalize()

        raise RuntimeError(
            "No mode selected"
        )

    except Exception as exc:

        banner(
            "SCRIPT 75 — ABORTED"
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

        if WARNINGS:

            print()
            print(
                "Warnings:"
            )

            for message in WARNINGS:

                print(
                    f"  - {message}"
                )

        print()
        print(
            "Do NOT change the frozen "
            "queries or retrievals."
        )

        print()
        print(
            "Do NOT proceed to Script 76."
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )