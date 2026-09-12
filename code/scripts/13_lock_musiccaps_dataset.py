from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

ACQUISITION_PATH = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_full_acquisition.parquet"
)

CANONICAL_PARQUET = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_canonical.parquet"
)

CANONICAL_CSV = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_canonical.csv"
)

EXCLUDED_CSV = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_excluded.csv"
)

LOCK_REPORT = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_dataset_lock.json"
)

EXPECTED_TOTAL = 5521
MINIMUM_USABLE = 3000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_bool(value) -> bool:
    if value is None or pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes", "y"}
    )


def main() -> None:

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    target_sr = int(
        config["audio"]["sample_rate"]
    )

    target_duration = float(
        config["audio"]["duration_seconds"]
    )

    tolerance = float(
        config["dataset"]["audit"][
            "duration_tolerance_seconds"
        ]
    )

    df = pd.read_parquet(
        ACQUISITION_PATH
    ).copy()

    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"found {len(df)}."
        )

    if df["track_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate track_id detected."
        )

    if "usable_check" not in df.columns:
        raise RuntimeError(
            "Independent validation missing. "
            "Run script 10 first."
        )

    usable_mask = (
        df["usable_check"]
        .map(parse_bool)
    )

    usable = (
        df.loc[usable_mask]
        .copy()
        .reset_index(drop=True)
    )

    excluded = (
        df.loc[~usable_mask]
        .copy()
        .reset_index(drop=True)
    )

    if len(usable) < MINIMUM_USABLE:
        raise RuntimeError(
            f"Only {len(usable)} usable clips. "
            "Dataset gate FAILED."
        )

    # --------------------------------------------------
    # Final physical-file sanity check
    # --------------------------------------------------

    for _, row in tqdm(
        usable.iterrows(),
        total=len(usable),
        desc="Lock-checking MusicCaps",
    ):
        audio_path = (
            ROOT
            / str(row["audio_path"])
        )

        if not audio_path.exists():
            raise RuntimeError(
                f"Missing usable audio: {audio_path}"
            )

        info = sf.info(audio_path)

        duration = (
            info.frames
            / info.samplerate
        )

        if info.samplerate != target_sr:
            raise RuntimeError(
                f"{row['track_id']}: "
                "sample-rate mismatch."
            )

        if info.channels != 1:
            raise RuntimeError(
                f"{row['track_id']}: "
                "channel mismatch."
            )

        if info.format != "WAV":
            raise RuntimeError(
                f"{row['track_id']}: "
                "not WAV."
            )

        if info.subtype != "PCM_16":
            raise RuntimeError(
                f"{row['track_id']}: "
                "not PCM_16."
            )

        if abs(
            duration - target_duration
        ) > tolerance:
            raise RuntimeError(
                f"{row['track_id']}: "
                f"duration {duration:.6f}s "
                "outside tolerance."
            )

    # --------------------------------------------------
    # Canonical ingestion schema
    # --------------------------------------------------

    required = [
        "track_id",
        "source_id",
        "ytid",
        "start_s",
        "end_s",
        "audio_path",
        "caption",
        "aspect_list",
    ]

    missing = [
        column
        for column in required
        if column not in usable.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing canonical columns: {missing}"
        )

    canonical = usable[
        required
    ].copy()

    canonical = canonical.rename(
        columns={
            "aspect_list": "aspects",
        }
    )

    # No label column yet.
    # Labels are constructed only after splitting
    # from TRAIN data, per Pipeline v6.

    canonical = canonical.sort_values(
        "track_id"
    ).reset_index(drop=True)

    if canonical["caption"].isna().any():
        raise RuntimeError(
            "Missing caption in canonical set."
        )

    if canonical["aspects"].isna().any():
        raise RuntimeError(
            "Missing aspects in canonical set."
        )

    CANONICAL_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical.to_parquet(
        CANONICAL_PARQUET,
        index=False,
    )

    canonical.to_csv(
        CANONICAL_CSV,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_CSV,
        index=False,
    )

    report = {
        "dataset": "MusicCaps",
        "metadata_total": EXPECTED_TOTAL,
        "usable_tracks": int(
            len(canonical)
        ),
        "excluded_tracks": int(
            len(excluded)
        ),
        "usable_rate": float(
            len(canonical)
            / EXPECTED_TOTAL
        ),
        "minimum_required": (
            MINIMUM_USABLE
        ),
        "dataset_gate_passed": True,
        "further_acquisition_retries": False,
        "canonical_schema": [
            "track_id",
            "source_id",
            "ytid",
            "start_s",
            "end_s",
            "audio_path",
            "caption",
            "aspects",
        ],
        "canonical_sha256": (
            sha256_file(
                CANONICAL_PARQUET
            )
        ),
        "source_manifest_sha256": (
            sha256_file(
                ACQUISITION_PATH
            )
        ),
    }

    LOCK_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("MusicCaps dataset LOCKED")
    print("=" * 44)

    print(
        f"Metadata rows    : "
        f"{EXPECTED_TOTAL}"
    )

    print(
        f"Usable tracks    : "
        f"{len(canonical)}"
    )

    print(
        f"Excluded tracks  : "
        f"{len(excluded)}"
    )

    print(
        f"Usable rate      : "
        f"{len(canonical) / EXPECTED_TOTAL * 100:.2f}%"
    )

    print(
        f">= 3000 gate     : True"
    )

    print()
    print(
        f"Canonical parquet: "
        f"{CANONICAL_PARQUET}"
    )

    print(
        f"Canonical CSV    : "
        f"{CANONICAL_CSV}"
    )

    print(
        f"Excluded CSV     : "
        f"{EXCLUDED_CSV}"
    )

    print(
        f"Lock report      : "
        f"{LOCK_REPORT}"
    )


if __name__ == "__main__":
    main()