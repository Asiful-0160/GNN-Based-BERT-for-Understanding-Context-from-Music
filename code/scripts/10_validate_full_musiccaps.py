from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

MANIFEST_PATH = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_full_acquisition.parquet"
)

AUDIO_DIR = (
    ROOT
    / "data"
    / "raw"
    / "musiccaps"
    / "audio"
    / "full"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "full_acquisition_report.json"
)

FAILURES_PATH = (
    ROOT
    / "data"
    / "audit"
    / "full_acquisition_failures.csv"
)

EXPECTED_ROWS = 5521


def inspect_audio(
    path: Path,
    target_sr: int,
    target_duration: float,
    tolerance: float,
) -> dict:

    if not path.exists():
        return {
            "file_present_check": False,
            "audio_load_check": False,
            "actual_duration_check": None,
            "duration_ok_check": False,
            "sample_rate_check": None,
            "sample_rate_ok_check": False,
            "channels_check": None,
            "channels_ok_check": False,
            "frames_check": None,
            "format_check": "",
            "format_ok_check": False,
            "subtype_check": "",
            "subtype_ok_check": False,
            "usable_check": False,
            "validation_failure_reason": "file_missing",
        }

    try:
        info = sf.info(path)

        duration = (
            float(info.frames)
            / float(info.samplerate)
        )

        duration_ok = (
            abs(
                duration - target_duration
            )
            <= tolerance
        )

        sample_rate_ok = (
            info.samplerate
            == target_sr
        )

        channels_ok = (
            info.channels == 1
        )

        format_ok = (
            info.format == "WAV"
        )

        subtype_ok = (
            info.subtype == "PCM_16"
        )

        usable = all([
            duration_ok,
            sample_rate_ok,
            channels_ok,
            format_ok,
            subtype_ok,
        ])

        reasons = []

        if not duration_ok:
            reasons.append(
                "duration_out_of_tolerance"
            )

        if not sample_rate_ok:
            reasons.append(
                f"unexpected_sample_rate={info.samplerate}"
            )

        if not channels_ok:
            reasons.append(
                f"unexpected_channels={info.channels}"
            )

        if not format_ok:
            reasons.append(
                f"unexpected_format={info.format}"
            )

        if not subtype_ok:
            reasons.append(
                f"unexpected_subtype={info.subtype}"
            )

        return {
            "file_present_check": True,
            "audio_load_check": True,
            "actual_duration_check": duration,
            "duration_ok_check": duration_ok,
            "sample_rate_check": int(
                info.samplerate
            ),
            "sample_rate_ok_check": (
                sample_rate_ok
            ),
            "channels_check": int(
                info.channels
            ),
            "channels_ok_check": (
                channels_ok
            ),
            "frames_check": int(
                info.frames
            ),
            "format_check": str(
                info.format
            ),
            "format_ok_check": format_ok,
            "subtype_check": str(
                info.subtype
            ),
            "subtype_ok_check": subtype_ok,
            "usable_check": usable,
            "validation_failure_reason": (
                "; ".join(reasons)
            ),
        }

    except Exception as exc:
        return {
            "file_present_check": True,
            "audio_load_check": False,
            "actual_duration_check": None,
            "duration_ok_check": False,
            "sample_rate_check": None,
            "sample_rate_ok_check": False,
            "channels_check": None,
            "channels_ok_check": False,
            "frames_check": None,
            "format_check": "",
            "format_ok_check": False,
            "subtype_check": "",
            "subtype_ok_check": False,
            "usable_check": False,
            "validation_failure_reason": (
                f"audio_load_failed: {exc}"
            ),
        }


def save_manifest(df: pd.DataFrame) -> None:
    temporary = MANIFEST_PATH.with_suffix(
        ".validation.tmp.parquet"
    )

    df.to_parquet(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        MANIFEST_PATH,
    )


def main():

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
        MANIFEST_PATH
    )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(df)}."
        )

    if df["track_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate track_id detected."
        )

    records = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Validating full MusicCaps",
    ):
        track_id = str(
            row["track_id"]
        )

        path = (
            AUDIO_DIR
            / f"{track_id}.wav"
        )

        result = inspect_audio(
            path,
            target_sr,
            target_duration,
            tolerance,
        )

        result[
            "track_id"
        ] = track_id

        records.append(
            result
        )

    validation = pd.DataFrame(
        records
    )

    validation_columns = [
        column
        for column in validation.columns
        if column != "track_id"
    ]

    old_columns = [
        column
        for column in validation_columns
        if column in df.columns
    ]

    if old_columns:
        df = df.drop(
            columns=old_columns
        )

    df = df.merge(
        validation,
        on="track_id",
        how="left",
        validate="one_to_one",
    )

    usable = (
        df["usable_check"]
        .fillna(False)
        .astype(bool)
    )

    loaded = (
        df["audio_load_check"]
        .fillna(False)
        .astype(bool)
    )

    present = (
        df["file_present_check"]
        .fillna(False)
        .astype(bool)
    )

    durations = pd.to_numeric(
        df.loc[
            loaded,
            "actual_duration_check",
        ],
        errors="coerce",
    ).dropna()

    report = {
        "total_rows": int(
            len(df)
        ),

        "unique_track_ids": int(
            df["track_id"].nunique()
        ),

        "files_present": int(
            present.sum()
        ),

        "files_missing": int(
            (~present).sum()
        ),

        "audio_load_success": int(
            loaded.sum()
        ),

        "usable_clips": int(
            usable.sum()
        ),

        "unusable_clips": int(
            (~usable).sum()
        ),

        "usable_rate": float(
            usable.mean()
        ),

        "usable_rate_percent": float(
            usable.mean() * 100
        ),

        "dataset_gate_3000_passed": bool(
            usable.sum() >= 3000
        ),
    }

    if len(durations) > 0:
        report[
            "duration_statistics"
        ] = {
            "min": float(
                durations.min()
            ),
            "max": float(
                durations.max()
            ),
            "mean": float(
                durations.mean()
            ),
            "median": float(
                durations.median()
            ),
            "std": float(
                durations.std(ddof=0)
            ),
            "p05": float(
                np.percentile(
                    durations,
                    5,
                )
            ),
            "p95": float(
                np.percentile(
                    durations,
                    95,
                )
            ),
        }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    df.loc[
        ~usable
    ].to_csv(
        FAILURES_PATH,
        index=False,
    )

    save_manifest(
        df
    )

    print()
    print("Full MusicCaps independent validation")
    print("=" * 48)

    print(
        f"Total rows         : "
        f"{report['total_rows']}"
    )

    print(
        f"Files present      : "
        f"{report['files_present']}"
    )

    print(
        f"Files missing      : "
        f"{report['files_missing']}"
    )

    print(
        f"Audio load success : "
        f"{report['audio_load_success']}"
    )

    print(
        f"Usable clips       : "
        f"{report['usable_clips']}"
    )

    print(
        f"Unusable clips     : "
        f"{report['unusable_clips']}"
    )

    print(
        f"Usable rate        : "
        f"{report['usable_rate_percent']:.2f}%"
    )

    print(
        f">= 3000 gate       : "
        f"{report['dataset_gate_3000_passed']}"
    )

    if "duration_statistics" in report:

        stats = report[
            "duration_statistics"
        ]

        print()
        print("Duration statistics")
        print("-" * 48)

        print(
            f"Min    : "
            f"{stats['min']:.6f}"
        )

        print(
            f"Max    : "
            f"{stats['max']:.6f}"
        )

        print(
            f"Mean   : "
            f"{stats['mean']:.6f}"
        )

        print(
            f"Median : "
            f"{stats['median']:.6f}"
        )

        print(
            f"P05    : "
            f"{stats['p05']:.6f}"
        )

        print(
            f"P95    : "
            f"{stats['p95']:.6f}"
        )

    print()
    print(f"Report   : {REPORT_PATH}")
    print(f"Failures : {FAILURES_PATH}")


if __name__ == "__main__":
    main()