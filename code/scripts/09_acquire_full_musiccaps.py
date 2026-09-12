from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

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

FULL_AUDIO_DIR = (
    ROOT
    / "data"
    / "raw"
    / "musiccaps"
    / "audio"
    / "full"
)

TMP_DIR = (
    ROOT
    / "data"
    / "raw"
    / "musiccaps"
    / "tmp"
    / "full"
)


def clean_error(text, limit=1000):
    text = " ".join(
        str(text).split()
    )

    return text[-limit:]


def cleanup(track_id):
    for path in TMP_DIR.glob(
        f"{track_id}.*"
    ):
        try:
            path.unlink()
        except OSError:
            pass


def inspect_audio(
    path,
    target_sr,
    target_duration,
    tolerance,
):
    try:
        info = sf.info(path)

        duration = (
            info.frames
            / info.samplerate
        )

        usable = (
            info.samplerate == target_sr
            and info.channels == 1
            and info.format == "WAV"
            and info.subtype == "PCM_16"
            and abs(
                duration - target_duration
            ) <= tolerance
        )

        return {
            "audio_load_success": True,
            "actual_duration": duration,
            "duration_ok": (
                abs(
                    duration
                    - target_duration
                )
                <= tolerance
            ),
            "sample_rate": (
                info.samplerate
            ),
            "channels": (
                info.channels
            ),
            "frames": (
                info.frames
            ),
            "usable": usable,
        }

    except Exception as exc:
        return {
            "audio_load_success": False,
            "actual_duration": None,
            "duration_ok": False,
            "sample_rate": None,
            "channels": None,
            "frames": None,
            "usable": False,
            "error": clean_error(
                str(exc)
            ),
        }


def acquire_one(
    row,
    target_sr,
    target_duration,
    tolerance,
):
    track_id = str(
        row["track_id"]
    )

    ytid = str(
        row["ytid"]
    )

    start_s = float(
        row["start_s"]
    )

    end_s = float(
        row["end_s"]
    )

    result = {
        "track_id": track_id,
        "download_success": False,
        "audio_load_success": False,
        "actual_duration": None,
        "duration_ok": False,
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "usable": False,
        "attempts": int(
            row.get(
                "attempts",
                0,
            )
            or 0
        ) + 1,
        "acquisition_status": "failed",
        "failure_stage": "",
        "failure_reason": "",
    }

    cleanup(
        track_id
    )

    final_path = (
        FULL_AUDIO_DIR
        / f"{track_id}.wav"
    )

    url = (
        "https://www.youtube.com/watch?v="
        + ytid
    )

    section = (
        f"*{start_s:.3f}-{end_s:.3f}"
    )

    template = str(
        TMP_DIR
        / f"{track_id}.%(ext)s"
    )

    command = [
        "yt-dlp",
        "--ignore-config",
        "--no-playlist",
        "--quiet",
        "--no-warnings",

        "--retries", "1",
        "--fragment-retries", "1",
        "--socket-timeout", "10",

        "-f",
        "bestaudio/best",

        "--download-sections",
        section,

        # IMPORTANT:
        # Verified necessary for accurate
        # MusicCaps section extraction.
        "--force-keyframes-at-cuts",

        "-o",
        template,

        "--print",
        "after_move:filepath",

        url,
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if process.returncode != 0:
            raise RuntimeError(
                clean_error(
                    process.stderr
                    or process.stdout
                    or "yt-dlp failed."
                )
            )

        candidates = []

        for line in (
            process.stdout.splitlines()
        ):
            line = line.strip()

            if not line:
                continue

            path = Path(line)

            if path.exists():
                candidates.append(
                    path
                )

        if not candidates:
            candidates = [
                path
                for path in TMP_DIR.glob(
                    f"{track_id}.*"
                )
                if path.suffix not in {
                    ".part",
                    ".ytdl",
                }
            ]

        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one source file; "
                f"found {len(candidates)}."
            )

        source = candidates[0]

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",

            "-i",
            str(source),

            "-vn",

            "-ac",
            "1",

            "-ar",
            str(target_sr),

            "-c:a",
            "pcm_s16le",

            str(final_path),
        ]

        conversion = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            timeout=60,
        )

        cleanup(
            track_id
        )

        if conversion.returncode != 0:
            raise RuntimeError(
                clean_error(
                    conversion.stderr
                    or conversion.stdout
                    or "FFmpeg failed."
                )
            )

        result[
            "download_success"
        ] = True

        inspection = inspect_audio(
            final_path,
            target_sr,
            target_duration,
            tolerance,
        )

        result.update(
            inspection
        )

        if inspection["usable"]:
            result[
                "acquisition_status"
            ] = "success"

        else:
            result[
                "acquisition_status"
            ] = "validation_failed"

            result[
                "failure_stage"
            ] = "validation"

            result[
                "failure_reason"
            ] = (
                "duration="
                f"{inspection['actual_duration']}"
            )

    except subprocess.TimeoutExpired:
        cleanup(
            track_id
        )

        result[
            "failure_stage"
        ] = "timeout"

        result[
            "failure_reason"
        ] = (
            "First-pass acquisition "
            "exceeded 60 seconds."
        )

    except Exception as exc:
        cleanup(
            track_id
        )

        result[
            "failure_stage"
        ] = "acquisition"

        result[
            "failure_reason"
        ] = clean_error(
            str(exc)
        )

    return result


def save_manifest(df):
    temporary = (
        MANIFEST_PATH
        .with_suffix(
            ".tmp.parquet"
        )
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.workers < 1 or args.workers > 4:
        raise ValueError(
            "Use between 1 and 4 workers."
        )

    for program in [
        "yt-dlp",
        "ffmpeg",
    ]:
        if shutil.which(
            program
        ) is None:
            raise RuntimeError(
                f"{program} not found."
            )

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

    FULL_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending = df.loc[
        ~df["usable"].fillna(False)
    ].copy()

    print()
    print("Full MusicCaps acquisition")
    print("=" * 45)

    print(
        f"Total tracks       : "
        f"{len(df)}"
    )

    print(
        f"Already usable     : "
        f"{int(df['usable'].sum())}"
    )

    print(
        f"Pending            : "
        f"{len(pending)}"
    )

    print(
        f"Workers            : "
        f"{args.workers}"
    )

    print()

    if pending.empty:
        print(
            "Nothing left to acquire."
        )
        return

    row_lookup = {
        str(row["track_id"]): idx
        for idx, row
        in df.iterrows()
    }

    completed_since_save = 0

    executor = ThreadPoolExecutor(
        max_workers=args.workers
    )

    futures = {
        executor.submit(
            acquire_one,
            row,
            target_sr,
            target_duration,
            tolerance,
        ): str(
            row["track_id"]
        )
        for _, row
        in pending.iterrows()
    }

    try:
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Acquiring MusicCaps",
        ):
            result = future.result()

            track_id = result[
                "track_id"
            ]

            idx = row_lookup[
                track_id
            ]

            for key, value in (
                result.items()
            ):
                if (
                    key != "track_id"
                    and key in df.columns
                ):
                    df.at[
                        idx,
                        key
                    ] = value

            completed_since_save += 1

            # Periodic atomic checkpoint.
            if completed_since_save >= 10:
                save_manifest(
                    df
                )

                completed_since_save = 0

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Saving progress..."
        )

        for future in futures:
            future.cancel()

    finally:
        save_manifest(
            df
        )

        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

    usable = int(
        df["usable"]
        .fillna(False)
        .sum()
    )

    print()
    print("First pass complete")
    print("-" * 45)

    print(
        f"Usable            : "
        f"{usable}/{len(df)}"
    )

    print(
        f"Usable rate       : "
        f"{usable / len(df) * 100:.2f}%"
    )

    print(
        f"Still unsuccessful: "
        f"{len(df) - usable}"
    )

    print(
        f"Manifest          : "
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()