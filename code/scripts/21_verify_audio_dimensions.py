from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

CANONICAL_PATH = (
    ROOT
    / "data"
    / "interim"
    / "musiccaps_canonical.parquet"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "audio_dimension_audit.json"
)

DETAIL_PATH = (
    ROOT
    / "data"
    / "audit"
    / "audio_dimension_audit.csv"
)

EXPECTED_TRACKS = 5140


def fix_length(
    y: np.ndarray,
    fixed_samples: int,
) -> tuple[np.ndarray, str, int]:

    original_samples = len(y)

    if original_samples > fixed_samples:

        # Remove excess from the end.
        y = y[:fixed_samples]

        action = "trim"

    elif original_samples < fixed_samples:

        # Right-pad with silence.
        y = np.pad(
            y,
            (
                0,
                fixed_samples
                - original_samples,
            ),
            mode="constant",
        )

        action = "pad"

    else:
        action = "none"

    return (
        y,
        action,
        original_samples,
    )


def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    sr = int(
        config["audio"]["sample_rate"]
    )

    fixed_samples = int(
        config["audio"]["fixed_samples"]
    )

    n_fft = int(
        config["audio"]["n_fft"]
    )

    hop_length = int(
        config["audio"]["hop_length"]
    )

    n_mels = int(
        config["audio"]["n_mels"]
    )

    n_chroma = int(
        config["audio"]["n_chroma"]
    )

    center = bool(
        config["audio"]["center"]
    )

    expected_frames = int(
        config["audio"]["expected_mel_frames"]
    )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    if len(df) != EXPECTED_TRACKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRACKS} tracks, "
            f"found {len(df)}."
        )

    records = []

    mel_frame_counts = Counter()
    chroma_frame_counts = Counter()
    actions = Counter()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Checking audio dimensions",
    ):

        track_id = str(
            row["track_id"]
        )

        path = (
            ROOT
            / str(row["audio_path"])
        )

        y, file_sr = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )

        if file_sr != sr:
            raise RuntimeError(
                f"{track_id}: sr={file_sr}, "
                f"expected {sr}"
            )

        if y.ndim != 1:
            raise RuntimeError(
                f"{track_id}: audio is not mono."
            )

        y, action, original_samples = (
            fix_length(
                y,
                fixed_samples,
            )
        )

        if len(y) != fixed_samples:
            raise RuntimeError(
                f"{track_id}: fixed-length failure."
            )

        # ---------------------------------------------
        # Log-mel
        # ---------------------------------------------

        mel_power = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            power=2.0,
        )

        log_mel = librosa.power_to_db(
            mel_power,
            ref=np.max,
        )

        # Per-track z-score normalization.
        mean = float(
            log_mel.mean()
        )

        std = float(
            log_mel.std()
        )

        epsilon = float(
            config["audio"]["mel"]["epsilon"]
        )

        log_mel_z = (
            log_mel - mean
        ) / (
            std + epsilon
        )

        # ---------------------------------------------
        # Chroma
        # Keep native values for cosine graph similarity.
        # ---------------------------------------------

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma,
            center=center,
        )

        mel_frames = int(
            log_mel_z.shape[1]
        )

        chroma_frames = int(
            chroma.shape[1]
        )

        mel_frame_counts[
            mel_frames
        ] += 1

        chroma_frame_counts[
            chroma_frames
        ] += 1

        actions[
            action
        ] += 1

        records.append({
            "track_id": track_id,
            "original_samples": (
                original_samples
            ),
            "standardized_samples": (
                len(y)
            ),
            "length_action": action,
            "mel_bins": int(
                log_mel_z.shape[0]
            ),
            "mel_frames": mel_frames,
            "chroma_bins": int(
                chroma.shape[0]
            ),
            "chroma_frames": (
                chroma_frames
            ),
            "mel_finite": bool(
                np.isfinite(
                    log_mel_z
                ).all()
            ),
            "chroma_finite": bool(
                np.isfinite(
                    chroma
                ).all()
            ),
        })

    results = pd.DataFrame(
        records
    )

    # ---------------------------------------------
    # Hard checks
    # ---------------------------------------------

    if not (
        results["standardized_samples"]
        == fixed_samples
    ).all():
        raise RuntimeError(
            "Not all clips standardized "
            "to fixed_samples."
        )

    if not (
        results["mel_bins"]
        == n_mels
    ).all():
        raise RuntimeError(
            "Unexpected mel dimension."
        )

    if not (
        results["chroma_bins"]
        == n_chroma
    ).all():
        raise RuntimeError(
            "Unexpected chroma dimension."
        )

    if not (
        results["mel_finite"]
    ).all():
        raise RuntimeError(
            "Non-finite log-mel value detected."
        )

    if not (
        results["chroma_finite"]
    ).all():
        raise RuntimeError(
            "Non-finite chroma value detected."
        )

    unique_mel_frames = sorted(
        results[
            "mel_frames"
        ].unique().tolist()
    )

    unique_chroma_frames = sorted(
        results[
            "chroma_frames"
        ].unique().tolist()
    )

    if unique_mel_frames != [
        expected_frames
    ]:
        raise RuntimeError(
            "Mel frame count differs from "
            f"expected {expected_frames}: "
            f"{unique_mel_frames}"
        )

    if unique_chroma_frames != [
        expected_frames
    ]:
        raise RuntimeError(
            "Chroma frame count differs from "
            f"expected {expected_frames}: "
            f"{unique_chroma_frames}"
        )

    # ---------------------------------------------
    # Save result
    # ---------------------------------------------

    report = {
        "tracks_checked": int(
            len(results)
        ),
        "sample_rate": sr,
        "fixed_samples": fixed_samples,
        "fixed_duration_seconds": (
            fixed_samples / sr
        ),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "n_chroma": n_chroma,
        "center": center,
        "length_actions": {
            str(key): int(value)
            for key, value
            in actions.items()
        },
        "mel_frame_counts": {
            str(key): int(value)
            for key, value
            in mel_frame_counts.items()
        },
        "chroma_frame_counts": {
            str(key): int(value)
            for key, value
            in chroma_frame_counts.items()
        },
        "verified_fixed_frames": (
            expected_frames
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_PATH,
        index=False,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------
    # Lock frame count into config
    # ---------------------------------------------

    current = config[
        "audio"
    ].get(
        "fixed_mel_frames"
    )

    if (
        current is not None
        and int(current)
        != expected_frames
    ):
        raise RuntimeError(
            "Existing fixed_mel_frames "
            "conflicts with verified result."
        )

    config[
        "audio"
    ][
        "fixed_mel_frames"
    ] = expected_frames

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    print()
    print(
        "MusicCaps audio dimension audit"
    )
    print("=" * 52)

    print(
        f"Tracks checked        : "
        f"{len(results)}"
    )

    print(
        f"Fixed samples         : "
        f"{fixed_samples}"
    )

    print(
        f"Fixed duration        : "
        f"{fixed_samples / sr:.6f} s"
    )

    print()
    print("Length standardization")
    print("-" * 52)

    for key in [
        "none",
        "trim",
        "pad",
    ]:
        print(
            f"{key:5s}: "
            f"{actions.get(key, 0)}"
        )

    print()
    print(
        f"Log-mel shape         : "
        f"({n_mels}, {expected_frames})"
    )

    print(
        f"Chroma shape          : "
        f"({n_chroma}, {expected_frames})"
    )

    print(
        "Finite log-mel values : PASS"
    )

    print(
        "Finite chroma values  : PASS"
    )

    print()
    print(
        f"Config fixed_mel_frames="
        f"{expected_frames}"
    )

    print()
    print(
        f"Report : {REPORT_PATH}"
    )

    print(
        f"Details: {DETAIL_PATH}"
    )


if __name__ == "__main__":
    main()