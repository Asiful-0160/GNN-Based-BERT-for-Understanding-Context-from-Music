from __future__ import annotations

import json
import os
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

MEL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "mel"
)

CHROMA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "chroma"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "audio_feature_manifest.parquet"
)

MANIFEST_CSV_PATH = (
    ROOT
    / "data"
    / "processed"
    / "audio_feature_manifest.csv"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "audio_feature_cache_report.json"
)


EXPECTED_TRACKS = 5140


def fix_length(
    y: np.ndarray,
    fixed_samples: int,
):
    original_samples = len(y)

    if original_samples > fixed_samples:
        y = y[:fixed_samples]
        action = "trim"

    elif original_samples < fixed_samples:
        y = np.pad(
            y,
            (
                0,
                fixed_samples - original_samples,
            ),
            mode="constant",
        )
        action = "pad"

    else:
        action = "none"

    return y, action


def save_npy_atomic(
    array: np.ndarray,
    destination: Path,
):
    temp = destination.with_suffix(
        ".tmp.npy"
    )

    np.save(
        temp,
        array,
        allow_pickle=False,
    )

    os.replace(
        temp,
        destination,
    )


def existing_feature_valid(
    path: Path,
    expected_shape: tuple[int, int],
):
    if not path.exists():
        return False

    try:
        array = np.load(
            path,
            allow_pickle=False,
        )

        return (
            array.shape == expected_shape
            and array.dtype == np.float32
            and np.isfinite(array).all()
        )

    except Exception:
        return False


def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    audio = config["audio"]

    sr = int(
        audio["sample_rate"]
    )

    fixed_samples = int(
        audio["fixed_samples"]
    )

    n_fft = int(
        audio["n_fft"]
    )

    hop_length = int(
        audio["hop_length"]
    )

    n_mels = int(
        audio["n_mels"]
    )

    n_chroma = int(
        audio["n_chroma"]
    )

    center = bool(
        audio["center"]
    )

    fixed_frames = int(
        audio["fixed_mel_frames"]
    )

    epsilon = float(
        audio["mel"]["epsilon"]
    )

    if fixed_frames != 431:
        raise RuntimeError(
            f"Expected fixed_mel_frames=431, "
            f"found {fixed_frames}."
        )

    df = pd.read_parquet(
        CANONICAL_PATH
    )

    if len(df) != EXPECTED_TRACKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRACKS} tracks, "
            f"found {len(df)}."
        )

    if df["track_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate track IDs."
        )

    MEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    actions = Counter()

    generated = 0
    reused = 0

    records = []

    mel_shape = (
        n_mels,
        fixed_frames,
    )

    chroma_shape = (
        n_chroma,
        fixed_frames,
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Caching audio features",
    ):

        track_id = str(
            row["track_id"]
        )

        audio_path = (
            ROOT
            / str(row["audio_path"])
        )

        mel_path = (
            MEL_DIR
            / f"{track_id}.npy"
        )

        chroma_path = (
            CHROMA_DIR
            / f"{track_id}.npy"
        )

        mel_valid = existing_feature_valid(
            mel_path,
            mel_shape,
        )

        chroma_valid = existing_feature_valid(
            chroma_path,
            chroma_shape,
        )

        if mel_valid and chroma_valid:

            reused += 1

            records.append({
                "track_id": track_id,
                "mel_path": str(
                    mel_path.relative_to(ROOT)
                ),
                "chroma_path": str(
                    chroma_path.relative_to(ROOT)
                ),
                "mel_bins": n_mels,
                "mel_frames": fixed_frames,
                "chroma_bins": n_chroma,
                "chroma_frames": fixed_frames,
                "cache_status": "reused",
            })

            continue

        y, file_sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )

        if file_sr != sr:
            raise RuntimeError(
                f"{track_id}: unexpected "
                f"sample rate {file_sr}."
            )

        if y.ndim != 1:
            raise RuntimeError(
                f"{track_id}: audio is not mono."
            )

        y, action = fix_length(
            y,
            fixed_samples,
        )

        actions[action] += 1

        if len(y) != fixed_samples:
            raise RuntimeError(
                f"{track_id}: fixed-length "
                "standardization failed."
            )

        # -----------------------------------------
        # Log-mel
        # -----------------------------------------

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

        mean = float(
            log_mel.mean()
        )

        std = float(
            log_mel.std()
        )

        log_mel = (
            log_mel - mean
        ) / (
            std + epsilon
        )

        log_mel = log_mel.astype(
            np.float32,
            copy=False,
        )

        # -----------------------------------------
        # Native chroma
        # -----------------------------------------

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_chroma=n_chroma,
            center=center,
        )

        chroma = chroma.astype(
            np.float32,
            copy=False,
        )

        # -----------------------------------------
        # Validation
        # -----------------------------------------

        if log_mel.shape != mel_shape:
            raise RuntimeError(
                f"{track_id}: mel shape "
                f"{log_mel.shape}, expected "
                f"{mel_shape}."
            )

        if chroma.shape != chroma_shape:
            raise RuntimeError(
                f"{track_id}: chroma shape "
                f"{chroma.shape}, expected "
                f"{chroma_shape}."
            )

        if not np.isfinite(
            log_mel
        ).all():
            raise RuntimeError(
                f"{track_id}: non-finite mel."
            )

        if not np.isfinite(
            chroma
        ).all():
            raise RuntimeError(
                f"{track_id}: non-finite chroma."
            )

        save_npy_atomic(
            log_mel,
            mel_path,
        )

        save_npy_atomic(
            chroma,
            chroma_path,
        )

        # Verify written files.
        if not existing_feature_valid(
            mel_path,
            mel_shape,
        ):
            raise RuntimeError(
                f"{track_id}: saved mel "
                "validation failed."
            )

        if not existing_feature_valid(
            chroma_path,
            chroma_shape,
        ):
            raise RuntimeError(
                f"{track_id}: saved chroma "
                "validation failed."
            )

        generated += 1

        records.append({
            "track_id": track_id,
            "mel_path": str(
                mel_path.relative_to(ROOT)
            ),
            "chroma_path": str(
                chroma_path.relative_to(ROOT)
            ),
            "mel_bins": n_mels,
            "mel_frames": fixed_frames,
            "chroma_bins": n_chroma,
            "chroma_frames": fixed_frames,
            "cache_status": "generated",
        })

    manifest = pd.DataFrame(
        records
    )

    if len(manifest) != EXPECTED_TRACKS:
        raise RuntimeError(
            "Feature manifest row-count mismatch."
        )

    if manifest[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate manifest track IDs."
        )

    # Final exhaustive cache check.
    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Validating feature cache",
    ):

        mel_path = (
            ROOT
            / row["mel_path"]
        )

        chroma_path = (
            ROOT
            / row["chroma_path"]
        )

        if not existing_feature_valid(
            mel_path,
            mel_shape,
        ):
            raise RuntimeError(
                f"Invalid cached mel: "
                f"{row['track_id']}"
            )

        if not existing_feature_valid(
            chroma_path,
            chroma_shape,
        ):
            raise RuntimeError(
                f"Invalid cached chroma: "
                f"{row['track_id']}"
            )

    manifest = manifest.sort_values(
        "track_id"
    ).reset_index(drop=True)

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_parquet(
        MANIFEST_PATH,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_CSV_PATH,
        index=False,
    )

    report = {
        "tracks": EXPECTED_TRACKS,
        "generated": generated,
        "reused": reused,
        "sample_rate": sr,
        "fixed_samples": fixed_samples,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "n_chroma": n_chroma,
        "fixed_frames": fixed_frames,
        "mel_shape": [
            n_mels,
            fixed_frames,
        ],
        "chroma_shape": [
            n_chroma,
            fixed_frames,
        ],
        "mel_dtype": "float32",
        "chroma_dtype": "float32",
        "mel_normalization": (
            "per-track z-score"
        ),
        "chroma_normalization": "native",
        "length_actions_for_generated": {
            str(k): int(v)
            for k, v
            in actions.items()
        },
        "cache_validation": "PASS",
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "MusicCaps audio feature cache complete"
    )
    print("=" * 56)

    print(
        f"Tracks       : {EXPECTED_TRACKS}"
    )

    print(
        f"Generated    : {generated}"
    )

    print(
        f"Reused       : {reused}"
    )

    print(
        f"Mel shape    : {mel_shape}"
    )

    print(
        f"Chroma shape : {chroma_shape}"
    )

    print(
        "Cache validation: PASS"
    )

    print()
    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"Report  : {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()