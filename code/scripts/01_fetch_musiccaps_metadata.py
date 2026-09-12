from pathlib import Path
import json

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data" / "raw" / "musiccaps" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = OUT_DIR / "musiccaps_metadata.parquet"
CSV_PATH = OUT_DIR / "musiccaps_metadata.csv"
REPORT_PATH = OUT_DIR / "metadata_report.json"


def main():
    print("Loading official Google MusicCaps metadata...")

    dataset = load_dataset(
        "google/MusicCaps",
        split="train"
    )

    df = dataset.to_pandas()

    required_columns = [
        "ytid",
        "start_s",
        "end_s",
        "caption",
        "aspect_list",
        "author_id",
        "is_balanced_subset",
        "is_audioset_eval",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise RuntimeError(
            f"MusicCaps schema changed. Missing columns: {missing}"
        )

    # Preserve original YouTube ID.
    df["source_id"] = df["ytid"].astype(str)

    # Deterministic internal track identifier.
    df["track_id"] = (
        "mc_"
        + df["ytid"].astype(str)
        + "_"
        + df["start_s"].astype(int).astype(str)
        + "_"
        + df["end_s"].astype(int).astype(str)
    )

    df["clip_duration"] = (
        df["end_s"].astype(float)
        - df["start_s"].astype(float)
    )

    if df["track_id"].duplicated().any():
        duplicates = df.loc[
            df["track_id"].duplicated(keep=False),
            ["track_id", "ytid", "start_s", "end_s"]
        ]

        raise RuntimeError(
            "Duplicate track IDs detected:\n"
            + duplicates.to_string(index=False)
        )

    empty_caption = (
        df["caption"].isna()
        | df["caption"].astype(str).str.strip().eq("")
    )

    empty_aspects = (
        df["aspect_list"].isna()
        | df["aspect_list"].astype(str).str.strip().eq("")
    )

    report = {
        "rows": int(len(df)),
        "unique_track_ids": int(df["track_id"].nunique()),
        "unique_source_ids": int(df["source_id"].nunique()),
        "missing_captions": int(empty_caption.sum()),
        "missing_aspect_lists": int(empty_aspects.sum()),
        "duration_min": float(df["clip_duration"].min()),
        "duration_max": float(df["clip_duration"].max()),
        "duration_mean": float(df["clip_duration"].mean()),
    }

    print("\nMetadata audit:")
    for key, value in report.items():
        print(f"{key}: {value}")

    df.to_parquet(PARQUET_PATH, index=False)
    df.to_csv(CSV_PATH, index=False)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(PARQUET_PATH)
    print(CSV_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()