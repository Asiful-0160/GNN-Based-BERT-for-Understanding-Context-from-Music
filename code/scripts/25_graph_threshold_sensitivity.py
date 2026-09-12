from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

FEATURE_MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "audio_feature_manifest.parquet"
)

SANITY_REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_sanity_report.json"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_threshold_sensitivity.csv"
)

SUMMARY_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_threshold_sensitivity_summary.json"
)


THRESHOLDS = [
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def add_bidirectional(
    edges: set[tuple[int, int]],
    i: int,
    j: int,
):

    if i == j:
        return

    edges.add((i, j))
    edges.add((j, i))


def main():

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    audio = config["audio"]
    graph = config["graph"]

    sr = int(
        audio["sample_rate"]
    )

    hop_length = int(
        audio["hop_length"]
    )

    fixed_frames = int(
        audio["fixed_mel_frames"]
    )

    n_chroma = int(
        audio["n_chroma"]
    )

    duration = float(
        audio["duration_seconds"]
    )

    window = float(
        graph["segment_window_seconds"]
    )

    stride = float(
        graph["segment_stride_seconds"]
    )

    expected_nodes = int(
        graph["expected_nodes_per_track"]
    )

    density_limit = float(
        graph["sanity"]["max_density_ratio"]
    )

    # --------------------------------------------
    # Reuse exactly the same five sanity tracks.
    # --------------------------------------------

    with SANITY_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        sanity = json.load(f)

    track_ids = [
        str(item["track_id"])
        for item in sanity["tracks"]
    ]

    if len(track_ids) != 5:
        raise RuntimeError(
            f"Expected 5 sanity tracks, "
            f"found {len(track_ids)}."
        )

    manifest = pd.read_parquet(
        FEATURE_MANIFEST_PATH
    )

    manifest = manifest.set_index(
        "track_id"
    )

    segment_starts = np.arange(
        0.0,
        duration - window + 1e-9,
        stride,
    )

    if len(segment_starts) != expected_nodes:
        raise RuntimeError(
            "Unexpected node count."
        )

    frame_times = (
        np.arange(fixed_frames)
        * hop_length
        / sr
    )

    rows = []

    # --------------------------------------------
    # Evaluate each track.
    # --------------------------------------------

    for track_id in track_ids:

        row = manifest.loc[
            track_id
        ]

        chroma = np.load(
            ROOT
            / str(row["chroma_path"]),
            allow_pickle=False,
        )

        if chroma.shape != (
            n_chroma,
            fixed_frames,
        ):
            raise RuntimeError(
                f"{track_id}: invalid "
                f"chroma shape {chroma.shape}."
            )

        node_chroma = []

        for start in segment_starts:

            end = start + window

            mask = (
                (frame_times >= start)
                & (frame_times < end)
            )

            if not mask.any():
                raise RuntimeError(
                    f"{track_id}: empty node."
                )

            node_chroma.append(
                chroma[
                    :,
                    mask,
                ]
                .mean(axis=1)
                .astype(np.float32)
            )

        node_chroma = np.stack(
            node_chroma
        )

        # Calculate pairwise similarities once.
        similarities = {}

        for i in range(expected_nodes):

            for j in range(
                i + 1,
                expected_nodes,
            ):

                similarities[
                    (i, j)
                ] = cosine_similarity(
                    node_chroma[i],
                    node_chroma[j],
                )

        for threshold in THRESHOLDS:

            g1_edges = set()

            for i in range(
                expected_nodes - 1
            ):
                add_bidirectional(
                    g1_edges,
                    i,
                    i + 1,
                )

            g2_edges = set(
                g1_edges
            )

            added_pairs = 0

            for (
                i,
                j,
            ), similarity in (
                similarities.items()
            ):

                if similarity > threshold:

                    before = len(
                        g2_edges
                    )

                    add_bidirectional(
                        g2_edges,
                        i,
                        j,
                    )

                    if len(g2_edges) > before:
                        added_pairs += 1

            max_edges = (
                expected_nodes
                * (
                    expected_nodes - 1
                )
            )

            density = (
                len(g2_edges)
                / max_edges
            )

            rows.append({
                "track_id": track_id,
                "threshold": threshold,
                "g1_directed_edges": (
                    len(g1_edges)
                ),
                "g2_directed_edges": (
                    len(g2_edges)
                ),
                "similarity_pairs_added": (
                    added_pairs
                ),
                "g2_density": density,
                "density_pass": (
                    density
                    <= density_limit
                ),
            })

    results = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------
    # Threshold-level summary
    # --------------------------------------------

    summary_rows = []

    for threshold in THRESHOLDS:

        subset = results.loc[
            results["threshold"]
            == threshold
        ]

        summary_rows.append({
            "threshold": threshold,

            "mean_density": float(
                subset[
                    "g2_density"
                ].mean()
            ),

            "max_density": float(
                subset[
                    "g2_density"
                ].max()
            ),

            "tracks_over_density_limit": int(
                (
                    subset[
                        "g2_density"
                    ]
                    > density_limit
                ).sum()
            ),

            "tracks_with_extra_edges": int(
                (
                    subset[
                        "similarity_pairs_added"
                    ]
                    > 0
                ).sum()
            ),

            "mean_similarity_pairs_added": float(
                subset[
                    "similarity_pairs_added"
                ].mean()
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # Lowest threshold satisfying:
    # 1. all tracks under density limit
    # 2. similarity graph still adds something
    #    on at least 4 of 5 tracks
    valid = summary_df.loc[
        (
            summary_df[
                "tracks_over_density_limit"
            ] == 0
        )
        &
        (
            summary_df[
                "tracks_with_extra_edges"
            ] >= 4
        )
    ]

    provisional_threshold = None

    if not valid.empty:
        provisional_threshold = float(
            valid.iloc[0][
                "threshold"
            ]
        )

    summary = {
        "tracks_checked": len(
            track_ids
        ),

        "thresholds_tested": (
            THRESHOLDS
        ),

        "density_limit": (
            density_limit
        ),

        "selection_rule": (
            "lowest threshold where all "
            "5 tracks have density <= limit "
            "and at least 4/5 retain one or "
            "more non-temporal similarity pairs"
        ),

        "provisional_threshold": (
            provisional_threshold
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "G2 threshold sensitivity"
    )
    print("=" * 72)

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "mean_density":
                    lambda x:
                    f"{x:.3f}",

                "max_density":
                    lambda x:
                    f"{x:.3f}",

                "mean_similarity_pairs_added":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    print()

    if provisional_threshold is None:

        print(
            "No threshold satisfies the "
            "current selection rule."
        )

    else:

        print(
            "Provisional threshold: "
            f">{provisional_threshold:.2f}"
        )

    print()
    print(
        f"Details : {OUTPUT_PATH}"
    )

    print(
        f"Summary : {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()