from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

TRAIN_SPLIT_PATH = (
    ROOT
    / "data"
    / "splits"
    / "train.json"
)

FEATURE_MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "audio_feature_manifest.parquet"
)

DETAIL_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_threshold_validation_100.csv"
)

SUMMARY_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_threshold_validation_100_summary.json"
)


SAMPLE_SIZE = 100
SAMPLE_SEED = 42

THRESHOLDS = [
    0.90,
    0.92,
    0.94,
    0.95,
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
) -> None:

    if i == j:
        return

    edges.add((i, j))
    edges.add((j, i))


def main() -> None:

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

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
        graph["sanity"][
            "max_density_ratio"
        ]
    )

    # --------------------------------------------------
    # Frozen TRAIN split
    # --------------------------------------------------

    with TRAIN_SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    train_ids = sorted(
        str(x)
        for x in split["track_ids"]
    )

    if len(train_ids) != 4112:
        raise RuntimeError(
            f"Expected 4112 training tracks, "
            f"found {len(train_ids)}."
        )

    # Deterministic 100-track sample.
    rng = np.random.default_rng(
        SAMPLE_SEED
    )

    sample_ids = (
        rng.choice(
            np.array(train_ids),
            size=SAMPLE_SIZE,
            replace=False,
        )
        .tolist()
    )

    # --------------------------------------------------
    # Feature manifest
    # --------------------------------------------------

    manifest = pd.read_parquet(
        FEATURE_MANIFEST_PATH
    )

    if manifest[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate feature manifest IDs."
        )

    manifest = manifest.set_index(
        "track_id"
    )

    # --------------------------------------------------
    # Segment geometry
    # --------------------------------------------------

    segment_starts = np.arange(
        0.0,
        duration - window + 1e-9,
        stride,
    )

    if len(segment_starts) != expected_nodes:
        raise RuntimeError(
            f"Expected {expected_nodes} nodes, "
            f"got {len(segment_starts)}."
        )

    frame_times = (
        np.arange(fixed_frames)
        * hop_length
        / sr
    )

    max_directed_edges = (
        expected_nodes
        * (expected_nodes - 1)
    )

    expected_g1_edges = (
        2 * (expected_nodes - 1)
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    records = []

    for track_id in tqdm(
        sample_ids,
        desc="Validating G2 thresholds",
    ):

        if track_id not in manifest.index:
            raise RuntimeError(
                f"Missing feature row: "
                f"{track_id}"
            )

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
                f"{track_id}: bad chroma "
                f"shape {chroma.shape}."
            )

        # ----------------------------------------------
        # 9 node-level mean chroma vectors
        # ----------------------------------------------

        node_chroma = []

        for start in segment_starts:

            end = start + window

            mask = (
                (frame_times >= start)
                & (frame_times < end)
            )

            if not mask.any():
                raise RuntimeError(
                    f"{track_id}: empty segment."
                )

            vector = (
                chroma[:, mask]
                .mean(axis=1)
                .astype(np.float32)
            )

            node_chroma.append(
                vector
            )

        node_chroma = np.stack(
            node_chroma
        )

        # ----------------------------------------------
        # Calculate pairwise similarities once
        # ----------------------------------------------

        similarities = {}

        for i in range(
            expected_nodes
        ):
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

        # ----------------------------------------------
        # Test thresholds
        # ----------------------------------------------

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

            if len(g1_edges) != (
                expected_g1_edges
            ):
                raise RuntimeError(
                    "Unexpected G1 edge count."
                )

            g2_edges = set(
                g1_edges
            )

            extra_pairs = 0

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

                    if len(
                        g2_edges
                    ) > before:
                        extra_pairs += 1

            density = (
                len(g2_edges)
                / max_directed_edges
            )

            records.append({
                "track_id": track_id,
                "threshold": threshold,

                "g1_directed_edges": (
                    len(g1_edges)
                ),

                "g2_directed_edges": (
                    len(g2_edges)
                ),

                "extra_similarity_pairs": (
                    extra_pairs
                ),

                "g2_density": (
                    density
                ),

                "density_pass": bool(
                    density
                    <= density_limit
                ),

                "has_extra_edges": bool(
                    extra_pairs > 0
                ),

                "complete_graph": bool(
                    len(g2_edges)
                    == max_directed_edges
                ),
            })

    results = pd.DataFrame(
        records
    )

    DETAIL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Threshold summary
    # --------------------------------------------------

    summary_rows = []

    for threshold in THRESHOLDS:

        subset = results.loc[
            results["threshold"]
            == threshold
        ].copy()

        densities = (
            subset[
                "g2_density"
            ]
            .to_numpy()
        )

        extra = (
            subset[
                "extra_similarity_pairs"
            ]
            .to_numpy()
        )

        summary_rows.append({
            "threshold": threshold,

            "mean_density": float(
                densities.mean()
            ),

            "median_density": float(
                np.median(
                    densities
                )
            ),

            "p90_density": float(
                np.percentile(
                    densities,
                    90,
                )
            ),

            "p95_density": float(
                np.percentile(
                    densities,
                    95,
                )
            ),

            "p99_density": float(
                np.percentile(
                    densities,
                    99,
                )
            ),

            "max_density": float(
                densities.max()
            ),

            "tracks_over_density_limit": int(
                (
                    densities
                    > density_limit
                ).sum()
            ),

            "complete_graphs": int(
                subset[
                    "complete_graph"
                ].sum()
            ),

            "tracks_with_extra_edges": int(
                subset[
                    "has_extra_edges"
                ].sum()
            ),

            "tracks_with_extra_edges_percent":
                float(
                    subset[
                        "has_extra_edges"
                    ].mean()
                    * 100.0
                ),

            "mean_extra_pairs": float(
                extra.mean()
            ),

            "median_extra_pairs": float(
                np.median(
                    extra
                )
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------
    # Final candidate rule
    #
    # Choose LOWEST threshold where:
    #   1. no track exceeds density 0.90
    #   2. no complete graph
    #   3. >= 90% of tracks retain extra G2 edges
    # --------------------------------------------------

    candidates = summary_df.loc[
        (
            summary_df[
                "tracks_over_density_limit"
            ] == 0
        )
        &
        (
            summary_df[
                "complete_graphs"
            ] == 0
        )
        &
        (
            summary_df[
                "tracks_with_extra_edges_percent"
            ] >= 90.0
        )
    ]

    if len(candidates):
        recommended = float(
            candidates.iloc[0][
                "threshold"
            ]
        )
    else:
        recommended = None

    output_summary = {
        "sample_source": (
            "train_only"
        ),

        "sample_size": (
            SAMPLE_SIZE
        ),

        "sample_seed": (
            SAMPLE_SEED
        ),

        "thresholds": (
            THRESHOLDS
        ),

        "density_limit": (
            density_limit
        ),

        "selection_rule": (
            "Lowest threshold with zero "
            "density-limit violations, "
            "zero complete graphs, and "
            "at least 90% of tracks "
            "retaining non-temporal "
            "similarity edges."
        ),

        "recommended_threshold": (
            recommended
        ),

        "summary": (
            summary_rows
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            output_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Console
    # --------------------------------------------------

    print()
    print(
        "100-track G2 threshold validation"
    )
    print("=" * 95)

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "mean_density":
                    lambda x: f"{x:.3f}",

                "median_density":
                    lambda x: f"{x:.3f}",

                "p90_density":
                    lambda x: f"{x:.3f}",

                "p95_density":
                    lambda x: f"{x:.3f}",

                "p99_density":
                    lambda x: f"{x:.3f}",

                "max_density":
                    lambda x: f"{x:.3f}",

                "tracks_with_extra_edges_percent":
                    lambda x: f"{x:.1f}%",

                "mean_extra_pairs":
                    lambda x: f"{x:.2f}",

                "median_extra_pairs":
                    lambda x: f"{x:.2f}",
            },
        )
    )

    print()

    if recommended is None:

        print(
            "FINAL THRESHOLD CANDIDATE: NONE"
        )

        print(
            "Do not modify config.yaml."
        )

    else:

        print(
            "FINAL THRESHOLD CANDIDATE: "
            f">{recommended:.2f}"
        )

    print()

    print(
        f"Details : {DETAIL_PATH}"
    )

    print(
        f"Summary : {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()