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
    / "sparse_g2_validation_100.csv"
)

SUMMARY_PATH = (
    ROOT
    / "data"
    / "audit"
    / "sparse_g2_validation_100_summary.json"
)


SAMPLE_SIZE = 100
SAMPLE_SEED = 42

SIMILARITY_THRESHOLD = 0.85

EXTRA_DEGREE_CAPS = [
    1,
    2,
    3,
]


def cosine_similarity(a, b):

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
    edges,
    i,
    j,
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
        graph[
            "sanity"
        ][
            "max_density_ratio"
        ]
    )

    # --------------------------------------------------
    # Deterministic 100-track TRAIN sample
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
            "Expected 4112 train tracks."
        )

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

    manifest = pd.read_parquet(
        FEATURE_MANIFEST_PATH
    )

    if manifest[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate feature IDs."
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
            "Unexpected node count."
        )

    frame_times = (
        np.arange(
            fixed_frames
        )
        * hop_length
        / sr
    )

    max_directed_edges = (
        expected_nodes
        * (
            expected_nodes - 1
        )
    )

    expected_g1_edges = (
        2
        * (
            expected_nodes - 1
        )
    )

    records = []

    # --------------------------------------------------
    # Tracks
    # --------------------------------------------------

    for track_id in tqdm(
        sample_ids,
        desc="Testing sparse G2",
    ):

        row = manifest.loc[
            track_id
        ]

        chroma = np.load(
            ROOT
            / str(
                row["chroma_path"]
            ),
            allow_pickle=False,
        )

        if chroma.shape != (
            n_chroma,
            fixed_frames,
        ):
            raise RuntimeError(
                f"{track_id}: invalid chroma."
            )

        node_chroma = []

        for start in segment_starts:

            end = (
                start + window
            )

            mask = (
                (frame_times >= start)
                & (frame_times < end)
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

        # --------------------------------------------------
        # G1 temporal edges
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Eligible NON-LOCAL similarity pairs
        #
        # Adjacent nodes are already represented in G1.
        # --------------------------------------------------

        candidates = []

        for i in range(
            expected_nodes
        ):

            for j in range(
                i + 1,
                expected_nodes,
            ):

                # Exclude temporal neighbors.
                if abs(i - j) <= 1:
                    continue

                similarity = (
                    cosine_similarity(
                        node_chroma[i],
                        node_chroma[j],
                    )
                )

                if (
                    similarity
                    > SIMILARITY_THRESHOLD
                ):
                    candidates.append(
                        (
                            similarity,
                            i,
                            j,
                        )
                    )

        # Strongest first.
        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # --------------------------------------------------
        # Evaluate each extra-degree cap
        # --------------------------------------------------

        for cap in EXTRA_DEGREE_CAPS:

            g2_edges = set(
                g1_edges
            )

            extra_degree = [
                0
                for _ in range(
                    expected_nodes
                )
            ]

            added_pairs = 0

            selected_similarities = []

            for (
                similarity,
                i,
                j,
            ) in candidates:

                if (
                    extra_degree[i] >= cap
                    or
                    extra_degree[j] >= cap
                ):
                    continue

                add_bidirectional(
                    g2_edges,
                    i,
                    j,
                )

                extra_degree[i] += 1
                extra_degree[j] += 1

                added_pairs += 1

                selected_similarities.append(
                    similarity
                )

            density = (
                len(g2_edges)
                / max_directed_edges
            )

            records.append({
                "track_id": track_id,

                "threshold": (
                    SIMILARITY_THRESHOLD
                ),

                "extra_degree_cap": (
                    cap
                ),

                "eligible_nonlocal_pairs": (
                    len(candidates)
                ),

                "selected_extra_pairs": (
                    added_pairs
                ),

                "g1_directed_edges": (
                    len(g1_edges)
                ),

                "g2_directed_edges": (
                    len(g2_edges)
                ),

                "g2_density": (
                    density
                ),

                "has_extra_edges": bool(
                    added_pairs > 0
                ),

                "mean_selected_similarity": (
                    float(
                        np.mean(
                            selected_similarities
                        )
                    )
                    if selected_similarities
                    else None
                ),

                "max_extra_degree": int(
                    max(extra_degree)
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
    # Summary
    # --------------------------------------------------

    summary_rows = []

    for cap in EXTRA_DEGREE_CAPS:

        subset = results.loc[
            results[
                "extra_degree_cap"
            ] == cap
        ]

        density = subset[
            "g2_density"
        ].to_numpy()

        added = subset[
            "selected_extra_pairs"
        ].to_numpy()

        summary_rows.append({
            "extra_degree_cap": (
                cap
            ),

            "mean_density": float(
                density.mean()
            ),

            "median_density": float(
                np.median(density)
            ),

            "p95_density": float(
                np.percentile(
                    density,
                    95,
                )
            ),

            "max_density": float(
                density.max()
            ),

            "tracks_over_density_limit": int(
                (
                    density
                    > density_limit
                ).sum()
            ),

            "tracks_with_extra_edges": int(
                (
                    added > 0
                ).sum()
            ),

            "tracks_with_extra_edges_percent":
                float(
                    (
                        added > 0
                    ).mean()
                    * 100.0
                ),

            "mean_extra_pairs": float(
                added.mean()
            ),

            "median_extra_pairs": float(
                np.median(
                    added
                )
            ),

            "mean_eligible_pairs": float(
                subset[
                    "eligible_nonlocal_pairs"
                ].mean()
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------
    # Selection rule
    #
    # Prefer smallest cap that:
    #  - gives zero density violations
    #  - retains G2 additions in >=90% of tracks
    #  - averages >=3 extra non-local pairs
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
                "tracks_with_extra_edges_percent"
            ] >= 90.0
        )
        &
        (
            summary_df[
                "mean_extra_pairs"
            ] >= 3.0
        )
    ]

    recommended_cap = None

    if not candidates.empty:
        recommended_cap = int(
            candidates.iloc[0][
                "extra_degree_cap"
            ]
        )

    summary = {
        "sample_source": "train_only",
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,

        "similarity_threshold": (
            SIMILARITY_THRESHOLD
        ),

        "candidate_rule": (
            "non-local node pairs only; "
            "cosine similarity > 0.85; "
            "strongest pairs selected greedily "
            "subject to per-node extra-degree cap"
        ),

        "density_limit": (
            density_limit
        ),

        "recommended_extra_degree_cap": (
            recommended_cap
        ),

        "summary": summary_rows,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Console
    # --------------------------------------------------

    print()
    print(
        "100-track sparse G2 validation"
    )
    print("=" * 92)

    print(
        f"Similarity threshold: "
        f">{SIMILARITY_THRESHOLD}"
    )

    print()

    print(
        summary_df.to_string(
            index=False,
            formatters={
                "mean_density":
                    lambda x:
                    f"{x:.3f}",

                "median_density":
                    lambda x:
                    f"{x:.3f}",

                "p95_density":
                    lambda x:
                    f"{x:.3f}",

                "max_density":
                    lambda x:
                    f"{x:.3f}",

                "tracks_with_extra_edges_percent":
                    lambda x:
                    f"{x:.1f}%",

                "mean_extra_pairs":
                    lambda x:
                    f"{x:.2f}",

                "median_extra_pairs":
                    lambda x:
                    f"{x:.2f}",

                "mean_eligible_pairs":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    print()

    if recommended_cap is None:

        print(
            "RECOMMENDED EXTRA DEGREE CAP: NONE"
        )

    else:

        print(
            "RECOMMENDED EXTRA DEGREE CAP: "
            f"{recommended_cap}"
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