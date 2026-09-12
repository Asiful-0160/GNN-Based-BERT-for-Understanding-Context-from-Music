from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "config.yaml"
)

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

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_sanity_report.json"
)

DETAIL_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_sanity_details.csv"
)


EXPECTED_NODE_DIM = 140


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


def add_bidirectional_edge(
    edges: set[tuple[int, int]],
    i: int,
    j: int,
):
    if i == j:
        return

    edges.add(
        (i, j)
    )

    edges.add(
        (j, i)
    )


def main():

    # --------------------------------------------------
    # Config
    # --------------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    audio = config[
        "audio"
    ]

    graph = config[
        "graph"
    ]

    sr = int(
        audio[
            "sample_rate"
        ]
    )

    hop_length = int(
        audio[
            "hop_length"
        ]
    )

    fixed_frames = int(
        audio[
            "fixed_mel_frames"
        ]
    )

    n_mels = int(
        audio[
            "n_mels"
        ]
    )

    n_chroma = int(
        audio[
            "n_chroma"
        ]
    )

    window_seconds = float(
        graph[
            "segment_window_seconds"
        ]
    )

    stride_seconds = float(
        graph[
            "segment_stride_seconds"
        ]
    )

    expected_nodes = int(
        graph[
            "expected_nodes_per_track"
        ]
    )

    similarity_threshold = float(
        graph[
            "similarity_threshold"
        ]
    )

    sample_tracks = int(
        graph[
            "sanity"
        ][
            "sample_tracks"
        ]
    )

    max_density_ratio = float(
        graph[
            "sanity"
        ][
            "max_density_ratio"
        ]
    )

    if not graph[
        "temporal_edges_bidirectional"
    ]:
        raise RuntimeError(
            "Expected bidirectional temporal edges."
        )

    if not graph[
        "similarity_edges_bidirectional"
    ]:
        raise RuntimeError(
            "Expected bidirectional similarity edges."
        )

    if (
        n_mels + n_chroma
        != EXPECTED_NODE_DIM
    ):
        raise RuntimeError(
            "Expected node input dimension 140."
        )

    if fixed_frames != 431:
        raise RuntimeError(
            "Expected 431 cached frames."
        )

    # --------------------------------------------------
    # Load frozen TRAIN IDs
    # --------------------------------------------------

    with TRAIN_SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        train_split = json.load(f)

    train_ids = [
        str(x)
        for x in train_split[
            "track_ids"
        ]
    ]

    # Deterministic sanity sample.
    rng = np.random.default_rng(
        42
    )

    sample_ids = (
        rng.choice(
            np.array(
                sorted(train_ids)
            ),
            size=sample_tracks,
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

    manifest = (
        manifest
        .set_index(
            "track_id"
        )
    )

    # --------------------------------------------------
    # Segment starts
    #
    # 10 s, 2 s window, 1 s stride:
    # 0-2, 1-3, ..., 8-10 = 9 nodes
    # --------------------------------------------------

    duration_seconds = float(
        audio[
            "duration_seconds"
        ]
    )

    segment_starts = np.arange(
        0.0,
        duration_seconds
        - window_seconds
        + 1e-9,
        stride_seconds,
    )

    if (
        len(segment_starts)
        != expected_nodes
    ):
        raise RuntimeError(
            f"Expected {expected_nodes} nodes, "
            f"derived {len(segment_starts)}."
        )

    # Frame-center times.
    frame_times = (
        np.arange(
            fixed_frames
        )
        * hop_length
        / sr
    )

    detail_rows = []

    track_reports = []

    # --------------------------------------------------
    # 5-track sanity
    # --------------------------------------------------

    for track_id in sample_ids:

        if track_id not in manifest.index:
            raise RuntimeError(
                f"Missing feature manifest row: "
                f"{track_id}"
            )

        row = manifest.loc[
            track_id
        ]

        mel = np.load(
            ROOT
            / str(
                row[
                    "mel_path"
                ]
            ),
            allow_pickle=False,
        )

        chroma = np.load(
            ROOT
            / str(
                row[
                    "chroma_path"
                ]
            ),
            allow_pickle=False,
        )

        if mel.shape != (
            n_mels,
            fixed_frames,
        ):
            raise RuntimeError(
                f"{track_id}: bad mel shape "
                f"{mel.shape}"
            )

        if chroma.shape != (
            n_chroma,
            fixed_frames,
        ):
            raise RuntimeError(
                f"{track_id}: bad chroma shape "
                f"{chroma.shape}"
            )

        node_features = []
        node_chroma = []
        node_frame_counts = []

        # ----------------------------------------------
        # Node features
        # 128 mean mel + 12 mean chroma = 140
        # ----------------------------------------------

        for start in segment_starts:

            end = (
                start
                + window_seconds
            )

            mask = (
                (frame_times >= start)
                & (frame_times < end)
            )

            frame_count = int(
                mask.sum()
            )

            if frame_count == 0:
                raise RuntimeError(
                    f"{track_id}: empty segment "
                    f"{start:.1f}-{end:.1f}s"
                )

            mel_mean = (
                mel[
                    :,
                    mask,
                ]
                .mean(
                    axis=1
                )
            )

            chroma_mean = (
                chroma[
                    :,
                    mask,
                ]
                .mean(
                    axis=1
                )
            )

            feature = np.concatenate(
                [
                    mel_mean,
                    chroma_mean,
                ]
            ).astype(
                np.float32
            )

            if feature.shape != (
                EXPECTED_NODE_DIM,
            ):
                raise RuntimeError(
                    f"{track_id}: "
                    "node feature dimension "
                    "is not 140."
                )

            if not np.isfinite(
                feature
            ).all():
                raise RuntimeError(
                    f"{track_id}: non-finite "
                    "node feature."
                )

            node_features.append(
                feature
            )

            node_chroma.append(
                chroma_mean.astype(
                    np.float32
                )
            )

            node_frame_counts.append(
                frame_count
            )

        x = np.stack(
            node_features,
            axis=0,
        )

        node_chroma = np.stack(
            node_chroma,
            axis=0,
        )

        if x.shape != (
            expected_nodes,
            EXPECTED_NODE_DIM,
        ):
            raise RuntimeError(
                f"{track_id}: node matrix "
                f"{x.shape}"
            )

        # ----------------------------------------------
        # G1: temporal edges
        # ----------------------------------------------

        g1_edges = set()

        for node in range(
            expected_nodes - 1
        ):
            add_bidirectional_edge(
                g1_edges,
                node,
                node + 1,
            )

        expected_g1_edges = (
            2
            * (
                expected_nodes - 1
            )
        )

        if (
            len(g1_edges)
            != expected_g1_edges
        ):
            raise RuntimeError(
                f"{track_id}: expected "
                f"{expected_g1_edges} G1 edges, "
                f"got {len(g1_edges)}."
            )



        # ----------------------------------------------
        # G2:
        # temporal edges
        # +
        # strongest NON-LOCAL chroma-similarity edges
        # subject to a per-node extra-degree cap
        # ----------------------------------------------

        g2_edges = set(
            g1_edges
        )

        similarity_pairs_added = 0

        similarity_values = []

        extra_degree_cap = int(
            graph[
                "similarity_extra_degree_cap"
            ]
        )

        nonlocal_only = bool(
            graph[
                "similarity_nonlocal_only"
            ]
        )

        selection = str(
            graph[
                "similarity_selection"
            ]
        )

        if selection != "strongest_first":
            raise RuntimeError(
                "Expected similarity_selection="
                "'strongest_first'."
            )

        candidates = []

        for i in range(
            expected_nodes
        ):

            for j in range(
                i + 1,
                expected_nodes,
            ):

                similarity = (
                    cosine_similarity(
                        node_chroma[i],
                        node_chroma[j],
                    )
                )

                similarity_values.append(
                    similarity
                )

                # Temporal neighbors are already
                # represented by G1.
                if (
                    nonlocal_only
                    and abs(i - j) <= 1
                ):
                    continue

                if (
                    similarity
                    > similarity_threshold
                ):
                    candidates.append(
                        (
                            similarity,
                            i,
                            j,
                        )
            )

        # Highest similarity first.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        extra_degree = [
            0
            for _ in range(
                expected_nodes
            )
        ]

        selected_similarities = []

        for (
            similarity,
            i,
            j,
        ) in candidates:

            if (
                extra_degree[i]
                >= extra_degree_cap
                or
                extra_degree[j]
                >= extra_degree_cap
            ):
                continue

            before = len(
                g2_edges
            )

            add_bidirectional_edge(
                g2_edges,
                i,
                j,
            )

            after = len(
                g2_edges
            )

            if after > before:

                extra_degree[i] += 1
                extra_degree[j] += 1

                similarity_pairs_added += 1

                selected_similarities.append(
                    similarity
                )

        # Directed graph, no self loops.
        max_directed_edges = (
            expected_nodes
            * (
                expected_nodes - 1
            )
        )

        density = (
            len(g2_edges)
            / max_directed_edges
        )

        density_ok = (
            density
            <= max_density_ratio
        )

        if not density_ok:
            print(
                f"WARNING: {track_id}: "
                f"G2 too dense "
                f"({density:.3f} > "
                f"{max_density_ratio:.3f})"
            )

        if len(g2_edges) < len(
            g1_edges
        ):
            raise RuntimeError(
                "G2 cannot contain fewer "
                "edges than G1."
            )

        similarities = np.array(
            similarity_values,
            dtype=np.float32,
        )

        detail_rows.append({
            "track_id": track_id,

            "nodes": expected_nodes,

            "node_feature_dim": (
                EXPECTED_NODE_DIM
            ),

            "min_frames_per_node": int(
                min(
                    node_frame_counts
                )
            ),

            "max_frames_per_node": int(
                max(
                    node_frame_counts
                )
            ),

            "g1_directed_edges": int(
                len(g1_edges)
            ),

            "g2_directed_edges": int(
                len(g2_edges)
            ),

            "similarity_pairs_added": int(
                similarity_pairs_added
            ),

            "g2_density": float(
                density
            ),

            "mean_pairwise_chroma_similarity":
                float(
                    similarities.mean()
                ),

            "max_pairwise_chroma_similarity":
                float(
                    similarities.max()
                ),

            "density_ok": bool(
                density_ok
            ),

            "eligible_similarity_pairs": int(
                len(candidates)
            ),

            "max_extra_similarity_degree": int(
                max(extra_degree)
            ),

            "extra_degree_cap": int(
                extra_degree_cap
            ),
        })

        track_reports.append({
            "track_id": track_id,

            "node_shape": [
                int(x.shape[0]),
                int(x.shape[1]),
            ],

            "node_frame_counts": [
                int(v)
                for v
                in node_frame_counts
            ],

            "g1_directed_edges": int(
                len(g1_edges)
            ),

            "g2_directed_edges": int(
                len(g2_edges)
            ),

            "similarity_pairs_added": int(
                similarity_pairs_added
            ),

            "g2_density": float(
                density
            ),
        })

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    details = pd.DataFrame(
        detail_rows
    )

    all_pass = bool(
        details[
            "density_ok"
        ].all()
    )

    report = {
        "sample_source": (
            "train_only"
        ),

        "sample_seed": 42,

        "tracks_checked": int(
            sample_tracks
        ),

        "window_seconds": (
            window_seconds
        ),

        "stride_seconds": (
            stride_seconds
        ),

        "nodes_per_track": (
            expected_nodes
        ),

        "node_input_dim": (
            EXPECTED_NODE_DIM
        ),

        "temporal_edges_bidirectional": (
            True
        ),

        "similarity_edges_bidirectional": (
            True
        ),

        "similarity_threshold": (
            similarity_threshold
        ),

        "max_allowed_density": (
            max_density_ratio
        ),

        "all_tracks_passed": (
            all_pass
        ),

        "tracks": (
            track_reports
        ),

        "similarity_nonlocal_only": (
            nonlocal_only
        ),

        "similarity_selection": (
            selection
        ),

        "similarity_extra_degree_cap": (
            extra_degree_cap
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    details.to_csv(
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

    # --------------------------------------------------
    # Console
    # --------------------------------------------------

    print()
    print(
        "MusicCaps G1/G2 graph sanity check"
    )
    print("=" * 70)

    print(
        f"Tracks checked       : "
        f"{sample_tracks}"
    )

    print(
        f"Nodes per track      : "
        f"{expected_nodes}"
    )

    print(
        f"Node feature shape   : "
        f"({expected_nodes}, "
        f"{EXPECTED_NODE_DIM})"
    )

    print(
        f"G1 temporal edges    : "
        f"{2 * (expected_nodes - 1)}"
    )

    print(
        f"G2 threshold         : "
        f">{similarity_threshold}"
    )

    print()

    print(
        details[
            [
                "track_id",
                "nodes",
                "g1_directed_edges",
                "g2_directed_edges",
                "similarity_pairs_added",
                "g2_density",
                "min_frames_per_node",
                "max_frames_per_node",
            ]
        ].to_string(
            index=False,
            formatters={
                "g2_density":
                    lambda x:
                    f"{x:.3f}"
            },
        )
    )

    print()
    print(
        f"Density gate "
        f"(<= {max_density_ratio:.2f}): "
        f"{'PASS' if all_pass else 'FAIL'}"
    )

    print()
    print(
        f"Report : {REPORT_PATH}"
    )

    print(
        f"Details: {DETAIL_PATH}"
    )

    print()

    if all_pass:
        print(
            "GRAPH SANITY GATE: PASS"
        )

        print(
            "Safe to proceed to "
            "full graph construction."
        )

    else:
        raise RuntimeError(
            "Graph sanity gate failed."
        )


if __name__ == "__main__":
    main()