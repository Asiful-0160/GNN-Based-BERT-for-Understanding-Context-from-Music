from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "config.yaml"

FEATURE_MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "audio_feature_manifest.parquet"
)

TRAIN_PATH = ROOT / "data" / "splits" / "train.json"
VAL_PATH = ROOT / "data" / "splits" / "val.json"
TEST_PATH = ROOT / "data" / "splits" / "test.json"

GRAPH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "graphs"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "graph_manifest.parquet"
)

MANIFEST_CSV_PATH = (
    ROOT
    / "data"
    / "processed"
    / "graph_manifest.csv"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "audit"
    / "graph_cache_report.json"
)


EXPECTED_TRACKS = 5140
EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_TEST = 514

GRAPH_RULE_VERSION = 2


def load_split(
    path: Path,
    expected: int,
) -> list[str]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    ids = [
        str(x)
        for x in data["track_ids"]
    ]

    if len(ids) != expected:
        raise RuntimeError(
            f"{path.name}: expected "
            f"{expected}, got {len(ids)}."
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{path.name}: duplicate IDs."
        )

    return ids


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


def edge_index_from_set(
    edges: set[tuple[int, int]],
) -> np.ndarray:

    ordered = sorted(edges)

    array = np.asarray(
        ordered,
        dtype=np.int64,
    )

    return array.T


def validate_edge_index(
    edge_index: np.ndarray,
    expected_nodes: int,
) -> set[tuple[int, int]]:

    if edge_index.ndim != 2:
        raise RuntimeError(
            "edge_index must be 2-D."
        )

    if edge_index.shape[0] != 2:
        raise RuntimeError(
            "edge_index first dimension "
            "must be 2."
        )

    edges = {
        (
            int(edge_index[0, i]),
            int(edge_index[1, i]),
        )
        for i in range(
            edge_index.shape[1]
        )
    }

    if len(edges) != edge_index.shape[1]:
        raise RuntimeError(
            "Duplicate directed edges detected."
        )

    for src, dst in edges:

        if src == dst:
            raise RuntimeError(
                "Self-loop detected."
            )

        if not (
            0 <= src < expected_nodes
            and
            0 <= dst < expected_nodes
        ):
            raise RuntimeError(
                "Edge node index out of range."
            )

        if (dst, src) not in edges:
            raise RuntimeError(
                "Graph edge is not bidirectional."
            )

    return edges


def save_graph_atomic(
    path: Path,
    **arrays,
) -> None:

    temp = path.with_suffix(
        ".tmp.npz"
    )

    np.savez_compressed(
        temp,
        **arrays,
    )

    os.replace(
        temp,
        path,
    )


def validate_cached_graph(
    path: Path,
    expected_nodes: int,
    node_dim: int,
    expected_g1_edges: int,
    threshold: float,
    extra_degree_cap: int,
    density_limit: float,
):

    if not path.exists():
        return None

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as cache:

            required = {
                "x",
                "edge_index_g1",
                "edge_index_g2",
                "rule_version",
                "similarity_threshold",
                "extra_degree_cap",
                "nonlocal_only",
            }

            if not required.issubset(
                set(cache.files)
            ):
                return None

            x = cache["x"]

            edge_g1 = cache[
                "edge_index_g1"
            ]

            edge_g2 = cache[
                "edge_index_g2"
            ]

            if x.shape != (
                expected_nodes,
                node_dim,
            ):
                return None

            if x.dtype != np.float32:
                return None

            if not np.isfinite(
                x
            ).all():
                return None

            if int(
                cache[
                    "rule_version"
                ][0]
            ) != GRAPH_RULE_VERSION:
                return None

            if not np.isclose(
                float(
                    cache[
                        "similarity_threshold"
                    ][0]
                ),
                threshold,
            ):
                return None

            if int(
                cache[
                    "extra_degree_cap"
                ][0]
            ) != extra_degree_cap:
                return None

            if not bool(
                cache[
                    "nonlocal_only"
                ][0]
            ):
                return None

            g1_edges = validate_edge_index(
                edge_g1,
                expected_nodes,
            )

            g2_edges = validate_edge_index(
                edge_g2,
                expected_nodes,
            )

            if len(g1_edges) != (
                expected_g1_edges
            ):
                return None

            if not g1_edges.issubset(
                g2_edges
            ):
                return None

            density = (
                len(g2_edges)
                /
                (
                    expected_nodes
                    * (
                        expected_nodes - 1
                    )
                )
            )

            if density > density_limit:
                return None

            extra_directed = (
                len(g2_edges)
                - len(g1_edges)
            )

            if extra_directed < 0:
                return None

            if extra_directed % 2 != 0:
                return None

            extra_pairs = (
                extra_directed // 2
            )

            return {
                "nodes": expected_nodes,
                "node_feature_dim": node_dim,
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
            }

    except Exception:
        return None


def main():

    # --------------------------------------------------
    # Configuration
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

    n_mels = int(
        audio["n_mels"]
    )

    n_chroma = int(
        audio["n_chroma"]
    )

    duration = float(
        audio["duration_seconds"]
    )

    window = float(
        graph[
            "segment_window_seconds"
        ]
    )

    stride = float(
        graph[
            "segment_stride_seconds"
        ]
    )

    expected_nodes = int(
        graph[
            "expected_nodes_per_track"
        ]
    )

    node_dim = int(
        graph[
            "node_input_dim"
        ]
    )

    threshold = float(
        graph[
            "similarity_threshold"
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

    extra_degree_cap = int(
        graph[
            "similarity_extra_degree_cap"
        ]
    )

    density_limit = float(
        graph[
            "sanity"
        ][
            "max_density_ratio"
        ]
    )

    # --------------------------------------------------
    # Enforce reviewed/final graph rule
    # --------------------------------------------------

    if node_dim != (
        n_mels + n_chroma
    ):
        raise RuntimeError(
            "Node dimension must equal "
            "n_mels + n_chroma."
        )

    if expected_nodes != 9:
        raise RuntimeError(
            "Expected 9 nodes."
        )

    if node_dim != 140:
        raise RuntimeError(
            "Expected node dimension 140."
        )

    if threshold != 0.85:
        raise RuntimeError(
            "Reviewed G2 threshold is 0.85."
        )

    if not nonlocal_only:
        raise RuntimeError(
            "G2 must use non-local "
            "similarity pairs only."
        )

    if selection != "strongest_first":
        raise RuntimeError(
            "G2 selection must be "
            "'strongest_first'."
        )

    if extra_degree_cap != 2:
        raise RuntimeError(
            "Reviewed G2 extra-degree "
            "cap is 2."
        )

    # --------------------------------------------------
    # Frozen split verification
    # --------------------------------------------------

    train_ids = load_split(
        TRAIN_PATH,
        EXPECTED_TRAIN,
    )

    val_ids = load_split(
        VAL_PATH,
        EXPECTED_VAL,
    )

    test_ids = load_split(
        TEST_PATH,
        EXPECTED_TEST,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if train_set & val_set:
        raise RuntimeError(
            "Train/val overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test overlap."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Val/test overlap."
        )

    split_lookup = {}

    for track_id in train_ids:
        split_lookup[track_id] = "train"

    for track_id in val_ids:
        split_lookup[track_id] = "val"

    for track_id in test_ids:
        split_lookup[track_id] = "test"

    all_split_ids = (
        train_set
        | val_set
        | test_set
    )

    if len(all_split_ids) != (
        EXPECTED_TRACKS
    ):
        raise RuntimeError(
            "Frozen split union is not 5140."
        )

    # --------------------------------------------------
    # Audio feature manifest
    # --------------------------------------------------

    features = pd.read_parquet(
        FEATURE_MANIFEST_PATH
    )

    if len(features) != (
        EXPECTED_TRACKS
    ):
        raise RuntimeError(
            "Audio feature manifest "
            "must contain 5140 rows."
        )

    if features[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate audio feature IDs."
        )

    feature_ids = set(
        features[
            "track_id"
        ].astype(str)
    )

    if feature_ids != all_split_ids:
        raise RuntimeError(
            "Audio feature IDs do not "
            "exactly match frozen splits."
        )

    features = (
        features
        .assign(
            track_id=lambda d:
                d[
                    "track_id"
                ].astype(str)
        )
        .set_index(
            "track_id"
        )
    )

    # --------------------------------------------------
    # Segment geometry
    # --------------------------------------------------

    segment_starts = np.arange(
        0.0,
        duration
        - window
        + 1e-9,
        stride,
    )

    if len(segment_starts) != (
        expected_nodes
    ):
        raise RuntimeError(
            f"Expected {expected_nodes} "
            f"segment nodes, found "
            f"{len(segment_starts)}."
        )

    frame_times = (
        np.arange(
            fixed_frames
        )
        * hop_length
        / sr
    )

    # --------------------------------------------------
    # Static G1
    # --------------------------------------------------

    g1_edges = set()

    for node in range(
        expected_nodes - 1
    ):
        add_bidirectional(
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

    if len(g1_edges) != (
        expected_g1_edges
    ):
        raise RuntimeError(
            "Unexpected G1 edge count."
        )

    edge_index_g1 = (
        edge_index_from_set(
            g1_edges
        )
    )

    # --------------------------------------------------
    # Theoretical sparse-G2 maximum
    # --------------------------------------------------

    max_extra_directed = (
        expected_nodes
        * extra_degree_cap
    )

    theoretical_max_g2_edges = (
        expected_g1_edges
        + max_extra_directed
    )

    max_possible_directed = (
        expected_nodes
        * (
            expected_nodes - 1
        )
    )

    theoretical_max_density = (
        theoretical_max_g2_edges
        / max_possible_directed
    )

    print()
    print(
        "Building MusicCaps graph cache"
    )
    print("=" * 62)

    print(
        f"Tracks                 : "
        f"{EXPECTED_TRACKS}"
    )

    print(
        f"Nodes / track          : "
        f"{expected_nodes}"
    )

    print(
        f"Node feature dim       : "
        f"{node_dim}"
    )

    print(
        f"G1 directed edges      : "
        f"{expected_g1_edges}"
    )

    print(
        f"G2 similarity threshold: "
        f">{threshold}"
    )

    print(
        f"G2 extra-degree cap    : "
        f"{extra_degree_cap}"
    )

    print(
        f"G2 theoretical max     : "
        f"{theoretical_max_g2_edges} edges "
        f"(density "
        f"{theoretical_max_density:.3f})"
    )

    print()

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    generated = 0
    reused = 0

    # --------------------------------------------------
    # Full graph construction
    # --------------------------------------------------

    for track_id in tqdm(
        sorted(all_split_ids),
        desc="Building graphs",
    ):

        row = features.loc[
            track_id
        ]

        graph_path = (
            GRAPH_DIR
            / f"{track_id}.npz"
        )

        existing = validate_cached_graph(
            graph_path,
            expected_nodes,
            node_dim,
            expected_g1_edges,
            threshold,
            extra_degree_cap,
            density_limit,
        )

        if existing is not None:

            reused += 1

            records.append({
                "track_id": track_id,
                "split": (
                    split_lookup[
                        track_id
                    ]
                ),
                "graph_path": str(
                    graph_path.relative_to(
                        ROOT
                    )
                ),
                **existing,
                "cache_status": "reused",
            })

            continue

        # ----------------------------------------------
        # Load shared audio features
        # ----------------------------------------------

        mel = np.load(
            ROOT
            / str(
                row["mel_path"]
            ),
            allow_pickle=False,
        )

        chroma = np.load(
            ROOT
            / str(
                row["chroma_path"]
            ),
            allow_pickle=False,
        )

        if mel.shape != (
            n_mels,
            fixed_frames,
        ):
            raise RuntimeError(
                f"{track_id}: bad mel "
                f"shape {mel.shape}."
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
        # Build 9 × 140 node feature matrix
        # ----------------------------------------------

        node_features = []
        node_chroma = []

        for start in segment_starts:

            end = start + window

            mask = (
                (frame_times >= start)
                &
                (frame_times < end)
            )

            if not mask.any():
                raise RuntimeError(
                    f"{track_id}: empty "
                    "segment."
                )

            mel_mean = (
                mel[:, mask]
                .mean(axis=1)
            )

            chroma_mean = (
                chroma[:, mask]
                .mean(axis=1)
            )

            feature = np.concatenate(
                [
                    mel_mean,
                    chroma_mean,
                ]
            ).astype(
                np.float32
            )

            node_features.append(
                feature
            )

            node_chroma.append(
                chroma_mean.astype(
                    np.float32
                )
            )

        x = np.stack(
            node_features,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        node_chroma = np.stack(
            node_chroma,
            axis=0,
        )

        if x.shape != (
            expected_nodes,
            node_dim,
        ):
            raise RuntimeError(
                f"{track_id}: bad node "
                f"shape {x.shape}."
            )

        if not np.isfinite(
            x
        ).all():
            raise RuntimeError(
                f"{track_id}: non-finite "
                "node features."
            )

        # ----------------------------------------------
        # Eligible non-local similarity pairs
        # ----------------------------------------------

        candidates = []

        for i in range(
            expected_nodes
        ):

            for j in range(
                i + 1,
                expected_nodes,
            ):

                # Adjacent nodes already
                # represented by G1.
                if abs(i - j) <= 1:
                    continue

                similarity = (
                    cosine_similarity(
                        node_chroma[i],
                        node_chroma[j],
                    )
                )

                if similarity > threshold:

                    candidates.append(
                        (
                            similarity,
                            i,
                            j,
                        )
                    )

        # Deterministic strongest-first ordering.
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            )
        )

        # ----------------------------------------------
        # Sparse G2 construction
        # ----------------------------------------------

        g2_edges = set(
            g1_edges
        )

        extra_degree = [
            0
            for _ in range(
                expected_nodes
            )
        ]

        selected_pairs = 0

        for similarity, i, j in candidates:

            if (
                extra_degree[i]
                >= extra_degree_cap
                or
                extra_degree[j]
                >= extra_degree_cap
            ):
                continue

            add_bidirectional(
                g2_edges,
                i,
                j,
            )

            extra_degree[i] += 1
            extra_degree[j] += 1

            selected_pairs += 1

        edge_index_g2 = (
            edge_index_from_set(
                g2_edges
            )
        )

        # ----------------------------------------------
        # Hard graph validation
        # ----------------------------------------------

        checked_g1 = (
            validate_edge_index(
                edge_index_g1,
                expected_nodes,
            )
        )

        checked_g2 = (
            validate_edge_index(
                edge_index_g2,
                expected_nodes,
            )
        )

        if not checked_g1.issubset(
            checked_g2
        ):
            raise RuntimeError(
                f"{track_id}: G1 is not "
                "a subset of G2."
            )

        if max(extra_degree) > (
            extra_degree_cap
        ):
            raise RuntimeError(
                f"{track_id}: extra-degree "
                "cap violated."
            )

        if len(checked_g2) > (
            theoretical_max_g2_edges
        ):
            raise RuntimeError(
                f"{track_id}: theoretical "
                "G2 edge maximum violated."
            )

        density = (
            len(checked_g2)
            / max_possible_directed
        )

        if density > density_limit:
            raise RuntimeError(
                f"{track_id}: density "
                f"{density:.3f} exceeds "
                f"{density_limit:.3f}."
            )

        # ----------------------------------------------
        # Save
        # ----------------------------------------------

        save_graph_atomic(
            graph_path,

            x=x,

            edge_index_g1=(
                edge_index_g1
            ),

            edge_index_g2=(
                edge_index_g2
            ),

            rule_version=np.array(
                [GRAPH_RULE_VERSION],
                dtype=np.int16,
            ),

            similarity_threshold=np.array(
                [threshold],
                dtype=np.float32,
            ),

            extra_degree_cap=np.array(
                [extra_degree_cap],
                dtype=np.int16,
            ),

            nonlocal_only=np.array(
                [True],
                dtype=np.bool_,
            ),
        )

        # Reload once after writing.
        verified = (
            validate_cached_graph(
                graph_path,
                expected_nodes,
                node_dim,
                expected_g1_edges,
                threshold,
                extra_degree_cap,
                density_limit,
            )
        )

        if verified is None:
            raise RuntimeError(
                f"{track_id}: saved graph "
                "failed validation."
            )

        generated += 1

        records.append({
            "track_id": track_id,
            "split": (
                split_lookup[
                    track_id
                ]
            ),
            "graph_path": str(
                graph_path.relative_to(
                    ROOT
                )
            ),
            **verified,
            "cache_status": "generated",
        })

    # --------------------------------------------------
    # Manifest
    # --------------------------------------------------

    manifest = pd.DataFrame(
        records
    ).sort_values(
        "track_id"
    ).reset_index(
        drop=True
    )

    if len(manifest) != (
        EXPECTED_TRACKS
    ):
        raise RuntimeError(
            "Graph manifest count mismatch."
        )

    if manifest[
        "track_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate graph manifest IDs."
        )

    split_counts = (
        manifest[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        split_counts.get(
            "train",
            0,
        )
        != EXPECTED_TRAIN
    ):
        raise RuntimeError(
            "Train graph count mismatch."
        )

    if (
        split_counts.get(
            "val",
            0,
        )
        != EXPECTED_VAL
    ):
        raise RuntimeError(
            "Val graph count mismatch."
        )

    if (
        split_counts.get(
            "test",
            0,
        )
        != EXPECTED_TEST
    ):
        raise RuntimeError(
            "Test graph count mismatch."
        )

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

    # --------------------------------------------------
    # Final exhaustive cache validation
    # --------------------------------------------------

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Validating graph cache",
    ):

        graph_path = (
            ROOT
            / str(
                row[
                    "graph_path"
                ]
            )
        )

        valid = validate_cached_graph(
            graph_path,
            expected_nodes,
            node_dim,
            expected_g1_edges,
            threshold,
            extra_degree_cap,
            density_limit,
        )

        if valid is None:
            raise RuntimeError(
                f"Final graph cache "
                f"validation failed: "
                f"{row['track_id']}"
            )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    density_values = (
        manifest[
            "g2_density"
        ].to_numpy()
    )

    extra_values = (
        manifest[
            "extra_similarity_pairs"
        ].to_numpy()
    )

    g2_edge_values = (
        manifest[
            "g2_directed_edges"
        ].to_numpy()
    )

    report = {
        "tracks": EXPECTED_TRACKS,

        "generated": generated,
        "reused": reused,

        "split_counts": {
            str(k): int(v)
            for k, v
            in split_counts.items()
        },

        "nodes_per_track": (
            expected_nodes
        ),

        "node_feature_dim": (
            node_dim
        ),

        "g1_directed_edges": (
            expected_g1_edges
        ),

        "g2_rule": {
            "similarity_threshold": (
                threshold
            ),
            "nonlocal_only": (
                nonlocal_only
            ),
            "selection": (
                selection
            ),
            "extra_degree_cap": (
                extra_degree_cap
            ),
        },

        "g2_directed_edges": {
            "min": int(
                g2_edge_values.min()
            ),
            "mean": float(
                g2_edge_values.mean()
            ),
            "median": float(
                np.median(
                    g2_edge_values
                )
            ),
            "max": int(
                g2_edge_values.max()
            ),
        },

        "extra_similarity_pairs": {
            "min": int(
                extra_values.min()
            ),
            "mean": float(
                extra_values.mean()
            ),
            "median": float(
                np.median(
                    extra_values
                )
            ),
            "max": int(
                extra_values.max()
            ),
            "tracks_with_none": int(
                (
                    extra_values == 0
                ).sum()
            ),
        },

        "g2_density": {
            "min": float(
                density_values.min()
            ),
            "mean": float(
                density_values.mean()
            ),
            "median": float(
                np.median(
                    density_values
                )
            ),
            "p95": float(
                np.percentile(
                    density_values,
                    95,
                )
            ),
            "max": float(
                density_values.max()
            ),
        },

        "theoretical_max_g2_edges": (
            theoretical_max_g2_edges
        ),

        "theoretical_max_density": (
            theoretical_max_density
        ),

        "cache_validation": "PASS",
    }

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
        "MusicCaps full graph cache complete"
    )
    print("=" * 62)

    print(
        f"Tracks       : "
        f"{EXPECTED_TRACKS}"
    )

    print(
        f"Generated    : "
        f"{generated}"
    )

    print(
        f"Reused       : "
        f"{reused}"
    )

    print()

    print(
        f"Node shape   : "
        f"({expected_nodes}, "
        f"{node_dim})"
    )

    print(
        f"G1 edges     : "
        f"{expected_g1_edges}"
    )

    print()

    print(
        "G2 directed edges"
    )

    print(
        f"  min        : "
        f"{g2_edge_values.min()}"
    )

    print(
        f"  mean       : "
        f"{g2_edge_values.mean():.2f}"
    )

    print(
        f"  median     : "
        f"{np.median(g2_edge_values):.2f}"
    )

    print(
        f"  max        : "
        f"{g2_edge_values.max()}"
    )

    print()

    print(
        "Extra similarity pairs"
    )

    print(
        f"  mean       : "
        f"{extra_values.mean():.2f}"
    )

    print(
        f"  median     : "
        f"{np.median(extra_values):.2f}"
    )

    print(
        f"  max        : "
        f"{extra_values.max()}"
    )

    print(
        f"  none       : "
        f"{(extra_values == 0).sum()}"
    )

    print()

    print(
        "G2 density"
    )

    print(
        f"  mean       : "
        f"{density_values.mean():.3f}"
    )

    print(
        f"  p95        : "
        f"{np.percentile(density_values, 95):.3f}"
    )

    print(
        f"  max        : "
        f"{density_values.max():.3f}"
    )

    print()

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