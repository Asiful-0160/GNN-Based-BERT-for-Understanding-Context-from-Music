from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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

OLD_LOCK_PATH = (
    ROOT
    / "data"
    / "splits"
    / "frozen_hashes.json"
)

AUDIO_MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "audio_feature_manifest.parquet"
)

BERT_TOKEN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "bert_tokens"
    / "bert_tokens.npz"
)

BERT_MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "bert_tokens"
    / "bert_token_manifest.parquet"
)

GRAPH_MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "graph_manifest.parquet"
)

TARGETS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "labels"
    / "musiccaps_targets.parquet"
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

GRAPH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "graphs"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "splits"
    / "pretraining_frozen_hashes.json"
)


EXPECTED_TRACKS = 5140


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def tree_hash(
    directory: Path,
    pattern: str,
    expected_count: int,
):

    files = sorted(
        directory.glob(pattern)
    )

    if len(files) != expected_count:
        raise RuntimeError(
            f"{directory}: expected "
            f"{expected_count} files, "
            f"found {len(files)}."
        )

    aggregate = hashlib.sha256()

    file_hashes = {}

    for path in tqdm(
        files,
        desc=f"Hashing {directory.name}",
    ):

        digest = sha256_file(
            path
        )

        relative = str(
            path.relative_to(ROOT)
        )

        file_hashes[
            relative
        ] = digest

        aggregate.update(
            relative.encode(
                "utf-8"
            )
        )

        aggregate.update(
            b"\0"
        )

        aggregate.update(
            digest.encode(
                "ascii"
            )
        )

        aggregate.update(
            b"\n"
        )

    return {
        "file_count": len(files),
        "tree_sha256": (
            aggregate.hexdigest()
        ),
        "files": file_hashes,
    }


def main():

    # --------------------------------------------------
    # Required artifacts
    # --------------------------------------------------

    required = [
        CONFIG_PATH,
        CANONICAL_PATH,
        OLD_LOCK_PATH,
        AUDIO_MANIFEST_PATH,
        BERT_TOKEN_PATH,
        BERT_MANIFEST_PATH,
        GRAPH_MANIFEST_PATH,
        TARGETS_PATH,
    ]

    for path in required:

        if not path.exists():
            raise RuntimeError(
                f"Missing required artifact: "
                f"{path}"
            )

    # --------------------------------------------------
    # Verify final configuration
    # --------------------------------------------------

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if int(
        config["text"]["max_length"]
    ) != 112:
        raise RuntimeError(
            "Expected BERT max_length=112."
        )

    if int(
        config["audio"][
            "fixed_samples"
        ]
    ) != 220500:
        raise RuntimeError(
            "Expected fixed_samples=220500."
        )

    if int(
        config["audio"][
            "fixed_mel_frames"
        ]
    ) != 431:
        raise RuntimeError(
            "Expected fixed_mel_frames=431."
        )

    graph = config["graph"]

    if int(
        graph[
            "expected_nodes_per_track"
        ]
    ) != 9:
        raise RuntimeError(
            "Expected 9 graph nodes."
        )

    if int(
        graph["node_input_dim"]
    ) != 140:
        raise RuntimeError(
            "Expected graph node_input_dim=140."
        )

    if float(
        graph[
            "similarity_threshold"
        ]
    ) != 0.85:
        raise RuntimeError(
            "Expected similarity_threshold=0.85."
        )

    if not bool(
        graph[
            "similarity_nonlocal_only"
        ]
    ):
        raise RuntimeError(
            "Expected similarity_nonlocal_only=true."
        )

    if str(
        graph[
            "similarity_selection"
        ]
    ) != "strongest_first":
        raise RuntimeError(
            "Expected strongest_first selection."
        )

    if int(
        graph[
            "similarity_extra_degree_cap"
        ]
    ) != 2:
        raise RuntimeError(
            "Expected similarity_extra_degree_cap=2."
        )

    # --------------------------------------------------
    # Cross-manifest track-ID verification
    # --------------------------------------------------

    canonical = pd.read_parquet(
        CANONICAL_PATH,
        columns=["track_id"],
    )

    audio_manifest = pd.read_parquet(
        AUDIO_MANIFEST_PATH,
        columns=["track_id"],
    )

    bert_manifest = pd.read_parquet(
        BERT_MANIFEST_PATH,
        columns=["track_id"],
    )

    graph_manifest = pd.read_parquet(
        GRAPH_MANIFEST_PATH,
        columns=["track_id"],
    )

    datasets = {
        "canonical": canonical,
        "audio_manifest": audio_manifest,
        "bert_manifest": bert_manifest,
        "graph_manifest": graph_manifest,
    }

    id_sets = {}

    for name, df in datasets.items():

        ids = (
            df["track_id"]
            .astype(str)
        )

        if len(ids) != EXPECTED_TRACKS:
            raise RuntimeError(
                f"{name}: expected "
                f"{EXPECTED_TRACKS} rows, "
                f"found {len(ids)}."
            )

        if ids.duplicated().any():
            raise RuntimeError(
                f"{name}: duplicate track IDs."
            )

        id_sets[
            name
        ] = set(ids)

    canonical_ids = id_sets[
        "canonical"
    ]

    for name, ids in id_sets.items():

        if ids != canonical_ids:
            raise RuntimeError(
                f"{name}: track-ID set "
                "does not match canonical."
            )

    # --------------------------------------------------
    # Verify BERT cache dimensions
    # --------------------------------------------------

    with np.load(
        BERT_TOKEN_PATH,
        allow_pickle=False,
    ) as cache:

        expected_shape = (
            EXPECTED_TRACKS,
            112,
        )

        for key in [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]:

            if key not in cache.files:
                raise RuntimeError(
                    f"BERT cache missing {key}."
                )

            if cache[key].shape != (
                expected_shape
            ):
                raise RuntimeError(
                    f"{key}: expected "
                    f"{expected_shape}, "
                    f"found {cache[key].shape}."
                )

    # --------------------------------------------------
    # Graph-manifest checks
    # --------------------------------------------------

    required_graph_columns = {
        "nodes",
        "node_feature_dim",
        "g1_directed_edges",
        "g2_directed_edges",
        "g2_density",
    }

    full_graph_manifest = (
        pd.read_parquet(
            GRAPH_MANIFEST_PATH
        )
    )

    missing_columns = (
        required_graph_columns
        - set(
            full_graph_manifest.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Graph manifest missing columns: "
            f"{sorted(missing_columns)}"
        )

    if not (
        full_graph_manifest[
            "nodes"
        ] == 9
    ).all():
        raise RuntimeError(
            "Graph node-count mismatch."
        )

    if not (
        full_graph_manifest[
            "node_feature_dim"
        ] == 140
    ).all():
        raise RuntimeError(
            "Graph feature-dimension mismatch."
        )

    if not (
        full_graph_manifest[
            "g1_directed_edges"
        ] == 16
    ).all():
        raise RuntimeError(
            "G1 edge-count mismatch."
        )

    if (
        full_graph_manifest[
            "g2_directed_edges"
        ].min()
        < 16
    ):
        raise RuntimeError(
            "Invalid G2 minimum edge count."
        )

    if (
        full_graph_manifest[
            "g2_directed_edges"
        ].max()
        > 34
    ):
        raise RuntimeError(
            "Invalid G2 maximum edge count."
        )

    if (
        full_graph_manifest[
            "g2_density"
        ].max()
        > 0.90
    ):
        raise RuntimeError(
            "G2 density gate violated."
        )

    print()
    print(
        "Cross-artifact validation: PASS"
    )

    print()
    print(
        "Hashing final preprocessing artifacts..."
    )
    print()

    # --------------------------------------------------
    # Individual critical-file hashes
    # --------------------------------------------------

    critical_files = {
        "config": CONFIG_PATH,
        "canonical_dataset": (
            CANONICAL_PATH
        ),
        "original_dataset_lock": (
            OLD_LOCK_PATH
        ),
        "audio_feature_manifest": (
            AUDIO_MANIFEST_PATH
        ),
        "bert_tokens": (
            BERT_TOKEN_PATH
        ),
        "bert_token_manifest": (
            BERT_MANIFEST_PATH
        ),
        "graph_manifest": (
            GRAPH_MANIFEST_PATH
        ),
        "targets": (
            TARGETS_PATH
        ),
    }

    critical_hashes = {}

    for name, path in critical_files.items():

        print(
            f"Hashing {name}..."
        )

        critical_hashes[
            name
        ] = {
            "path": str(
                path.relative_to(ROOT)
            ),
            "sha256": sha256_file(
                path
            ),
        }

    # --------------------------------------------------
    # Exact cache tree hashes
    # --------------------------------------------------

    mel_lock = tree_hash(
        MEL_DIR,
        "*.npy",
        EXPECTED_TRACKS,
    )

    chroma_lock = tree_hash(
        CHROMA_DIR,
        "*.npy",
        EXPECTED_TRACKS,
    )

    graph_lock = tree_hash(
        GRAPH_DIR,
        "*.npz",
        EXPECTED_TRACKS,
    )

    # --------------------------------------------------
    # Hash preprocessing/graph scripts
    # --------------------------------------------------

    script_hashes = {}

    for number in range(
        20,
        30,
    ):

        matches = sorted(
            (
                ROOT
                / "scripts"
            ).glob(
                f"{number}_*.py"
            )
        )

        for path in matches:

            relative = str(
                path.relative_to(ROOT)
            )

            script_hashes[
                relative
            ] = sha256_file(
                path
            )

    # --------------------------------------------------
    # Save lock
    # --------------------------------------------------

    output = {
        "lock_type": (
            "pretraining_artifact_lock"
        ),

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "dataset_tracks": (
            EXPECTED_TRACKS
        ),

        "final_dimensions": {
            "bert_max_length": 112,
            "audio_samples": 220500,
            "audio_frames": 431,
            "mel_bins": 128,
            "chroma_bins": 12,
            "graph_nodes": 9,
            "graph_node_dim": 140,
        },

        "final_graph_rule": {
            "g1_directed_edges": 16,
            "similarity_threshold": 0.85,
            "nonlocal_only": True,
            "selection": (
                "strongest_first"
            ),
            "extra_degree_cap": 2,
            "maximum_g2_directed_edges": 34,
        },

        "critical_files": (
            critical_hashes
        ),

        "cache_trees": {
            "mel": mel_lock,
            "chroma": chroma_lock,
            "graphs": graph_lock,
        },

        "preprocessing_scripts": (
            script_hashes
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Hash lock file itself for console record.
    lock_digest = sha256_file(
        OUTPUT_PATH
    )

    print()
    print(
        "PRETRAINING ARTIFACTS FROZEN"
    )
    print("=" * 62)

    print(
        f"Tracks             : "
        f"{EXPECTED_TRACKS}"
    )

    print(
        f"Mel files          : "
        f"{mel_lock['file_count']}"
    )

    print(
        f"Chroma files       : "
        f"{chroma_lock['file_count']}"
    )

    print(
        f"Graph files        : "
        f"{graph_lock['file_count']}"
    )

    print()

    print(
        f"Mel tree SHA256    : "
        f"{mel_lock['tree_sha256']}"
    )

    print(
        f"Chroma tree SHA256 : "
        f"{chroma_lock['tree_sha256']}"
    )

    print(
        f"Graph tree SHA256  : "
        f"{graph_lock['tree_sha256']}"
    )

    print()

    print(
        f"Lock file          : "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Lock SHA256        : "
        f"{lock_digest}"
    )

    print()

    print(
        "PRETRAINING LOCK: PASS"
    )

    print(
        "Safe to begin model training."
    )


if __name__ == "__main__":
    main()