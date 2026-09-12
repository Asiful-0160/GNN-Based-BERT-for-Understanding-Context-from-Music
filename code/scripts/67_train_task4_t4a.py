#!/usr/bin/env python3
"""
SCRIPT 67 — FORMAL TASK-4 T4-A CONTRASTIVE TRAINING

T4-A:
    Frozen T1-B representation:
        768-D CLS
            -> Linear(768, 256, bias=False)

    Frozen A4/G2 representation:
        128-D pooled GAT
            -> Linear(128, 256, bias=False)

    Both:
        -> L2 normalize
        -> cosine similarity
        -> symmetric InfoNCE

Formal frozen protocol from Script 65:
    seed                  = 42
    embedding dimension   = 256
    temperature           = 0.07
    batch size            = 32
    optimizer             = AdamW
    learning rate         = 1e-3
    weight decay          = 1e-4
    max epochs            = 30
    patience              = 5
    gradient clip norm    = 1.0
    checkpoint metric     = VAL mR
    tie rule              = earliest epoch

Development data:
    TRAIN = 4112
    VAL   = 514
    TEST  = 0

IMPORTANT
---------
This is the FIRST FORMAL TASK-4 MODEL RUN.

- NO BERT instantiation
- NO GAT instantiation
- NO encoder fine-tuning
- NO TEST examples
- NO TEST labels
- NO TEST retrieval
- NO threshold tuning
- NO architecture changes
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ROOT / PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DESIGN_PATH = (
    ROOT
    / "results/runs/task4/task4_contrastive_design.json"
)

DESIGN_LOCK_PATH = (
    ROOT
    / "results/runs/task4/task4_contrastive_design_lock.json"
)

SMOKE_PATH = (
    ROOT
    / "results/runs/task4/task4_script66_smoke.json"
)

BERT_SEQUENCE_PATH = (
    ROOT
    / "data/processed/bert_seq_frozen/"
      "bert_sequence_trainval.npy"
)

BERT_MANIFEST_PATH = (
    ROOT
    / "data/processed/bert_seq_frozen/"
      "bert_seq_manifest.parquet"
)

GAT_EMBEDDING_PATH = (
    ROOT
    / "data/processed/gat_g2_frozen/"
      "gat_g2_embedding_trainval.npy"
)

GAT_MANIFEST_PATH = (
    ROOT
    / "data/processed/gat_g2_frozen/"
      "gat_g2_manifest.parquet"
)

MODEL_DIR = (
    ROOT / "models/task4_candidates"
)

RESULT_DIR = (
    ROOT / "results/runs/task4"
)

CHECKPOINT_PATH = (
    MODEL_DIR / "task4_T4-A_seed42_best.pt"
)

HISTORY_CSV_PATH = (
    RESULT_DIR / "task4_T4-A_seed42_history.csv"
)

SUMMARY_PATH = (
    RESULT_DIR / "task4_T4-A_seed42_summary.json"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_DESIGN_SHA256 = (
    "61ed1263476e631374790c9669ab4aa96"
    "e3b5cb295436d4e273da66d4da556db"
)

EXPECTED_DESIGN_LOCK_SHA256 = (
    "53f50205e712833ec4efb506b9b6d911"
    "8dcf46f51b72c5907f33cd61ddca612c"
)

EXPECTED_SMOKE_SHA256 = (
    "0151e229f7691284deea9a005f3f357d"
    "55f4c6af8417418ee42c0948bb6b2329"
)

EXPECTED_BERT_SEQUENCE_SHA256 = (
    "a1c723ec939392b8b0d9bafadb8327d"
    "8a0967ecb9bd8e89ffcd559306633e133"
)

EXPECTED_GAT_EMBEDDING_SHA256 = (
    "3d39a343d1acc678576f9dd3d88b088"
    "ca18a81017190246b0b32841292b92852"
)


# =============================================================================
# FROZEN FORMAL T4-A PROTOCOL
# =============================================================================

EXPECTED_TRAIN = 4112
EXPECTED_VAL = 514
EXPECTED_DEV = 4626

SEED = 42

TEXT_DIM = 768
GRAPH_DIM = 128
SHARED_DIM = 256

TEMPERATURE = 0.07

BATCH_SIZE = 32

LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4

MAX_EPOCHS = 30
PATIENCE = 5

GRAD_CLIP_NORM = 1.0


# =============================================================================
# STATE
# =============================================================================

FAILURES: list[str] = []


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(message: str) -> None:

    print(f"PASS  {message}")


def failed(message: str) -> None:

    FAILURES.append(message)
    print(f"FAIL  {message}")


def require(
    condition: bool,
    message: str,
) -> None:

    if condition:
        passed(message)
    else:
        failed(message)


def safe_relative(path: Path) -> str:

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def verify_hash(
    path: Path,
    expected: str,
    label: str,
) -> None:

    if not path.exists():

        failed(f"{label} exists")
        return

    observed = sha256_file(path)

    print(
        f"{label}:\n"
        f"  observed = {observed}\n"
        f"  expected = {expected}"
    )

    require(
        observed == expected,
        f"{label} SHA256 matches frozen identity",
    )


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    temp = Path(
        str(path) + ".tmp"
    )

    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(
        temp,
        path,
    )


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(
        torch.backends.cuda.matmul,
        "allow_tf32",
    ):
        torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(
        torch.backends.cudnn,
        "allow_tf32",
    ):
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(
        True
    )


def split_counts(
    manifest: pd.DataFrame,
) -> dict[str, int]:

    counts = (
        manifest["split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )

    return {
        str(k): int(v)
        for k, v in counts.items()
    }


# =============================================================================
# MODEL
# =============================================================================

class Task4T4A(
    nn.Module,
):

    def __init__(self) -> None:

        super().__init__()

        self.text_projection = nn.Linear(
            TEXT_DIM,
            SHARED_DIM,
            bias=False,
        )

        self.graph_projection = nn.Linear(
            GRAPH_DIM,
            SHARED_DIM,
            bias=False,
        )

        nn.init.xavier_uniform_(
            self.text_projection.weight
        )

        nn.init.xavier_uniform_(
            self.graph_projection.weight
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        z_text = self.text_projection(
            text_cls
        )

        z_graph = self.graph_projection(
            graph_embedding
        )

        z_text = F.normalize(
            z_text,
            p=2,
            dim=-1,
        )

        z_graph = F.normalize(
            z_graph,
            p=2,
            dim=-1,
        )

        return (
            z_text,
            z_graph,
        )


# =============================================================================
# INFONCE
# =============================================================================

def symmetric_infonce(
    z_text: torch.Tensor,
    z_graph: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    logits = (
        z_graph
        @ z_text.T
    ) / TEMPERATURE

    targets = torch.arange(
        z_text.shape[0],
        device=z_text.device,
        dtype=torch.long,
    )

    loss_a2t = F.cross_entropy(
        logits,
        targets,
    )

    loss_t2a = F.cross_entropy(
        logits.T,
        targets,
    )

    loss = (
        0.5
        * (
            loss_a2t
            + loss_t2a
        )
    )

    return (
        loss,
        loss_a2t,
        loss_t2a,
    )


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieval_metrics(
    z_text: torch.Tensor,
    z_graph: torch.Tensor,
) -> dict[str, float]:

    """
    Full candidate-pool retrieval.

    Matrix convention:
        rows    = audio
        columns = caption

    Rank convention frozen in Script 65:
        rank = 1 + count(similarity > correct_similarity)
    """

    similarity_a2t = (
        z_graph
        @ z_text.T
    )

    similarity_c2a = (
        similarity_a2t.T
    )

    def ranks(
        matrix: torch.Tensor,
    ) -> torch.Tensor:

        correct = torch.diagonal(
            matrix
        )

        return (
            1
            + (
                matrix
                > correct.unsqueeze(1)
            )
            .sum(dim=1)
        )

    a2t_rank = ranks(
        similarity_a2t
    )

    c2a_rank = ranks(
        similarity_c2a
    )

    metrics: dict[str, float] = {}

    for k in (
        1,
        5,
        10,
    ):

        metrics[
            f"audio_to_caption_R@{k}"
        ] = float(
            (
                a2t_rank <= k
            )
            .float()
            .mean()
            .item()
        )

        metrics[
            f"caption_to_audio_R@{k}"
        ] = float(
            (
                c2a_rank <= k
            )
            .float()
            .mean()
            .item()
        )

    metrics["mR"] = float(
        np.mean(
            [
                metrics[
                    "audio_to_caption_R@1"
                ],
                metrics[
                    "audio_to_caption_R@5"
                ],
                metrics[
                    "audio_to_caption_R@10"
                ],
                metrics[
                    "caption_to_audio_R@1"
                ],
                metrics[
                    "caption_to_audio_R@5"
                ],
                metrics[
                    "caption_to_audio_R@10"
                ],
            ],
            dtype=np.float64,
        )
    )

    metrics[
        "audio_to_caption_mean_rank"
    ] = float(
        a2t_rank
        .float()
        .mean()
        .item()
    )

    metrics[
        "caption_to_audio_mean_rank"
    ] = float(
        c2a_rank
        .float()
        .mean()
        .item()
    )

    return metrics


# =============================================================================
# EMBEDDING
# =============================================================================

@torch.no_grad()
def embed_full_split(
    model: Task4T4A,
    text_np: np.ndarray,
    graph_np: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    model.eval()

    text_outputs: list[
        torch.Tensor
    ] = []

    graph_outputs: list[
        torch.Tensor
    ] = []

    total = text_np.shape[0]

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        text_batch = torch.from_numpy(
            text_np[start:end]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        graph_batch = torch.from_numpy(
            graph_np[start:end]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        z_text, z_graph = model(
            text_batch,
            graph_batch,
        )

        text_outputs.append(
            z_text.cpu()
        )

        graph_outputs.append(
            z_graph.cpu()
        )

    return (
        torch.cat(
            text_outputs,
            dim=0,
        ),
        torch.cat(
            graph_outputs,
            dim=0,
        ),
    )


# =============================================================================
# HEADER
# =============================================================================

section(
    "SCRIPT 67 — FORMAL TASK-4 T4-A TRAINING"
)

print(f"Project root: {ROOT}")
print()
print("FORMAL TASK-4 RESULT:       YES")
print("Variant:                    T4-A")
print("TRAIN:                      4112")
print("VAL:                        514")
print("TEST:                       0")
print("Seed:                       42")
print("BERT instantiated:          NO")
print("GAT instantiated:           NO")
print("Trainable encoder params:   0")
print("Trainable projection heads: YES")
print("Checkpoint metric:          VAL mR")
print("TEST retrieval:             FORBIDDEN")


# =============================================================================
# 1. OUTPUT PROTECTION
# =============================================================================

section(
    "1. OUTPUT PROTECTION"
)

for path, label in [
    (
        CHECKPOINT_PATH,
        "T4-A checkpoint",
    ),
    (
        HISTORY_CSV_PATH,
        "T4-A history CSV",
    ),
    (
        SUMMARY_PATH,
        "T4-A summary",
    ),
]:

    require(
        not path.exists(),
        f"{label} does not already exist",
    )

if FAILURES:

    print()
    print(
        "Refusing to overwrite an existing formal T4-A result."
    )

    sys.exit(1)


# =============================================================================
# 2. VERIFY FROZEN TASK-4 LINEAGE
# =============================================================================

section(
    "2. VERIFY FROZEN TASK-4 LINEAGE"
)

verify_hash(
    DESIGN_PATH,
    EXPECTED_DESIGN_SHA256,
    "Task-4 design",
)

verify_hash(
    DESIGN_LOCK_PATH,
    EXPECTED_DESIGN_LOCK_SHA256,
    "Task-4 design lock",
)

verify_hash(
    SMOKE_PATH,
    EXPECTED_SMOKE_SHA256,
    "Script-66 smoke report",
)

verify_hash(
    BERT_SEQUENCE_PATH,
    EXPECTED_BERT_SEQUENCE_SHA256,
    "Script-47 BERT sequence cache",
)

verify_hash(
    GAT_EMBEDDING_PATH,
    EXPECTED_GAT_EMBEDDING_SHA256,
    "Script-48 GAT embedding cache",
)

if FAILURES:

    print()
    print(
        "Frozen Task-4 lineage mismatch."
    )

    print(
        "Do NOT begin formal training."
    )

    sys.exit(1)


# =============================================================================
# 3. VERIFY SCRIPT-65 DESIGN VALUES
# =============================================================================

section(
    "3. VERIFY FORMAL T4-A DESIGN"
)

design = load_json(
    DESIGN_PATH
)

require(
    design[
        "shared_embedding_space"
    ][
        "dimension"
    ]
    == SHARED_DIM,
    "shared dimension = 256",
)

require(
    math.isclose(
        float(
            design[
                "contrastive_objective"
            ][
                "temperature"
            ]
        ),
        TEMPERATURE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ),
    "temperature = 0.07",
)

require(
    design[
        "t4_a"
    ][
        "batch_size"
    ]
    == BATCH_SIZE,
    "batch size = 32",
)

require(
    math.isclose(
        float(
            design[
                "t4_a"
            ][
                "learning_rate"
            ]
        ),
        LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ),
    "learning rate = 1e-3",
)

require(
    math.isclose(
        float(
            design[
                "t4_a"
            ][
                "weight_decay"
            ]
        ),
        WEIGHT_DECAY,
        rel_tol=0.0,
        abs_tol=1e-15,
    ),
    "weight decay = 1e-4",
)

require(
    design[
        "t4_a"
    ][
        "max_epochs"
    ]
    == MAX_EPOCHS,
    "max epochs = 30",
)

require(
    design[
        "t4_a"
    ][
        "early_stopping_patience"
    ]
    == PATIENCE,
    "patience = 5",
)

require(
    design[
        "t4_a"
    ][
        "formal_selection_seed"
    ]
    == SEED,
    "formal seed = 42",
)

require(
    design[
        "validation_retrieval"
    ][
        "primary_model_selection_metric"
    ]
    == "mR",
    "primary selection metric = VAL mR",
)

require(
    design[
        "dataset"
    ][
        "test_during_development"
    ]
    is False,
    "TEST forbidden during Task-4 development",
)

if FAILURES:

    print()
    print(
        "Formal design mismatch."
    )

    sys.exit(1)


# =============================================================================
# 4. MANIFEST ALIGNMENT
# =============================================================================

section(
    "4. TRAIN / VAL MANIFEST ALIGNMENT"
)

bert_manifest = pd.read_parquet(
    BERT_MANIFEST_PATH
)

gat_manifest = pd.read_parquet(
    GAT_MANIFEST_PATH
)

require(
    len(bert_manifest)
    == EXPECTED_DEV,
    "BERT manifest rows = 4626",
)

require(
    len(gat_manifest)
    == EXPECTED_DEV,
    "GAT manifest rows = 4626",
)

bert_ids = (
    bert_manifest["track_id"]
    .astype(str)
    .tolist()
)

gat_ids = (
    gat_manifest["track_id"]
    .astype(str)
    .tolist()
)

require(
    bert_ids == gat_ids,
    "BERT/GAT rowwise track_id alignment exact",
)

bert_splits = (
    bert_manifest["split"]
    .astype(str)
    .str.lower()
    .to_numpy()
)

gat_splits = (
    gat_manifest["split"]
    .astype(str)
    .str.lower()
    .to_numpy()
)

require(
    np.array_equal(
        bert_splits,
        gat_splits,
    ),
    "BERT/GAT rowwise split alignment exact",
)

train_rows = np.flatnonzero(
    bert_splits == "train"
)

val_rows = np.flatnonzero(
    bert_splits == "val"
)

require(
    len(train_rows)
    == EXPECTED_TRAIN,
    "TRAIN rows = 4112",
)

require(
    len(val_rows)
    == EXPECTED_VAL,
    "VAL rows = 514",
)

require(
    int(
        np.sum(
            bert_splits == "test"
        )
    )
    == 0,
    "development cache contains zero TEST rows",
)

if FAILURES:

    sys.exit(1)


# =============================================================================
# 5. LOAD TRAIN / VAL FROZEN REPRESENTATIONS
# =============================================================================

section(
    "5. LOAD FROZEN TRAIN / VAL REPRESENTATIONS"
)

bert_sequence = np.load(
    BERT_SEQUENCE_PATH,
    mmap_mode="r",
)

gat_embedding = np.load(
    GAT_EMBEDDING_PATH,
    mmap_mode="r",
)

require(
    tuple(
        bert_sequence.shape
    )
    == (
        EXPECTED_DEV,
        112,
        TEXT_DIM,
    ),
    "BERT sequence shape = (4626,112,768)",
)

require(
    tuple(
        gat_embedding.shape
    )
    == (
        EXPECTED_DEV,
        GRAPH_DIM,
    ),
    "GAT embedding shape = (4626,128)",
)

train_text = np.asarray(
    bert_sequence[
        train_rows,
        0,
        :,
    ],
    dtype=np.float32,
).copy()

train_graph = np.asarray(
    gat_embedding[
        train_rows,
        :,
    ],
    dtype=np.float32,
).copy()

val_text = np.asarray(
    bert_sequence[
        val_rows,
        0,
        :,
    ],
    dtype=np.float32,
).copy()

val_graph = np.asarray(
    gat_embedding[
        val_rows,
        :,
    ],
    dtype=np.float32,
).copy()

require(
    train_text.shape
    == (
        EXPECTED_TRAIN,
        TEXT_DIM,
    ),
    "TRAIN text shape = (4112,768)",
)

require(
    train_graph.shape
    == (
        EXPECTED_TRAIN,
        GRAPH_DIM,
    ),
    "TRAIN graph shape = (4112,128)",
)

require(
    val_text.shape
    == (
        EXPECTED_VAL,
        TEXT_DIM,
    ),
    "VAL text shape = (514,768)",
)

require(
    val_graph.shape
    == (
        EXPECTED_VAL,
        GRAPH_DIM,
    ),
    "VAL graph shape = (514,128)",
)

require(
    np.isfinite(
        train_text
    ).all(),
    "TRAIN text finite",
)

require(
    np.isfinite(
        train_graph
    ).all(),
    "TRAIN graph finite",
)

require(
    np.isfinite(
        val_text
    ).all(),
    "VAL text finite",
)

require(
    np.isfinite(
        val_graph
    ).all(),
    "VAL graph finite",
)


# =============================================================================
# 6. REPRODUCIBILITY / DEVICE
# =============================================================================

section(
    "6. REPRODUCIBILITY / DEVICE"
)

require(
    torch.cuda.is_available(),
    "CUDA available",
)

if FAILURES:

    sys.exit(1)


set_seed(
    SEED
)

device = torch.device(
    "cuda"
)

print(
    "Device:",
    torch.cuda.get_device_name(
        device
    ),
)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA runtime:",
    torch.version.cuda,
)

passed(
    "seed fixed at 42"
)

passed(
    "deterministic algorithms enabled"
)

passed(
    "TF32 disabled"
)

passed(
    "AMP disabled for T4-A formal run"
)


# =============================================================================
# 7. BUILD FORMAL T4-A MODEL
# =============================================================================

section(
    "7. BUILD FORMAL T4-A MODEL"
)

model = Task4T4A().to(
    device
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

require(
    trainable_parameters
    == 229376,
    "T4-A trainable parameters = 229376",
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

print(
    "Trainable parameters:",
    trainable_parameters,
)

print(
    "Optimizer: AdamW"
)

print(
    f"LR: {LEARNING_RATE}"
)

print(
    f"Weight decay: {WEIGHT_DECAY}"
)


# =============================================================================
# 8. PREPARE OUTPUT DIRECTORIES
# =============================================================================

section(
    "8. PREPARE FORMAL OUTPUT DIRECTORIES"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

passed(
    "formal output directories ready"
)


# =============================================================================
# 9. FORMAL TRAINING
# =============================================================================

section(
    "9. FORMAL T4-A TRAINING"
)

rng = np.random.default_rng(
    SEED
)

history: list[
    dict[str, Any]
] = []

best_mr = -math.inf
best_epoch = None
epochs_without_improvement = 0

bert_hash_before = sha256_file(
    BERT_SEQUENCE_PATH
)

gat_hash_before = sha256_file(
    GAT_EMBEDDING_PATH
)

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    model.train()

    permutation = rng.permutation(
        EXPECTED_TRAIN
    )

    running_loss_sum = 0.0
    running_a2t_sum = 0.0
    running_t2a_sum = 0.0
    running_examples = 0

    batch_count = 0

    for start in range(
        0,
        EXPECTED_TRAIN,
        BATCH_SIZE,
    ):

        batch_indices = permutation[
            start:
            start + BATCH_SIZE
        ]

        text_batch = torch.from_numpy(
            train_text[
                batch_indices
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        graph_batch = torch.from_numpy(
            train_graph[
                batch_indices
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        z_text, z_graph = model(
            text_batch,
            graph_batch,
        )

        (
            loss,
            loss_a2t,
            loss_t2a,
        ) = symmetric_infonce(
            z_text,
            z_graph,
        )

        if not bool(
            torch.isfinite(
                loss
            ).item()
        ):

            raise RuntimeError(
                f"Non-finite training loss at "
                f"epoch={epoch}, batch={batch_count}"
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )
        )

        if not math.isfinite(
            float(
                grad_norm
            )
        ):

            raise RuntimeError(
                f"Non-finite gradient norm at "
                f"epoch={epoch}, batch={batch_count}"
            )

        optimizer.step()

        current_batch_size = int(
            len(
                batch_indices
            )
        )

        running_loss_sum += (
            float(
                loss.item()
            )
            * current_batch_size
        )

        running_a2t_sum += (
            float(
                loss_a2t.item()
            )
            * current_batch_size
        )

        running_t2a_sum += (
            float(
                loss_t2a.item()
            )
            * current_batch_size
        )

        running_examples += (
            current_batch_size
        )

        batch_count += 1

    train_loss = (
        running_loss_sum
        / running_examples
    )

    train_a2t_loss = (
        running_a2t_sum
        / running_examples
    )

    train_t2a_loss = (
        running_t2a_sum
        / running_examples
    )

    # -------------------------------------------------------------------------
    # FULL 514 x 514 VALIDATION RETRIEVAL
    # -------------------------------------------------------------------------

    val_z_text, val_z_graph = (
        embed_full_split(
            model=model,
            text_np=val_text,
            graph_np=val_graph,
            device=device,
            batch_size=256,
        )
    )

    val_metrics = retrieval_metrics(
        val_z_text,
        val_z_graph,
    )

    val_mr = float(
        val_metrics[
            "mR"
        ]
    )

    improved = (
        val_mr > best_mr
    )

    if improved:

        best_mr = val_mr
        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint = {

            "artifact_type":
                "task4_T4-A_checkpoint",

            "variant":
                "T4-A",

            "seed":
                SEED,

            "epoch":
                epoch,

            "val_metrics":
                val_metrics,

            "model_state_dict":
                {
                    key:
                        value.detach().cpu()
                    for key, value
                    in model.state_dict().items()
                },

            "architecture": {

                "text_input_dim":
                    TEXT_DIM,

                "graph_input_dim":
                    GRAPH_DIM,

                "shared_dim":
                    SHARED_DIM,

                "text_projection":
                    "Linear(768,256,bias=False)",

                "graph_projection":
                    "Linear(128,256,bias=False)",

                "normalization":
                    "L2",

                "temperature":
                    TEMPERATURE,
            },

            "training": {

                "optimizer":
                    "AdamW",

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "batch_size":
                    BATCH_SIZE,

                "gradient_clip_norm":
                    GRAD_CLIP_NORM,

                "max_epochs":
                    MAX_EPOCHS,

                "patience":
                    PATIENCE,
            },

            "provenance": {

                "script67_sha256":
                    sha256_file(
                        Path(__file__).resolve()
                    ),

                "design_sha256":
                    EXPECTED_DESIGN_SHA256,

                "design_lock_sha256":
                    EXPECTED_DESIGN_LOCK_SHA256,

                "script66_smoke_sha256":
                    EXPECTED_SMOKE_SHA256,

                "bert_sequence_sha256":
                    EXPECTED_BERT_SEQUENCE_SHA256,

                "gat_embedding_sha256":
                    EXPECTED_GAT_EMBEDDING_SHA256,
            },

            "test_access":
                False,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

    else:

        epochs_without_improvement += 1

    row = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_audio_to_caption_loss":
            train_a2t_loss,

        "train_caption_to_audio_loss":
            train_t2a_loss,

        "val_audio_to_caption_R@1":
            val_metrics[
                "audio_to_caption_R@1"
            ],

        "val_audio_to_caption_R@5":
            val_metrics[
                "audio_to_caption_R@5"
            ],

        "val_audio_to_caption_R@10":
            val_metrics[
                "audio_to_caption_R@10"
            ],

        "val_caption_to_audio_R@1":
            val_metrics[
                "caption_to_audio_R@1"
            ],

        "val_caption_to_audio_R@5":
            val_metrics[
                "caption_to_audio_R@5"
            ],

        "val_caption_to_audio_R@10":
            val_metrics[
                "caption_to_audio_R@10"
            ],

        "val_mR":
            val_metrics[
                "mR"
            ],

        "val_audio_to_caption_mean_rank":
            val_metrics[
                "audio_to_caption_mean_rank"
            ],

        "val_caption_to_audio_mean_rank":
            val_metrics[
                "caption_to_audio_mean_rank"
            ],

        "best_so_far":
            int(
                improved
            ),

        "epochs_without_improvement":
            epochs_without_improvement,
    }

    history.append(
        row
    )

    print()
    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS}"
    )

    print(
        f"  TRAIN InfoNCE = "
        f"{train_loss:.8f}"
    )

    print(
        f"  VAL A→C  "
        f"R@1={val_metrics['audio_to_caption_R@1']:.6f}  "
        f"R@5={val_metrics['audio_to_caption_R@5']:.6f}  "
        f"R@10={val_metrics['audio_to_caption_R@10']:.6f}"
    )

    print(
        f"  VAL C→A  "
        f"R@1={val_metrics['caption_to_audio_R@1']:.6f}  "
        f"R@5={val_metrics['caption_to_audio_R@5']:.6f}  "
        f"R@10={val_metrics['caption_to_audio_R@10']:.6f}"
    )

    print(
        f"  VAL mR = "
        f"{val_mr:.8f}"
    )

    if improved:

        print(
            "  BEST CHECKPOINT UPDATED"
        )

    else:

        print(
            f"  no improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# =============================================================================
# 10. WRITE HISTORY
# =============================================================================

section(
    "10. WRITE TRAINING HISTORY"
)

if not history:

    raise RuntimeError(
        "No training history produced."
    )

with HISTORY_CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            history[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        history
    )

passed(
    "formal T4-A epoch history written"
)

print(
    f"History: "
    f"{safe_relative(HISTORY_CSV_PATH)}"
)

print(
    f"SHA256: "
    f"{sha256_file(HISTORY_CSV_PATH)}"
)


# =============================================================================
# 11. BEST CHECKPOINT RELOAD VALIDATION
# =============================================================================

section(
    "11. BEST CHECKPOINT RELOAD VALIDATION"
)

require(
    CHECKPOINT_PATH.exists(),
    "best T4-A checkpoint exists",
)

if FAILURES:

    sys.exit(1)


best_checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

reload_model = Task4T4A()

reload_model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ],
    strict=True,
)

reload_model = reload_model.to(
    device
)

reload_z_text, reload_z_graph = (
    embed_full_split(
        model=reload_model,
        text_np=val_text,
        graph_np=val_graph,
        device=device,
        batch_size=256,
    )
)

reload_metrics = retrieval_metrics(
    reload_z_text,
    reload_z_graph,
)

saved_metrics = (
    best_checkpoint[
        "val_metrics"
    ]
)

for key in [
    "audio_to_caption_R@1",
    "audio_to_caption_R@5",
    "audio_to_caption_R@10",
    "caption_to_audio_R@1",
    "caption_to_audio_R@5",
    "caption_to_audio_R@10",
    "mR",
]:

    require(
        math.isclose(
            float(
                reload_metrics[key]
            ),
            float(
                saved_metrics[key]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"checkpoint reload reproduces {key}",
    )

require(
    int(
        best_checkpoint[
            "epoch"
        ]
    )
    == int(
        best_epoch
    ),
    "checkpoint epoch equals selected best epoch",
)

require(
    math.isclose(
        float(
            reload_metrics[
                "mR"
            ]
        ),
        float(
            best_mr
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ),
    "checkpoint reload reproduces best VAL mR exactly",
)


# =============================================================================
# 12. UPSTREAM FROZEN-CACHE INTEGRITY
# =============================================================================

section(
    "12. UPSTREAM FROZEN-CACHE INTEGRITY"
)

bert_hash_after = sha256_file(
    BERT_SEQUENCE_PATH
)

gat_hash_after = sha256_file(
    GAT_EMBEDDING_PATH
)

require(
    bert_hash_before
    == bert_hash_after
    == EXPECTED_BERT_SEQUENCE_SHA256,
    "BERT cache remained byte-identical",
)

require(
    gat_hash_before
    == gat_hash_after
    == EXPECTED_GAT_EMBEDDING_SHA256,
    "GAT cache remained byte-identical",
)

verify_hash(
    DESIGN_PATH,
    EXPECTED_DESIGN_SHA256,
    "Task-4 design after T4-A",
)

verify_hash(
    DESIGN_LOCK_PATH,
    EXPECTED_DESIGN_LOCK_SHA256,
    "Task-4 design lock after T4-A",
)


# =============================================================================
# 13. TEST BOUNDARY
# =============================================================================

section(
    "13. TEST-BOUNDARY CONFIRMATION"
)

print(
    "TRAIN examples optimized: 4112"
)

print(
    "VAL examples used for checkpoint selection: 514"
)

print(
    "TEST examples loaded: 0"
)

print(
    "TEST captions loaded: 0"
)

print(
    "TEST graphs loaded: 0"
)

print(
    "TEST labels loaded: 0"
)

print(
    "TEST retrieval calculations: 0"
)

print(
    "TEST-driven hyperparameter changes: 0"
)

passed(
    "formal T4-A preserved the frozen TEST boundary"
)


# =============================================================================
# 14. SUMMARY ARTIFACT
# =============================================================================

section(
    "14. WRITE FORMAL T4-A SUMMARY"
)

completed_utc = datetime.now(
    timezone.utc
).isoformat()

summary: dict[str, Any] = {

    "artifact_type":
        "task4_T4-A_formal_summary",

    "status":
        "formal_validation_result",

    "completed_utc":
        completed_utc,

    "variant":
        "T4-A",

    "seed":
        SEED,

    "train_examples":
        EXPECTED_TRAIN,

    "val_examples":
        EXPECTED_VAL,

    "test_examples":
        0,

    "epochs_completed":
        len(history),

    "early_stopping_triggered":
        len(history) < MAX_EPOCHS,

    "best_epoch":
        int(
            best_epoch
        ),

    "best_val_metrics":
        reload_metrics,

    "primary_selection_metric":
        "VAL mR",

    "best_val_mR":
        float(
            reload_metrics[
                "mR"
            ]
        ),

    "checkpoint":
        safe_relative(
            CHECKPOINT_PATH
        ),

    "checkpoint_sha256":
        sha256_file(
            CHECKPOINT_PATH
        ),

    "history":
        safe_relative(
            HISTORY_CSV_PATH
        ),

    "history_sha256":
        sha256_file(
            HISTORY_CSV_PATH
        ),

    "formal_protocol": {

        "embedding_dim":
            SHARED_DIM,

        "temperature":
            TEMPERATURE,

        "batch_size":
            BATCH_SIZE,

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "gradient_clip_norm":
            GRAD_CLIP_NORM,

        "amp":
            False,
    },

    "provenance": {

        "script67":
            safe_relative(
                Path(__file__).resolve()
            ),

        "script67_sha256":
            sha256_file(
                Path(__file__).resolve()
            ),

        "design_sha256":
            EXPECTED_DESIGN_SHA256,

        "design_lock_sha256":
            EXPECTED_DESIGN_LOCK_SHA256,

        "script66_smoke_sha256":
            EXPECTED_SMOKE_SHA256,

        "bert_sequence_sha256":
            EXPECTED_BERT_SEQUENCE_SHA256,

        "gat_embedding_sha256":
            EXPECTED_GAT_EMBEDDING_SHA256,
    },

    "boundaries": {

        "test_access":
            False,

        "test_retrieval":
            False,

        "test_model_selection":
            False,
    },
}

atomic_write_json(
    SUMMARY_PATH,
    summary,
)

passed(
    "formal T4-A summary written"
)

summary_sha256 = sha256_file(
    SUMMARY_PATH
)

print(
    f"Summary: "
    f"{safe_relative(SUMMARY_PATH)}"
)

print(
    f"SHA256: "
    f"{summary_sha256}"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section(
    "SCRIPT 67 SUMMARY"
)

print(
    f"Failures: {len(FAILURES)}"
)

for message in FAILURES:
    print(
        f"  - {message}"
    )

print()

if FAILURES:

    print(
        "RESULT: FAIL"
    )

    print()
    print(
        "Do NOT proceed to T4-B."
    )

    sys.exit(1)


print(
    "RESULT: PASS"
)

print()
print(
    "FORMAL T4-A TRAINING COMPLETE."
)

print()
print(
    f"Epochs completed : {len(history)}"
)

print(
    f"Best epoch       : {best_epoch}"
)

print()
print(
    "BEST VALIDATION RETRIEVAL"
)

print(
    f"Audio→Caption R@1  = "
    f"{reload_metrics['audio_to_caption_R@1']:.8f}"
)

print(
    f"Audio→Caption R@5  = "
    f"{reload_metrics['audio_to_caption_R@5']:.8f}"
)

print(
    f"Audio→Caption R@10 = "
    f"{reload_metrics['audio_to_caption_R@10']:.8f}"
)

print(
    f"Caption→Audio R@1  = "
    f"{reload_metrics['caption_to_audio_R@1']:.8f}"
)

print(
    f"Caption→Audio R@5  = "
    f"{reload_metrics['caption_to_audio_R@5']:.8f}"
)

print(
    f"Caption→Audio R@10 = "
    f"{reload_metrics['caption_to_audio_R@10']:.8f}"
)

print(
    f"VAL mR              = "
    f"{reload_metrics['mR']:.8f}"
)

print()
print(
    "Checkpoint:"
)

print(
    f"  {safe_relative(CHECKPOINT_PATH)}"
)

print(
    f"  SHA256 = "
    f"{sha256_file(CHECKPOINT_PATH)}"
)

print()
print(
    "History:"
)

print(
    f"  {safe_relative(HISTORY_CSV_PATH)}"
)

print()
print(
    "Summary:"
)

print(
    f"  {safe_relative(SUMMARY_PATH)}"
)

print(
    f"  SHA256 = {summary_sha256}"
)

print()
print(
    "TEST retrieval observed: NO"
)

print()
print(
    "STOP HERE."
)

print(
    "Do not interpret T4-A versus T4-B yet."
)

print(
    "Next step, after reviewing this result:"
)

print(
    "Script 68 — formal T4-B partial dual-encoder fine-tuning."
)