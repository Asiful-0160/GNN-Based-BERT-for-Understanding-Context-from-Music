#!/usr/bin/env python3

"""
Script 36 — Formal Task-2 A1 CNN model-selection run

Pipeline Plan v6:
    A1 — CNN audio-only baseline

Architecture:
    Input (1,128,431)
    [Conv3x3 -> BN -> ReLU -> MaxPool2] x4
    channels: 1 -> 32 -> 64 -> 128 -> 256
    AdaptiveAvgPool(1,1)
    Linear(256,20)

Formal settings:
    AdamW
    LR = 1e-3
    weight decay = 1e-4
    batch size = 32
    early-stop patience = 3
    shared TRAIN-only BCEWithLogitsLoss(pos_weight)
    selection = validation macro mean per-label AP

Task-2 operational epoch ceiling:
    15 epochs

This fixes the ~15-epoch v6 planning guidance BEFORE
seeing any formal Task-2 validation results and will be
kept identical for CNN / GraphSAGE / GAT model selection.

Important:
- TRAIN is used for optimization.
- VAL is used for checkpoint selection / early stopping.
- TEST targets and inference are never used.
- No threshold sweep is performed.
- No scheduler is introduced.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

AUDIO_MANIFEST_PATH = (
    ROOT
    / "data/processed/audio_feature_manifest.parquet"
)

TARGETS_PATH = (
    ROOT
    / "data/processed/labels/musiccaps_targets.parquet"
)

VOCAB_PATH = (
    ROOT
    / "data/splits/label_vocab.json"
)

PRETRAINING_LOCK_PATH = (
    ROOT
    / "data/splits/pretraining_frozen_hashes.json"
)

TASK1_SELECTION_PATH = (
    ROOT
    / "results/runs/task1/task1_model_selection.json"
)

RESULTS_CSV_DIR = (
    ROOT
    / "results/csv"
)

EPOCH_METRICS_PATH = (
    RESULTS_CSV_DIR
    / "epoch_metrics.csv"
)

TRIALS_PATH = (
    RESULTS_CSV_DIR
    / "hyperparameter_trials.csv"
)

RUNS_DIR = (
    ROOT
    / "results/runs/task2"
)

MODEL_DIR = (
    ROOT
    / "models/task2_candidates"
)


# ============================================================
# FROZEN IDENTITIES
# ============================================================

EXPECTED_LOCK_SHA256 = (
    "caffa90d9de726a6a28fb15b2ee0e28f4abac8f0766c90d5b0e027a6c917b4e8"
)

EXPECTED_TASK1_WINNER_SHA256 = (
    "3b675acbacb7191d514e3cc51366944271e0d5a1b4d98f767c0ccab163f4e275"
)

EXPECTED_DATASET_TRACKS = 5140
EXPECTED_TRAIN_TRACKS = 4112
EXPECTED_VAL_TRACKS = 514

NUM_LABELS = 20

N_MELS = 128
MEL_FRAMES = 431

TARGET_COLUMNS = [
    f"y_{i:02d}"
    for i in range(NUM_LABELS)
]


# ============================================================
# FORMAL A1 SETTINGS
# ============================================================

TASK = "task2"
VARIANT = "A1"
MODEL_NAME = "CNN"

SEED = 42

BATCH_SIZE = 32

MAX_EPOCHS = 15

LEARNING_RATE = 1.0e-3

WEIGHT_DECAY = 1.0e-4

EARLY_STOP_PATIENCE = 3

POS_WEIGHT_CAP = 10.0

SELECTION_METRIC_NAME = (
    "val_auc_pr_macro_mean_ap"
)

IMPROVEMENT_EPSILON = 1.0e-12


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(
    raw_path: str,
) -> Path:

    path = Path(
        str(raw_path)
    )

    if not path.is_absolute():
        path = ROOT / path

    return path


def relative(path: Path) -> str:

    try:
        return str(
            path.relative_to(ROOT)
        )

    except ValueError:
        return str(path)


def get_git_commit() -> str:

    try:

        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        value = (
            result.stdout.strip()
        )

        if value:
            return value

    except Exception:
        pass

    return "UNAVAILABLE"


# ============================================================
# CSV HELPERS
# ============================================================

def write_dataframe_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )


def append_rows_to_csv(
    rows: list[dict],
    path: Path,
) -> None:

    new_frame = pd.DataFrame(
        rows
    )

    if path.exists():

        old_frame = pd.read_csv(
            path
        )

        # Preserve union of columns across task families.
        combined = pd.concat(
            [
                old_frame,
                new_frame,
            ],
            ignore_index=True,
            sort=False,
        )

    else:

        combined = new_frame

    write_dataframe_atomic(
        combined,
        path,
    )


# ============================================================
# DUPLICATE RUN PROTECTION
# ============================================================

def assert_no_completed_duplicate() -> None:

    if not TRIALS_PATH.exists():
        return

    trials = pd.read_csv(
        TRIALS_PATH
    )

    required = {
        "task",
        "variant",
        "seed",
        "status",
    }

    if not required.issubset(
        trials.columns
    ):
        return

    duplicate = trials.loc[
        (trials["task"] == TASK)
        & (trials["variant"] == VARIANT)
        & (trials["seed"] == SEED)
        & (trials["status"] == "COMPLETE")
    ]

    require(
        len(duplicate) == 0,
        (
            "A completed formal A1 CNN "
            "seed-42 run already exists. "
            "Refusing accidental rerun."
        ),
    )


# ============================================================
# FROZEN ARTIFACT CHECKS
# ============================================================

def verify_frozen_inputs():

    section(
        "1. FROZEN ARTIFACT VERIFICATION"
    )

    lock_sha = sha256_file(
        PRETRAINING_LOCK_PATH
    )

    print(
        "Expected lock SHA256:",
        EXPECTED_LOCK_SHA256,
    )

    print(
        "Actual lock SHA256:  ",
        lock_sha,
    )

    require(
        lock_sha
        == EXPECTED_LOCK_SHA256,
        "Pretraining lock changed.",
    )

    with PRETRAINING_LOCK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        lock = json.load(
            file
        )

    require(
        lock.get(
            "lock_type"
        )
        == "pretraining_artifact_lock",
        "Unexpected lock type.",
    )

    require(
        lock.get(
            "dataset_tracks"
        )
        == EXPECTED_DATASET_TRACKS,
        "Frozen dataset count changed.",
    )

    critical_files = (
        lock.get(
            "critical_files"
        )
    )

    require(
        isinstance(
            critical_files,
            dict,
        ),
        "Malformed critical_files.",
    )

    for key in [
        "audio_feature_manifest",
        "targets",
        "config",
        "original_dataset_lock",
    ]:

        record = critical_files.get(
            key
        )

        require(
            isinstance(record, dict),
            (
                f"Missing critical file "
                f"record: {key}"
            ),
        )

        path = (
            ROOT
            / record["path"]
        )

        require(
            path.is_file(),
            (
                f"Missing frozen file: "
                f"{path}"
            ),
        )

        actual_sha = sha256_file(
            path
        )

        require(
            actual_sha
            == record["sha256"],
            (
                f"Frozen artifact changed: "
                f"{key}"
            ),
        )

        print(
            f"PASS  {key:<24} "
            f"{record['path']}"
        )

    # Task 1 winner is not an input to CNN,
    # but its frozen selection must remain intact.
    require(
        TASK1_SELECTION_PATH.is_file(),
        (
            "Missing frozen Task-1 "
            "selection artifact."
        ),
    )

    with TASK1_SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        task1 = json.load(
            file
        )

    require(
        task1.get(
            "status"
        )
        == "FROZEN",
        "Task-1 selection is not frozen.",
    )

    winner = task1.get(
        "winner"
    )

    require(
        isinstance(
            winner,
            dict,
        ),
        "Task-1 winner malformed.",
    )

    require(
        winner.get(
            "checkpoint_sha256"
        )
        == EXPECTED_TASK1_WINNER_SHA256,
        "Task-1 winner changed.",
    )

    winner_checkpoint = (
        ROOT
        / winner[
            "checkpoint_path"
        ]
    )

    require(
        winner_checkpoint.is_file(),
        (
            "Frozen Task-1 winning "
            "checkpoint missing."
        ),
    )

    actual_winner_sha = (
        sha256_file(
            winner_checkpoint
        )
    )

    require(
        actual_winner_sha
        == EXPECTED_TASK1_WINNER_SHA256,
        (
            "Task-1 winning checkpoint "
            "was modified."
        ),
    )

    print(
        "PASS  task1_winner_checkpoint  "
        + relative(
            winner_checkpoint
        )
    )

    print()
    print(
        "Frozen-artifact verification: PASS"
    )

    print(
        "TEST examples/targets loaded: 0"
    )

    return {
        "pretraining_lock_sha256":
            lock_sha,

        "task1_winner_checkpoint_sha256":
            actual_winner_sha,
    }


# ============================================================
# DEVELOPMENT DATA
# ============================================================

def load_development_data():

    section(
        "2. TRAIN + VAL DATA"
    )

    manifest = pd.read_parquet(
        AUDIO_MANIFEST_PATH,
        columns=[
            "track_id",
            "mel_path",
            "mel_bins",
            "mel_frames",
            "cache_status",
        ],
    )

    require(
        len(manifest)
        == EXPECTED_DATASET_TRACKS,
        "Audio manifest size mismatch.",
    )

    require(
        manifest[
            "track_id"
        ].is_unique,
        (
            "Duplicate audio-manifest "
            "track_id."
        ),
    )

    require(
        (
            manifest[
                "mel_bins"
            ]
            == N_MELS
        ).all(),
        "Unexpected mel-bin count.",
    )

    require(
        (
            manifest[
                "mel_frames"
            ]
            == MEL_FRAMES
        ).all(),
        "Unexpected mel-frame count.",
    )

    print(
        "cache_status values:",
        sorted(
            manifest[
                "cache_status"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    train_targets = pd.read_parquet(
        TARGETS_PATH,
        filters=[
            (
                "split",
                "==",
                "train",
            )
        ],
    )

    val_targets = pd.read_parquet(
        TARGETS_PATH,
        filters=[
            (
                "split",
                "==",
                "val",
            )
        ],
    )

    require(
        len(train_targets)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN target count mismatch.",
    )

    require(
        len(val_targets)
        == EXPECTED_VAL_TRACKS,
        "VAL target count mismatch.",
    )

    train_frame = (
        train_targets.merge(
            manifest,
            on="track_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("track_id")
        .reset_index(drop=True)
    )

    val_frame = (
        val_targets.merge(
            manifest,
            on="track_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("track_id")
        .reset_index(drop=True)
    )

    require(
        len(train_frame)
        == EXPECTED_TRAIN_TRACKS,
        "TRAIN audio merge failed.",
    )

    require(
        len(val_frame)
        == EXPECTED_VAL_TRACKS,
        "VAL audio merge failed.",
    )

    require(
        set(
            train_frame[
                "track_id"
            ]
        ).isdisjoint(
            set(
                val_frame[
                    "track_id"
                ]
            )
        ),
        "TRAIN/VAL overlap found.",
    )

    print(
        "TRAIN:",
        len(train_frame),
    )

    print(
        "VAL:  ",
        len(val_frame),
    )

    print(
        "TEST target rows loaded: 0"
    )

    print(
        "Development mapping: PASS"
    )

    return (
        train_frame,
        val_frame,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_pos_weight(
    train_frame: pd.DataFrame,
):

    section(
        "3. TRAIN-ONLY CLASS WEIGHTS"
    )

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        vocab = json.load(
            file
        )

    labels = vocab[
        "labels"
    ]

    require(
        len(labels)
        == NUM_LABELS,
        "Expected 20 frozen labels.",
    )

    train_y = (
        train_frame[
            TARGET_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    support = (
        train_y.sum(
            axis=0
        )
        .astype(
            np.int64
        )
    )

    frozen_support = (
        np.asarray(
            [
                int(
                    item[
                        "train_support"
                    ]
                )
                for item
                in labels
            ],
            dtype=np.int64,
        )
    )

    require(
        np.array_equal(
            support,
            frozen_support,
        ),
        (
            "TRAIN support differs "
            "from frozen vocabulary."
        ),
    )

    negatives = (
        EXPECTED_TRAIN_TRACKS
        - support
    )

    pos_weight = (
        np.minimum(
            negatives / support,
            POS_WEIGHT_CAP,
        )
        .astype(
            np.float32
        )
    )

    require(
        np.isfinite(
            pos_weight
        ).all(),
        "Non-finite class weight.",
    )

    print(
        "TRAIN support == frozen vocabulary: PASS"
    )

    print(
        "pos_weight = min(negative/positive,10): PASS"
    )

    label_names = [
        item[
            "label"
        ]
        for item
        in labels
    ]

    return (
        pos_weight,
        label_names,
    )


# ============================================================
# DATASET
# ============================================================

class MelDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
    ) -> None:

        self.paths = [
            resolve_path(
                path
            )
            for path
            in frame[
                "mel_path"
            ].tolist()
        ]

        self.targets = (
            frame[
                TARGET_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

    def __len__(self):

        return len(
            self.paths
        )

    def __getitem__(
        self,
        index: int,
    ):

        path = self.paths[
            index
        ]

        mel = np.load(
            path,
            allow_pickle=False,
        )

        if mel.shape != (
            N_MELS,
            MEL_FRAMES,
        ):

            raise RuntimeError(
                (
                    "Mel shape changed: "
                    f"{path} -> "
                    f"{mel.shape}"
                )
            )

        if not np.isfinite(
            mel
        ).all():

            raise RuntimeError(
                (
                    "Non-finite mel values: "
                    f"{path}"
                )
            )

        mel = np.array(
            mel,
            dtype=np.float32,
            copy=True,
        )

        return (
            torch.from_numpy(
                mel
            ).unsqueeze(0),

            torch.tensor(
                self.targets[
                    index
                ],
                dtype=torch.float32,
            ),
        )


# ============================================================
# CNN
# ============================================================

class CNNBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2,
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2,
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2,
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                2,
                2,
            ),
        )

        self.pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = (
            nn.Linear(
                256,
                NUM_LABELS,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        x = self.features(
            x
        )

        x = self.pool(
            x
        )

        x = torch.flatten(
            x,
            1,
        )

        return self.classifier(
            x
        )


# ============================================================
# METRICS
# ============================================================

def compute_ap(
    y_true: np.ndarray,
    y_probability: np.ndarray,
):

    require(
        y_true.shape
        == y_probability.shape,
        "Metric shape mismatch.",
    )

    per_label = (
        average_precision_score(
            y_true,
            y_probability,
            average=None,
        )
    )

    require(
        len(per_label)
        == NUM_LABELS,
        (
            "Expected 20 per-label "
            "AP values."
        ),
    )

    require(
        np.isfinite(
            per_label
        ).all(),
        "Non-finite AP value.",
    )

    macro = float(
        np.mean(
            per_label
        )
    )

    micro = float(
        average_precision_score(
            y_true,
            y_probability,
            average="micro",
        )
    )

    return (
        macro,
        micro,
        per_label.astype(
            np.float64
        ),
    )


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_examples = 0

    for mel, targets in loader:

        mel = mel.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            mel
        )

        require(
            logits.shape
            == (
                targets.shape[0],
                NUM_LABELS,
            ),
            (
                "Unexpected CNN "
                "logit shape."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            (
                "Non-finite training "
                "logits."
            ),
        )

        loss = criterion(
            logits,
            targets,
        )

        require(
            bool(
                torch.isfinite(
                    loss
                )
            ),
            "Non-finite training loss.",
        )

        loss.backward()

        for name, parameter in (
            model.named_parameters()
        ):

            if parameter.grad is None:
                continue

            require(
                bool(
                    torch.isfinite(
                        parameter.grad
                    ).all()
                ),
                (
                    "Non-finite gradient: "
                    f"{name}"
                ),
            )

        optimizer.step()

        batch_size = int(
            targets.shape[0]
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
            )
            * batch_size
        )

        total_examples += (
            batch_size
        )

    require(
        total_examples
        == EXPECTED_TRAIN_TRACKS,
        (
            "TRAIN epoch did not "
            "process exactly 4112 tracks."
        ),
    )

    return (
        total_loss
        / total_examples
    )


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    target_batches = []
    probability_batches = []

    for mel, targets in loader:

        mel = mel.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        logits = model(
            mel
        )

        require(
            bool(
                torch.isfinite(
                    logits
                ).all()
            ),
            (
                "Non-finite validation "
                "logits."
            ),
        )

        loss = criterion(
            logits,
            targets,
        )

        require(
            bool(
                torch.isfinite(
                    loss
                )
            ),
            (
                "Non-finite validation "
                "loss."
            ),
        )

        probability = (
            torch.sigmoid(
                logits
            )
        )

        batch_size = int(
            targets.shape[0]
        )

        total_loss += (
            float(
                loss.detach()
                .cpu()
            )
            * batch_size
        )

        total_examples += (
            batch_size
        )

        target_batches.append(
            targets.detach()
            .cpu()
            .numpy()
        )

        probability_batches.append(
            probability.detach()
            .cpu()
            .numpy()
        )

    require(
        total_examples
        == EXPECTED_VAL_TRACKS,
        (
            "VAL did not process "
            "exactly 514 tracks."
        ),
    )

    y_true = np.concatenate(
        target_batches,
        axis=0,
    )

    y_probability = np.concatenate(
        probability_batches,
        axis=0,
    )

    (
        macro_ap,
        micro_ap,
        per_label_ap,
    ) = compute_ap(
        y_true,
        y_probability,
    )

    return {
        "loss":
            total_loss
            / total_examples,

        "macro_ap":
            macro_ap,

        "micro_ap":
            micro_ap,

        "per_label_ap":
            per_label_ap,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    epoch,
    val_macro_ap,
    val_loss,
    run_id,
    pos_weight,
    lock_sha,
):

    state_dict = {
        key:
            value.detach()
            .cpu()
        for key, value
        in model.state_dict().items()
    }

    checkpoint = {
        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "run_id":
            run_id,

        "seed":
            SEED,

        "epoch":
            epoch,

        "val_auc_pr_macro_mean_ap":
            val_macro_ap,

        "val_loss":
            val_loss,

        "architecture": {
            "input":
                [1, 128, 431],

            "channels":
                [1, 32, 64, 128, 256],

            "conv":
                {
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },

            "pool":
                {
                    "kernel_size": 2,
                    "stride": 2,
                },

            "adaptive_pool":
                [1, 1],

            "classifier":
                "Linear(256,20)",
        },

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "early_stop_patience":
            EARLY_STOP_PATIENCE,

        "scheduler":
            None,

        "pos_weight":
            pos_weight.tolist(),

        "pretraining_lock_sha256":
            lock_sha,

        "model_state_dict":
            state_dict,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "SCRIPT 36 — FORMAL TASK-2 A1 CNN"
    )
    print("=" * 78)

    print()
    print(
        "TRAIN -> optimization"
    )

    print(
        "VAL -> early stopping / checkpoint selection"
    )

    print(
        "TEST -> NOT LOADED / NOT EVALUATED"
    )

    print(
        "Threshold sweep -> NOT RUN"
    )

    assert_no_completed_duplicate()

    artifact_info = (
        verify_frozen_inputs()
    )

    (
        train_frame,
        val_frame,
    ) = load_development_data()

    (
        pos_weight_np,
        label_names,
    ) = compute_pos_weight(
        train_frame
    )

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    section(
        "4. ENVIRONMENT"
    )

    require(
        torch.cuda.is_available(),
        "CUDA required.",
    )

    device = torch.device(
        "cuda"
    )

    set_seed(
        SEED
    )

    gpu_name = (
        torch.cuda
        .get_device_name(0)
    )

    git_commit = (
        get_git_commit()
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "scikit-learn:",
        sklearn.__version__,
    )

    print(
        "GPU:",
        gpu_name,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "cudnn.deterministic:",
        torch.backends
        .cudnn.deterministic,
    )

    print(
        "cudnn.benchmark:",
        torch.backends
        .cudnn.benchmark,
    )

    print(
        "Git commit:",
        git_commit,
    )


    # ========================================================
    # RUN ID
    # ========================================================

    timestamp = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    run_id = (
        f"task2_A1_CNN_"
        f"seed{SEED}_"
        f"{timestamp}"
    )

    run_dir = (
        RUNS_DIR
        / run_id
    )

    checkpoint_path = (
        MODEL_DIR
        / (
            f"{run_id}_best.pt"
        )
    )

    require(
        not run_dir.exists(),
        "Run directory already exists.",
    )

    require(
        not checkpoint_path.exists(),
        (
            "Checkpoint path "
            "already exists."
        ),
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_CSV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    section(
        "5. DATALOADERS"
    )

    train_dataset = MelDataset(
        train_frame
    )

    val_dataset = MelDataset(
        val_frame
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    require(
        len(train_loader) == 129,
        (
            "Unexpected TRAIN "
            "batch count."
        ),
    )

    require(
        len(val_loader) == 17,
        (
            "Unexpected VAL "
            "batch count."
        ),
    )

    print(
        "TRAIN batches:",
        len(train_loader),
    )

    print(
        "VAL batches:",
        len(val_loader),
    )


    # ========================================================
    # MODEL
    # ========================================================

    section(
        "6. MODEL"
    )

    model = CNNBaseline()

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    require(
        total_parameters
        == 393940,
        (
            "CNN parameter count "
            "differs from Script 35."
        ),
    )

    require(
        trainable_parameters
        == total_parameters,
        (
            "Unexpected frozen "
            "CNN parameters."
        ),
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    model = model.to(
        device
    )


    # ========================================================
    # OPTIMIZATION
    # ========================================================

    section(
        "7. FORMAL OPTIMIZATION CONFIG"
    )

    pos_weight = torch.tensor(
        pos_weight_np,
        dtype=torch.float32,
        device=device,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Maximum epochs:",
        MAX_EPOCHS,
    )

    print(
        "Early-stop patience:",
        EARLY_STOP_PATIENCE,
    )

    print(
        "Scheduler: NONE"
    )

    print(
        "Selection metric:",
        SELECTION_METRIC_NAME,
    )

    print(
        "Loss: BCEWithLogitsLoss("
        "TRAIN-only pos_weight)"
    )


    # ========================================================
    # SAVE RUN SPECIFICATION BEFORE TRAINING
    # ========================================================

    run_spec = {
        "run_id":
            run_id,

        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "formal_model_selection_run":
            True,

        "seed":
            SEED,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "architecture": {
            "input":
                [1, 128, 431],

            "channels":
                [1, 32, 64, 128, 256],

            "block":
                (
                    "Conv3x3 -> BN -> "
                    "ReLU -> MaxPool2"
                ),

            "conv_kernel":
                3,

            "conv_stride":
                1,

            "conv_padding":
                1,

            "pool_kernel":
                2,

            "pool_stride":
                2,

            "adaptive_pool":
                [1, 1],

            "classifier":
                "Linear(256,20)",

            "parameter_count":
                total_parameters,
        },

        "training": {
            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "epoch_ceiling_basis":
                (
                    "Operationalized from "
                    "Pipeline v6 Task-2 "
                    "planning guidance <=~15; "
                    "fixed before formal "
                    "Task-2 results."
                ),

            "early_stop_patience":
                EARLY_STOP_PATIENCE,

            "optimizer":
                "torch.optim.AdamW",

            "scheduler":
                None,
        },

        "loss": {
            "name":
                "BCEWithLogitsLoss",

            "pos_weight_source":
                "TRAIN only",

            "formula":
                "min(negative/positive,10)",

            "pos_weight":
                pos_weight_np.tolist(),
        },

        "selection": {
            "metric":
                SELECTION_METRIC_NAME,

            "aggregation":
                "mean over 20 labels",

            "implementation":
                (
                    "sklearn.metrics."
                    "average_precision_score"
                    "(average=None), then mean"
                ),

            "threshold_free":
                True,
        },

        "frozen_artifacts":
            artifact_info,

        "environment": {
            "torch":
                torch.__version__,

            "sklearn":
                sklearn.__version__,

            "gpu":
                gpu_name,

            "git_commit":
                git_commit,
        },
    }

    with (
        run_dir
        / "run_spec.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            run_spec,
            file,
            indent=2,
        )


    # ========================================================
    # TRAINING
    # ========================================================

    section(
        "8. FORMAL TRAINING"
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    start_time = (
        time.perf_counter()
    )

    best_metric = -math.inf

    best_epoch = None

    best_val_loss = None

    epochs_without_improvement = 0

    stopped_early = False

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = float(
            validation[
                "loss"
            ]
        )

        val_macro_ap = float(
            validation[
                "macro_ap"
            ]
        )

        val_micro_ap = float(
            validation[
                "micro_ap"
            ]
        )

        improved = (
            val_macro_ap
            > (
                best_metric
                + IMPROVEMENT_EPSILON
            )
        )

        if improved:

            best_metric = (
                val_macro_ap
            )

            best_epoch = (
                epoch
            )

            best_val_loss = (
                val_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                epoch=epoch,
                val_macro_ap=(
                    val_macro_ap
                ),
                val_loss=val_loss,
                run_id=run_id,
                pos_weight=(
                    pos_weight_np
                ),
                lock_sha=(
                    artifact_info[
                        "pretraining_lock_sha256"
                    ]
                ),
            )

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        history.append(
            {
                "run_id":
                    run_id,

                "task":
                    TASK,

                "variant":
                    VARIANT,

                "model":
                    MODEL_NAME,

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "val_auc_pr_macro":
                    val_macro_ap,

                "val_auc_pr_micro_diagnostic":
                    val_micro_ap,

                "learning_rate_end":
                    LEARNING_RATE,

                "improved":
                    improved,

                "best_val_auc_pr_so_far":
                    best_metric,

                "epochs_without_improvement":
                    epochs_without_improvement,

                "epoch_seconds":
                    epoch_seconds,

                "test_split_used":
                    False,
            }
        )

        write_dataframe_atomic(
            pd.DataFrame(
                history
            ),
            run_dir
            / "history.csv",
        )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss="
            f"{train_loss:.6f} | "
            f"val_loss="
            f"{val_loss:.6f} | "
            f"val_macro_AP="
            f"{val_macro_ap:.6f} | "
            f"val_micro_AP="
            f"{val_micro_ap:.6f} | "
            f"time="
            f"{epoch_seconds:.1f}s"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            stopped_early = True

            print()
            print(
                "Early stopping triggered "
                "after "
                f"{EARLY_STOP_PATIENCE} "
                "consecutive non-improving "
                "epochs."
            )

            break


    # ========================================================
    # REVALIDATE BEST CHECKPOINT
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    require(
        best_epoch is not None,
        "No best epoch selected.",
    )

    require(
        checkpoint_path.is_file(),
        (
            "Best CNN checkpoint "
            "was not saved."
        ),
    )

    section(
        "9. BEST CHECKPOINT REVALIDATION"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    require(
        checkpoint[
            "variant"
        ]
        == VARIANT,
        (
            "Checkpoint variant "
            "mismatch."
        ),
    )

    require(
        int(
            checkpoint[
                "seed"
            ]
        )
        == SEED,
        (
            "Checkpoint seed "
            "mismatch."
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    revalidated = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    reproduced_metric = float(
        revalidated[
            "macro_ap"
        ]
    )

    require(
        abs(
            reproduced_metric
            - best_metric
        )
        <= 1.0e-10,
        (
            "Reloaded best CNN "
            "checkpoint failed to "
            "reproduce validation AP."
        ),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Recorded macro AP:",
        f"{best_metric:.8f}",
    )

    print(
        "Reloaded macro AP:",
        f"{reproduced_metric:.8f}",
    )

    print(
        "Checkpoint revalidation: PASS"
    )


    # ========================================================
    # PER-LABEL AP
    # ========================================================

    per_label_frame = (
        pd.DataFrame(
            {
                "index":
                    np.arange(
                        NUM_LABELS
                    ),

                "label":
                    label_names,

                "val_average_precision":
                    revalidated[
                        "per_label_ap"
                    ],
            }
        )
    )

    write_dataframe_atomic(
        per_label_frame,
        run_dir
        / "best_val_per_label_ap.csv",
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    epochs_completed = len(
        history
    )

    peak_vram_gb = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / (1024 ** 3)
    )

    result = {
        "status":
            "COMPLETE",

        "run_id":
            run_id,

        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "seed":
            SEED,

        "best_epoch":
            best_epoch,

        "best_val_auc_pr_macro":
            best_metric,

        "best_val_loss":
            best_val_loss,

        "epochs_completed":
            epochs_completed,

        "stopped_early":
            stopped_early,

        "test_split_used":
            False,

        "threshold_tuning_used":
            False,

        "checkpoint_path":
            relative(
                checkpoint_path
            ),

        "runtime_seconds":
            runtime,

        "peak_vram_gb":
            peak_vram_gb,

        "pretraining_lock_sha256":
            artifact_info[
                "pretraining_lock_sha256"
            ],
    }

    with (
        run_dir
        / "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # Only append globally after successful completion.
    append_rows_to_csv(
        history,
        EPOCH_METRICS_PATH,
    )

    trial_row = {
        "run_id":
            run_id,

        "status":
            "COMPLETE",

        "task":
            TASK,

        "variant":
            VARIANT,

        "model":
            MODEL_NAME,

        "strategy":
            "CNN audio-only baseline",

        "seed":
            SEED,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "max_epochs":
            MAX_EPOCHS,

        "epochs_completed":
            epochs_completed,

        "weight_decay":
            WEIGHT_DECAY,

        "scheduler":
            "NONE",

        "early_stop_patience":
            EARLY_STOP_PATIENCE,

        "selection_metric":
            SELECTION_METRIC_NAME,

        "best_epoch":
            best_epoch,

        "best_val_auc_pr_macro":
            best_metric,

        "best_val_loss":
            best_val_loss,

        "stopped_early":
            stopped_early,

        "trainable_parameters":
            trainable_parameters,

        "total_parameters":
            total_parameters,

        "runtime_seconds":
            runtime,

        "peak_vram_gb":
            peak_vram_gb,

        "checkpoint_path":
            relative(
                checkpoint_path
            ),

        "pretraining_lock_sha256":
            artifact_info[
                "pretraining_lock_sha256"
            ],

        "git_commit":
            git_commit,

        "test_split_used":
            False,
    }

    append_rows_to_csv(
        [
            trial_row
        ],
        TRIALS_PATH,
    )


    # ========================================================
    # FINISH
    # ========================================================

    section(
        "FORMAL TASK-2 A1 CNN COMPLETE"
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation macro mean AP:",
        f"{best_metric:.8f}",
    )

    print(
        "Epochs completed:",
        epochs_completed,
    )

    print(
        "Stopped early:",
        stopped_early,
    )

    print(
        "Runtime:",
        f"{runtime / 60:.2f} min",
    )

    print(
        "Peak allocated VRAM:",
        f"{peak_vram_gb:.3f} GB",
    )

    print(
        "Checkpoint:",
        relative(
            checkpoint_path
        ),
    )

    print(
        "Run directory:",
        relative(
            run_dir
        ),
    )

    print()

    print(
        "TEST examples used for "
        "labels/inference: 0"
    )

    print(
        "Threshold tuning performed: NO"
    )

    print()

    print(
        "DO NOT begin GraphSAGE "
        "until this CNN run has "
        "been reviewed."
    )


if __name__ == "__main__":
    main()