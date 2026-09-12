# GNN-Based BERT for Understanding Context from Music

**Supervised Context Tagging and Contrastive MusicCaps Alignment**

A multimodal deep-learning project investigating whether **graph-based audio representations add useful information beyond a strong BERT text baseline for music-context understanding**.

## Overview

The project uses a single MusicCaps dataset across four connected tasks:

1. **Task 1 — Text:** BERT-based multi-label music-context classification
2. **Task 2 — Audio:** CNN, GraphSAGE, and GAT audio classification
3. **Task 3 — Fusion:** GNN–BERT multimodal classification
4. **Task 4 — Alignment:** Contrastive audio–caption learning, retrieval, prototype tag readout, qualitative analysis, and human evaluation

The main research question is:

> **Does combining BERT text representations with graph-based audio representations improve music-context prediction over single-modality models?**

---

## Dataset

The project uses **MusicCaps**, which provides timestamped music clips, human-written captions, and aspect descriptions.

| Property | Value |
|---|---:|
| Original metadata entries | 5,521 |
| Usable clips | 5,140 |
| Excluded clips | 381 |
| Train | 4,112 |
| Validation | 514 |
| Test | 514 |
| Final labels | 20 |
| Clip duration | 10 s |
| Sample rate | 22,050 Hz |

The dataset split was frozen **before label construction**, and the final 20-label vocabulary was constructed using **TRAIN only**.

Raw MusicCaps audio is not distributed in this repository.

---

## Models

### Task 1 — BERT

Two text models were evaluated:

- Frozen BERT + linear classifier
- Fully fine-tuned `bert-base-uncased`

The fine-tuned BERT model was selected as the Task-1 winner.

### Task 2 — Audio

Audio-only models included:

- CNN on log-mel spectrograms
- GraphSAGE with temporal graph G1
- GraphSAGE with sparse temporal + similarity graph G2
- GAT with G2

CNN was the strongest audio-only classifier, while GAT was retained as the graph encoder for multimodal experiments.

### Task 3 — Multimodal Fusion

Two fusion approaches were evaluated:

- **M3 — Early Fusion:** projected BERT and GAT representations are concatenated
- **M4 — Cross-Attention:** the pooled graph representation attends over BERT token representations

M3 Early Fusion was selected as the best fusion model.

### Task 4 — Contrastive Learning

Audio and caption representations are projected into a shared **256-dimensional normalized embedding space** and trained using a symmetric InfoNCE objective.

Task 4 evaluates:

- audio → caption retrieval
- caption → audio retrieval
- prototype-based tag prediction
- qualitative retrieval
- human evaluation

---

## Main Results

### Validation Macro AP

| Model | Macro AP |
|---|---:|
| Frozen BERT | 0.559 |
| Fine-tuned BERT | **0.841** |
| CNN | **0.229** |
| GraphSAGE G1 | 0.178 |
| GraphSAGE G2 | 0.184 |
| GAT G2 | 0.188 |
| Early Fusion M3 | **0.840** |
| Cross-Attention M4 | 0.832 |

### Final TEST Results

Three-run models report **mean ± sample SD**.

| Model | Input | Macro AP |
|---|---|---:|
| Majority baseline | TRAIN prior | 0.1033 |
| CNN | Audio | 0.2232 ± 0.0060 |
| **BERT** | Text | **0.8313 ± 0.0044** |
| GraphSAGE | Audio graph | 0.1944 |
| GAT | Audio graph | 0.1877 |
| **Early Fusion** | Audio + Text | **0.8319 ± 0.0020** |
| Cross-Attention | Audio + Text | 0.8270 |

The Early Fusion–BERT difference was only **+0.00058 Macro AP** and changed sign across seeds. The experiments therefore **do not establish a reliable overall improvement of GNN–BERT fusion over BERT alone**.

---

## Task 4 Retrieval

Final three-seed TEST retrieval:

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Audio → Caption | ~0.0227 | 0.0856 | 0.1368 |
| Caption → Audio | ~0.0195 | 0.0707 | 0.1420 |

**Mean Recall:** `0.0796 ± 0.0065`

Random retrieval over 514 candidates is approximately `0.0104`, so the learned model performs about **7.7× above random mean recall**, although exact-pair R@1 remains low.

---

## Human Evaluation

The Task-4 listening study used:

- 5 listeners
- 10 fixed retrieval queries
- 50 ratings
- 1–5 rating scale

**Pooled mean:** `3.02 / 5`  
**Sample SD:** `≈ 1.58`

This suggests moderate but variable semantic correspondence between retrieved audio and captions.

---

## Repository Structure

```text
.
├── code/
│   ├── configs/
│   │   └── config.yaml
│   ├── scripts/
│   ├── requirements-lock.txt
│   └── .gitignore
│
├── RESULTS/
├── report/
├── human_evaluation/
├── Human Eval Audio/
├── GNN_BERT_Report_Overleaf.zip
└── Human_Evaluation_Supplement.zip
```

The public `code/scripts/` directory contains the **important scripts required to reconstruct the experimental pipeline**. Development-only debug, retry, smoke-test, and superseded scripts were intentionally excluded.

---

## Reproducibility

Create an environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements-lock.txt
```

Main configuration:

```text
code/configs/config.yaml
```

Core implementation:

```text
code/scripts/
```

The repository intentionally excludes:

- raw MusicCaps audio
- generated feature caches
- BERT token caches
- graph caches
- intermediate tensors
- trained model checkpoints
- virtual environments
- development logs

---

## Experimental Controls

The project uses several controls to reduce leakage and post-hoc selection:

- fixed TRAIN / VAL / TEST split
- TRAIN-only label-vocabulary construction
- validation-only model selection
- frozen thresholds before final TEST evaluation
- aligned track IDs across modalities
- multi-seed final evaluation
- preprocessing artifact locking
- Task-4 protocol frozen before TEST retrieval

---

## Main Conclusion

- **Fine-tuned BERT strongly outperforms the tested audio-only models**
- **Early GNN–BERT fusion matches BERT closely but does not reliably outperform it**
- **Contrastive learning produces meaningful audio–text alignment above random retrieval**
- **Prototype-based tag readout remains substantially weaker than supervised classification**

The main contribution is a **controlled and reproducible comparison of text, audio, graph, fusion, and contrastive approaches for music-context understanding**.

---

## Limitations

Key limitations include:

- possible semantic overlap between MusicCaps captions and aspect-derived labels
- no independently verified artist/recording-level duplicate isolation across splits
- compressed nine-node graph representations
- stronger pretraining available to BERT than the audio models
- prototype tagging inherits previously supervised encoders
- small human-evaluation sample

---

## Authors

**Mohammed Ali Hossain**  
**Jarin Akter Mou**  
**Asiful Kanzan Auishik**

Department of Computer Science and Engineering  
BRAC University, Dhaka, Bangladesh

---

## Project Status

**Completed — Tasks 1 through 4**

This repository contains the public code, experimental results, report, and human-evaluation materials for the project.
