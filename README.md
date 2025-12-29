# Unveiling the Semantic Trail

This repository contains the source code and evaluation artifacts for the paper **Unveiling the Semantic Trail** submitted to **ACM CCS**.

These artifacts are designed to verify the reproducibility of our results, including the SALT feature extraction efficacy and the CoFoNet model performance.

## Repository Contents

* **`evaluate.py`**: The main entry point for training and evaluation.
* **`pyproject.toml` / `uv.lock`**: Dependency definitions for exact environment reproduction.
* **`data/`**: Directory for storing the pre-processed feature datasets.
* **`results/`**: Directory where logs, metrics, and visualization plots will be saved.

---

##  Environment Setup

We use **[uv](https://github.com/astral-sh/uv)** for ultra-fast, reliable, and cross-platform dependency management.

### 1. Install uv
If you do not have `uv` installed, please install it via pip or curl:

```bash
# Via pip (Recommended)
pip install uv

# Or via curl (Linux/macOS)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

### 2. Sync Environment
Navigate to the project root directory and run the following command. This will automatically create a virtual environment and install all dependencies (PyTorch, Lightning, Scikit-learn, etc.) with locked versions.
```bash
uv sync
```

To ensure the scripts run correctly, please verify that the pre-processed datasets  are placed in the ./data/processed/ directory.

```bash
7z x data.zip
```

The expected directory structure is:
``` bash
.
├── evaluate.py
├── pyproject.toml
├── uv.lock
├── results/
└── data/
    └── processed/
        └── SALT/
            ├── monitored_tor/
            │   ├── train_features.npy
            │   ├── train_label_vid_adjusted.npy
            │   └── ...
            └── unmonitored_tor/
                ├── train_features.npy
                └── ...
```

## Usage Instructions

1. Training (Closed World)
To train the CoFoNet model from scratch using the SALT representation on the Tor-YouTube dataset:

```bash
uv run evaluate.py --mode train --rep SALT --proto tor --plat youtube
```

2. Evaluation (Closed World)

To evaluate a pre-trained checkpoint on the test set and generate metrics:
```bash
# Please replace the path with the actual checkpoint file provided
uv run evaluate.py --mode eval \
    --rep SALT \
    --proto tor \
    --plat youtube \
    --checkpoint ./results/checkpoints/best_model.ckpt
```

3. Open World Evaluation (Dataset too big)
o evaluate the model in the Open World setting (identifying unmonitored background traffic):
```bash 
uv run evaluate.py --mode eval \
    --rep SALT \
    --proto tor \
    --plat youtube \
    --open_world \
    --checkpoint ./results/checkpoints/best_model_openworld.ckpt
```

After running the evaluation, the following results will be generated in the ./results/ directory

## Anonymity Statement

This repository has been fully anonymized for double-blind review. All personal identifiers, institutional metadata, and absolute paths have been removed.