# CoFonet

This repository contains the evaluation artifacts for the paper.

For thereview period, this release is intentionally limited to checkpoint-based evaluation. It includes the public validation/test splits and a pretrained CoFoNet checkpoint.

## Review-Period Release Scope

The current release provides:

* **`evaluate.py`**: Evaluation-only entry point for loading the released checkpoint and running validation/test inference.
* **`data.7z`**: Public validation and test splits used for review.
* **`model/`**: Released checkpoint directory. The default checkpoint is `model/f1_vid=0.5064.ckpt`.
* **`pyproject.toml` / `uv.lock`**: Dependency definitions for environment reproduction.
* **`results/`**: Directory where evaluation metrics and visualization plots are written.

After the paper is accepted, we will publicly release the complete dataset and the full code.

---

## Environment Setup

We use **[uv](https://github.com/astral-sh/uv)** for fast and reproducible dependency management.

### 1. Install uv

If `uv` is not installed, install it with one of the following commands:

```bash
# Via pip
pip install uv

# Or via curl on Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync the Environment

From the project root, run:

```bash
uv sync
```

### 3. Extract the Public Validation/Test Data

Extract `data.7z` into the project root:

```bash
7z x data.7z
```

The evaluator supports either of the following extracted layouts:

```bash
data/
└── SALT/
    └── monitored_tor/
        ├── val_features.npy
        ├── val_label_vid_adjusted.npy
        ├── val_label_playlist_code.npy
        ├── test_features.npy
        ├── test_label_vid_adjusted.npy
        └── test_label_playlist_code.npy
```

or:

```bash
data/
└── processed/
    └── SALT/
        └── monitored_tor/
            ├── val_features.npy
            ├── val_label_vid_adjusted.npy
            ├── val_label_playlist_code.npy
            ├── test_features.npy
            ├── test_label_vid_adjusted.npy
            └── test_label_playlist_code.npy
```

If open-world evaluation data is present, place the corresponding files under `unmonitored_tor/` with the same `val_*` or `test_*` naming convention.

## Usage Instructions

### Evaluate the Released Checkpoint on the Test Set

The default command loads the checkpoint from `model/` and evaluates the public test split:

```bash
uv run evaluate.py --rep SALT --proto tor --plat youtube
```

Equivalent explicit command:

```bash
uv run evaluate.py \
    --rep SALT \
    --proto tor \
    --plat youtube \
    --split test \
    --checkpoint ./model/f1_vid=0.5064.ckpt
```

### Evaluate the Validation Split

```bash
uv run evaluate.py \
    --rep SALT \
    --proto tor \
    --plat youtube \
    --split val
```

 The complete dataset and full source code will be released after paper acceptance.
