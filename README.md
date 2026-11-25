# Drug Molecule Generation (`FingerprintToSMILES`: ZINC250K → Tox21)

End-to-end machine learning pipeline for **fingerprint-conditioned de novo molecule generation** pre-trained on **ZINC250K** and fine-tuned on **non-toxic Tox21** molecules, managed with `uv` and PyTorch (CUDA + AMP).

---

## Model Architecture (`FingerprintToSMILES`)

An autoregressive causal Transformer decoder that conditions SMILES generation on a molecule's **2048-bit Morgan fingerprint** (radius 2) projected as a prefix token.

| Component | Configuration |
| :--- | :--- |
| **Input Conditioning** | 2048-bit Morgan Fingerprint (radius 2) → `Linear(2048, 256) + GELU + LayerNorm` |
| **Tokenizer / Vocab** | Atom-wise SMILES regex tokenizer (`52` tokens, `MAX_LEN = 92`) |
| **Transformer Backbone** | 4-layer causal `TransformerEncoder` (`d_model=256`, `nhead=8`, `dim_feedforward=1024`, `dropout=0.1`, Pre-LN) |
| **Total Parameters** | `3,734,836` (~14.27 MB checkpoint) |
| **Compute Environment** | NVIDIA GeForce RTX 4050 Laptop GPU (5.69 GiB VRAM), PyTorch `2.15.0+cu132`, AMP (`float16`) |

---

## Pipeline Stages ([`src/molecule_generation.ipynb`](src/molecule_generation.ipynb))

1. **Environment Setup & Configuration** — Imports, CUDA pre-flight check + smoke test, reproducibility seeds (`SEED=42`), and directory setup.
2. **Data Gathering** — Loads **ZINC250K** (`yairschiff/zinc250k` via Hugging Face `datasets`) and downloads **Tox21** (`tox21.csv.gz` from DeepChem S3).
3. **Data Preprocessing (ZINC250K)** — Subsets 50,000 train / 2,000 val / 2,000 test molecules, builds the 52-token SMILES vocabulary, computes 2048-bit Morgan fingerprints, and creates PyTorch `DataLoader`s.
4. **Training (ZINC250K Base Model)** — Trains `zinc_model` for 12 epochs (`AdamW`, `lr=1e-3`, `OneCycleLR`, `batch_size=512`, AMP) and restores the best validation checkpoint.
5. **Evaluation Metrics Setup** — Configures RDKit `SA_Score` and `quick_gen_metrics` for validity, uniqueness, fingerprint Precision/Recall/F1, logP, QED, SAS, and drug suitability.
6. **Evaluating & Saving Base Model** — Evaluates `zinc_model` on ZINC250K validation data, saves `models/zinc250k_fp2smiles_v1.pt`, and runs a round-trip load check.
7. **Data Preprocessing (Tox21 Non-Toxic Subset)** — Strips salts, filters for molecules with zero positive toxicity outcomes across all 12 Tox21 assays, filters to ZINC vocabulary compatibility (4,697 molecules), and splits 80/10/10 into Train/Val/Test.
8. **Fine-Tuning (Tox21 Non-Toxic Data)** — Loads `models/zinc250k_fp2smiles_v1.pt` into `tox_model`, logs zero-shot validation baseline, and fine-tunes for 10 epochs (`AdamW`, `lr=2e-4`, `CosineAnnealingLR`, `batch_size=64`).
9. **Evaluating F1 & Generation Metrics (Tox21 Test Data)** — Evaluates the fine-tuned `tox_model` on the held-out Tox21 test set.

---

## Dataset Splits

| Dataset | Raw Size | Filtered / Used | Train | Validation | Test |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **ZINC250K** | 249,455 | 54,000 | 50,000 | 2,000 | 2,000 |
| **Tox21 (Non-Toxic)** | 7,831 | 4,697 | 3,757 | 470 | 470 |

---

## Training & Fine-Tuning Results

### 1. Base Pre-Training on ZINC250K (12 Epochs, Peak VRAM: 1.92 GiB)

| Epoch | Train Loss | Val Loss | Val PPL | Val TokAcc | Validity | Uniqueness | FP Precision | FP Recall | FP F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1.9003 | 1.2143 | 3.37 | 59.83% | 8.59% | 100.00% | 0.3359 | 0.2689 | 0.2974 |
| 04 | 0.6129 | 0.4846 | 1.62 | 82.60% | 28.12% | 100.00% | 0.5643 | 0.4907 | 0.5210 |
| 08 | 0.4062 | 0.3451 | 1.41 | 87.49% | 47.66% | 100.00% | 0.6651 | 0.6403 | 0.6509 |
| **12 (Best)** | **0.3564** | **0.3173** | **1.37** | **88.59%** | **53.91%** | **100.00%** | **0.6950** | **0.6658** | **0.6778** |

### 2. Fine-Tuning on Non-Toxic Tox21 (10 Epochs, Peak VRAM: 0.44 GiB)

| Stage / Epoch | Train Loss | Val Loss | Val PPL | Val TokAcc | Validity | FP F1 | QED | SAS | Suitable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Before Fine-Tuning (Zero-Shot)** | — | 1.0123 | 2.75 | 74.42% | — | — | — | — | — |
| Epoch 01 | 0.7110 | 0.5786 | 1.78 | 80.24% | 79.69% | 0.6408 | 0.522 | 2.30 | 54.90% |
| Epoch 05 | 0.4352 | 0.4894 | 1.63 | 83.45% | 85.94% | 0.6993 | 0.500 | 2.49 | 50.91% |
| Epoch 08 | 0.3757 | 0.4820 | 1.62 | 83.91% | 78.12% | 0.7351 | 0.495 | 2.49 | 46.00% |
| **Epoch 10 (Best Val Loss)** | **0.3539** | **0.4790** | **1.61** | **84.27%** | **84.38%** | **0.7112** | **0.500** | **2.52** | **44.44%** |

### 3. Final Held-Out Tox21 Test Evaluation (`tox_test_loader`)

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Validity** | `78.12%` | Valid RDKit-parseable SMILES |
| **Uniqueness** | `100.00%` | Unique canonical SMILES among valid generations |
| **Fingerprint Precision** | `0.7202` | Bit-level Morgan FP match precision |
| **Fingerprint Recall** | `0.7013` | Bit-level Morgan FP match recall |
| **Fingerprint F1** | **`0.7046`** | Harmonic mean of FP precision & recall |
| **logP** | `2.1305` | Target range: `[-0.4, 5.0]` |
| **QED** | `0.5272` | Drug-likeness (`>= 0.5`) |
| **SAS** | `2.4875` | Synthetic accessibility (`<= 4.5`, lower is easier) |
| **Overall Suitability** | **`58.00%`** | Meets all three logP, QED, and SAS criteria |

**Sample valid generated SMILES:** `CN(C)/C=C/c1ccc(Br)cc1`

---

## Quick Start

```bash
# Sync dependencies
uv sync

# Launch interactive JupyterLab
uv run jupyter lab

# Execute full pipeline from CLI
uv run jupyter nbconvert --to notebook --execute --inplace src/molecule_generation.ipynb
```
