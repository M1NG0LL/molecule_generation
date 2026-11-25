# Drug Molecule Generation (ZINC250K → Tox21)

End-to-end machine learning pipeline for **de novo drug-like, non-toxic molecule generation** pre-trained on **ZINC250K** and fine-tuned on **Tox21**, managed with `uv` and versioned via Git.

## Pipeline Stages (`molecule_generation.ipynb`)

1. **Data Gather** — Loads ZINC250K via Hugging Face `datasets` (`load_dataset("yairschiff/zinc250k")`) and downloads/caches MoleculeNet Tox21 (`data/tox21.csv.gz`).
2. **Data Processing (ZINC250K)** — RDKit SMILES validation & canonicalization, atom-wise chemical regex tokenization, and Train/Val/Test splitting.
3. **Training the Model (ZINC250K)** — Trains an autoregressive 2-layer GRU SMILES language model (`SMILESGeneratorGRU`) with mixed precision on GPU.
4. **Evaluate (ZINC250K)** — Computes held-out Test Loss, Perplexity, Next-Token F1, plus Validity, Uniqueness, Novelty, QED, and Lipinski pass rate on generated molecules.
5. **Extract File of the Model (ZINC250K)** — Saves self-contained pre-trained checkpoint to `models/zinc250k_pretrained_v1.pt`.
6. **Data Processing (Tox21)** — Canonicalizes Tox21 molecules across 12 toxicity assays, constructs a verified safe & drug-like fine-tuning split, and trains an ECFP4 Morgan-fingerprint Tox21 Safety Evaluator on held-out splits.
7. **Fine-Tune (Tox21)** — Loads `models/zinc250k_pretrained_v1.pt` and fine-tunes on safe, drug-like Tox21 molecules with KL-divergence regularization toward the ZINC250K prior to preserve syntax validity while reducing predicted toxicity.
8. **Evaluate (Tox21)** — Evaluates held-out Tox21 metrics and de novo molecule generation safety, drug-likeness, uniqueness, and novelty.
9. **Extract File of the Model (Tox21)** — Saves fine-tuned checkpoint to `models/tox21_finetuned_v1.pt`.
10. **Evaluate Using F1 Score (Best Version Selection)** — Compares extracted checkpoints using **Safe-Drug Generation F1** (harmonic mean of Valid-Safe-Druglike Precision and Reference Chemical-Space Recall) and **Held-Out Token F1**, logging version history to `metrics/version_leaderboard.json`.

## Quick Start

```bash
# Sync dependencies
uv sync

# Launch interactive JupyterLab
uv run jupyter lab

# Execute full pipeline from CLI
uv run jupyter nbconvert --to notebook --execute --inplace molecule_generation.ipynb
```
