# Emotion Classification with Imbalance-Aware Transformers

This repository contains the implementation and experimental artefacts for an MSc dissertation on fine-grained multi-label emotion classification using the GoEmotions dataset.

## Contents
- `train_baseline_optimized-FINAL.py` – Main training script (DeBERTa-v3-base)
- `figures/` – Generated figures used in the dissertation
- `tables/` – CSV tables used in appendices and analysis

## Reproducibility
Due to file size constraints, trained model checkpoints are not included.
All results can be reproduced by running the training script with the provided configuration.

Experiments were conducted using:
- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Google Colab (GPU)

Random seeds were fixed to ensure reproducibility.  
