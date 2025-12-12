# Intelligent Credit Card Fraud Detection

This repository contains a PyTorch implementation of various Deep Learning models for Credit Card Fraud Detection. The project addresses the problem of highly imbalanced datasets using Weighted Loss MLPs, SMOTE, and Autoencoder/VAE architectures.

## Project Overview

The goal is to detect fraudulent transactions (Class 1) among a vast majority of normal transactions (Class 0). The codebase compares:

1.  **Simple MLP with Weighted Loss**: Handles class imbalance by penalizing mistakes on the minority class more heavily.
2.  **Variational Autoencoder (VAE)**: Treats fraud detection as an anomaly detection problem.

## Directory Structure

- `src/`: Contains source code for models, utility functions, and the training loop.
- `data/`: Directory for the dataset (not included in repo, see instructions below).
- `checkpoints/`: Directory where trained model weights are saved.
- `results/`: Training metrics and logs.
- `demo/`: Inference script for demonstration.

## Setup Instructions

### 1. Environment

It is recommended to use a virtual environment.

```bash
# Create env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2\. Dataset

This project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

1.  Download `creditcard.csv` from the link above.
2.  Place `creditcard.csv` inside the `data/` folder.

## How to Run

### Training

To train the main MLP model and evaluate it on the test set:

```bash
python src/main.py --epochs 10
```

This will:

- Preprocess the data.
- Train the model.
- Save the weights to `checkpoints/mlp_weighted.pth`.
- Print evaluation metrics (AUPRC, Recall, Precision, F1).

### Demo

After training, run the demo script to simulate inference on random samples:

```bash
python demo.py
```

## Expected Output

Running the demo script should produce output similar to:

```plaintext
Loading data sample...
Running Inference...

--- Demo Results ---
Sample 1: Fraud Probability: 0.0003 => Prediction: Normal
Sample 2: Fraud Probability: 0.9812 => Prediction: Fraud
...
```

## Pre-trained Model

You can download a pre-trained model checkpoint here: **(https://colab.research.google.com/drive/1l8F5WD3G_jwO_147vo6FMxInT3vNkljN)**.

> **Note:** Run `src/main.py` locally to generate `checkpoints/mlp_weighted.pth` immediately.


## Reproducibility & Configuration
To ensure reproducibility, we adhered to the following setup:

- Data Split:

    Train/Val/Test split: 60% / 20% / 20%.

    Stratified Sampling was used to maintain the 0.17% fraud ratio across all splits.

- Preprocessing:

    Time and Amount features were scaled using StandardScaler (fit on training data only to avoid leakage).

    V1-V28 features were left as-is (already PCA transformed).

- Hyperparameters:

    Random Forest: n_estimators=100, n_jobs=-1, random_state=42.

    MLP: Layers [64, 32, 1], Dropout=0.3, Optimizer=Adam(lr=0.001), Batch_Size=2048.

    Autoencoder: Encoder [30 -> 16 -> 8], Decoder [8 -> 16 -> 30], Loss=MSE.

- Seed: random_state=42 is used globally for all splits and initializations.


## Acknowledgments

- Dataset provided by Machine Learning Group - ULB on Kaggle.
- Implementation inspired by standard anomaly detection techniques in PyTorch.
