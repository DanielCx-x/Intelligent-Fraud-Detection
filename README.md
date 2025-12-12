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
python demo/demo_inference.py
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

## Acknowledgments

- Dataset provided by Machine Learning Group - ULB on Kaggle.
- Implementation inspired by standard anomaly detection techniques in PyTorch.
