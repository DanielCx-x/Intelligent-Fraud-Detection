# Intelligent Credit Card Fraud Detection

This repository contains a PyTorch implementation of various Deep Learning models for Credit Card Fraud Detection. The project addresses the problem of highly imbalanced datasets using Weighted Loss MLPs, SMOTE, and Autoencoder/VAE architectures.

## Project Overview

The goal is to detect fraudulent transactions (Class 1) among a vast majority of normal transactions (Class 0). The codebase compares:

1.  **Random Forest (Baseline):** A strong baseline using standard supervised learning.
2.  **MLP with Weighted Loss:** Handles class imbalance by penalizing mistakes on the minority class more heavily.
3.  **MLP with SMOTE:** Uses synthetic oversampling to balance the training data before feeding it into the MLP.
4.  **Autoencoder (AE):** Unsupervised learning trained only on normal transactions; detects fraud based on high reconstruction error.
5.  **Variational Autoencoder (VAE):** A probabilistic approach to anomaly detection.

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

# Activate env
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Dataset

This project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Download creditcard.csv from the link above.
Place creditcard.csv inside the data/ folder.
- Note: If you only want to run the demo.py, you do not need the dataset.

## How to Run

### Training

The `main.py` script supports training specific models using the `--model` argument.

**Usage:**
```bash
python src/main.py --model <MODEL_NAME> --epochs <NUM_EPOCHS>
```

**Available Options:**

**rf:** Random Forest

**mlp_weighted:** MLP with Weighted Loss

**mlp_smote:** MLP with SMOTE Oversampling

**ae:** Autoencoder

**vae:** Variational Autoencoder

**all:** Train and evaluate ALL models sequentially.


**Examples:**

Train the Weighted MLP:
```bash
python src/main.py --model mlp_weighted --epochs 10
```
Train all models to compare results:
```bash
python src/main.py --model all --epochs 10
```
### Demo (Inference)

The `demo.py` script works **out-of-the-box** (no dataset required). It generates synthetic data to simulate transactions and uses your trained models for inference.

You can select which model to use for the demo using the `--model` argument.

**Usage:**
```bash
python demo/demo.py --model <MODEL_NAME>
```

**Examples:**
Run demo with the Weighted MLP (default):
```bash
python demo/demo.py --model mlp_weighted
```
Run demo with Autoencoder (Anomaly Detection):
```bash
python demo/demo.py --model ae
```
Run demo with Random Forest:
```bash
python demo/demo.py --model rf
```

## Expected Output
The results will be printed to the console and saved to `results/demo_<model>.txt`.

*Example output (works for ALL models):*
```text
Sample 1: Risk Score: 0.0003 => Prediction: Normal
Sample 2: Risk Score: 0.9812 => Prediction: Fraud
Sample 3: Risk Score: 0.1240 => Prediction: Normal
...
```

## Pre-trained Model
You can download the pre-trained model weights from the link below and place them in the checkpoints/ folder: Download Link Placeholder.


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
