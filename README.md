# Intelligent Credit Card Fraud Detection

This repository contains a PyTorch implementation of various Deep Learning models for Credit Card Fraud Detection. The project addresses the problem of highly imbalanced datasets using Weighted Loss MLPs, SMOTE, and Autoencoder/VAE architectures.

## Project Overview

The goal is to detect fraudulent transactions (Class 1) among a vast majority of normal transactions (Class 0). The codebase compares:
1.  **Baseline:** Random Forest Classifier (Supervised).
2.  **Deep Learning:** MLP with SMOTE (Synthetic Minority Over-sampling).
3.  **Anomaly Detection:** Autoencoder (Unsupervised).

## Directory Structure

* `src/`: Contains source code for models, utility functions, and the training loop.
* `data/`: Link for the dataset (not included in repo, see instructions below).
* `checkpoints/`: Directory where trained model weights are saved.
* `results/`: Training metrics and logs.
* `demo/`: Inference script for demonstration.

## Setup Instructions

### 1. Environment
It is recommended to use a virtual environment.

```bash
# Create env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

