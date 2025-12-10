import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, recall_score, precision_score, f1_score
import os

def load_and_preprocess_data(file_path, test_size=0.2, val_size=0.25):
    """
    Loads csv, handles missing values, splits data, and scales features.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Reading file: {file_path} ...")
    df = pd.read_csv(file_path)
    
    # Handle missing values
    if df.isnull().sum().max() > 0:
        df = df.fillna(0)

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split: Train/Val/Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def to_loader(X, y, batch_size=2048, shuffle=True):
    """Converts numpy arrays to PyTorch DataLoader."""
    tensor_x = torch.Tensor(X)
    # Handle y depending on whether it's Series or numpy array
    y_values = y.values if isinstance(y, pd.Series) else y
    tensor_y = torch.Tensor(y_values).unsqueeze(1) # [batch, 1]
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_model_metrics(model_name, y_true, y_pred_prob, save_path=None):
    """Calculates metrics, prints them, and optionally saves confusion matrix."""
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []

    for thresh in thresholds:
        y_pred_temp = (y_pred_prob > thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_temp))

    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx]
    best_f1 = f1_scores[best_thresh_idx]

    y_pred_class = (y_pred_prob > best_threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(recall, precision)
    rec_score = recall_score(y_true, y_pred_class)
    prec_score = precision_score(y_true, y_pred_class)

    print(f"\n--- {model_name} ---")
    print(f"Best Threshold: {best_threshold:.2f} (Max F1: {best_f1:.4f})")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Recall: {rec_score:.4f}")
    print(f"Precision: {prec_score:.4f}")
    cm = confusion_matrix(y_true, y_pred_class)
    print("Confusion Matrix:")
    print(cm)
    
    if save_path:
        with open(os.path.join(save_path, f"{model_name}_results.txt"), "w") as f:
            f.write(f"Best Threshold: {best_threshold:.2f}\n")
            f.write(f"AUPRC: {auprc:.4f}\n")
            f.write(f"Recall: {rec_score:.4f}\n")
            f.write(f"Precision: {prec_score:.4f}\n")

    return auprc, rec_score, prec_score, best_threshold

def loss_function_vae(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD