import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import numpy as np
import joblib # 用于保存 sklearn 模型
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from model import SimpleMLP, VAE, Autoencoder
from utils import load_and_preprocess_data, to_loader, evaluate_model_metrics, loss_function_vae

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Functions ---

def train_random_forest(X_train, y_train, output_dir):
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # Save model
    save_path = os.path.join(output_dir, 'random_forest.joblib')
    joblib.dump(rf, save_path)
    print(f"Random Forest saved to {save_path}")
    return rf

def train_mlp_weighted(X_train, y_train, X_val, y_val, input_dim, epochs, output_dir):
    print("Training MLP (Weighted)...")
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    train_loader = to_loader(X_train, y_train)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    model = SimpleMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    save_path = os.path.join(output_dir, 'mlp_weighted.pth')
    torch.save(model.state_dict(), save_path)
    return model

def train_mlp_smote(X_train, y_train, X_val, y_val, input_dim, epochs, output_dir):
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Data shape after SMOTE: {X_train_smote.shape}")

    print("Training MLP (SMOTE)...")
    train_loader = to_loader(X_train_smote, y_train_smote)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    model = SimpleMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss() # No weights needed, data is balanced
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    save_path = os.path.join(output_dir, 'mlp_smote.pth')
    torch.save(model.state_dict(), save_path)
    return model

def train_autoencoder(X_train, y_train, input_dim, epochs, output_dir):
    print("Training Autoencoder (Unsupervised on Normal Data)...")
    # Only train on normal transactions
    X_train_normal = X_train[y_train == 0]
    train_loader = to_loader(X_train_normal, X_train_normal, batch_size=256)

    model = Autoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs * 2): # AE usually needs more epochs
        model.train()
        train_loss = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f}")

    save_path = os.path.join(output_dir, 'autoencoder.pth')
    torch.save(model.state_dict(), save_path)
    return model

def train_vae(X_train, y_train, input_dim, epochs, output_dir):
    print("Training VAE (Unsupervised on Normal Data)...")
    X_train_normal = X_train[y_train == 0]
    train_loader = to_loader(X_train_normal, X_train_normal, batch_size=256)

    model = VAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs * 2):
        model.train()
        train_loss = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_X)
            loss = loss_function_vae(recon_batch, batch_X, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f}")

    save_path = os.path.join(output_dir, 'vae.pth')
    torch.save(model.state_dict(), save_path)
    return model

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Project")
    parser.add_argument('--data_path', type=str, default='data/creditcard.csv')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['rf', 'mlp_weighted', 'mlp_smote', 'ae', 'vae', 'all'],
                        help='Choose which model to train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. Load Data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(args.data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.data_path}")
        return

    input_dim = X_train.shape[1]
    X_test_tensor = torch.Tensor(X_test).to(device)

    # 2. Train & Evaluate Selected Model
    
    # --- Random Forest ---
    if args.model in ['rf', 'all']:
        rf_model = train_random_forest(X_train, y_train, args.output_dir)
        probs = rf_model.predict_proba(X_test)[:, 1]
        evaluate_model_metrics("RandomForest", y_test, probs, save_path='results')

    # --- MLP Weighted ---
    if args.model in ['mlp_weighted', 'all']:
        mlp_w = train_mlp_weighted(X_train, y_train, X_val, y_val, input_dim, args.epochs, args.output_dir)
        mlp_w.eval()
        with torch.no_grad():
            logits = mlp_w(X_test_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        evaluate_model_metrics("MLP_Weighted", y_test, probs, save_path='results')

    # --- MLP SMOTE ---
    if args.model in ['mlp_smote', 'all']:
        mlp_s = train_mlp_smote(X_train, y_train, X_val, y_val, input_dim, args.epochs, args.output_dir)
        mlp_s.eval()
        with torch.no_grad():
            logits = mlp_s(X_test_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        evaluate_model_metrics("MLP_SMOTE", y_test, probs, save_path='results')

    # --- Autoencoder ---
    if args.model in ['ae', 'all']:
        ae = train_autoencoder(X_train, y_train, input_dim, args.epochs, args.output_dir)
        ae.eval()
        with torch.no_grad():
            recon = ae(X_test_tensor)
            # MSE per sample
            mse = torch.mean(torch.pow(X_test_tensor - recon, 2), dim=1).cpu().numpy()
        # Normalize MSE to 0-1 for probability-like scoring
        probs = (mse - mse.min()) / (mse.max() - mse.min())
        evaluate_model_metrics("Autoencoder", y_test, probs, save_path='results')

    # --- VAE ---
    if args.model in ['vae', 'all']:
        vae = train_vae(X_train, y_train, input_dim, args.epochs, args.output_dir)
        vae.eval()
        with torch.no_grad():
            recon, _, _ = vae(X_test_tensor)
            mse = torch.mean(torch.pow(X_test_tensor - recon, 2), dim=1).cpu().numpy()
        probs = (mse - mse.min()) / (mse.max() - mse.min())
        evaluate_model_metrics("VAE", y_test, probs, save_path='results')

if __name__ == "__main__":
    main()
