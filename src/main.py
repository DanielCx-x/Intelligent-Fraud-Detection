import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import numpy as np
from model import SimpleMLP, VAE
from utils import load_and_preprocess_data, to_loader, evaluate_model_metrics, loss_function_vae

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=6):
    # Calculate weights for imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    # Loaders
    train_loader = to_loader(X_train, y_train)
    val_loader = to_loader(X_val, y_val, shuffle=False)

    model = SimpleMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training MLP...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training")
    parser.add_argument('--data_path', type=str, default='data/creditcard.csv', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save model')
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load Data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data(args.data_path)
    except FileNotFoundError:
        print("Dataset not found. Please ensure 'data/creditcard.csv' exists.")
        return

    input_dim = X_train.shape[1]

    # --- Train MLP (Weighted) ---
    mlp_model = train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=args.epochs)
    
    # Save Model
    save_path = os.path.join(args.output_dir, 'mlp_weighted.pth')
    torch.save(mlp_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Evaluate
    X_test_tensor = torch.Tensor(X_test).to(device)
    mlp_model.eval()
    with torch.no_grad():
        logits = mlp_model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    
    evaluate_model_metrics("MLP_Weighted", y_test, probs, save_path='results')

if __name__ == "__main__":
    main()