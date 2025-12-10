import torch
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import SimpleMLP

def run_demo():
    # Configuration
    model_path = 'checkpoints/mlp_weighted.pth'
    data_path = 'data/creditcard.csv' # Using original data just to sample random rows for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please run 'python src/main.py' first.")
        return

    # 1. Load Data Sample (Simulating new incoming data)
    print("Loading data sample...")
    df = pd.read_csv(data_path)
    # Get features only
    X_raw = df.drop('Class', axis=1).values
    
    # NOTE: In a real production environment, you should save the fitted scaler 
    # from the training phase (e.g., using joblib) and load it here.
    # For this demo, we refit on the whole dataset to approximate the scaling.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Select 5 random samples
    indices = np.random.choice(len(X_scaled), 5, replace=False)
    sample_inputs = X_scaled[indices]
    sample_tensor = torch.Tensor(sample_inputs).to(device)

    # 2. Load Model
    input_dim = X_scaled.shape[1]
    model = SimpleMLP(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    print("\nRunning Inference...")
    with torch.no_grad():
        logits = model(sample_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    # 4. Results
    print("\n--- Demo Results ---")
    for i, prob in enumerate(probs):
        pred_label = 1 if prob > 0.9 else 0 # Threshold from training (approx)
        print(f"Sample {i+1}: Fraud Probability: {prob:.4f} => Prediction: {'Fraud' if pred_label == 1 else 'Normal'}")

if __name__ == "__main__":
    run_demo()
