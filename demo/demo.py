import torch
import sys
import os
import argparse
import numpy as np
import joblib

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import SimpleMLP, Autoencoder, VAE

def get_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Demo Inference")
    parser.add_argument('--model', type=str, default='mlp_weighted', 
                        choices=['rf', 'mlp_weighted', 'mlp_smote', 'ae', 'vae'],
                        help='Choose which model to use for demo')
    return parser.parse_args()

def run_demo():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 30  # Standard for Credit Card dataset (Time + V1-V28 + Amount)
    
    # Define model path and type mapping.
    # 'type': 'supervised' or 'unsupervised'
    model_config = {
        'rf':           {'path': 'checkpoints/random_forest.joblib', 'type': 'sklearn'},
        'mlp_weighted': {'path': 'checkpoints/mlp_weighted.pth',   'type': 'pytorch_sup', 'class': SimpleMLP},
        'mlp_smote':    {'path': 'checkpoints/mlp_smote.pth',      'type': 'pytorch_sup', 'class': SimpleMLP},
        'ae':           {'path': 'checkpoints/autoencoder.pth',    'type': 'pytorch_unsup', 'class': Autoencoder},
        'vae':          {'path': 'checkpoints/vae.pth',            'type': 'pytorch_unsup', 'class': VAE}
    }

    config = model_config[args.model]
    model_path = config['path']

    # 1. Check if the model file exists.
    if not os.path.exists(model_path):
        print(f"\n[Error] Model checkpoint not found at: {model_path}")
        print(f"Please run training first: python src/main.py --model {args.model}")
        return

    print(f"\n--- Running Demo for Model: {args.model.upper()} ---")

    # 2. Generate simulated data (5 samples)
    print("Generating synthetic random transaction data...")
    # Simulate data processed by StandardScaler (mean 0, variance 1).
    sample_inputs = np.random.randn(5, input_dim).astype(np.float32)
    
    # 3. Load the model and perform inference.
    predictions = []
    
    # --- Case A: Scikit-Learn (Random Forest) ---
    if config['type'] == 'sklearn':
        print("Loading Random Forest model...")
        model = joblib.load(model_path)
        # Get the probability of belonging to class 1 (Fraud).
        probs = model.predict_proba(sample_inputs)[:, 1]
        predictions = probs

    # --- Case B: PyTorch Models ---
    else:
        print(f"Loading PyTorch model ({config['class'].__name__})...")
        model = config['class'](input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        sample_tensor = torch.Tensor(sample_inputs).to(device)
        
        with torch.no_grad():
            # -- B1: Supervised (MLP) --
            if config['type'] == 'pytorch_sup':
                logits = model(sample_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                predictions = probs
            
            # -- B2: Unsupervised (AE / VAE) --
            elif config['type'] == 'pytorch_unsup':
                if args.model == 'vae':
                    recon_x, _, _ = model(sample_tensor)
                else: # ae
                    recon_x = model(sample_tensor)
                
                # Calculate the reconstruction error (MSE) as anomaly score.
                # dim=1 Calculate the average error for all features of each sample.
                mse = torch.mean(torch.pow(sample_tensor - recon_x, 2), dim=1).cpu().numpy()
                # In real-world applications, this should be divided by the maximum MSE of the training set, or determined based on a threshold.
                # For the demo purposes, we'll use Min-Max Scaling.
                if mse.max() - mse.min() > 0:
                    fraud_scores = (mse - mse.min()) / (mse.max() - mse.min())
                else:
                    fraud_scores = np.zeros_like(mse)

    # 4. Print results.
    os.makedirs('results', exist_ok=True)
    output_file = f'results/demo_{args.model}.txt'

    print("\n--- Inference Results ---")
    with open(output_file, 'w') as f:
        header = f"Model: {args.model.upper()}\n"
        f.write(header)
        
        for i, score in enumerate(fraud_scores):
            # Set a threshold for demo
            threshold = 0.5 
            
            label = "Fraud" if score > threshold else "Normal"
            
            # Format the output
            msg = f"Sample {i+1}: Risk Score: {score:.4f} => Prediction: {label}"
            
            print(msg)
            f.write(msg + "\n")

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    run_demo()
