import torch
import numpy as np
import joblib
import ast
import socket
from HandPoseClass import *
import argparse

def load_config(info_path):
    """Load all parameters from the configuration file"""
    with open(info_path, 'r') as f:
        config = {}
        for line in f:
            if ':' in line:  # Only parse valid key:value lines
                key, val = line.split(':', 1)
                key = key.strip()
                try:
                    # Try to parse numeric or list values (e.g., 45, [1,2,3])
                    config[key] = ast.literal_eval(val.strip())
                except:
                    # Otherwise store them as strings
                    config[key] = val.strip()
        return config
    
def reconstruct_output(output, fix_indices, original_dim=45):
    """
    Dynamically reconstruct the output.
    
    - For indices in fix_indices, the model predicts sin and cos values.
      These are converted back to angles in degrees using atan2.
    - For all other indices, the output is directly taken from the model.
    """
    reconstructed = np.zeros(original_dim)  # Placeholder for reconstructed output
    mixed_idx = 0  # Counter for iterating through model output
    
    for i in range(original_dim):
        if i in fix_indices:
            # Convert sin/cos pair back to an angle in degrees
            sin_val = output[mixed_idx]
            cos_val = output[mixed_idx+1]
            reconstructed[i] = np.rad2deg(np.arctan2(sin_val, cos_val))
            mixed_idx += 2
        else:
            # Directly assign the predicted value
            reconstructed[i] = output[mixed_idx]
            mixed_idx += 1
    return reconstructed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Server for Hand Pose Estimation")
    parser.add_argument('--info_path', type=str, default='training_info.txt', help='Path to training info file')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_args()
    # Load configuration parameters from file
    config = load_config(args.info_path)

    # Print help prompt examples
    print("\nHelping prompt : ") 
    print("python main.py --info_path training_results\training_20250521_150116\training_info.txt \n") 
    print("python main.py --info_path training_results\training_20250521_150116\training_info.txt \n")
    
    # Extract config parameters
    fix_Indices = config.get("Fixed Indices", [])
    print(f"fix_indices: {fix_Indices}\n")

    pca_components = config.get("PCA Components", 0)
    print(f"pca_components: {pca_components}\n")

    model_type = config.get("Model", "FCNN")
    print(f"model_type: {model_type}\n")

    # Initialize model depending on type
    if model_type == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=45+len(fix_Indices))
    elif model_type == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_Indices, pca_dim=0)

    # Load trained model weights and scaler
    weights_path = config.get("Model Save Path", "model.pth")
    scaler_path = config.get("Scaler Path", "scaler.save")

    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    scaler = joblib.load(scaler_path)

    # ---------------------- SERVER SETUP ----------------------
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # Bind server to localhost:65432
        server.bind(('127.0.0.1', 65432))
        server.listen(1)
        print(f"Server ready (Model: {model_type} | Fixed indices: {fix_Indices})...")

        # Accept client connection
        conn, _ = server.accept()
        with conn:
            while True:
                # Receive 16 bytes (4 float32 values) from client
                data = conn.recv(16)
                if not data: 
                    break

                # Convert received bytes into numpy array of float32
                input = np.frombuffer(data, dtype=np.float32).copy()
                
                # Run model inference
                output = model(torch.FloatTensor(input.reshape(1, -1))).detach().numpy()
                
                # Reverse scaling (to original value range)
                output = scaler.inverse_transform(output).flatten()
                
                # Reconstruct the final hand pose (angles + fixed indices conversion)
                reconstructed = reconstruct_output(output, fix_Indices).astype(np.float32)
                
                # Send result back to client as raw bytes
                conn.sendall(reconstructed.tobytes())
                
