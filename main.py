# FINAL run_server_fixed.py for output_dim=42
# Thumb = 0–8 (sin) + 9–17 (cos)
# PIP/DIP = sin/cos decoded at correct indices: 13,14,16,17,25,26

import torch
import numpy as np
import joblib
import socket
from HandPoseClass import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Server for Hand Pose Estimation")
    parser.add_argument('--model', type=str, help='Apply PCA to the target dataset')
    parser.add_argument('--pca_components', type=int, default=0, help='Number of PCA components to keep')
    parser.add_argument('--weights_path', type=str, help='Path to weights file')
    parser.add_argument('--scaler_path', type=str, help='Path to scaler file')
    parser.add_argument('--pca_path', type=str, help='Path to PCA file')
  
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\nHelping prompt : ")
    print("python main.py --model Transformer --pca_components 32 --weights_path weights_path --scaler_path scaler_path --pca_path pca_path\n")

    if args.pca_components == 0:
        print(f"Model selected: {args.model}, without PCA\n")
        if args.model == 'FCNN':
            model = HandPoseFCNN(input_dim=4, output_dim=42)
        elif args.model == 'Transformer':
            model = HandPoseTransformer(input_dim=4, output_dim=42)

    else:
        print(f"Model selected: {args.model}, with PCA\n")
        if args.model == 'FCNN':
            model = HandPoseFCNN(input_dim=4, output_dim=args.pca_components)
        elif args.model == 'Transformer':
            model = HandPoseTransformer(input_dim=4, output_dim=args.pca_components)

    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    
    scaler_y = joblib.load(args.scaler_path)

    if args.pca_components != 0:
        pca = joblib.load(args.pca_path)

    HOST = '127.0.0.1'
    PORT = 65432
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Server listening on {HOST}:{PORT}...")

    conn, addr = server.accept()
    print(f"Connected to {addr}")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            inputs = np.frombuffer(data, dtype=np.float32).copy()
            if inputs.shape[0] != 4:
                print("Expected 4 float inputs.")
                continue

            input_tensor = torch.FloatTensor(inputs.reshape(1, -1))

            with torch.no_grad():
                output = model(input_tensor).numpy()

            if args.pca_components != 0:
                output = pca.inverse_transform(output)

            output = scaler_y.inverse_transform(output).flatten()

            # Decode thumb (0–8 and 9–17)
            thumb_sin = output[0:9]
            thumb_cos = output[9:18]
            thumb_deg = np.rad2deg(np.arctan2(thumb_sin, thumb_cos))

            # Decode PIP/DIP (indices [13,14,16,17,25,26]) from sin/cos pairs 18–29
            pip_y = np.rad2deg(np.arctan2(output[18], output[24]))  # index 13
            pip_z = np.rad2deg(np.arctan2(output[19], output[25]))  # index 14
            dip_y = np.rad2deg(np.arctan2(output[20], output[26]))  # index 16
            dip_z = np.rad2deg(np.arctan2(output[21], output[27]))  # index 17
            mid_y = np.rad2deg(np.arctan2(output[22], output[28]))  # index 25
            mid_z = np.rad2deg(np.arctan2(output[23], output[29]))  # index 26

            # Remaining 12 raw angles (starts at index 30)
            raw = output[30:]

            # Build 27-angle output in correct index order
            final_output = []
            raw_i = 0
            for i in range(27):
                if i < 9:
                    final_output.append(thumb_deg[i])
                elif i == 13:
                    final_output.append(pip_y)
                elif i == 14:
                    final_output.append(pip_z)
                elif i == 16:
                    final_output.append(dip_y)
                elif i == 17:
                    final_output.append(dip_z)
                elif i == 25:
                    final_output.append(mid_y)
                elif i == 26:
                    final_output.append(mid_z)
                else:
                    final_output.append(raw[raw_i])
                    raw_i += 1

            conn.sendall(np.array(final_output, dtype=np.float32).tobytes())

    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
        server.close()



        