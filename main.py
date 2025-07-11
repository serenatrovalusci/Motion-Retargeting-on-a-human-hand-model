import torch
import numpy as np
import joblib
import socket
from HandPoseClass import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Server for Hand Pose Estimation")
    parser.add_argument('--model', type=str, help='Choose the model')
    parser.add_argument('--weights_path', type=str, help='Path to weights file')
    parser.add_argument('--scaler_path', type=str, help='Path to scaler file')
  
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\nHelping prompt : ")
    print("python main.py --model Transformer --weights_path weights_path --scaler_path scaler_path\n")
    
    # angoli fixati per una performance migliore
    fix_Indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43] # all the thumb angles(9) + 4 index angles + 2 middle angles

    print(f"Model selected: {args.model}\n")
    if args.model == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=45+len(fix_Indices))
    elif args.model == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_Indices)


    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    
    scaler_y = joblib.load(args.scaler_path)

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
        

            output = scaler_y.inverse_transform(output).flatten()

            
            original_output = np.zeros(45)
            mixed_col = 0
            original_col = 0

            for i in range(45):
                if i in fix_Indices:
                    # Estrai sin e cos, poi calcola l'angolo con arctan2
                    sin_vals = output[mixed_col]
                    cos_vals = output[mixed_col + 1]
                    angles_rad = np.arctan2(sin_vals, cos_vals) 
                    original_output[original_col] = np.rad2deg(angles_rad) # Converti in gradi
                    mixed_col += 2
                    original_col += 1
                else:
                    # Copia direttamente il valore raw
                    original_output[original_col] = output[mixed_col]
                    mixed_col += 1
                    original_col += 1

            conn.sendall(np.array(original_output, dtype=np.float32).tobytes())

    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
        server.close()



        