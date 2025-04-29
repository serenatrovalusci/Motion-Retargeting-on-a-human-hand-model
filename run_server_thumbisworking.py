import torch
import torch.nn as nn
import numpy as np
import joblib
import socket
from HandPoseClass import *

# ----------------------------
# 2. Load model & scaler
# ----------------------------
model = HandPoseFCNN(input_dim=4, output_dim=36)  # 18 sin/cos thumb + 18 fingers
model.load_state_dict(torch.load("hand_pose_fcnn.pth"))
model.eval()

scaler_y = joblib.load("scaler_y.save")  # Used to inverse scale outputs

# ----------------------------
# 3. Setup TCP server
# ----------------------------
HOST = '127.0.0.1'
PORT = 65432
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"Server listening on {HOST}:{PORT}...")

conn, addr = server.accept()
print(f"Connected to {addr}")

# ----------------------------
# 4. Real-time communication loop
# ----------------------------
try:
    while True:
        data = conn.recv(1024)
        if not data:
            break

        # Convert to float array
        inputs = np.frombuffer(data, dtype=np.float32)
        if inputs.shape[0] != 4:
            print("Expected 4 float inputs.")
            continue

        # Predict
        input_tensor = torch.FloatTensor(inputs.reshape(1, -1))
        with torch.no_grad():
            output = model(input_tensor).numpy()

        # Inverse transform
        output = scaler_y.inverse_transform(output).flatten()

        # --- Decode thumb sin/cos back to angles ---
        sin_vals = output[:9]
        cos_vals = output[9:18]
        thumb_angles_rad = np.arctan2(sin_vals, cos_vals)
        thumb_angles_deg = np.rad2deg(thumb_angles_rad)

        # Get fingers outputs
        fingers = output[18:]

        # Combine thumb + fingers into 27 joint angles
        real_output = np.concatenate([thumb_angles_deg, fingers], axis=0)

        # Send back 27 floats
        conn.sendall(real_output.astype(np.float32).tobytes())

except Exception as e:
    print("Error:", e)
finally:
    conn.close()
    server.close()
