import torch
import numpy as np
import socket
import joblib
from HandPoseClass import *

model = HandPoseFCNN(input_dim=4, output_dim=11)
model.load_state_dict(torch.load("hand_pose_fcnn_PCA.pth"))
model.eval()

scaler = joblib.load("scaler_joint.save")
pca = joblib.load("pca_joint.save")

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
        inputs = np.frombuffer(data, dtype=np.float32)
        input_tensor = torch.FloatTensor(inputs.reshape(1, -1))

        with torch.no_grad():
            pca_output = model(input_tensor).numpy()

        synergy_vec = scaler.inverse_transform(pca.inverse_transform(pca_output)).flatten()

        # Decode thumb (0–8)
        thumb_sin, thumb_cos = synergy_vec[0:9], synergy_vec[9:18]
        thumb_deg = np.rad2deg(np.arctan2(thumb_sin, thumb_cos))

        # Decode PIP/DIP
        fix_pairs = [(18, 24), (19, 25), (20, 26), (21, 27), (22, 28), (23, 29)]
        decoded_fix = [np.rad2deg(np.arctan2(synergy_vec[s], synergy_vec[c])) for s, c in fix_pairs]

        # Raw joints
        raw = synergy_vec[30:]
        raw_i = 0
        output = []
        for i in range(27):
            if i < 9:
                output.append(thumb_deg[i])
            elif i in [13, 14, 16, 17, 25, 26]:
                output.append(decoded_fix.pop(0))
            else:
                output.append(raw[raw_i])
                raw_i += 1

        conn.sendall(np.array(output, dtype=np.float32).tobytes())

except Exception as e:
    print("Error:", e)
finally:
    conn.close()
    server.close()
