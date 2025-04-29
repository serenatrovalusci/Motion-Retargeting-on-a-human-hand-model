# # run_server.py
# import torch
# import torch.nn as nn
# import numpy as np
# import joblib
# import socket
# from HandPoseClass import *
# # ----------------------------
# # 2. Load model & scaler
# # ----------------------------
# model = HandPoseFCNN_PCA()
# model.load_state_dict(torch.load("hand_pose_fcnn_PCA_16.pth"))
# model.eval()

# scaler_y = joblib.load("scaler_PCA_16.save")  # Used to inverse scale outputs

# # ----------------------------
# # 3. Setup TCP server
# # ----------------------------
# HOST = '127.0.0.1'
# PORT = 65432
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind((HOST, PORT))
# server.listen(1)
# print(f"Server listening on {HOST}:{PORT}...")

# conn, addr = server.accept()
# print(f"Connected to {addr}")

# # ----------------------------
# # 4. Real-time communication loop
# # ----------------------------
# try:
#     while True:
#         data = conn.recv(1024)
#         if not data:
#             break

#         # Convert to float array
#         inputs = np.frombuffer(data, dtype=np.float32)
#         if inputs.shape[0] != 4:
#             print("Expected 4 float inputs.")
#             continue

#         # Predict
#         input_tensor = torch.FloatTensor(inputs.reshape(1, -1))
#         with torch.no_grad():
#             output = model(input_tensor).numpy()
#         # Inverse transform to joint space
#         pca = joblib.load('pca_joint_16.save')

#         print("Model output shape:", output.shape)
#         print("Expected by PCA:", pca.n_components_)

#         reconstructed_stand_joints = pca.inverse_transform(output)
#         print(" reconstructed_stand_joints:",  reconstructed_stand_joints)
#         reconstructed_joints = scaler_y.inverse_transform(reconstructed_stand_joints).flatten()
#         print(" reconstructed_joints:",  reconstructed_joints)
   
#         #scopriamo i limiti angolari della rotatione intorno a z della prima falange del thumb (colonna numero 2!!)
#         #scopriamo i le componenti della pca

#         #leviamo le rotazioni intorno a asse z del terzo joint del medio (componente 28, 27 nell originale), e del secondo (15 cioè 16) e del terzo joint dell'indice (18 cioè 19)
#         # Inverse transform to real values
#         #real_output = scaler_y.inverse_transform(output).flatten()

#         # Send back 27 floats
#         conn.sendall(reconstructed_joints.astype(np.float32).tobytes())

# except Exception as e:
#     print("Error:", e)
# finally:
#     conn.close()
#     server.close()



# PS C:\Users\LENOVO\Desktop\MR_project\Motion-Retargeting-on-a-human-hand-model> & C:/Users/LENOVO/AppData/Local/Microsoft/WindowsApps/python/Motion-Retargeting-on-a-human-hand-model/run_server.py
# C:\Users\LENOVO\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\base.py:380: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from versioC:\Users\LENOVO\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\base.py:380: InconsistentVerc:\Users\LENOVO\Desktop\MR_project\Motion-Retargeting-on-a-human-hand-model\run_server.py:46: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:209.)       
#   input_tensor = torch.FloatTensor(inputs.reshape(1, -1))
# Traceback (most recent call last):


# run_server_no_pca.py
import torch
import torch.nn as nn
import numpy as np
import joblib
import socket
from HandPoseClass import *

# ----------------------------
# 2. Load model & scaler (No PCA)
# ----------------------------
model = HandPoseFCNN(input_dim=4, output_dim=40)  # Adjust output_dim to the number of features after sin/cos
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

        print("Model output shape:", output.shape)

        # Inverse transform to joint space
        reconstructed_joints_scaled = output  # The model output is already the scaled sin/cos values
        reconstructed_joints = scaler_y.inverse_transform(reconstructed_joints_scaled).flatten()
        print("Reconstructed (scaled) joints:", reconstructed_joints_scaled)
        print("Reconstructed (unscaled) joints:", reconstructed_joints)

        # Send back 27 floats (original joint angles)
        # The 'reconstructed_joints' now contains the inverse-transformed sin/cos and x values.
        # You might need to process 'reconstructed_joints' to get back the original 27 joint angles
        # if your downstream application expects angles.

        # For now, we'll send back the 40 inverse-transformed values.
        conn.sendall(reconstructed_joints.astype(np.float32).tobytes())

except Exception as e:
    print("Error:", e)
finally:
    conn.close()
    server.close()