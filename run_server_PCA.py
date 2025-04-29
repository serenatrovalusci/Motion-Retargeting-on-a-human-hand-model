# run_server.py
import torch
import torch.nn as nn
import numpy as np
import joblib
import socket
from HandPoseClass import *
# ----------------------------
# 2. Load model & scaler
# ----------------------------
model = HandPoseFCNN_PCA()
model.load_state_dict(torch.load("hand_pose_fcnn_PCA_16.pth"))
model.eval()

scaler = joblib.load("scaler_PCA_16.save")  # Used to inverse scale outputs

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
        # Inverse transform to joint space
        pca = joblib.load('pca_joint_16.save')

        print("Model output shape:", output.shape)
        print("Expected by PCA:", pca.n_components_)

        reconstructed_stand_joints = pca.inverse_transform(output)
        print(" reconstructed_stand_joints:",  reconstructed_stand_joints)
        reconstructed_joints = scaler.inverse_transform(reconstructed_stand_joints).flatten()
        print(" reconstructed_joints:",  reconstructed_joints)

        # Clamp the thumb 1y,2y,3y 

      # Bound the thumb joints [0], [1], [2] to avoid negative values
        # You can customize thresholds based on what's safe for your avatar

        # min_thumb_angle = 110.0  # or use a small positive value if 0 is still problematic
        # max_thumb_angle = 165.0  # example max, adjust based on your range
        # min_middle_angle = 155.0  # or use a small positive value if 0 is still problematic
        # max_middle_angle = 162.0
        # Apply clamping to thumb 1,2,3
        # reconstructed_joints[1] = np.clip(reconstructed_joints[1], min_thumb_angle, max_thumb_angle)
        # reconstructed_joints[4] = np.clip(reconstructed_joints[4], min_thumb_angle, max_thumb_angle)
        # reconstructed_joints[7] = np.clip(reconstructed_joints[7], min_thumb_angle, max_thumb_angle)
        #reconstructed_joints[26] = np.clip(reconstructed_joints[26], min_middle_angle, max_middle_angle)

        # Fix index 
        #reconstructed_joints[13] = -90.0
        
        # Change sign index to z component of 2,3
        # reconstructed_joints[14] = -reconstructed_joints[14]
        # reconstructed_joints[17] = -reconstructed_joints[17]

        # if input_tensor[0, 0].item() == 0.0 and input_tensor[0, 3].item() < 0.87 :
        #    reconstructed_joints[1] = -reconstructed_joints[1]
        #    reconstructed_joints[4] = -reconstructed_joints[4]
        #    reconstructed_joints[7] = -reconstructed_joints[7]
         
        # middle post processing joint 3
        # print(input_tensor)

        # if input_tensor[0, 2].item() > 0.9:
        #    #reconstructed_joints[25] = 95.0
        #    reconstructed_joints[24] = -70.0
        #    reconstructed_joints[26] = np.clip(reconstructed_joints[26], min_middle_angle, max_middle_angle)
        # if input_tensor[0, 2].item() > 0.85:
        #    reconstructed_joints[25] = 95.0
         
        # Inverse transform to real values
        #real_output = scaler_y.inverse_transform(output).flatten()

        # Send back 27 floats
        conn.sendall(reconstructed_joints.astype(np.float32).tobytes())

except Exception as e:
    print("Error:", e)
finally:
    conn.close()
    server.close()



# PS C:\Users\LENOVO\Desktop\MR_project\Motion-Retargeting-on-a-human-hand-model> & C:/Users/LENOVO/AppData/Local/Microsoft/WindowsApps/python/Motion-Retargeting-on-a-human-hand-model/run_server.py
# C:\Users\LENOVO\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\base.py:380: InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from versioC:\Users\LENOVO\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\sklearn\base.py:380: InconsistentVerc:\Users\LENOVO\Desktop\MR_project\Motion-Retargeting-on-a-human-hand-model\run_server.py:46: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:209.)       
#   input_tensor = torch.FloatTensor(inputs.reshape(1, -1))
# Traceback (most recent call last):