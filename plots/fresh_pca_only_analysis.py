
import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from HandPoseClass import HandPoseFCNN

os.chdir(os.path.dirname(os.path.abspath(__file__)))
closure_cols = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
fix_indices = [13, 14, 16, 17, 25, 26]
thumb_indices = list(range(9))
raw_indices = [i for i in range(27) if i not in thumb_indices + fix_indices]

# Load dataset
df = pd.read_csv("last_dataset.csv")
df.columns = df.columns.str.strip()
X = df[closure_cols].values
Y = df[[c for c in df.columns if c not in closure_cols]].values

# === Ground truth ===
def decode_gt(y):
    thumb = np.rad2deg(np.arctan2(np.sin(np.deg2rad(y[:, :9])), np.cos(np.deg2rad(y[:, :9]))))
    fix_sin = np.sin(np.deg2rad(y[:, fix_indices]))
    fix_cos = np.cos(np.deg2rad(y[:, fix_indices]))
    fix = np.rad2deg(np.arctan2(fix_sin, fix_cos))
    raw = y[:, raw_indices]
    output = []
    for i in range(27):
        if i < 9:
            output.append(thumb[:, i])
        elif i in fix_indices:
            j = fix_indices.index(i)
            output.append(fix[:, j])
        else:
            j = raw_indices.index(i)
            output.append(raw[:, j])
    return np.stack(output, axis=1)

Y_true = decode_gt(Y)

# === Rebuild synergy like in FCNN_PCA_EPIClo.py ===
def build_synergy_targets(y):
    thumb = np.concatenate([np.sin(np.deg2rad(y[:, :9])), np.cos(np.deg2rad(y[:, :9]))], axis=1)
    fix = np.concatenate([np.sin(np.deg2rad(y[:, fix_indices])), np.cos(np.deg2rad(y[:, fix_indices]))], axis=1)
    raw = y[:, raw_indices]
    return np.concatenate([thumb, fix, raw], axis=1)

Y_synergy = build_synergy_targets(Y)

# === Load PCA model ===
model_pca = HandPoseFCNN(input_dim=4, output_dim=11)
model_pca.load_state_dict(torch.load("hand_pose_fcnn_PCA_11_new.pth", map_location="cpu"))
model_pca.eval()

scaler = joblib.load("scaler_joint_11.save")
pca = joblib.load("pca_joint_11.save")

# === Inference ===
X_tensor = torch.FloatTensor(X)
with torch.no_grad():
    pca_out = model_pca(X_tensor).numpy()

# === Reverse PCA and scaler ===
Y_synergy_pred = scaler.inverse_transform(pca.inverse_transform(pca_out))

# === Decode synergy to 27 joint angles ===
def decode_synergy(preds):
    thumb = np.rad2deg(np.arctan2(preds[:, 0:9], preds[:, 9:18]))
    fix_pairs = [(18, 24), (19, 25), (20, 26), (21, 27), (22, 28), (23, 29)]
    fix = np.stack([np.rad2deg(np.arctan2(preds[:, s], preds[:, c])) for s, c in fix_pairs], axis=1)
    raw = preds[:, 30:]
    output = []
    for i in range(27):
        if i < 9:
            output.append(thumb[:, i])
        elif i in fix_indices:
            j = fix_indices.index(i)
            output.append(fix[:, j])
        else:
            j = raw_indices.index(i)
            output.append(raw[:, j])
    return np.stack(output, axis=1)

Y_pred_pca = decode_synergy(Y_synergy_pred)

# === Error metrics ===
mae_pca = np.mean(np.abs(Y_true - Y_pred_pca), axis=0)
rmse_pca = np.sqrt(np.mean((Y_true - Y_pred_pca)**2, axis=0))

# === Plot ===
os.makedirs("ComparisonPlots", exist_ok=True)
plt.figure(figsize=(14, 6))
plt.plot(list(range(27)), mae_pca, label="MAE (PCA)", marker='o')
plt.plot(list(range(27)), rmse_pca, label="RMSE (PCA)", marker='s')
plt.xlabel("Joint Index")
plt.ylabel("Error (degrees)")
plt.title("Fresh PCA-Only Per-Joint Errors (Trusted Build)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ComparisonPlots/fresh_pca_per_joint_error.png")
plt.close()

print("✅ PCA-only trusted plot saved to ComparisonPlots/fresh_pca_per_joint_error.png")
print(f"Average MAE (PCA): {mae_pca.mean():.4f}")
print(f"Average RMSE (PCA): {rmse_pca.mean():.4f}")
