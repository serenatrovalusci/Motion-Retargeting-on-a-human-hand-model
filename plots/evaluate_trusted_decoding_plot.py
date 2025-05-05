
import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from HandPoseClass import HandPoseFCNN

# Ensure same working folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
df = pd.read_csv("last_dataset.csv")
df.columns = df.columns.str.strip()
closure_cols = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
X = df[closure_cols].values
Y = df[[c for c in df.columns if c not in closure_cols]].values

# Joint mapping
fix_indices = [13, 14, 16, 17, 25, 26]
thumb_indices = list(range(9))
raw_indices = [i for i in range(27) if i not in thumb_indices + fix_indices]

# Ground truth decoding
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

# Decode predictions from synergy
def decode(preds):
    thumb_deg = np.rad2deg(np.arctan2(preds[:, 0:9], preds[:, 9:18]))
    fix_pairs = [(18, 24), (19, 25), (20, 26), (21, 27), (22, 28), (23, 29)]
    fix_decoded = np.stack([np.rad2deg(np.arctan2(preds[:, s], preds[:, c])) for s, c in fix_pairs], axis=1)
    raw = preds[:, 30:]
    final = []
    for i in range(27):
        if i < 9:
            final.append(thumb_deg[:, i])
        elif i in fix_indices:
            j = fix_indices.index(i)
            final.append(fix_decoded[:, j])
        else:
            j = raw_indices.index(i)
            final.append(raw[:, j])
    return np.stack(final, axis=1)

# Build synergy target from raw joints
def build_synergy_targets(y):
    thumb = np.concatenate([np.sin(np.deg2rad(y[:, :9])), np.cos(np.deg2rad(y[:, :9]))], axis=1)
    fix = np.concatenate([np.sin(np.deg2rad(y[:, fix_indices])), np.cos(np.deg2rad(y[:, fix_indices]))], axis=1)
    raw = y[:, raw_indices]
    return np.concatenate([thumb, fix, raw], axis=1)

Y_true = decode_gt(Y)
Y_synergy = build_synergy_targets(Y)

# Non-PCA model
model_nopca = HandPoseFCNN(input_dim=4, output_dim=42)
model_nopca.load_state_dict(torch.load("hand_pose_fcnn_epic_nopca.pth", map_location="cpu"))
model_nopca.eval()
scaler_nopca = joblib.load("scaler_y_epic_nopca.save")

with torch.no_grad():
    pred_nopca = model_nopca(torch.FloatTensor(X)).numpy()
pred_nopca = scaler_nopca.inverse_transform(pred_nopca)
decoded_nopca = decode(pred_nopca)

# PCA model
model_pca = HandPoseFCNN(input_dim=4, output_dim=11)
model_pca.load_state_dict(torch.load("hand_pose_fcnn_PCA_11_new.pth", map_location="cpu"))
model_pca.eval()
scaler_pca = joblib.load("scaler_joint_11.save")
pca = joblib.load("pca_joint_11.save")

with torch.no_grad():
    pred_pca = model_pca(torch.FloatTensor(X)).numpy()
decoded_synergy_pca = scaler_pca.inverse_transform(pca.inverse_transform(pred_pca))
decoded_pca = decode(decoded_synergy_pca)

# Compute MAE and RMSE
mae_nopca = np.mean(np.abs(Y_true - decoded_nopca), axis=0)
rmse_nopca = np.sqrt(np.mean((Y_true - decoded_nopca)**2, axis=0))
mae_pca = np.mean(np.abs(Y_true - decoded_pca), axis=0)
rmse_pca = np.sqrt(np.mean((Y_true - decoded_pca)**2, axis=0))

# Plotting
os.makedirs("ComparisonPlots", exist_ok=True)
joints = list(range(27))
plt.figure(figsize=(14, 6))
plt.plot(joints, mae_nopca, label='MAE (No PCA)', marker='o')
plt.plot(joints, mae_pca, label='MAE (PCA)', marker='o')
plt.plot(joints, rmse_nopca, label='RMSE (No PCA)', marker='s')
plt.plot(joints, rmse_pca, label='RMSE (PCA)', marker='s')
plt.xlabel("Joint Index")
plt.ylabel("Error (degrees)")
plt.title("Per-Joint MAE and RMSE: PCA vs No PCA (Trusted Version)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ComparisonPlots/per_joint_error_trusted.png")
plt.close()

print("✅ Trusted decoding plot saved to ComparisonPlots/per_joint_error_trusted.png")
print(f"Average MAE (No PCA): {mae_nopca.mean():.4f}")
print(f"Average MAE (PCA):    {mae_pca.mean():.4f}")
print(f"Average RMSE (No PCA): {rmse_nopca.mean():.4f}")
print(f"Average RMSE (PCA):    {rmse_pca.mean():.4f}")
