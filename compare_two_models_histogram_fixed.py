import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from HandPoseClass import HandPoseFCNN, HandPoseTransformer
from scipy.stats import zscore

def decode_output_sincos(output, fix_indices):
    output = output.flatten()
    final_angles = []
    i = 0
    angle_idx = 0
    while i < len(output):
        if angle_idx in fix_indices:
            sin_val = output[i]
            cos_val = output[i + 1]
            angle = np.rad2deg(np.arctan2(sin_val, cos_val))
            final_angles.append(angle)
            i += 2
        else:
            final_angles.append(output[i])
            i += 1
        angle_idx += 1
    return np.array(final_angles)

def decode_output_raw(output, _):
    return output.flatten()

def generate_sincos_dataset(Y, fix_indices):
    mixed_columns = []
    for i in range(Y.shape[1]):
        if i in fix_indices:
            sin_val = np.sin(np.deg2rad(Y[:, [i]]))
            cos_val = np.cos(np.deg2rad(Y[:, [i]]))
            mixed_columns.extend([sin_val, cos_val])
        else:
            mixed_columns.append(Y[:, [i]])
    return np.concatenate(mixed_columns, axis=1)

def prepare_ground_truth(Y_raw, fix_indices):
    Y_sincos = generate_sincos_dataset(Y_raw, fix_indices)
    return np.array([decode_output_sincos(y, fix_indices) for y in Y_sincos])

def load_and_evaluate(model_path, scaler_path, pca_path, model_type, fix_indices, X, decode_fn):
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path) if pca_path else None
    output_dim = pca.n_components_ if pca else scaler.mean_.shape[0]

    if model_type == "Transformer":
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_indices, pca_dim=output_dim if pca else 0)
    else:
        model = HandPoseFCNN(input_dim=4, output_dim=output_dim)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).numpy()

    if pca:
        preds = pca.inverse_transform(preds)
    preds = scaler.inverse_transform(preds)

    decoded = np.array([decode_fn(p, fix_indices) for p in preds])
    return decoded

def display_mae_histogram(mae1, mae2, label1, label2):
    num_joints = len(mae1)
    indices = np.arange(num_joints)
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(indices - width / 2, mae1, width, label=label1, color="orange")
    plt.bar(indices + width / 2, mae2, width, label=label2, color="green")

    plt.xlabel("Joint Index")
    plt.ylabel("Mean Absolute Error (degrees)")
    plt.title("Per-Joint MAE Comparison (Degrees)")
    plt.xticks(indices)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("histograms/latent_space_15.png")
    plt.show()

if __name__ == "__main__":
    fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']

    df = pd.read_csv("dataset/hand_dataset_all_fingers.csv").iloc[:, 1:]
    X = df[closure_columns].values
    Y = df.drop(columns=closure_columns).values
    mask = (np.abs(zscore(Y, axis=0)) < 2.5).all(axis=1)
    X, Y = X[mask], Y[mask]

    Y = prepare_ground_truth(Y, fix_indices)

    # Comparison Synergies Models
    # model_1 = {
    #     "label": "FCNN Non-PCA",
    #     "model": "training_results/training_synergies_results/training_20250607_131503/FCNN.pth",
    #     "scaler": "training_results/training_synergies_results/training_20250607_131503/scaler.save",
    #     "pca": None,
    #     "type": "FCNN",
    #     "decode": decode_output_sincos
    # }

    # model_2 = {
    #     "label": "FCNN PCA 45",
    #     "model": "training_results/training_synergies_results/training_20250610_174333/FCNN_45_PCA.pth",
    #     "scaler": "training_results/training_synergies_results/training_20250610_174333/scaler.save",
    #     "pca": "training_results/training_synergies_results/training_20250610_174333/pca.save",
    #     "type": "FCNN",
    #     "decode": decode_output_sincos
    # }

    # Compararison Latent Space Models
    shared_scaler_path = "training_results/training_reduction_models/training_20250605_180718/scaler_AE.save"

    model_1 = {
        "label": "FCNN AE 15",
        "model": "training_results/training_latentspace_results/FCNN/training_20250606_185214/FCNN_AE_regression.pth",
        "scaler": shared_scaler_path,
        "pca": None,
        "type": "FCNN",
        "decode": decode_output_sincos
    }

    model_2 = {
        "label": "Transformer AE 15",
        "model": "training_results/training_latentspace_results/Transformer/training_20250606_191502/Transformer_AE_regression.pth",
        "scaler": shared_scaler_path,
        "pca": None,
        "type": "Transformer",
        "decode": decode_output_sincos
    }


    Y_pred1 = load_and_evaluate(model_1["model"], model_1["scaler"], model_1["pca"], model_1["type"], fix_indices, X, model_1["decode"])
    Y_pred2 = load_and_evaluate(model_2["model"], model_2["scaler"], model_2["pca"], model_2["type"], fix_indices, X, model_2["decode"])

    mae1 = np.mean(np.abs(Y_pred1 - Y), axis=0)
    mae2 = np.mean(np.abs(Y_pred2 - Y), axis=0)

    display_mae_histogram(mae1, mae2, model_1["label"], model_2["label"])