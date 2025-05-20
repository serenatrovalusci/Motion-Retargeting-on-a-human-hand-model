import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import zscore

def remove_outliers_zscore(X, Y, threshold):
    z_scores = np.abs(zscore(Y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], Y[mask]

def generate_sincos_dataset(Y, fix_indices):
    # In questo modo avremo un dataset tipo: [sin1, cos1, sin2, cos2, ang3, ang4, ...] 
    # l'ordine degli angoli non è mischiato, i primi sono del pollice, poi ci sono quelli dell'indice ecc...
    
    mixed_columns = []
    for i in range(Y.shape[1]):
        if i in fix_indices:
            sin_val = np.sin(np.deg2rad(Y[:, [i]]))
            cos_val = np.cos(np.deg2rad(Y[:, [i]]))
            mixed_columns.extend([sin_val, cos_val])
        else:
            mixed_columns.append(Y[:, [i]])
    return np.concatenate(mixed_columns, axis=1)

def load_data(csv_path, closure_columns, fix_indices, z_thresh=2.5):
    print("Loading Dataset from csv file...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data = data.iloc[:, 1:] # remuve the first column (time)
    joint_columns = [c for c in data.columns if c not in closure_columns]
    X = data[closure_columns].values
    Y = data[joint_columns].values

    print("Dataset size:", len(TensorDataset(torch.FloatTensor(Y))))

    _, Y = remove_outliers_zscore(X, Y, z_thresh)
    Y_sincos = generate_sincos_dataset(Y, fix_indices)

    scaler = StandardScaler().fit(Y_sincos)
    Y_scaled = scaler.transform(Y_sincos)

    print("All components:", Y_scaled.shape[1])

    return Y_scaled, scaler

def find_min_max(Y):

    min_vals = np.min(Y, axis=0)
    max_vals = np.max(Y, axis=0)
    return min_vals, max_vals


closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
Y_sc, _ = load_data("hand_dataset_all_fingers.csv", closure_columns, fix_indices, z_thresh=2.5)

min_vals, max_vals = find_min_max(Y_sc)

print("Min values:", min_vals)
print("Max values:", max_vals)

np.save("min_vals.npy", min_vals)
np.save("max_vals.npy", max_vals)


