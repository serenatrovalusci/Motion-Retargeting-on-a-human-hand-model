import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import joblib

class HandPoseFCNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=27):
        super().__init__()
        self.net = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.LeakyReLU(),
    nn.BatchNorm1d(512),

    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Dropout(0.3),

    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.BatchNorm1d(64),

    nn.Linear(64, output_dim)

        )

    def forward(self, x):
        return self.net(x)
    
class HandPoseFCNN_PCA(nn.Module):
    def __init__(self, input_dim=4, output_dim=11):
        super().__init__()
        self.net = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.LeakyReLU(),
    nn.BatchNorm1d(512),

    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Dropout(0.3),

    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.BatchNorm1d(64),

    nn.Linear(64, output_dim)

        )

    def forward(self, x):
        return self.net(x)
    
    
