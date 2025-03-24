import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Caricamento e preprocessing dei dati
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Rimuovere spazi dai nomi delle colonne
    df.columns = df.columns.str.strip()
    
    # Separare input (parametri di closure) e output (rotazioni giunti)
    input_columns = ["ThumbClosure", "IndexClosure", "MiddleClosure"]
    output_columns = [col for col in df.columns if col not in input_columns]

    print("Colonne di input:", input_columns)
    print("Colonne di output:", output_columns)
    
    X = df[input_columns].values.astype('float32')
    y = df[output_columns].values.astype('float32')
    
    return X, y

# Definizione Dataset PyTorch
class HandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Definizione della rete neurale MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Funzione per il training e logging
def train_model(model, dataloader, epochs=1000, lr=0.001, save_path="mlp_model.pth"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Salvataggio del modello
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")
    
    # Plot della loss
    plt.plot(range(1, epochs+1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss durante il training')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Percorso al file CSV ----- da modificare in funzione di dove è salvato il file del dataset
    # csv_path = "/content/hand_dataset.csv" ----- usare questo percorso se si usa colab
    csv_path = "C:/Users/Giordano/OneDrive/Documenti/UniSap/Medical Robotics/HandProject_DM/hand_dataset.csv"

    X, y = load_data(csv_path)
    
    # Creazione dataset e dataloader
    dataset = HandDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Creazione e training del modello
    model = MLP(input_dim=3, output_dim=27)
    train_model(model, dataloader)