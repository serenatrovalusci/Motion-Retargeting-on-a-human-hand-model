import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Caricamento e preprocessing dei dati
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Rimuove spazi dai nomi colonne
    
    input_columns = ["ThumbClosure", "IndexClosure", "MiddleClosure"]
    output_columns = [col for col in df.columns if col not in input_columns]
    
    X = df[input_columns].values.astype('float32')
    y = df[output_columns].values.astype('float32')
    
    return X, y

# Dataset PyTorch
class HandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Definizione del VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Funzione di training
def train_vae(model, dataloader, epochs=1000, lr=0.001, save_path="vae_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)
            
            # Calcolo della VAE loss
            mse_loss = loss_fn(outputs, targets)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / inputs.size(0)
            loss = mse_loss + 0.001 * kl_loss  # Peso del termine KL
            
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
    vae = VAE(input_dim=3, latent_dim=5, output_dim=y.shape[1])
    train_vae(vae, dataloader)
