import torch.nn as nn
import torch
import math

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
    def __init__(self, input_dim=4, output_dim=9):
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
    
    
class HandPoseTransformer(nn.Module):
    def __init__(self, input_dim=4, output_dim=27, d_model=128, num_heads=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input Embedding ottimizzato per 4 parametri
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional Encoding (opzionale per singoli valori, ma utile per estensioni future)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output Decoding anatomico dinamico
        self.output_heads = nn.ModuleDict({
            'thumb': FingerHead(d_model, 9),    # 3 falangi × 3 assi
            'index': FingerHead(d_model, 9),    # 3 falangi × 3 assi
            'middle': FingerHead(d_model, 9)    # 3 falangi × 3 assi
        })

        # Dynamic Output
        if output_dim != 27:
            self.dyn_out = nn.Linear(27,output_dim)
        

    def forward(self, x):
        # x shape: [batch_size, 4]
        
        # 1. Embedding dell'input
        x = self.input_embedding(x)  # [batch_size, d_model]
        x = x.unsqueeze(1)          # [batch_size, 1, d_model] (aggiunge dimensione sequenza)
        
        # 2. Positional encoding
        x = self.pos_encoder(x)
        
        # 3. Transformer
        encoded = self.transformer(x)
        encoded = encoded.squeeze(1)  # [batch_size, d_model]
        
        # 4. Output strutturato
        thumb = self.output_heads['thumb'](encoded)
        index = self.output_heads['index'](encoded)
        middle = self.output_heads['middle'](encoded)
        
        # 5. Combina gli output 
        output = torch.cat([thumb, index, middle], dim=-1)  # [batch_size, 27]

        # 6. Output Dinamico
        if self.dyn_out:
            output = self.dyn_out(output)
        
        return output

class FingerHead(nn.Module):
    """Testa di output per ogni dito"""
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.joint_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.GELU(),
                nn.Linear(d_model//2, 3)  # 3 valori x,y,z per ogni falange
            ) for _ in range(3)  # 3 falange per dito
        ])
    
    def forward(self, x):
        outputs = [decoder(x) for decoder in self.joint_decoders]
        return torch.cat(outputs, dim=-1)

class PositionalEncoding(nn.Module):
    """Versione semplificata per singolo elemento di sequenza"""
    def __init__(self, d_model, dropout, max_len=1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)