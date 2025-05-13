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
    
    
class HandPoseTransformer(nn.Module):
    def __init__(self, input_dim=4, fix_indices=[], d_model=128, num_heads=8, 
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
        
        # calcolo delle dimensioni di output delle testate
        thu = 0
        ind = 0
        mid = 0
        rin = 0
        pin = 0
        for i in range(45):
            if i<9:
                if i in fix_indices:
                    thu += 2
                else:
                    thu += 1
            elif i<18:
                if i in fix_indices:
                    ind += 2
                else:
                    ind += 1
            elif i<27:
                if i in fix_indices:
                    mid += 2
                else:
                    mid += 1
            elif i<36:
                if i in fix_indices:
                    rin += 2
                else:
                    rin += 1
            else:
                if i in fix_indices:
                    pin += 2
                else:
                    pin += 1
        #print(f" thumb={thu}, index={ind}, middle={mid}, ring={rin}, pinky={pin}")

        # Output Decoding anatomico
        self.output_heads = nn.ModuleDict({
            'thumb': FingerHead(d_model, thu),    
            'index': FingerHead(d_model, ind),    
            'middle': FingerHead(d_model, mid),    
            'ring': FingerHead(d_model, rin),    
            'pinky': FingerHead(d_model, pin)    
        })


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
        ring = self.output_heads['ring'](encoded)
        pinky = self.output_heads['pinky'](encoded)
        
        # 5. Combina gli output 
        output = torch.cat([thumb, index, middle, ring, pinky], dim=-1)  # [batch_size, 45+fixed]
        
        return output

class FingerHead(nn.Module):
    """Testa di output per ogni dito"""
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
                                nn.Linear(d_model, d_model//2),
                                nn.GELU(),
                                nn.Linear(d_model//2, output_dim) 
                                ) 
    
    def forward(self, x):
        return self.decoder(x)

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