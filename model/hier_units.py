import torch
import torch.nn as nn
from model.common import Embeddings



class HierBlockEncoder(nn.Module):
    def __init__(self, config):
        super(HierBlockEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )
        
        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, src_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x



class HierLayerEncoder(nn.Module):
    def __init__(self, config):
        super(HierLayerEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )
        
        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, src_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x



class HierEncoder(nn.Module):
    def __init__(self, config):
        super(HierEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )
        
        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, src_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x                