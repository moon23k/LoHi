import torch
import torch.nn as nn
from .common import (
    clones,
    Embeddings, 
    PositionalEncoding, 
    MultiHeadAttention, 
    PositionwiseFeedForward,
    SublayerConnection
)



class BaseLayer(nn.Module):
    def __init__(self, config):
        super(BaseLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.sent_attn(q=x, k=x, v=x, mask=mask))  
        return self.sublayer[1](x, self.feed_forward)



class BaseEncoder(nn.Module):
    def __init__(self, config):
        super(BaseEncoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(BaseLayer(config), config.n_layers)

    def forward(self, x, mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

