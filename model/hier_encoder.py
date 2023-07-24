import torch
import torch.nn as nn
from model.common import (
    clones,
    Embeddings, 
    PositionalEncoding, 
    MultiHeadAttention, 
    PositionwiseFeedForward,
    SublayerConnection
)



class SentLayer(nn.Module):
    def __init__(self, config):
        super(SentLayer, self).__init__()
        self.sent_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

    def forward(self, x, sent_mask=None, text_mask=None):
        batch_size, seq_num, seq_len, hidden_dim = x.shape
        x = x.view(batch_size * seq_num, seq_len, hidden_dim)
        x = self.sublayer[0](x, lambda x: self.sent_attn(q=x, k=x, v=x, sent_mask=sent_mask, text_mask=None))
        return self.sublayer[2](x, self.feed_forward).view(batch_size, seq_num, seq_len, hidden_dim)


class TextLayer(nn.Module):
    def __init__(self, config):
        super(TextLayer, self).__init__()
        self.text_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, sent_mask=None, text_mask=None):
        batch_size, seq_num, seq_len, hidden_dim = x.shape
        x = x.view(batch_size * seq_num, seq_len, hidden_dim)
        x = self.sublayer[0](x, lambda x: self.text_attn(q=x, k=x, v=x, sent_mask=None, text_mask=text_mask))
        return self.sublayer[2](x, self.feed_forward)



class HierEncoder(nn.Module):
    def __init__(self, config):
        super(HierEncoder, self).__init__()

        sent_layer = SentLayer(config)
        text_layer = TextLayer(config)
        
        self.embeddings = Embeddings(config)
        self.sent_layers = clones(sent_layer, config.n_layers)
        self.text_pos = PositionalEncoding(config)
        self.text_layers = clones(text_layer, config.n_layers)


    def forward(self, x, sent_mask, text_mask):
        x = self.embeddings(x)
        for layer in enumerate(self.sent_layers):
            x = layer(x, sent_mask=sent_mask)

        x = self.text_pos(x)
        for layer in self.text_layers:
            x = layer(x, text_mask=text_mask)

        return x
