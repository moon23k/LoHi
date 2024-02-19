import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .components import (
    clones, Embeddings,
    PositionalEncoding, 
    PositionwiseFeedForward,
    SublayerConnection
)



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.n_heads,
            batch_first=True
        )
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, mask):
        is_lower = len(x.shape) == 4

        if is_lower:
            batch_size, seq_num, seq_len, hidden_dim = x.shape
            x = x.view(-1, seq_len, hidden_dim)
            mask = mask.view(-1, seq_len)

        x = self.sublayer[0](
            x, lambda x: self.attn(
                x, x, x, key_padding_mask=mask
                )[0]
            )

        if is_lower:
            x = x.view(batch_size, seq_num, seq_len, hidden_dim)

        return self.sublayer[1](x, self.pff)




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.model_type = config.model_type

        self.embeddings = Embeddings(config)
        layer = EncoderLayer(config)
        self.layers = clones(layer, config.n_layers)

        self.pos_enc = PositionalEncoding(config)
        if self.model_type == 'hier_lin':
            self.lin = PositionwiseFeedForward(config)
        elif self.model_type == 'hier_rnn':
            self.rnn = nn.GRU(config.hidden_dim, config.hidden_dim)
        elif self.model_type == 'hier_attn':
            self.high_layers = clones(layer, config.n_layers)            


    def forward(self, x, sent_mask, text_mask=None):
        #lower layer process
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, sent_mask)
        
        x = x[:, :, 0, :]
        x = self.pos_enc(x)

        if self.model_type == 'hier_lin':
            x = self.lin(x)
        elif self.model_type == 'hier_rnn':
            x = self.rnn(x)[0]
        elif self.model_type == 'hier_attn':
            for layer in self.high_layers:
                x = layer(x, text_mask)        
        
        return x




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, e_mask=None, d_mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x




class HierModel(nn.Module):
    def __init__(self, config):
        super(HierModel, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def enc_mask(self, x):
        sent_mask = x == self.pad_id
        text_mask = sent_mask.all(dim=-1)
        
        return sent_mask, text_mask


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, x, y):
        y, label = self.shift_y(y)

        #Masking
        sent_mask, text_mask = self.enc_mask(x)
        d_mask = self.dec_mask(y)
        
        #Actual Processing
        memory = self.encoder(x, sent_mask, text_mask)
        dec_out = self.decoder(y, memory, text_mask, d_mask)
        logit = self.generator(dec_out)
        
        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out