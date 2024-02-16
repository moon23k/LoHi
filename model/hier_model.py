import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from .common import (
    clones, Embeddings, Decoder,
    PositionalEncoding, 
    PositionwiseFeedForward,
    SublayerConnection
)



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.model_type = config.model_type

        self.attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.n_heads,
            batch_first=True
        )
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)

        self.sent_pos = PositionalEncoding(config)

        if self.model_type == 'hier_lin':
            self.sent_lin = PositionwiseFeedForward(config)

        elif self.model_type == 'hier_rnn':
            self.rnn = nn.GRU(config.hidden_dim, config.hidden_dim)
            self.act = F.gelu()
            self.dropout = nn.Dropout(config.dropout_ratio)

        elif self.model_type == 'hier_attn':
            self.hier_attn = nn.MultiheadAttention(
                config.hidden_dim,
                config.n_heads,
                batch_first=True
            )
            self.hier_pff = PositionwiseFeedForward(config)
            self.hier_sublayer = clones(SublayerConnection(config), 2)



    def forward(self, x, sent_mask, text_mask=None):

        batch_size, seq_num, seq_len, hidden_dim = x.shape
        x = x.view(batch_size * seq_num, seq_len, hidden_dim)

        x = self.sublayer[0](
            x, lambda x: self.attn(
                x, x, x, key_padding_mask=sent_mask
                )[0]
            )

        x = x.view(batch_size, seq_num, seq_len, hidden_dim)

        return self.hier_forward(x, text_mask)



    def hier_forward(self, x, text_mask=None):
        #process for hierarchical vector
        x = x[:, 0]
        x = x + self.sent_pos(x)

        #process for hierarchical network
        if self.model_type == 'hier_lin':
            out = self.dropout(self.linear(x))
        elif self.model_type == 'hier_rnn':
            out = self.dropout(self.rnn(x))
        elif self.model_type == 'hier_attn':
            out = self.dropout(self.hier_attn(x, x, x, text_mask))
        return out



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.embeddings = Embeddings(config)
        self.layer = EncoderLayer(config)


    def forward(self, x, sent_mask, text_mask=None):
        x = self.embeddings(x)
        for layer in enumerate(self.sent_layers):
            x = layer(x, sent_mask=sent_mask, text_mask=text_mask)
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