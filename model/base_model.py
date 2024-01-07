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



class BaseSummarizer(nn.Module):
    def __init__(self, config):
        super(BaseSummarizer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = BaseEncoder(config)
        self.decoder = Decoder(config)

        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    def shift_trg(self, x):
        return x[:, :-1], x[:, 1:]


    def src_mask(self, x):
        return


    def trg_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        

    def forward(self, src, trg):
        trg, label = self.shift_trg(trg)
        sent_mask, text_mask = self.src_mask(src)
        trg_mask = self.trg_mask(trg)
        
        memory = self.encoder(x, sent_mask, text_mask)
        dec_out = self.decoder(trg, memory, tgt_mask=trg_mask,  
                               memory_key_padding_mask=text_mask)
        logit = self.generator(dec_out)
        

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out        