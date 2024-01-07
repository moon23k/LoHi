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



class HierSummarizer(nn.Module):
    def __init__(self, config):
        super(HierSummarizer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = HierEncoder(config)
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