import copy, math, torch
import torch.nn as nn
from collections import namedtuple



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(config.max_len, config.emb_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(config.dropout_ratio)
        

    def forward(self, x):
        out = x + self.pe[:, :x.size(1)]
        return self.dropout(out)



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)

        self.use_fc_layer = (config.emb_dim != config.hidden_dim)
        if self.use_fc_layer:
            self.fc = nn.Linear(config.emb_dim, config.hidden_dim)
            self.fc_dropout = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)

        if not self.use_fc_layer:
            return out
        return self.fc_dropout(self.fc(out))




class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))




class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim, 
            num_heads=config.n_heads, 
            dropout=config.dropout_ratio,
            batch_first=True
        )        

    def forward(self, q, k, v, pad_mask=None, attn_mask=None):
        
        out, _ = self.attn(
            query=q, key=k, value=v,
            key_padding_mask=pad_mask, 
            attn_mask=attn_mask
        )

        return out




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return x        
    


class Summarizer(nn.Module):
    def __init__(self, config, encoder):
        super(Summarizer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size
        
        self.encoder = encoder
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