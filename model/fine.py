import math, copy, torch
import torch.nn as nn
from collections import namedtuple



class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout_ratio)
        
        pe = torch.zeros(max_len, config.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.fc = nn.Linear(config.emb_dim, config.hidden_dim)

    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)
        return self.fc(out)



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.embeddings = Embeddings(config)
        
        layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           batch_first=True, norm_first=True)
        
        self.decoder = nn.TransformerDecoder(decoder_layer=layer, 
                                             num_layers=config.n_layers,
                                             norm=nn.LayerNorm(config.hidden_dim))
        

    def forward(self, x, memory, x_sub_mask, x_pad_mask, m_pad_mask):
        return self.decoder(self.embeddings(x), memory, 
                            memory_key_padding_mask=m_pad_mask, 
                            tgt_key_padding_mask=x_pad_mask, 
                            tgt_mask=x_sub_mask)



class FineModel(nn.Module):
    def __init__(self, config, bert, bert_embeddings):
        super(FineModel, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.encoder = bert
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_id, 
                                             label_smoothing=0.1).to(self.device)
        self.outputs = namedtuple('outputs', ('logits', 'loss'))



    def forward(self, x, x_seg_mask, y):

        #shift y
        y_input = y[:, :-1]
        y_label = y[:, 1:]


        #create masks
        x_pad_mask = (x != self.pad_id).to(self.device)
        y_pad_mask = (y_input != self.pad_id).to(self.device)
        y_size = y_input.size(1)
        y_sub_mask = torch.triu(torch.full((y_size, y_size), float('-inf')), diagonal=1).to(self.device)
        

        memory = self.encoder(input_ids=x, token_type_ids=x_seg_mask, 
                              attention_mask=x_pad_mask).last_hidden_state
        
        d_out = self.decoder(x=y_input, memory=memory, 
                             x_sub_mask=y_sub_mask, 
                             x_pad_mask=y_pad_mask, 
                             m_pad_mask=x_pad_mask)
        
        logits = self.generator(d_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              y_label.contiguous().view(-1))

        return self.outputs(logits, loss)
