import numpy as np
import json, itertools, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split, pad_id):
        super().__init__()

        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']
        
        src_ids = [self.tokenizer.encode(x).ids for x in src]
        trg_ids = torch.LongTensor(self.tokenizer.encode(trg).ids)
        return src_ids, trg_ids, len(src_ids), max([len(x) for x in src_ids])



class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        src_batch, trg_batch, num_batch, len_batch = list(zip(*batch))

        src_batch = self.pad_src(src_batch, max(num_batch), max(len_batch))
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=self.pad_id)

        return {'src': src_batch, 
                'trg': trg_batch}
    
    
    def pad_src(self, batch, max_num, max_len):

        padded_batch = np.full(
            shape=(len(batch), max_num, max_len), 
            fill_value=self.pad_id, 
            dtype=int
        )        
        
        for batch_idx, text in enumerate(batch):
            for text_idx, sent in enumerate(text):
                sent_pad = [self.pad_id for _ in range(max_len - len(sent))]
                padded_batch[batch_idx, text_idx] = sent + sent_pad

        return torch.LongTensor(padded_batch)


def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(tokenizer, split, config.pad_id), 
        batch_size=config.batch_size, 
        shuffle=True if config.mode == 'train' else False,
        collate_fn=Collator(config.pad_id),
        num_workers=2
    )
    