import numpy as np
import json, itertools, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, split):
        super().__init__()

        self.pad_id = config.pad_id
        self.model_type = config.model_type
        self.tokenizer = tokenizer
        self.data = self.load_data(split)


    def load_data(self, split):
        with open(f"data/{self.model_type}_{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):        
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']

        if self.model_type == 'base':
            src_ids = self.tokenizer(src).ids
            trg_ids = self.tokenizer(trg).ids

            return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)
        
        elif self.model_type == 'hier':
            src_ids = [self.tokenizer.encode(x).ids for x in src]
            trg_ids = torch.LongTensor(self.tokenizer.encode(trg).ids)
            return src_ids, trg_ids, len(src_ids), max([len(x) for x in src_ids])




class Collator(object):
    def __init__(self, config):
        self.pad_id = pad_id
        self.model_type = config.model_type


    def __call__(self, batch):
        if self.model_type == 'base':
            return self.base_collate(batch)
        elif self.model_type == 'hier':
            return self.hier_collate(batch)


    def base_collate(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_batch = self.base_pad(src_batch)
        trg_batch = self.base_pad(trg_batch)

        return {'src': src_batch, 
                'trg': trg_batch}


    def base_pad(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


    def hier_collate(self, batch):
        src_batch, trg_batch, num_batch, len_batch = zip(*batch)

        src_batch = self.hier_pad(src_batch, max(num_batch), max(len_batch))
        trg_batch = self.base_pad(trg_batch)

        return {'src': src_batch, 
                'trg': trg_batch}

    
    def hier_pad(self, batch, max_num, max_len):

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
    is_train = True if split == 'train' else False
    return DataLoader(
        Dataset(config, tokenizer, split), 
        batch_size=config.batch_size if is_train else 1, 
        shuffle=True if is_train else False,
        collate_fn=Collator(config),
        num_workers=2
    )
    