import numpy as np
import json, itertools, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, split):
        super().__init__()

        self.data_type = 'base' if config.model_type == 'base' else 'hier'
        self.tokenizer = tokenizer
        self.data = self.load_data(split)


    def load_data(self, split):
        with open(f"data/{self.data_type}_{split}.json", 'r') as f:
            data = json.load(f)
        if split == 'train':
            return data[::10]
        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):        
        x, y = self.data[idx]['x'], self.data[idx]['y']

        if self.data_type == 'base':
            x = self.tokenizer.encode(x).ids
            y = self.tokenizer.encode(y).ids
            return torch.LongTensor(x), torch.LongTensor(y)
        
        else:
            x = [self.tokenizer.encode(sent).ids for sent in x]
            y = torch.LongTensor(self.tokenizer.encode(y).ids)
            return x, y, len(x), max([len(sent) for sent in x])




class Collator(object):
    def __init__(self, config):
        self.pad_id = config.pad_id
        self.model_type = config.model_type


    def __call__(self, batch):
        if self.model_type == 'base':
            return self.base_collate(batch)
        else:
            return self.hier_collate(batch)


    def base_collate(self, batch):
        x_batch, y_batch = zip(*batch)

        return {'x': self.base_pad(x_batch), 
                'y': self.base_pad(y_batch)}


    def hier_collate(self, batch):
        x_batch, y_batch, num_batch, len_batch = zip(*batch)

        return {'x': self.hier_pad(x_batch, max(num_batch), max(len_batch)), 
                'y': self.base_pad(y_batch)}

    
    def base_pad(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


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
    return DataLoader(
        Dataset(config, tokenizer, split), 
        batch_size=config.batch_size, 
        shuffle= split == 'train',
        collate_fn=Collator(config),
        num_workers=2
    )
    