import json, itertools, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, bert_tokenizer, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        summ = self.data[idx]['summ']

        text_seg = []
        text = self.bert_tokenizer(text).input_ids
        
        for idx, ids in enumerate(text):
            _len = len(ids)
            if idx % 2:
                temp = [1 for _ in range(_len)]
            else:
                temp = [0 for _ in range(_len)]
            text_seg.extend(temp)

        summ = self.tokenizer.EncodeAsIds(summ)
        
        return {'text': list(itertools.chain(*text)),
                'text_seg': text_seg,
                'summ': summ}



class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        text_batch, text_seg_batch, summ_batch = [], [], []
        
        for elem in batch:
            text_batch.append(torch.LongTensor(elem['text']))
            text_seg_batch.append(torch.LongTensor(elem['text_seg']))
            summ_batch.append(torch.LongTensor(elem['summ']))

        return {'text': self.pad_batch(text_batch),
                'text_seg': self.pad_batch(text_seg_batch),
                'summ': self.pad_batch(summ_batch)}

    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, tokenizer, bert_tokenizer, split):
    return DataLoader(Dataset(tokenizer, bert_tokenizer, split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(config.pad_id), 
                      num_workers=2)
    