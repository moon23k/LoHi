import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        with open(f"data/{self.split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        orig = self.data[idx]['orig']
        summ = self.data[idx]['summ']
        return orig, summ



def load_dataloader(config, tokenizer, split):
    
    def collate_fn(batch):
        orig_batch, summ_batch = [], []
        for orig, summ in batch:
            orig_batch.append(orig) 
            summ_batch.append(summ)

        orig_encodings = tokenizer(orig_batch, padding=True, truncation=True, return_tensors='pt')
        summ_encodings = tokenizer(summ_batch, padding=True, truncation=True, return_tensors='pt')

        return {'input_ids': orig_encodings.input_ids,
                'attention_mask': orig_encodings.attention_mask,
                'labels': summ_encodings.input_ids}


    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=collate_fn,
                      num_workers=2,
                      pin_memory=True)