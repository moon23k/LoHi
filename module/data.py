import json, torch
from torch.utils.data import DataLoader




class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)


    def load_data(self, split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)

        return data


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        orig = self.data[idx]['orig']
        summ = self.data[idx]['summ']
        return orig, summ




def load_dataloader(tokenizer, split, batch_size):

    def collate_fn(batch):
        
        orig_batch, summ_batch = [], []
        
        for orig, summ in batch:
            orig_batch.append(orig)
            summ_batch.append(summ)

        orig_batch = tokenizer(orig_batch, padding=True, truncation=True, return_tensors='pt')
        summ_batch = tokenizer(summ_batch, padding=True, truncation=True, return_tensors='pt')

        return {'input_ids': orig_batch.input_ids,
                'attention_mask': orig_batch.attention_mask,
                'labels': summ_batch.input_ids}


    return DataLoader(Dataset(split), 
                      batch_size=batch_size, 
                      collate_fn=collate_fn,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=True)