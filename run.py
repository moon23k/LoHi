import os, json, argparse, torch
from tqdm import tqdm

from module.data import load_dataloader
from module.model import load_model

from module.train import Trainer
from module.test import Tester

from transformers import (set_seed, 
						  BigBirdTokenizerFast,
						  LongformerTokenizerFast)



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.model_name = args.model
        
        if self.model_name == 'bigbird':
        	self.mname = "google/bigbird-roberta-base"        
        else:
	        self.mname = 'allenai/longformer-base-4096'

        self.clip = 1
        self.lr = 5e-5
        self.n_epochs = 10
        self.batch_size = 16
        self.iters_to_accumulate = 4
        
        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.ckpt = f'ckpt/{self.model_name}.pt'


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
	if config.mname == 'bigbird':
		return BigBirdTokenizerFast.from_pretrained(config.mname)

	elif config.mname == 'longformer':
		return LongformerTokenizerFast.from_pretrained(config.mname)
	



def train(config, model, tokenizer):
    train_dataloader = load_dataloader(tokenizer, 'train', config.batch_size)
    valid_dataloader = load_dataloader(tokenizer, 'valid', config.batch_size)
    trainer = Trainer(config, model, train_dataloader, valid_dataloader)
    trainer.train()


def test(config, model, tokenizer):
    test_dataloader = load_dataloader(tokenizer, 'test', config.batch_size)
    tester = Tester(config, model, tokenizer, test_dataloader)
    tester.test()    



def inference(model, tokenizer):
    model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = tokenizer(input_seq)['input_ids']
        output_ids = model.generate(input_ids, beam_size=4, do_sample=True, use_cache=True)
        output_seq = tokenizer.decode(output_ids, skip_special_tokens=True)

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")       



def main(args):
    set_seed(42)

    config = Config(args)    
    model = load_model(config)
    tokenizer = load_tokenizer(config)

    setattr(config, 'pad_id', tokenizer.pad_token_id)


    if config.mode == 'train':
        train(config, model, tokenizer)
    
    elif config.mode == 'test':
        test(config, model, tokenizer)
    
    elif config.mode == 'inference':
        inference(model, tokenizer)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'infernece']
    assert args.model in ['bigbird', 'longformer']

    if args.mode in ['test', 'infernece']:
        assert os.path.exists('ckpt/{args.model}.pt')
    
    main(args)