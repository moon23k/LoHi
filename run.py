import numpy as np
import os, yaml, random, argparse

import torch
import torch.backends.cudnn as cudnn

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module.test import Tester
from module.train import Trainer
from module.search import Search
from module.model import load_model
from module.data import load_dataloader



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.model_type = args.encoder
        self.ckpt = f"ckpt/{self.model_type}.pt"
        self.tokenizer_path = 'data/tokenizer.json'

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'
        
        if self.mode == 'inference':
            self.search_method = search
            self.device = torch.device('cpu')
        else:
            self.search = None
            self.device = torch.device(self.device_type)

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer



def inference(config, model, tokenizer):
    print('Type "quit" to terminate Summarization')
    
    while True:
        user_input = input('Please Type Text >> ')
        if user_input.lower() == 'quit':
            print('--- Terminate the Summarization ---')
            print('-' * 30)
            break

        src = config.src_tokenizer.Encode(user_input)
        src = torch.LongTensor(src).unsqueeze(0).to(config.device)

        if config.search == 'beam':
            pred_seq = config.search.beam_search(src)
        elif config.search == 'greedy':
            pred_seq = config.search.greedy_search(src)

        print(f" Original  Sequence: {user_input}")
        print(f'Summarized Sequence: {tokenizer.Decode(pred_seq)}\n')



def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer()


    if config.mode == 'train': 
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
        return

    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
        return
    
    elif config.mode == 'inference':
        inference(config, model, tokenizer)
        return
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-encoder', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.encoder in ['base', 'hier']

    if args.task == 'inference':
        import nltk
        nltk.download('punkt')
        assert args.search in ['greedy', 'beam']

    main(args)