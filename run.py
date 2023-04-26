import os, argparse, torch
from module.model import load_model
from module.data import load_dataloader
from module.test import Tester
from module.train import Trainer
from transformers import set_seed, AutoTokenizer 



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.model_type = args.model        
        self.strategy = args.strategy

        if self.model_type == 'big'
            self.mname = "google/bigbird-roberta-base"
        elif self.model_type == 'long':
            self.mname = "allenai/longformer-base-4096"
        
        #Training args
        self.clip = 1
        self.lr = 5e-4
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4
        self.ckpt = f"ckpt/{self.strategy}_{self.model_type}.pt"
        
        self.early_stop = True
        self.patience = 3

        #Model args
        self.n_heads = 8
        self.n_layers = 6
        self.pff_dim = 2048
        self.bert_dim = 768
        self.hidden_dim = 512
        self.dropout_ratio = 0.1
        self.model_max_length = 1024
        self.act = 'gelu'
        self.norm_first = True
        self.batch_first = True

        if self.mode == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')
        else:
            self.search_method = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")





def inference(config, model, tokenizer):
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
        output_ids = model.generate(input_ids, 
                                    beam_size=4,
                                    do_sample=True, 
                                    max_new_tokens=config.max_len, 
                                    use_cache=True)
        output_seq = tokenizer.decode(output_ids, skip_special_tokens=True)

        #Print Output Sequence
        print(f"Model Out Sequence >> {output_seq}")       




def main(args):
    set_seed(42)
    config = Config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.mname)
    setattr(config, 'pad_id', tokenizer.pad_token_id)
    setattr(config, 'vocab_size', tokenizer.vocab_size)
    model = load_model(config)


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
    parser.add_argument('-strategy', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.strategy in ['fine', 'fuse']
    assert args.model in ['long', 'big']

    if args.task == 'inference':
        assert args.search in ['greedy', 'beam']
    
    main(args)