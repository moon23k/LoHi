import os, re, yaml, json, nltk
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def load_data():
    data = []
    orig_data = load_dataset('cnn_dailymail', '3.0.0')

    for split in ['train', 'validation', 'test']:
        for elem in orig_data[split]:
            data.append({'article': elem['article'], 
                         'highlights': elem['highlights']})

    return data



def process_data(orig_data):
    volumn_cnt, data_volumn = 0, 53100
    corpus, hier_data = [], []

    src_min_len, src_max_len, trg_max_len = 1000, 2500, 500
    min_sent_num, max_sent_num, max_sent_len = 10, 30, 300
    

    for elem in orig_data:
        src = elem['article'].lower()
        trg = elem['highlights'].lower()

        if src_min_len < len(src) < src_max_len:
            if len(trg) < trg_max_len:
                sents = nltk.tokenize.sent_tokenize(src)

                sent_num_cond = min_sent_num <= len(sents) <= max_sent_num
                sent_len_cond = not sum([len(s) > max_sent_len for s in sents])
                if sent_num_cond and sent_len_cond:             
                    #Remove unnecessary characters in trg sequence
                    trg = re.sub(r'\n', ' ', trg)                 #remove \n
                    trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

                    hier_data.append({'x': sents, 'y': trg})
                    corpus.append(src)
                    corpus.append(trg)

                    volumn_cnt += 1
                    if volumn_cnt == data_volumn:
                        break

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))

    return base_data, hier_data           



def train_tokenizer():
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_config['vocab_size'], 
                         special_tokens=[vocab_config['pad_token'], 
                                         vocab_config['unk_token'],
                                         vocab_config['bos_token'],
                                         vocab_config['eos_token']])

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-3100], data_obj[-3100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)
        assert os.path.exists(f'data/{key}.json')



def main():
    nltk.download('punkt')

    orig_data = load_data()
    processed_data = process_data(orig_data)
    train_tokenizer()
    save_data(processed_data)



if __name__ == '__main__':
    main()