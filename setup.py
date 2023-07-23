import os, re, yaml, json
from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordPieceTrainer
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
    corpus, processed = [], []
    max_num, min_len, max_len = 35, 500, 3000


    for elem in orig_data:
        src = elem['article'].lower()
        trg = elem['highlights'].lower()

        if min_len < len(src) < max_len:
            if len(trg) < min_len:
                sents = nltk.tokenize.sent_tokenize(src)
                len_chk = not sum([len(s) > min_len for s in sents])

                if len(sents) < max_num and len_chk:
                    #Remove unnecessary characters in trg sequence
                    trg = re.sub(r'\n', ' ', trg)                 #remove \n
                    trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

                    processed.append({'src': sents, 'trg': trg})
                    corpus.append(src)
                    corpus.append(trg)


    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))
    
    return processed           



def train_tokenizer():
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(vocab_size=vocab_config['vocab_size'], 
                               special_tokens=[vocab_config['pad_token'], 
                                               vocab_config['unk_token'],
                                               vocab_config['bos_token'],
                                               vocab_config['eos_token']])

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save("data/tokenizer.json")



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-1100], data_obj[-1100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)
        assert os.path.exists(f'data//{key}.json')



def main():
    nltk.download('punkt')
    orig_data = load_data()
    processed = process_data(orig_data)
    train_tokenizer()
    save_data(processed)



if __name__ == '__main__':
    main()