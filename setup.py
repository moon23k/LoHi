import os, re, json
from datasets import load_dataset
from transformers import PegasusTokenizerFast



def preprocess_data(data_obj, volumn=34000):
    
    min_len, max_len = 2000, 3000
    orig_list, summ_list, volumn_cnt = [], [], 0

    for elem in data_obj:

        if volumn_cnt == 32000:
            break
        
        orig = elem['article'].lower()

        if not (min_len <= len(orig) <= max_len):
            continue
        
        summ = elem['highlights'].lower()
        summ = re.sub(r"\s([.](?:\s|$))", r'\1', summ)  #remove whitespace in front of dot
        summ = re.sub('\n', ' ', summ.strip())          #replace new line with whitespace

        orig_list.append(orig)
        summ_list.append(summ)

        volumn_cnt += 1

    return orig_list, summ_list
    


def train_tokenizer(train_data):
    
    mname = 'google/bigbird-pegasus-large-arxiv'
    old_tokenizer = PegasusTokenizerFast.from_pretrained(mname, model_max_length=128)

    new_tokenizer = old_tokenizer.train_new_from_iterator(train_data, max_vocab_size=30000)
    new_tokenizer.save_pretrained('data/tokenizer')
    
    del old_tokenizer 
    del new_tokenizer



def save_data(orig_data, summ_data):

    tot_data = []
    for orig, summ in zip(orig_data, summ_data):
        tot_data.append({'orig': orig, 'summ': summ})

    train, valid, test = tot_data[:-4000], tot_data[-4000:-1000], tot_data[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



def main():
    data = load_dataset('cnn_dailymail', '3.0.0', split='train')
    orig_data, summ_data = preprocess_data(data)
    train_tokenizer(orig_data + summ_data)
    save_data(orig_data, summ_data)



if __name__ == '__main__':
    main()