from tqdm import tqdm
import sentencepiece as spm
import re, os, json, nltk, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          BartForConditionalGeneration)



#Select and Tokenize Data
def process_data(orig_data, model, tokenizer):

    processed, corpus = [], []
    device = model.device
    cnt, volumn = 0, 12000
    min_len, max_len = 1000, 3000

    
    for elem in orig_data:
        text, summ = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(text) < max_len):
            continue

        #remove unnecessary characters in trg sequence
        summ = re.sub(r'\n', ' ', summ)                 #remove \n
        summ = re.sub(r"\s([.](?:\s|$))", r'\1', summ)  #remove whitespace in front of dot

        input_ids = tokenizer(summ, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            summ = model.generate(input_ids=input_ids, max_length=512, use_cache=True)
        summ = tokenizer.batch_decode(summ, skip_special_tokens=True)[0]

        text = nltk.tokenize.sent_tokenize(text)
        
        processed.append({"text": text, 'summ': summ})
        corpus.append(text)
        corpus.append(summ)
        
        cnt += 1
        if cnt == volumn:
            break


    with open('data/corpus.txt', 'w') as f:
        json.dump(corpus, f)

    return processed



def build_vocab():
    assert os.path.exists(f'data/corpus.txt')
    opt = f"--input=data/corpus.txt \
            --model_prefix=data/tokenizer \
            --vocab_size=30000 \
            --character_coverage=1 \
            --model_type=bpe \
            --pad_id=0 --pad_piece=[PAD] \
            --unk_id=1 --unk_piece=[UNK] \
            --bos_id=2 --bos_piece=[BOS] \
            --eos_id=3 --eos_piece=[EOS]".replace(' '*12, '')

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/corpus.txt')



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)                    



def main():
    mname = 'circulus/kobart-trans-en-ko-v2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mname, model_max_length=512)

    processed = process_data(orig, model, tokenizer)    
    save_data(processed)
    build_vocab()



if __name__ == '__main__':
    nltk.download('punkt')    
    main()