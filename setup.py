import json, argparse
from datasets import load_dataset
from transformers import AutoTokenizer



#Select and Tokenize Data
def process_data(orig_data, tokenizer):

    processed = []
    cnt, volumn = 0, 34000
    min_len, max_len = 1000, 3000

    
    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue

        processed.append({"input_ids": tokenizer(src).input_ids,
                          'labels': tokenizer(trg).input_ids})
        
        cnt += 1
        if cnt == volumn:
            break

    return processed



def save_data(data_obj, model_type):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-4000], data_obj[-4000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{model_type}_{key}.json', 'w') as f:
            json.dump(val, f)                    



def main(model_type):

    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')

    if model_type != 'big':
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        processed = process_data(orig, tokenizer)        
        save_data(processed)
    elif model_type != 'long':
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        processed = process_data(orig, tokenizer)
        save_data(processed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)    

    args = parser.parse_args()
    assert args.model in ['all', 'big', 'long']        
    main(args.model)