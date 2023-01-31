import os, re, json
from datasets import load_dataset



def filter_data(orig_data, volumn=36000):
    min_len=500 
    max_len=2000

    volumn_cnt = 0
    processed = []

    for elem in orig_data:
        orig, summ = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(orig) < max_len):
            continue
        if len(summ) > min_len:
            continue

        summ = re.sub(r'\n', ' ', summ.strip())         #remove \n
        summ = re.sub(r"\s([.](?:\s|$))", r'\1', summ)  #remove whitespace in front of dot
        
        processed.append({'orig': orig, 'summ': summ})

        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    return processed
    


def main():
    #Load and filter dataset
    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    filtered = filter_data(orig)


    #Split and save dataset
    train, valid, test = filtered[:-6000], filtered[-6000:-3000], filtered[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')



if __name__ == '__main__':
    main()