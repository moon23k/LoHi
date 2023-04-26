import torch, math, time
import torch.nn as nn
import evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.metric_module = evaluate.load('rouge')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"    


    def test(self):
        self.model.eval()
        tot_len, greedy_score, beam_score = 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader):

                greedy_pred = self.search.greedy_search()
                beam_pred = self.search.beam_search()

                greedy_score += self.metric_score(greedy_pred, trg)
                beam_score += self.metric_score(beam_pred, trg)                
        
        greedy_score = round(greedy_score/tot_len, 2)
        beam_score = round(beam_score/tot_len, 2)
        
        return greedy_score, beam_score



    def metric_score(self, pred, label):

        pred = self.tokenizer.batch_decode(pred)
        label = self.tokenizer.batch_decode(label.tolist())

        #For Translation and Summarization Tasks
        score = self.metric_module.compute(pred, label)['rouge2']

        return (score * 100)