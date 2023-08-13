import math, time, torch, evaluate
import torch.nn as nn
from module import Generator



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader, test_volumn=100):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.test_volumn = test_volumn
        self.dataloader = test_dataloader
        self.generator = Generator(config)
        self.metric_module = evaluate.load('rouge')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"    


    def test(self):
        self.model.eval()
        greedy_score, beam_score = 0, 0

        with torch.no_grad():
            for batch in self.dataloader:

                greedy_pred = self.generator.generate()
                beam_pred = self.generator.generate()

                greedy_score += self.metric_score(greedy_pred, trg)
                beam_score += self.metric_score(beam_pred, trg)                
        
        greedy_score = round(greedy_score/self.test_volumn, 2)
        beam_score = round(beam_score/self.test_volumn, 2)
        
        return greedy_score, beam_score



    def metric_score(self, pred, label):
        score = self.metric_module.compute(pred, [label])['rouge2']
        return (score * 100)