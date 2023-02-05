import torch, time, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.max_len = config.max_len
        self.dataloader = test_dataloader


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        metric_module = evaluate.load('rouge1')
        
        start_time = time.time()
        with torch.no_grad():
            for _, batch in enumerate(self.dataloader):   
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                                
                preds = self.model.generate(input_ids, 
                                            max_new_tokens=self.max_len, 
                                            use_cache=True)
                
                preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                metric_module.add_batch(predictions=preds, 
                                        references=[[l] for l in labels])    

        metric_score = metric_module.compute()['rouge1'] * 100

        print('Test Results')
        print(f"  >> ROUGE Score: {metric_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")
    
