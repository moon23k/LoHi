import torch
import torch.nn as nn
from transformers import BigBirdConfig, BigBirdModel
from transformers import LongformerConfig, LongformerModel




class Summarizer(nn.Module):
    def __init__(self, config, encoder):
        super(Summarizer, self).__init__()

        self.encoder = encoder
        self.device = config.device
        self.vocab_size = config.vocab_size


    def forward(self, input_ids, attention_mask, labels):
    	enc_out = self.encoder(input_ids=input_ids,
    						   attention_mask=attention_mask,
    						   labels=labels)
        return



def load_model(config):
	if config.mode == 'train':
	    if config.model == 'bigbird':
	    	encoder = BigBirdModel.from_pretrained(config.mname)
	    else:
	    	encoder = LongformerModel.from_pretrained(config.mname)
	
	    model = Summarizer(config, encoder)

	else:
        assert os.path.exists(config.ckpt)

	    if config.model == 'bigbird':
	    	enc_cfg = BigBirdConfig()
	    	encoder = BigBirdModel(enc_cfg)
	    else:
	    	enc_cfg = LongformerConfig()
	    	encoder = LongformerModel(enc_cfg)		

	    model = Summarizer(config, encoder)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model States has loaded on the Model")



    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params
        
    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    
    return model.to(config.device)

	