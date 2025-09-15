import torch
import os
from src.model import BigramLanguageModel
from src.model import decode

torch.manual_seed(1337)
pwd = os.getcwd()
_CKPT_PATH = "./out/ckpt.pt"

def model_infer(model_path):
    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    #load model
    model = BigramLanguageModel()
    checkpoint = torch.load(os.path.join(pwd, model_path), weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    m = model.to(device)

    context = torch.ones((1, 1), dtype=torch.long, device=device)*0
    token_list = m.generate(context, max_new_tokens=1000)[0].tolist()
    print(decode(token_list))

if __name__=="__main__":
    model_infer(_CKPT_PATH)