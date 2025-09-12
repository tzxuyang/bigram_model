import torch
from model import BigramLanguageModel

torch.manual_seed(1337)
_CKPT_PATH = "./out/ckpt.pt"
file_name = "input.txt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.mps.is_available() else 'cpu'

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


if __name__=="__main__":
    model = BigramLanguageModel()
    print(device)
    checkpoint = torch.load(_CKPT_PATH, weights_only=False)

    # Load the state_dict into your model
    # The key 'model_state_dict' is a common convention, but it might be different
    # depending on how the checkpoint was saved (e.g., 'model', 'state_dict').
    model.load_state_dict(checkpoint['model'])
    # If the checkpoint only saved the state_dict directly, you can do:
    # model.load_state_dict(torch.load(checkpoint_path))
    # print(model)
    model.eval()
    m = model.to(device)

    context = torch.ones((1, 1), dtype=torch.long, device=device)
    token_list = m.generate(context, max_new_tokens=1000)[0].tolist()
    print(decode(token_list))
    # logging.info(decode(m.generate(context, max_new_tokens=500)[0].tolist()))