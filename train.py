import torch
import os
from pytictoc import TicToc
import pickle
from model import BigramLanguageModel
import logging
from pretrain_data_prepare.prepare import get_all_files_in_directory

_FILE_PATH = "./pretrain_data_prepare/extract/openwebtext"
_NUM_FILE = 3
file_name = "input.txt"
# logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# hyperparameters
out_dir = 'out'
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200
epochs = 2

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

logging.info("Successfully read file")

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  # leverage Cuda if available
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# start training
logging.info(f"training started, device = {device}")

file_list = get_all_files_in_directory(_FILE_PATH)

for i in range(epochs):
    for file_name in file_list[:_NUM_FILE]:

        logging.info("Reading file")
        with open(file_name, 'r', encoding='utf-8') as f:
            text = f.read()

        logging.info("Successfully read file")
        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]

        t = TicToc() #create instance of class
        t.tic()
        running_mfu = -1.0
        for iter in range(max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0:
                losses = estimate_loss()
                logging.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model,
                    'iter_num': iter,
                    'loss': losses['train'],
                    # 'config': config,
                }
                logging.info(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

            # sample a batch of data
            xb, yb = get_batch('train')

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # generate from the model
    logging.info(f"epoch {i} completed successfully, with loss {losses['val']:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
logging.info(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
t.toc()
logging.info("training completed successfully")