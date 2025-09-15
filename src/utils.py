import torch
from src.model import encode
import logging
import json
import os

torch.manual_seed(1337)

# data loading
def get_batch(split, train_data, eval_data, block_size, batch_size, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    # logging.info(f"split is {split}")
    data = train_data if split == 'train' else eval_data
    # logging.info(f"data size is {len(data)})")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  # leverage Cuda if available
    return x, y

# estimate loss at each step
@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, eval_data, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, eval_data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# read data from file
def load_data(file_path):
    """
    Shakespeare file can be accessed from wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    Other files all in https://huggingface.co/datasets/Skylion007/openwebtext/tree/main
    """
    logging.info("Reading file")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    logging.info("Successfully read file")

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def load_train_config(path):
    with open(path, 'r') as f:
        train_config = json.load(f)
    batch_size = train_config['batch_size'] # how many independent sequences will we process in parallel?
    block_size = train_config['block_size'] # what is the maximum context length for predictions?
    max_iters = train_config['max_iters']
    eval_interval = train_config['eval_interval']
    learning_rate = train_config['learning_rate']
    eval_iters = train_config['eval_iters']
    num_files = train_config['num_file']
    epochs = train_config['epochs']
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters, device, num_files, epochs

def get_all_files_in_directory(directory_path):
    """
    Retrieves a list of all file paths within a given directory,
    including files in subdirectories.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list: A list of absolute paths to all files found.
    """
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


