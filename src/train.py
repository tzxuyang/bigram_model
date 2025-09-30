import sys

sys.path.append("../")
sys.path.append("./")
import torch
import os
import time
from src.model import BigramLanguageModel
from src.model import decode
import logging
from src.utils import (
    get_batch,
    estimate_loss,
    load_data,
    load_train_config,
    get_all_files_in_directory,
)

torch.manual_seed(1337)

# logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# hyperparameters
out_dir = "../out"
file_name = "../input.txt"
_CKPT_PATH = "./out/ckpt.pt"
_FILE_PATH = "./pretrain_data_prepare/extract/openwebtext"
pwd = os.getcwd()


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def post_train(file_name, out_dir, load_model=False):
    # load training hyperparameters
    (
        batch_size,
        block_size,
        max_iters,
        eval_interval,
        learning_rate,
        eval_iters,
        device,
        num_file,
        epochs,
    ) = load_train_config(os.path.join(pwd, "./config/train.json"))

    # load data
    train_data, val_data = load_data(file_name)

    # load model
    model = BigramLanguageModel()
    if load_model:
        checkpoint = torch.load(os.path.join(pwd, _CKPT_PATH), weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train()
    m = model.to(device)
    early_stopping = EarlyStopping(patience=1, delta=0.005)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # start training
    logging.info(f"training started, device = {device}")

    start_time = time.time()
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(
                model, eval_iters, train_data, val_data, block_size, batch_size, device
            )
            logging.info(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            early_stopping(losses["val"], model)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model": model,
                "iter_num": iter,
                "loss_train": losses["train"],
                "loss_val": losses["val"],
            }
            logging.info(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt_lite.pt"))

        # sample a batch of data
        xb, yb = get_batch(
            "train", train_data, val_data, block_size, batch_size, device
        )
        # xb, yb = get_batch('val')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    early_stopping.load_best_model(model)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    logging.info(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    end_time = time.time()
    duration_min = (end_time - start_time) / 60.0
    logging.info(f"training completed successfully in {duration_min} min")


# file_list = get_all_files_in_directory(_FILE_PATH)
def pre_train(file_list, out_dir):
    # load training hyperparameters
    (
        batch_size,
        block_size,
        max_iters,
        eval_interval,
        learning_rate,
        eval_iters,
        device,
        num_file,
        epochs,
    ) = load_train_config(os.path.join(pwd, "./config/train.json"))

    # load model
    model = BigramLanguageModel()
    m = model.to(device)

    logging.info(model)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # start training
    logging.info(f"training started, device = {device}")

    start_time = time.time()
    for i in range(epochs):
        for file_name in file_list[:num_file]:
            logging.info(file_name)
            early_stopping = EarlyStopping(patience=1, delta=0.005)
            train_data, val_data = load_data(file_name)

            for iter in range(max_iters + 1):

                # every once in a while evaluate the loss on train and val sets
                if iter % eval_interval == 0:
                    losses = estimate_loss(
                        model,
                        eval_iters,
                        train_data,
                        val_data,
                        block_size,
                        batch_size,
                        device,
                    )
                    logging.info(
                        f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )
                    early_stopping(losses["val"], model)
                    if early_stopping.early_stop:
                        logging.info("Early stopping")
                        break

                # sample a batch of data
                xb, yb = get_batch(
                    "train", train_data, val_data, block_size, batch_size, device
                )

                # evaluate the loss
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            early_stopping.load_best_model(model)

        # record checkpoint end of each epoch
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": model,
            "iter_num": iter,
            "loss_train": losses["train"],
            "loss_val": losses["val"],
        }
        logging.info(
            f"------------------saving checkpoint to {out_dir}------------------------"
        )
        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        logging.info(f"epoch {i} completed successfully, with loss {losses['val']:.4f}")

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    logging.info(
        "-------------------------sample text generation----------------------------"
    )
    logging.info(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
    end_time = time.time()
    duration_min = (end_time - start_time) / 60.0
    logging.info(f"training completed successfully in {duration_min} min")


if __name__ == "__main__":
    # post_train(file_name, out_dir)
    file_list = get_all_files_in_directory(_FILE_PATH)
    print(file_list)
    pre_train(file_list, out_dir)
