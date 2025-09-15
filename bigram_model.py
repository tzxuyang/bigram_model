import os
import argparse
from src.train import post_train, pre_train
from src.utils import get_all_files_in_directory
from src.infer import model_infer

_FILE_PATH = './input.txt'
_PRE_FILE_PATH = "./pretrain_data_prepare/extract/openwebtext"
_OUTPUT_PATH = './out'
_CKPT_PATH_PRE = "./out/ckpt.pt"
_CKPT_PATH_POST = "./out/ckpt_lite.pt"
PWD = os.getcwd()

def argment_parser():
    parser = argparse.ArgumentParser(
        description="CLI for bigram model"
    )
    parser.add_argument(
        "-m",
        "--mode",
        default = 'simple_train',
        choices = ['simple_train', 'train', 'infer_pretrain', 'infer'],
        help = "Run mode, simple train directly train with shakespeare's txt, train run pre-train and finetuning, infer direct generate txt",
    )
    return parser.parse_args()


if __name__=="__main__":
    args = argment_parser()

    if args.mode == "simple_train":
        post_train(os.path.join(PWD, _FILE_PATH), os.path.join(PWD, _OUTPUT_PATH))
    elif args.mode == "train":
        file_list = get_all_files_in_directory(os.path.join(PWD, _PRE_FILE_PATH))
        pre_train(file_list, os.path.join(PWD, _OUTPUT_PATH))
        post_train(os.path.join(PWD, _FILE_PATH), os.path.join(PWD, _OUTPUT_PATH), load_model=True)
    elif args.mode == "infer_pretrain":
        model_infer(_CKPT_PATH_PRE)
    else:
        model_infer(_CKPT_PATH_POST)
        


