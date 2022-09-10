import logging, time, sys, os
sys.path.append("..")
from argparse import ArgumentParser
from trainer import TrainerModel
from config.config import Configuration
import nltk
nltk.download('punkt')
from utils.utils import seed_everything

seed_everything(42)

if __name__ == '__main__':
    parser = ArgumentParser(description = "Model trainer")
    parser.add_argument("--train_data_path", type = str, default='../data/Reuters/train.json')
    parser.add_argument("--val_data_path", type = str, default='../data/Reuters/val.json')
    parser.add_argument("--test_data_path", type = str, default='../data/Reuters/test.json')
    parser.add_argument("--device", type = str, default='cuda')
    parser.add_argument("--pretrained", type = str, default='roberta-base',
                        help= "Pretrained Transformer Model")
    parser.add_argument("--checkpoint", type = str, default='../ckpt/checkpoint_7.pt')
    parser.add_argument("--tokenizer_name", type = str, default='roberta-base')
    parser.add_argument("--sbert", type = str, default='paraphrase-distilroberta-base-v1')
    parser.add_argument("--mode", type = str, default='train',
                        help= "train/test")
    args = parser.parse_args()
    
    config = Configuration.get_config("../config/config.yml")['train']
    model_trainer = TrainerModel(config, args)
    
    if args.mode == "train":
        model_trainer.train()
    elif args.mode == "test":
        model_trainer.test()