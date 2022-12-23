import torch
import logging

from config import Config
from nn import build_model
from tokenizer import BertTokenizer
from utils import evaluate

if __name__ == '__main__':

    # device = torch.device(Config.device)
    device = torch.device('cpu')

    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)

    tokenizer = BertTokenizer.from_pretrained(Config.model_name)

    model = build_model(Config).to(device)
    logging.info(state_dict['model'])
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()
    model.freeze()

    while True:
        s = input('You>')
        if s == 'q':
            break
        print('BOT>', end='')
        text = evaluate(Config, s, tokenizer, model, device, True)
