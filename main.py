import argparse
from torch import optim

from config import get_cfg_defaults
from utils import load_utils
from models import SQVAE


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dbg', '--debug', default=50, help='print loss per n batch')
    parser.add_argument(
        '-dev', '--device', default='cpu', help="device: cpu/cuda/mps")
    args = parser.parse_args()
    return args


def load_config(args):
    cfg = get_cfg_defaults()
    cfg.PRINT_PER_BATCH = args.debug
    cfg.DEVICE = args.device
    return cfg


if __name__ == '__main__':
    args = arg_parse()
    cfg = load_config(args)
    util = load_utils(cfg)

    train_loader, valid_loader, test_loader = util.get_loader()

    model = SQVAE(cfg, util)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    util.train(model, train_loader, valid_loader, optimizer)

