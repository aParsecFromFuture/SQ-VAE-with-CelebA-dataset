import argparse
from torch import optim

from config import get_cfg_defaults
from utils import load_utils
from models import SQVAE


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cbg', '--config_file', default='', help='config file path')
    args = parser.parse_args()
    return args


def load_config(args):
    cfg = get_cfg_defaults()

    if args.config_file != '':
        cfg = cfg.merge_from_file(args.config_file)

    return cfg


if __name__ == '__main__':
    args = arg_parse()
    cfg = load_config(args)
    util = load_utils(cfg)

    train_loader, valid_loader, test_loader = util.get_loader()

    model = SQVAE(cfg, util)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    util.train(model, train_loader, valid_loader, optimizer)

