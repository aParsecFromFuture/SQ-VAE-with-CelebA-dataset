from torch import optim

from config import get_cfg_defaults
from utils import get_util_defaults
from models import SQVAE


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    util = get_util_defaults()
    train_loader, valid_loader, test_loader = util.get_loader()

    model = SQVAE(cfg, util)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    util.train(model, train_loader, valid_loader, optimizer)

