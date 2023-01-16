import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from config import get_cfg_defaults

from torchvision.utils import make_grid


class Utils:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if not os.path.exists(cfg.CHECKPOINT_PATH):
            os.makedirs(cfg.CHECKPOINT_PATH)

        if not os.path.exists(cfg.SAMPLE_PATH):
            os.makedirs(cfg.SAMPLE_PATH)

    def get_loader(self):
        data_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(0.5)])

        train_set = ImageFolder(self.cfg.DATASET.TRAIN_PATH, transform=data_aug)
        train_loader = DataLoader(train_set, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True)

        valid_set = ImageFolder(self.cfg.DATASET.VALID_PATH, transform=data_aug)
        valid_loader = DataLoader(valid_set, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=False)

        test_set = ImageFolder(self.cfg.DATASET.TEST_PATH, transform=data_aug)
        test_loader = DataLoader(test_set, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=False)

        return train_loader, valid_loader, test_loader

    def calc_temperature(self, epoch, step, num_batches):
        coef = (epoch - 1) * num_batches + step + 1
        temperature_now = np.max([self.cfg.QUANTIZER.TEMPERATURE.INIT * np.exp(-self.cfg.QUANTIZER.TEMPERATURE.DECAY * coef), self.cfg.QUANTIZER.TEMPERATURE.MIN])

        return temperature_now

    @staticmethod
    def save_images(images, filename):
        plt.figure(figsize=(8, 8), frameon=False)
        images = images.numpy()
        fig = plt.imshow(np.transpose(images, (1, 2, 0)), interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(filename)

    @staticmethod
    def gumbel_softmax(logit, temperature):
        u = torch.rand(logit.shape)
        g = -torch.log(-torch.log(u))
        y = logit + g

        return F.softmax(y / temperature, dim=-1)

    @staticmethod
    def calc_distances(x, codebook, weight):
        x = x.view(-1, codebook.shape[1])
        distances = (torch.sum(x.pow(2), dim=1, keepdim=True)
                     + torch.sum(codebook.pow(2), dim=1)
                     - 2 * torch.matmul(x, codebook.t()))

        return weight * distances

    @staticmethod
    def weighted_mse(x1, x2, weight):
        return torch.sum((x1 - x2).pow(2) * weight, dim=(1, 2, 3))

    def train(self, model, train_loader, valid_loader, optimizer):
        for epoch in range(1, self.cfg.TRAIN.NUM_EPOCHS + 1):
            train_loss, valid_loss = 0.0, 0.0

            model.train()
            for step, (x, _) in enumerate(train_loader):
                model.quantizer.temperature = self.calc_temperature(epoch, step, len(train_loader))
                xhat, loss = model(x)
                train_loss += loss.item() / len(train_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 50 == 0:
                    self.save_images(make_grid(torch.cat([x[:32], xhat[:32]])),
                                     os.path.join(self.cfg.SAMPLE_PATH, f'sample({step}).png'))

            model.eval()
            with torch.no_grad():
                for step, (x, _) in enumerate(valid_loader):
                    xhat, loss = model(x)
                    valid_loss += loss.item() / len(valid_loader)

            torch.save(model.state_dict(),
                       os.path.join(self.cfg.CHECKPONT_PATH, f'checkpoint({epoch}).pt'))
            print(
                f'{epoch:2d}: '
                f'train_loss: {train_loss:.2f}, '
                f'valid_loss: {valid_loss:.2f}, '
                f'temperature: {model.quantizer.temperature:.2f}')


def get_util_defaults():
    return Utils(get_cfg_defaults())
