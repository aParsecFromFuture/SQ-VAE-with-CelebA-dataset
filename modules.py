from torch import nn
from torch.nn import functional as F


class MixerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        return self.model(x)


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels))

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.relu(self.model(x) + self.shortcut(x))


class ReverseDoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride, 1),
            nn.BatchNorm2d(out_channels))

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.relu(self.model(x) + self.shortcut(x))

