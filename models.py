import torch
from torch.distributions import Categorical

from modules import *


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            MixerBlock(in_channels, 32),
            DoubleConv2d(32, 32, 2),
            MixerBlock(32, 32),
            DoubleConv2d(32, 64, 2))

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ReverseDoubleConv2d(64, 32, 2),
            MixerBlock(32, 32),
            ReverseDoubleConv2d(32, 32, 2),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class GaussianVQ(nn.Module):
    def __init__(self, cfg, utils):
        super().__init__()

        self.cfg = cfg
        self.utils = utils

        self.num_embeddings = cfg.NUM_EMBEDDINGS
        self.embedding_dim = cfg.EMBEDDING_DIM
        self.temperature = cfg.TEMPERATURE.INIT
        self.log_var_q = torch.tensor(cfg.LOG_VAR_Q)
        self.log_var_q_scalar = nn.Parameter(torch.tensor(cfg.LOG_VAR_Q_SCALAR))
        self.codebook = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim))

    def forward(self, x):
        bs, dim_z, width, height = x.shape
        x_flatten = x.permute(0, 2, 3, 1).contiguous()
        var_q = self.log_var_q.exp() + self.log_var_q_scalar.exp()
        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self.utils.calc_distances(x_flatten, self.codebook, 0.5 * precision_q)
        probs = torch.softmax(logit, dim=-1)
        log_probs = torch.log_softmax(logit, dim=-1)
        encodings = self.utils.gumbel_softmax(logit, self.temperature)

        quantized = torch.matmul(encodings, self.codebook).view(bs, width, height, dim_z)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        kld_discrete = torch.sum(probs * log_probs, dim=(0, 1)) / bs
        kld_continuous = self.utils.weighted_mse(x, quantized, 0.5 * precision_q).mean()

        return quantized, (kld_discrete + kld_continuous)

    @torch.no_grad()
    def quantize(self, x):
        bs, dim_z, width, height = x.shape
        x_flatten = x.permute(0, 2, 3, 1).contiguous()
        var_q = self.log_var_q.exp() + self.log_var_q_scalar.exp()
        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self.utils.calc_distances(x_flatten, self.codebook, 0.5 * precision_q)
        probs = torch.softmax(logit, dim=-1)

        dist = Categorical(probs)
        indices = dist.sample().view(bs, width, height)
        encodings = F.one_hot(indices, num_classes=self.num_embeddings)

        quantized = torch.matmul(encodings, self.codebook).view(bs, width, height, dim_z)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized


class SQVAE(nn.Module):
    def __init__(self, cfg, utils):
        super().__init__()

        self.cfg = cfg
        self.utils = utils

        self.encoder = Encoder(cfg.TRAIN.NUM_CHANNELS)
        self.quantizer = GaussianVQ(cfg.QUANTIZER, utils)
        self.decoder = Decoder(cfg.TRAIN.NUM_CHANNELS)

    def forward(self, x):
        zhat_q = self.encoder(x)
        z_q, loss_latent = self.quantizer(zhat_q)
        xhat = self.decoder(z_q)

        mse = F.mse_loss(xhat, x, reduction='sum') / x.shape[0]
        loss_recon = self.cfg.TRAIN.X_DIM * torch.log(mse) / 2

        return xhat, (loss_latent + loss_recon)
