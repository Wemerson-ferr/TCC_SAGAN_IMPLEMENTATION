import torch.nn as nn
from torch.nn.utils import spectral_norm

from modules import Self_Attn

class ResBlockDiscriminator(nn.Module):
    """ Bloco Residual específico para o Discriminador (Downsampling). """
    def __init__(self, in_channels, out_channels):
        super(ResBlockDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.AvgPool2d(2)
        )
        self.skip = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class Discriminator(nn.Module):
    """ Discriminador baseado em ResNet para SAGAN. """
    def __init__(self, d_channels=64, image_size=128):
        super(Discriminator, self).__init__()

        self.initial_block = nn.Sequential(
            spectral_norm(nn.Conv2d(3, d_channels, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(d_channels, d_channels, kernel_size=3, padding=1)),
            nn.AvgPool2d(2)
        )
        self.res1 = ResBlockDiscriminator(d_channels, d_channels * 2)
        self.attn = Self_Attn(d_channels * 2)
        self.res2 = ResBlockDiscriminator(d_channels * 2, d_channels * 4)
        self.res3 = ResBlockDiscriminator(d_channels * 4, d_channels * 8)
        self.res4 = ResBlockDiscriminator(d_channels * 8, d_channels * 16)
        self.final_block = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(d_channels * 16, d_channels * 16, kernel_size=3, padding=1))
        )
        self.dense = spectral_norm(nn.Linear(d_channels * 16 * 4 * 4, 1))

    def forward(self, x):
        x0 = self.initial_block(x)
        x1 = self.res1(x0)
        x2, _ = self.attn(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.res4(x4)
        x6 = self.final_block(x5)
        out = x6.view(x6.size(0), -1)
        return self.dense(out)
