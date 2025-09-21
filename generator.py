import torch.nn as nn
from torch.nn.utils import spectral_norm


from modules import Self_Attn

class ResBlockGenerator(nn.Module):
    """ Bloco Residual específico para o Gerador (Upsampling). """
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class Generator(nn.Module):
    """ Gerador baseado em ResNet para SAGAN. """
    def __init__(self, z_dim=128, g_channels=64, image_size=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = spectral_norm(nn.Linear(self.z_dim, 4 * 4 * g_channels * 16))
        self.res1 = ResBlockGenerator(g_channels * 16, g_channels * 8)
        self.res2 = ResBlockGenerator(g_channels * 8, g_channels * 4)
        self.res3 = ResBlockGenerator(g_channels * 4, g_channels * 2)
        self.attn = Self_Attn(g_channels * 2)
        self.res4 = ResBlockGenerator(g_channels * 2, g_channels)
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(g_channels, 3, kernel_size=3, padding=1)),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.dense(z).view(-1, 1024, 4, 4)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x, _ = self.attn(x)
        x = self.res4(x)
        return self.final(x)
