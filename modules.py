import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Self_Attn(nn.Module):
    """ Camada de Self-Attention, peça central no SA-GAN """
    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels

        self.query_conv = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out, attention
