import torch
import torch.nn as nn
from model.layers import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MultiWorkReport(nn.Module):
    def __init__(self, modelConfig):
        super(MultiWorkReport, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w  -> b 1 (h w c) '),
            nn.Linear(1024, 1024),
        )
        self.report_mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            Swish(),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )

    def forward(self, report, latent_img):
        img = self.to_patch_embedding(latent_img)
        report = self.report_mlp(report)
        work = self.out_mlp(img + report)
        return work