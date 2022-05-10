import numpy as np
import torch
from torch import nn
from torch.nn import init


class Pixelattention(nn.Module):

    def __init__(self, channel ):
        super().__init__()
        self.conv=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=1,stride=1,bias=False)

    def forward(self, x):
        x1=self.conv(x)
        x2=nn.functional.sigmoid(x1)
        return x*x2

        



    