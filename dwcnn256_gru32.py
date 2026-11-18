# python
from math import ceil
# pytorch
import torch
from torch.nn import ModuleDict, Module, Sequential, Conv2d, GroupNorm, ReLU, Flatten, GRU, Linear
# 共有
from .abstract import AbstractRecurrentEncoder
def models(width, height):
    class Encoder(AbstractRecurrentEncoder, Module):
        image_features = 256*ceil(height/32)*ceil(width/32)
        hidden_features = 32
        state_features = image_features + hidden_features
        def __init__(self):
            super().__init__()
            self.vision = Sequential(
                # 3->16
                Conv2d(3, 16, 3, 2, 1, groups=1, bias=False),
                GroupNorm(4, 16),
                ReLU(),
                # 16->32
                Conv2d(16, 16, 3, 2, 1, groups=16, bias=False),
                GroupNorm(4, 16),
                ReLU(),
                Conv2d(16, 32, 1, 1, 0, groups=1, bias=False),
                GroupNorm(8, 32),
                ReLU(),
                # 32->64
                Conv2d(32, 32, 3, 2, 1, groups=32, bias=False),
                GroupNorm(8, 32),
                ReLU(),
                Conv2d(32, 64, 1, 1, 0, groups=1, bias=False),
                GroupNorm(16, 64),
                ReLU(),
                # 64->128
                Conv2d(64, 64, 3, 2, 1, groups=64, bias=False),
                GroupNorm(16, 64),
                ReLU(),
                Conv2d(64, 128, 1, 1, 0, groups=1, bias=False),
                GroupNorm(32, 128),
                ReLU(),
                # 128->256
                Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                GroupNorm(32, 128),
                ReLU(),
                Conv2d(128, 256, 1, 1, 0, groups=1, bias=False),
                GroupNorm(64, 256),
                Flatten()
            )
            self.recurrent = GRU(self.image_features, self.hidden_features, batch_first=True)
    encoder = Encoder()
    actor = Linear(encoder.state_features, 7)
    critic = Linear(encoder.state_features, 1)
    return ModuleDict(dict(
        encoder=encoder,
        actor=actor,
        critic=critic
    ))
