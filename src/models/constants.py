import torch.nn as nn

from .basic_blocks import ConvBlock, Res18Block, Res50Block, ResV2Block


BLOCKS = {
    "conv": ConvBlock,
    "res18": Res18Block,
    "res50": Res50Block,
    "resv2": ResV2Block,
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}
