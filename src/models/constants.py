import torch.nn as nn
import torch.nn.functional as F

from .basic_blocks import ConvBlock, Res18Block, Res50Block, ResV2Block
from .losses import dice_loss, bce_dice


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

LOSSES = {
    "bce": F.binary_cross_entropy_with_logits,
    "dice": dice_loss,
    "bce+dice": bce_dice,
}
