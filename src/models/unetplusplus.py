import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .basic_blocks import ConvBlock
from .constants import ACTIVATIONS, BLOCKS
from ..constants import PATCH_SIZE


class UnetPlusPlusModel(BaseModel):
    def create_model(self):
        # This will error if they are not well-defined configuration options
        act_fn = ACTIVATIONS[self.config["activation"]]
        block = BLOCKS[self.config["block"]]
        self.model = UnetPlusPlus(act_fn=act_fn, block=block, patch_size=PATCH_SIZE if self.config["predict_patches"] else 1)

    def step(self, input_BCHW):
        return self.model(input_BCHW).squeeze(1)

    def loss(self, pred_BHW, target_BHW):
        return F.binary_cross_entropy_with_logits(pred_BHW, target_BHW)

    def predict(self, input_BCHW):
        return F.sigmoid(self.model(input_BCHW).squeeze(1))


class UnetPlusPlus(nn.Module):
    def __init__(
        self,
        channels: list[int]=[3, 64, 128, 256, 512, 1024],
        act_fn: nn.Module=nn.ReLU,
        block: nn.Module=ConvBlock,
        patch_size: int=1,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
