import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .basic_blocks import ConvBlock
from .constants import ACTIVATIONS, BLOCKS
from ..constants import PATCH_SIZE


class UnetModel(BaseModel):
    def create_model(self):
        # This will error if they are not well-defined configuration options
        act_fn = ACTIVATIONS[self.config["activation"]]
        block = BLOCKS[self.config["block"]]
        self.model = Unet(act_fn=act_fn, block=block, patch_size=PATCH_SIZE if self.config["predict_patches"] else 1)

    def step(self, input_BCHW):
        return self.model(input_BCHW).squeeze(1)

    def loss(self, pred_BHW, target_BHW):
        # Do not use sigmoid in the model, because it is more numerically stable to use BCE with
        # logits, which combines the sigmoid and the BCE loss in a single function.
        return F.binary_cross_entropy_with_logits(pred_BHW, target_BHW)

    def predict(self, input_BCHW):
        return F.sigmoid(self.model(input_BCHW).squeeze(1))


class Unet(nn.Module):
    def __init__(
        self,
        channels: list[int]=[3, 64, 128, 256, 512, 1024],
        act_fn: nn.Module=nn.ReLU,
        block: nn.Module=ConvBlock,
        patch_size: int=1,
    ):
        super().__init__()
        enc_channels = channels[1:]
        dec_channels = channels[::-1][:-1]

        # Use a normal convolution for the first layer, because we want to keep the number of channels
        # a power of 2
        self.enc_blocks = nn.ModuleList([ConvBlock(channels[0], channels[1], act_fn)] + [block(in_c, out_c, act_fn) for in_c, out_c in zip(enc_channels[:-1], enc_channels[1:])])
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2) for in_c, out_c in zip(dec_channels[:-1], dec_channels[1:])])
        self.dec_blocks = nn.ModuleList([block(in_c, out_c, act_fn) for in_c, out_c in zip(dec_channels[:-1], dec_channels[1:])])

        # The final convolution has a patch size and stride of `patch_size`, such that we have a
        # single prediction for each patch of size `patch_size` x `patch_size`.
        self.final_conv = nn.Conv2d(dec_channels[-1], 1, kernel_size=patch_size, stride=patch_size, padding=0)

        self.apply(self.init_weights)

    def forward(self, x):
        # Encoding
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.enc_blocks[-1](x)

        # Decoding
        for block, up_conv, feature in zip(self.dec_blocks, self.up_convs, enc_features[::-1]):
            x = up_conv(x)
            x = torch.cat([x, feature], dim=1)
            x = block(x)

        return self.final_conv(x)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)


if __name__ == "__main__":
    model = Unet()
    x = torch.randn((20, 3, 400, 400))
    y: torch.Tensor = model(x)

    print("We want to make sure to initialize the model s.t. it preserves variance of the input.")
    print(f"Input std: {x.std().item():.3f}")
    print(f"Output std: {y.std().item():.3f}")
    print("Shape:", y.shape)
