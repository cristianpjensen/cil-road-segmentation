"""
Most simple model possible, a single linear layer. This model serves as a template for new models
and as a test model for the training pipeline, since it is very fast (even on the CPU). When creating
a new model, make sure to update `create_model()` in `models/create_model.py` to return the new
model.
"""

from .base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetModel(BaseModel):
    def create_model(self):
        self.model = Unet()

    def step(self, input_BCHW):
        return self.model(input_BCHW).squeeze(1)

    def loss(self, pred_BHW, target_BHW):
        # Do not use sigmoid in the model, because it is more numerically stable to use BCE with
        # logits, which combines the sigmoid and the BCE loss in a single function.
        return F.binary_cross_entropy_with_logits(pred_BHW, target_BHW, pos_weight=self.config["pos_weight"])

    def predict(self, input_BCHW):
        return F.sigmoid(self.model(input_BCHW).squeeze(1))


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=[64, 128, 256, 512]):
        super().__init__()
        enc_channels = channels
        dec_channels = channels[::-1]

        self.initial_conv = nn.Conv2d(in_channels, enc_channels[0], kernel_size=3, padding=1)
        self.enc_blocks = nn.ModuleList([Block(in_channels, out_channels) for in_channels, out_channels in zip(enc_channels[:-1], enc_channels[1:])])
        self.bottleneck = Block(enc_channels[-1], dec_channels[0])
        self.dec_blocks = nn.ModuleList([Block(in_channels * 2, out_channels) for in_channels, out_channels in zip(dec_channels[:-1], dec_channels[1:])])
        self.final_conv = nn.Conv2d(dec_channels[-1], out_channels, kernel_size=1)

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.initial_conv(x)
        enc_features = []
        maxpool_indices = []
        for block in self.enc_blocks:
            x = block(x)
            enc_features.append(x)
            x, indices = F.max_pool2d(x, 2, return_indices=True)
            maxpool_indices.append(indices)

        x = self.bottleneck(x)

        for block in self.dec_blocks:
            x = F.max_unpool2d(x, maxpool_indices.pop(), 2)
            x = torch.cat([x, enc_features.pop()], dim=1)
            x = block(x)

        return self.final_conv(x)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.01)


class Block(nn.Module):
    """
    Input: [B, C_in, H, W]
    Output: [B, C_out, H, W]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


if __name__ == "__main__":
    model = Unet()
    x = torch.randn((1, 3, 400, 400))
    y: torch.Tensor = model(x)

    print("We want to make sure to initialize the model s.t. it preserves variance of the input.")
    print(f"Input std: {x.std().item():.3f}")
    print(f"Output std: {y.std().item():.3f}")
    print("Shape:", y.shape)
