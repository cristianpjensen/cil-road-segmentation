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
        self.model = UnetPlusPlus(
            act_fn=act_fn,
            block=block,
            patch_size=PATCH_SIZE if self.config["predict_patches"] else 1,
            deep_supervision=self.config["deep_supervision"],
        )

    def step(self, input_BCHW):
        return self.model(input_BCHW).squeeze(2).mean(1)

    def loss(self, pred_BHW, target_BHW):
        return F.binary_cross_entropy_with_logits(pred_BHW, target_BHW)

    def predict(self, input_BCHW):
        return F.sigmoid(self.model(input_BCHW).squeeze(2).mean(1))


class UnetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=1,
        channels: list[int]=[64, 128, 256, 512, 1024],
        act_fn: nn.Module=nn.ReLU,
        block: nn.Module=ConvBlock,
        patch_size: int=1,
        deep_supervision: bool=False,
    ):
        super().__init__()

        self.channels = channels
        self.N = len(channels)

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.convs = []
        for i in range(self.N):
            convs_i = []
            for j in range(self.N - i):
                # (*, 0) has upsampling connection and no skip-connection
                if j == 0:
                    # (0, 0) has input image as input
                    if i == 0:
                        c_in = in_channels
                    else:
                        c_in = channels[i-1]
                else:
                    c_in = j * channels[i] + channels[i+1]

                convs_i.append(block(c_in, channels[i], act_fn))

            self.convs.append(nn.ModuleList(convs_i))

        self.convs = nn.ModuleList(self.convs)

        if deep_supervision:
            self.final_convs = nn.ModuleList([
                nn.Conv2d(channels[0], out_channels, kernel_size=patch_size, stride=patch_size, padding=0) for _ in range(self.N - 1)
            ])
        else:
            self.final_convs = nn.ModuleList([
                nn.Conv2d(channels[0], out_channels, kernel_size=patch_size, stride=patch_size, padding=0)
            ])

        self.apply(self.init_weights)

    def forward(self, x):
        xs = [[None for _ in range(self.N - i)] for i in range(self.N)]

        # First the backbone (because we only need the downsample here)
        xs[0][0] = self.convs[0][0](x)
        for i in range(1, self.N):
            xs[i][0] = self.convs[i][0](self.down(xs[i-1][0]))

        # Then the all the intermediary convolutions
        for j in range(1, self.N):
            for i in range(self.N - j):
                skip_connections = [xs[i][k] for k in range(j)]
                skip_connections.append(self.up(xs[i+1][j-1]))
                xs[i][j] = self.convs[i][j](torch.cat(skip_connections, dim=1))

        # Finally the dense layer that make the prediction per pixel/patch
        return torch.stack([final_conv(xs[0][i]) for i, final_conv in enumerate(self.final_convs)], dim=1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)