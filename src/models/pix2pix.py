import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .unet import Unet
from .constants import ACTIVATIONS, BLOCKS
from ..constants import PATCH_SIZE


class Pix2PixModel(BaseModel):
    def create_model(self):
        act_fn = ACTIVATIONS[self.config["activation"]]
        block = BLOCKS[self.config["block"]]
        self.generator = Unet(
            channels=self.config["channels"],
            bottleneck_mhsa_layers=self.config["bottleneck_mhsa_layers"],
            num_heads=self.config["num_heads"],
            act_fn=act_fn,
            block=block,
            blocks_per_layer=self.config["blocks_per_layer"],
            patch_size=PATCH_SIZE if self.config["predict_patches"] else 1,
        )
        if self.config["predict_patches"] and self.config["pix2pix"]["patch_discriminator"]:
            self.discriminator = PatchDiscriminator(input_channels=3, target_channels=1)
        else:
            self.discriminator = Discriminator(in_channels=4)

        self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=self.config["lr"])
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.config["lr"])

        self.l1_weight = self.config["pix2pix"]["l1_weight"]
        self.interleave_patches = self.config["predict_patches"] and not self.config["pix2pix"]["patch_discriminator"]

    def training_step(self, input_BCHW, target_BHW):
        # Train the discriminator
        self.d_optimizer.zero_grad()

        pred_BHW = F.sigmoid(self.generator(input_BCHW).detach().squeeze(1))
        if self.interleave_patches:
            pred_BHW = pred_BHW.repeat_interleave(PATCH_SIZE, dim=-2).repeat_interleave(PATCH_SIZE, dim=-1)
            target_BHW = target_BHW.repeat_interleave(PATCH_SIZE, dim=-2).repeat_interleave(PATCH_SIZE, dim=-1)

        real_pred = self.discriminator(input_BCHW, target_BHW.unsqueeze(1))
        fake_pred = self.discriminator(input_BCHW, pred_BHW.unsqueeze(1))
        d_real_loss, d_fake_loss = self._discriminator_loss(real_pred, fake_pred)
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.d_optimizer.step()

        # Train the generator
        self.g_optimizer.zero_grad()

        pred_BHW = F.sigmoid(self.generator(input_BCHW).squeeze(1))
        if self.interleave_patches:
            pred_BHW = pred_BHW.repeat_interleave(PATCH_SIZE, dim=-2).repeat_interleave(PATCH_SIZE, dim=-1)

        g_loss = self._generator_loss(input_BCHW, pred_BHW, target_BHW)

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_optimizer.step()

        return { "g_loss": g_loss.item(), "d_real_loss": d_real_loss.item(), "d_fake_loss": d_fake_loss.item() }

    def _generator_loss(self, input_BCHW: torch.Tensor, pred_BHW: torch.Tensor, target_BHW: torch.Tensor) -> torch.Tensor:
        disc_pred = self.discriminator(input_BCHW, pred_BHW.unsqueeze(1))
        disc_loss = F.binary_cross_entropy_with_logits(disc_pred, torch.ones_like(disc_pred))
        l1_loss = F.l1_loss(pred_BHW, target_BHW)
        return disc_loss + self.l1_weight * l1_loss

    def _discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # Every patch has to be classified correctly. If real, the discriminator should output 1, if
        # fake, 0.
        real_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        return real_loss, fake_loss

    def predict(self, input_BCHW):
        return F.sigmoid(self.generator(input_BCHW).squeeze(1))


class DiscriminatorDownsample(nn.Module):
    """An encoder block that is used in the pix2pix discriminator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """Discriminator that distinguishes between real and fake image segmentation maps.

    Args:
        in_channels (int): Input channels for both the input image and predicted segmentation map.

    """

    def __init__(self, in_channels: int=4):
        super().__init__()

        self.net = nn.Sequential(
            DiscriminatorDownsample(in_channels, 64, norm=False), # [64, 200, 200]
            DiscriminatorDownsample(64, 128), # [128, 100, 100]
            DiscriminatorDownsample(128, 256), # [256, 50, 50]
            DiscriminatorDownsample(256, 512), # [512, 25, 25]
            nn.Conv2d(512, 1, kernel_size=4, padding=2, bias=False), # [1, 25, 25]
        )
        self.apply(init_weights)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


class PatchDiscriminatorTargetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int=3, target_channels: int=1):
        super().__init__()

        # The image is 400x400, while the target is 25x25, because it contains 16x16 patchwise predictions
        self.input_net = nn.Sequential(
            DiscriminatorDownsample(input_channels, 64, norm=False), # [64, 200, 200]
            DiscriminatorDownsample(64, 128), # [128, 100, 100]
            DiscriminatorDownsample(128, 256), # [256, 50, 50]
            DiscriminatorDownsample(256, 512), # [512, 25, 25]
        )
        self.target_net = nn.Sequential(
            PatchDiscriminatorTargetBlock(target_channels, 64, norm=False), # [64, 25, 25]
            PatchDiscriminatorTargetBlock(64, 128), # [128, 25, 25]
            PatchDiscriminatorTargetBlock(128, 256), # [256, 25, 25]
            PatchDiscriminatorTargetBlock(256, 512), # [512, 25, 25]
        )
        self.combo_net = nn.Sequential(
            PatchDiscriminatorTargetBlock(1024, 512), # [512, 25, 25]
            PatchDiscriminatorTargetBlock(512, 512), # [512, 25, 25]
            nn.Conv2d(512, 1, kernel_size=4, padding=2, bias=False), # [1, 25, 25]
        )

        self.apply(init_weights)

    def forward(self, x, y):
        input_features = self.input_net(x)
        target_features = self.target_net(y)
        features = torch.cat([input_features, target_features], dim=1)
        return self.combo_net(features)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.normal_(m.bias, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)