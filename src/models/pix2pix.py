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
        self.discriminator = Discriminator(in_channels=4)

        self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=self.config["lr"])
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.config["lr"])

        self.l1_weight = self.config["pix2pix"]["l1_weight"]
        self.predict_patches = self.config["predict_patches"]

    def training_step(self, input_BCHW, target_BHW):
        # Train the discriminator
        self.d_optimizer.zero_grad()

        pred_BHW = F.sigmoid(self.generator(input_BCHW).detach().squeeze(1))
        if self.predict_patches:
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
        if self.predict_patches:
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
                padding=1
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
            DiscriminatorDownsample(in_channels, 64, norm=False),
            DiscriminatorDownsample(64, 128),
            DiscriminatorDownsample(128, 256),
            DiscriminatorDownsample(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False),
        )
        self.apply(self.init_weights)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.0, 0.02)

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
