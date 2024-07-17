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
        self.model = Unet(
            channels=self.config["channels"],
            bottleneck_mhsa_layers=self.config["bottleneck_mhsa_layers"],
            num_heads=self.config["num_heads"],
            act_fn=act_fn,
            block=block,
            blocks_per_layer=self.config["blocks_per_layer"],
            patch_size=PATCH_SIZE if self.config["predict_patches"] else 1,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        self.pos_weight = self.config["pos_weight"]

    def training_step(self, input_BCHW, target_BHW):
        self.optimizer.zero_grad()

        pred_BHW = self.model(input_BCHW).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(pred_BHW, target_BHW, pos_weight=self.pos_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return { "loss": loss.item() }

    def predict(self, input_BCHW):
        return F.sigmoid(self.model(input_BCHW).squeeze(1))


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: int=1,
        channels: list[int]=[64, 128, 256, 512, 1024],
        bottleneck_mhsa_layers: int=0,
        num_heads: int=1,
        act_fn: nn.Module=nn.ReLU,
        block: nn.Module=ConvBlock,
        blocks_per_layer: int=1,
        patch_size: int=1,
    ):
        super().__init__()
        enc_channels = channels
        dec_channels = channels[::-1]

        # Use a normal convolution for the first layer, because we want to keep the number of channels
        # a power of 2
        self.initial_conv = nn.Sequential(
            ConvBlock(in_channels, channels[0], act_fn),
            *[block(channels[0], channels[0], act_fn) for _ in range(blocks_per_layer - 1)],
        )
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(block(in_c, out_c, act_fn), *[block(out_c, out_c, act_fn) for _ in range(blocks_per_layer - 1)])
            for in_c, out_c in zip(enc_channels[:-1], enc_channels[1:])
        ])
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
            for in_c, out_c in zip(dec_channels[:-1], dec_channels[1:])
        ])
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(block(in_c, out_c, act_fn), *[block(out_c, out_c, act_fn) for _ in range(blocks_per_layer - 1)])
            for in_c, out_c in zip(dec_channels[:-1], dec_channels[1:])
        ])

        if bottleneck_mhsa_layers > 0:
            transformer_layer = nn.TransformerEncoderLayer(channels[-1], num_heads, dim_feedforward=channels[-1] * 2, batch_first=True)
            self.bottleneck = nn.Sequential(
                PosEnc(channels[-1], max_len=(400 // 2**(len(channels)-1)) ** 2),
                nn.TransformerEncoder(transformer_layer, bottleneck_mhsa_layers),
            )
        else:
            self.bottleneck = None

        # The final convolution has a patch size and stride of `patch_size`, such that we have a
        # single prediction for each patch of size `patch_size` x `patch_size`.
        self.final_conv = nn.Conv2d(dec_channels[-1], out_channels, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x: torch.Tensor):
        # Encoding
        x = self.initial_conv(x)
        enc_features = []
        for block in self.enc_blocks:
            enc_features.append(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = block(x)
        
        # MHSA bottleneck
        if self.bottleneck is not None:
            x_bot = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
            x_bot = self.bottleneck(x_bot)
            x = x_bot.transpose(1, 2).reshape_as(x)

        # Decoding
        for block, up_conv, feature in zip(self.dec_blocks, self.up_convs, enc_features[::-1]):
            x = up_conv(x)
            x = torch.cat([x, feature], dim=1)
            x = block(x)

        return self.final_conv(x)


class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int=512):
        super().__init__()
        self.register_buffer("pos_enc", self.create_pos_enc(d_model, max_len))

    def create_pos_enc(self, d_model: int, max_len: int):
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(pos.float() * div_term)
        pos_enc[:, 1::2] = torch.cos(pos.float() * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        return x + self.pos_enc[:, :x.shape[1]]
