import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import LOSSES
from .unet import UnetModel
from ..constants import DEVICE


class NeighborUnetModel(UnetModel):
    def create_model(self):
        super(NeighborUnetModel, self).create_model()

        self.neighbor_size = self.config["neighbor_unet"]["neighbor_kernel_size"]
        self.alpha = self.config["neighbor_unet"]["neighbor_loss_weight"]
        self.loss = LOSSES[self.config["loss"]]

        # Define neighborhood loss kernel
        self.neighbor_kernel = torch.ones((self.neighbor_size, self.neighbor_size), dtype=torch.float32)
        self.neighbor_kernel[self.neighbor_size // 2, self.neighbor_size // 2] = 0
        self.neighbor_kernel /= self.neighbor_kernel.sum()
        self.neighbor_kernel = self.neighbor_kernel.unsqueeze(0).unsqueeze(0)
        self.neighbor_kernel = nn.Parameter(self.neighbor_kernel, requires_grad=False).to(DEVICE)

    def training_step(self, input_BCHW, target_BHW):
        self.optimizer.zero_grad()

        pred_BHW = self.model(input_BCHW).squeeze(1)
        loss = self._loss(pred_BHW, target_BHW)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return { "loss": loss.item() }

    def _loss(self, pred_BHW, target_BHW):
        loss = self.loss(pred_BHW, target_BHW)

        pad = self.neighbor_size // 2
        neighbor_target_BHW = F.conv2d(
            F.pad(pred_BHW, [pad, pad, pad, pad], mode="replicate").unsqueeze(1),
            self.neighbor_kernel,
        ).squeeze(1).detach()
        neighbor_loss = F.mse_loss(pred_BHW, neighbor_target_BHW)

        return loss + self.alpha * neighbor_loss
