import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UnetModel


class NeighborUnetModel(UnetModel):
    def create_model(self):
        super(NeighborUnetModel, self).create_model()

        self.neighbor_size = self.config["neighbor_unet"]["neighbor_kernel_size"]
        self.alpha = self.config["neighbor_unet"]["neighbor_loss_weight"]

        # Define neighborhood loss kernel
        self.neighbor_kernel = torch.ones((self.neighbor_size, self.neighbor_size), dtype=torch.float32)
        self.neighbor_kernel[self.neighbor_size // 2, self.neighbor_size // 2] = 0
        self.neighbor_kernel /= self.neighbor_kernel.sum()
        self.neighbor_kernel = self.neighbor_kernel.unsqueeze(0).unsqueeze(0)
        self.neighbor_kernel = nn.Parameter(self.neighbor_kernel, requires_grad=False)

    def loss(self, pred_BHW, target_BHW):
        bce_loss = F.binary_cross_entropy_with_logits(pred_BHW, target_BHW)

        pad = self.neighbor_size // 2
        neighbor_target_BHW = F.conv2d(
            F.pad(pred_BHW, [pad, pad, pad, pad], mode="replicate").unsqueeze(1),
            self.neighbor_kernel,
        ).squeeze(1).detach()
        neighbor_loss = F.mse_loss(pred_BHW, neighbor_target_BHW)

        return bce_loss + self.alpha * neighbor_loss
