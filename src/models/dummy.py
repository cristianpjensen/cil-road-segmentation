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


class DummyModel(BaseModel):
    def create_model(self):
        self.model = Dummy(patch_size=16 if self.config["predict_patches"] else 1)
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

    def set_optimizer_lr(self, lr):
        self.config["lr"] = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class Dummy(nn.Module):
    def __init__(self, patch_size: int=1):
        super().__init__()
        self.net = nn.Conv2d(3, 1, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        return self.net(x)
