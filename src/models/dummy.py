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
        self.model = Dummy()

    def step(self, input_BCHW, _):
        return self.model(input_BCHW).squeeze(1)

    def loss(self, pred_BHW, target_BHW):
        # Do not use sigmoid in the model, because it is more numerically stable to use BCE with
        # logits, which combines the sigmoid and the BCE loss in a single function.
        return F.binary_cross_entropy_with_logits(pred_BHW, target_BHW, pos_weight=self.config["pos_weight"])

    def predict(self, input_BCHW):
        return F.sigmoid(self.model(input_BCHW).squeeze(1))


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        return self.net(x)
