from abc import abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.create_model()

    @abstractmethod
    def create_model(self):
        """Use this method to initialize the model."""
        raise NotImplementedError

    @abstractmethod
    def step(self, input_BCHW: torch.Tensor) -> torch.Tensor:
        """Make sure to put the model in train mode before calling this method."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred_BHW: torch.Tensor, target_BHW: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_BCHW: torch.Tensor) -> torch.Tensor:
        """Make sure to put the model in eval mode before calling this method."""
        raise NotImplementedError
