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
        """
        Use this method to initialize the model in any way that is necessary. You have access to
        the configuration by `self.config`.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, input_BCHW: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        The output of this method will be passed to the loss method. The reason for two separate
        methods (`step` and `predict`) is that the outputs may be different. E.g., in the case of a
        BCE loss, `step` should return the logits, while `predict` should return the probabilities,
        because logits are more numerically stable.

        NOTE: Make sure to put the model in train mode (call `model.train()`) before calling this
        method.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred_BHW: torch.Tensor, target_BHW: torch.Tensor) -> torch.Tensor:
        """
        The loss function. The output of this method will be used to compute the gradients and update
        the weights.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_BCHW: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Make sure to put the model in eval mode (call `model.eval()`) before calling this method.
        """
        raise NotImplementedError
