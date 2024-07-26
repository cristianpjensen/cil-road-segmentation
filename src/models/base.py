from abc import abstractmethod
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.optimizer = None
        self.create_model()

    def to_device(self, device: torch.device):
        """Iterate over attributes and add nn.Modules to device."""

        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.to(device)

    def num_params(self) -> int:
        """Iterate over attributes and add up the number of parameters of all nn.Modules."""

        total_params = 0
        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                total_params += sum(p.numel() for p in attr.parameters())

        return total_params

    def train(self):
        """Iterate over attributes and set the model to train mode."""

        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.train()

    def eval(self):
        """Iterate over attributes and set the model to eval mode."""

        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.eval()

    def set_optimizer_lr(self, new_lr: float):
        if self.optimizer is None:
            raise ValueError("No optimizer defined.")

        self.config["lr"] = new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def save(self, path: str):
        """Iterate over attributes and save the model to the given path."""

        state_dicts = {}
        for name, attr in self.__dict__.items():
            if isinstance(attr, nn.Module):
                state_dicts[name] = attr.state_dict()

        torch.save(state_dicts, path)

    def load(self, path: str):
        """Iterate over attributes and load the model from the given path."""

        state_dicts = torch.load(path)
        for name, attr in self.__dict__.items():
            if name in state_dicts:
                attr.load_state_dict(state_dicts[name])

    @abstractmethod
    def create_model(self):
        """
        Use this method to initialize the model in any way that is necessary. You have access to
        the configuration by `self.config`.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, input_BCHW: torch.Tensor, target_BHW: torch.Tensor) -> dict:
        """
        Perform a full training step, including the forward and backward pass. Return a dictionary
        with the loss values. These will be logged.

        NOTE: Make sure to put the model in train mode (call `model.train()`) before calling this method.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_BCHW: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Make sure to put the model in eval mode (call `model.eval()`) before calling this method.
        """
        raise NotImplementedError
