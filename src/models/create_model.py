from .dummy import DummyModel
from .base import BaseModel
from .unet import UnetModel


def create_model(model: str, config: dict) -> BaseModel:
    match model:
        case "dummy":
            return DummyModel(config)

        case "relu-unet":
            return UnetModel(config)

        case _:
            ValueError(f"Unknown model: {model}")
    