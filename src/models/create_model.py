from .dummy import DummyModel
from .base import BaseModel
from .unet import UnetModel
from .res18unet import Res18UnetModel
from .res50unet import Res50UnetModel
from .resv2unet import ResV2UnetModel


def create_model(model: str, config: dict) -> BaseModel:
    match model:
        case "dummy":
            return DummyModel(config)

        case "unet":
            return UnetModel(config)

        case "res18unet":
            return Res18UnetModel(config)

        case "res50unet":
            return Res50UnetModel(config)
    
        case "resv2unet":
            return ResV2UnetModel(config)

        case _:
            ValueError(f"Unknown model: {model}")
    