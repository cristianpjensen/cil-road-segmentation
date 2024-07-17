from .dummy import DummyModel
from .base import BaseModel
from .unet import UnetModel
from .neighbor_unet import NeighborUnetModel
from .unetplusplus import UnetPlusPlusModel
from .pix2pix import Pix2PixModel


def create_model(model: str, config: dict) -> BaseModel:
    match model:
        case "dummy":
            return DummyModel(config)

        case "unet":
            return UnetModel(config)

        case "neighbor_unet":
            return NeighborUnetModel(config)

        case "unet++":
            return UnetPlusPlusModel(config)

        case "pix2pix":
            return Pix2PixModel(config)

        case _:
            ValueError(f"Unknown model: {model}")
    