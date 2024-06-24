from models.dummy import DummyModel
from models.base import BaseModel


def create_model(model: str, config: dict) -> BaseModel:
    match model:
        case "dummy":
            return DummyModel(config)
        
        case _:
            ValueError(f"Unknown model: {model}")
    