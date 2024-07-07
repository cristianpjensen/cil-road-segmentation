import torch
import torchvision.transforms.functional as TF
from functools import reduce
import hashlib

def hash_fn(s: str, seed: int) -> int:
    return int(hashlib.md5(bytes(f"{s}{seed}", "utf-8")).hexdigest()[-8:], 16)


def alternating_transforms(
    x_HW: torch.Tensor,
    indices: list[str],
    transformations: list[callable],
    epoch: int,
    seed: int,
) -> torch.Tensor:
    """
    Input: [B, *, H, W]
    Output: [B, *, H, W]
    """

    hashed_indices = torch.tensor([hash_fn(i, seed) for i in indices])
    flip_mask = ((hashed_indices + epoch) % len(transformations)) 
    flip_mask = flip_mask.view(flip_mask.shape + (1,) * (x_HW.dim() - flip_mask.dim()))

    # Apply transformations
    out_HW = x_HW.clone()
    for i, transforms in enumerate(transformations):
        out_HW = torch.where(flip_mask == i, compose_funcs(out_HW, transforms), out_HW)

    return out_HW


def rotate90(img: torch.Tensor) -> torch.Tensor:
    return TF.rotate(img, 90)


def rotate180(img: torch.Tensor) -> torch.Tensor:
    return TF.rotate(img, 180)


def rotate270(img: torch.Tensor) -> torch.Tensor:
    return TF.rotate(img, 270)


def compose_funcs(obj, func_list: list[callable]):
    return reduce(lambda o, func: func(o), func_list, obj)


def compose_transforms(transforms: str) -> list[list[callable]]:
    transformations = [[]]
    if "v" in transforms:
        transformations += [fns + [TF.vflip] for fns in transformations]

    if "h" in transforms:
        transformations += [fns + [TF.hflip] for fns in transformations]

    if "r" in transforms:
        new_transformations = []
        new_transformations += [fns + [rotate90] for fns in transformations]
        new_transformations += [fns + [rotate180] for fns in transformations]
        new_transformations += [fns + [rotate270] for fns in transformations]
        transformations += new_transformations

    return transformations
