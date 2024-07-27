import torch
import torch.nn.functional as F


def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float=1) -> torch.Tensor:
    input = F.sigmoid(input)

    input = input.view(-1)
    target = target.view(-1)

    intersection = (input * target).sum()
    dice = (2 * intersection + eps) / (input.sum() + target.sum() + eps)

    return 1 - dice


def bce_dice(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Input: [B, H, W]
    Target: [B, H, W]
    Output: [1]
    """

    return 0.5 * (dice_loss(pred, target, eps) + F.binary_cross_entropy_with_logits(pred, target))
