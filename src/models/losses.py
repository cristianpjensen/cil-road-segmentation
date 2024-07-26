import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Input: [B, H, W]
    Target: [B, H, W]
    Output: [1]
    """

    pred = F.sigmoid(pred)

    pos_intersection = (pred * target).sum(dim=[-2, -1])
    pos_union = (pred + target).sum(dim=[-2, -1])

    neg_intersection = ((1 - pred) * (1 - target)).sum(dim=[-2, -1])
    neg_union = (2 - pred - target).sum(dim=[-2, -1])

    dice_loss = 1 - ((pos_intersection + eps) / (pos_union + eps)) - ((neg_intersection + eps) / (neg_union + eps))
    return dice_loss.mean()


def bce_dice(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    """
    Input: [B, H, W]
    Target: [B, H, W]
    Output: [1]
    """

    return dice_loss(pred, target, eps) + F.binary_cross_entropy_with_logits(pred, target)
