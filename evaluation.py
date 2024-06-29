"""
Suffix keys:
    P: Patch size
    M: Number of patches on vertical axis
    N: Number of patches on horizontal axis
"""

import torch
from constants import FOREGROUND_THRESHOLD, PATCH_SIZE


def patchify(x: torch.Tensor, patch_size=PATCH_SIZE) -> torch.Tensor:
    """
    Input: [*, H, W]
    Output: [*, M, N, P, P] 
    """

    P = patch_size
    return x.unfold(-2, P, P).unfold(-2, P, P)


def eval_f1_score(pred_BHW, target_BHW):
    """Accuracy for binary classification."""

    pred_patches_BMNPP = patchify(pred_BHW)
    target_patches_BMNPP = patchify(target_BHW)
    patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
    patchwise_target_BMN = target_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
    patchwise_pred_BD = patchwise_pred_BMN.view(pred_BHW.shape[0], -1)
    patchwise_target_BD = patchwise_target_BMN.view(pred_BHW.shape[0], -1)

    tp_B = (patchwise_pred_BD & patchwise_target_BD).float().sum(dim=1)
    fp_B = (patchwise_pred_BD & ~patchwise_target_BD).float().sum(dim=1)
    fn_B = (~patchwise_pred_BD & patchwise_target_BD).float().sum(dim=1)

    f1_B = 2 * tp_B / (2 * tp_B + fp_B + fn_B)

    return f1_B.mean()


def get_mask(pred_BHW):
    """
    Input: [B, H, W]
    Output: [B, H, W]
    """

    pred_patches_BMNPP = patchify(pred_BHW)
    patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
    mask_BHW = patchwise_pred_BMN.repeat_interleave(PATCH_SIZE, dim=-2).repeat_interleave(PATCH_SIZE, dim=-1)
    return mask_BHW.float()
