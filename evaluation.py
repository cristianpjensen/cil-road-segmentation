"""
Suffix keys:
    P: Patch size
    D: Number of patches
"""

import torch


FOREGROUND_THRESHOLD = 0.25


def patchify(x: torch.Tensor, patch_size=16) -> torch.Tensor:
    """
    Input: [*, H, W]
    Output: [*, D, P, P] 
    """

    P = patch_size
    return x.unfold(-2, P, P).unfold(-2, P, P).flatten(-4, -3)


def eval_f1_score(pred_BHW, target_BHW):
    """Accuracy for binary classification."""

    pred_patches_BDPP = patchify(pred_BHW)
    target_patches_BDPP = patchify(target_BHW)
    patchwise_pred_BD = pred_patches_BDPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
    patchwise_target_BD = target_patches_BDPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD

    tp_B = (patchwise_pred_BD & patchwise_target_BD).float().sum(dim=1)
    fp_B = (patchwise_pred_BD & ~patchwise_target_BD).float().sum(dim=1)
    fn_B = (~patchwise_pred_BD & patchwise_target_BD).float().sum(dim=1)

    f1_B = 2 * tp_B / (2 * tp_B + fp_B + fn_B)

    return f1_B.mean()
