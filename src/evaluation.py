"""
Suffix keys:
    P: Patch size
    M: Number of patches on vertical axis
    N: Number of patches on horizontal axis
"""

import os
import tempfile
import torch
from torchvision.io import write_png
from torch.utils.data import DataLoader
from sacred import Experiment

from .constants import FOREGROUND_THRESHOLD, PATCH_SIZE, DEVICE
from .models.base import BaseModel


def patchify(x: torch.Tensor, patch_size=PATCH_SIZE) -> torch.Tensor:
    """
    Input: [*, H, W]
    Output: [*, M, N, P, P] 
    """

    P = patch_size
    return x.unfold(-2, P, P).unfold(-2, P, P)


def depatchify(x: torch.Tensor) -> torch.Tensor:
    """
    Input: [*, M, N, P, P]
    Output: [*, H, W]
    """

    return x.transpose(-3, -2).flatten(-4, -3).flatten(-2, -1)


def patch_f1_score(pred_BHW: torch.Tensor, target_BHW: torch.Tensor, is_patches: bool=False) -> torch.Tensor:
    if not is_patches:
        pred_patches_BMNPP = patchify(pred_BHW)
        target_patches_BMNPP = patchify(target_BHW)
        patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
        patchwise_target_BMN = target_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
        patchwise_pred_BD = patchwise_pred_BMN.view(pred_BHW.shape[0], -1)
        patchwise_target_BD = patchwise_target_BMN.view(pred_BHW.shape[0], -1)
    else:
        patchwise_pred_BD = pred_BHW.round().bool().view(pred_BHW.shape[0], -1)
        patchwise_target_BD = target_BHW.round().bool().view(pred_BHW.shape[0], -1)

    tp = (patchwise_pred_BD & patchwise_target_BD).float().sum()
    fp = (patchwise_pred_BD & ~patchwise_target_BD).float().sum()
    fn = (~patchwise_pred_BD & patchwise_target_BD).float().sum()
    f1 = 2 * tp / (2 * tp + fp + fn)

    return f1


def patch_accuracy(pred_BHW: torch.Tensor, target_BHW: torch.Tensor, is_patches: bool=False) -> torch.Tensor:
    if not is_patches:
        pred_patches_BMNPP = patchify(pred_BHW)
        target_patches_BMNPP = patchify(target_BHW)
        patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
        patchwise_target_BMN = target_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
    else:
        patchwise_pred_BMN = pred_BHW.round()
        patchwise_target_BMN = target_BHW.round()

    return (patchwise_pred_BMN == patchwise_target_BMN).float().mean()


def pixel_accuracy(pred_BHW: torch.Tensor, target_BHW: torch.Tensor) -> torch.Tensor:
    return (pred_BHW.round() == target_BHW.round()).float().mean()


def get_mask(pred_BHW: torch.Tensor, is_patches: bool) -> torch.Tensor:
    """
    Input: [B, H, W]
    Output: [B, H, W]
    """

    if not is_patches:
        pred_patches_BMNPP = patchify(pred_BHW)
        patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
    else:
        patchwise_pred_BMN = pred_BHW.round()
    
    mask_BHW = patchwise_pred_BMN.repeat_interleave(PATCH_SIZE, dim=-2).repeat_interleave(PATCH_SIZE, dim=-1)
    return mask_BHW.float()


def output_mask_overlay(
    ex: Experiment,
    ex_dir: str,
    dir: str,
    epoch: int,
    file_names: tuple[str],
    input_BCHW: torch.Tensor,
    pred_BHW: torch.Tensor,
    is_patches: bool=False,
):
    """Output input images with the predicted patch-wise mask overlaid on top in red."""

    mask_BHW = get_mask(pred_BHW, is_patches)
    red_mask_BCHW = mask_BHW.unsqueeze(1) * torch.tensor([1, 0, 0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    overlay_BCHW = input_BCHW
    overlay_BCHW[mask_BHW.bool().unsqueeze(1).repeat(1, 3, 1, 1)] *= 0.5
    overlay_BCHW += 0.5 * red_mask_BCHW

    for i, overlay_img in enumerate(overlay_BCHW):
        with tempfile.NamedTemporaryFile() as tmp_file:
            write_png((overlay_img * 255).byte(), tmp_file.name, compression_level=0)
            ex.add_artifact(tmp_file.name, get_patch_overlay_dir(ex_dir, dir, epoch, file_names[i]))


def get_patch_overlay_dir(ex_dir: str, dir: str, epoch: int, file_name: str | None = None):
    if file_name is None:
        return os.path.join(ex_dir, f"{dir}_valid", str(epoch), "patch_overlay")
    else:
        return os.path.join(f"{dir}_valid", str(epoch), "patch_overlay", file_name)


def output_pixel_pred(
    ex: Experiment,
    ex_dir: str,
    dir: str,
    epoch: int,
    file_names: tuple[str],
    pred_BHW: torch.Tensor,
    is_patches: bool=False,
):
    """Output the per-pixel predictions as images."""

    if is_patches:
        pred_BHW = get_mask(pred_BHW, True)

    pred_BHW = (pred_BHW.unsqueeze(1) * 255).byte().cpu()
    for i, pred_img in enumerate(pred_BHW):
        with tempfile.NamedTemporaryFile() as tmp_file:
            write_png(pred_img, tmp_file.name, compression_level=0)
            ex.add_artifact(tmp_file.name, get_pixel_pred_dir(ex_dir, dir, epoch, file_names[i]))


def get_pixel_pred_dir(ex_dir: str, dir: str, epoch: int, file_name: str | None = None):
    if file_name is None:
        return os.path.join(ex_dir, f"{dir}_valid", str(epoch), "pixel_pred")
    else:
        return os.path.join(f"{dir}_valid", str(epoch), "pixel_pred", file_name)


def output_submission_file(
    ex: Experiment,
    dir: str,
    model: BaseModel,
    test_loader: DataLoader,
    predict_patches: bool=False,
):
    """Given a model and data loader, output a submission file for the test set. It assumes that the
    data loader does not contain targets."""

    os.makedirs(os.path.join(dir, "test"))

    with open(os.path.join(dir, "submission.csv"), "w") as f:
        f.write("id,prediction\n")

        for (input_BCHW, input_files) in test_loader:
            input_BCHW = input_BCHW.to(DEVICE)
            pred_BHW = model.predict(input_BCHW)
            
            if not predict_patches:
                pred_patches_BMNPP = patchify(pred_BHW)
                patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD
            else:
                patchwise_pred_BMN = pred_BHW.round().bool()

            for i in range(patchwise_pred_BMN.shape[0]):
                # Output prediction map
                with tempfile.NamedTemporaryFile() as tmp_file:
                    write_png((pred_BHW[i].unsqueeze(0) * 255).byte().cpu(), tmp_file.name, compression_level=0)
                    ex.add_artifact(tmp_file.name, os.path.join("test", input_files[i]))

                # Add to submission file
                for x in range(patchwise_pred_BMN.shape[1]):
                    for y in range(patchwise_pred_BMN.shape[2]):
                        image_id = int(input_files[i].split("_")[-1].split(".")[0])
                        f.write(f"{image_id:03d}_{y * PATCH_SIZE}_{x * PATCH_SIZE},{int(patchwise_pred_BMN[i, x, y])}\n")
