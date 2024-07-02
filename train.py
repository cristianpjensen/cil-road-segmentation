import os
import tempfile
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.io import write_png
import sacred
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm

from src.models.create_model import create_model
from src.models.base import BaseModel
from src.dataset import ImageSegmentationDataset
from src.evaluation import eval_f1_score, get_mask, patchify
from src.constants import DEVICE, FOREGROUND_THRESHOLD

# Get rid of warnings
sacred.SETTINGS["CAPTURE_MODE"] = "sys"

ex = Experiment()
observer = FileStorageObserver(basedir="experiments", resource_dir="data")
ex.observers.append(observer)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    model_name = "dummy"
    epochs = 1000
    batch_size = 32
    lr = 1e-3
    is_pbar = True
    is_early_stopping = True
    early_stopping_config = {
        "patience": 50,
        "min_delta": 1e-4,
    }
    output_images_every = 10
    train_split = 0.8


@ex.capture
def get_iterator(iterator, is_pbar, **kwargs):
    return tqdm(iterator, **kwargs) if is_pbar else iterator


def output_mask_overlay(epoch: int, file_names: tuple[str], input_BCHW: torch.Tensor, pred_BHW: torch.Tensor):
    """Output input images with the predicted patch-wise mask overlaid on top in red."""

    mask_BHW = get_mask(pred_BHW)
    red_mask_BCHW = mask_BHW.unsqueeze(1) * torch.tensor([1, 0, 0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    overlay_BCHW = input_BCHW
    overlay_BCHW[mask_BHW.bool().unsqueeze(1).repeat(1, 3, 1, 1)] *= 0.5
    overlay_BCHW += 0.5 * red_mask_BCHW

    for i, overlay_img in enumerate(overlay_BCHW):
        with tempfile.NamedTemporaryFile() as tmp_file:
            write_png((overlay_img * 255).byte(), tmp_file.name)
            ex.add_artifact(tmp_file.name, get_patch_overlay_dir(epoch, file_names[i]))


def get_patch_overlay_dir(epoch: int, file_name: str | None = None):
    if file_name is None:
        return os.path.join(observer.dir, "validation", str(epoch), "patch_overlay")
    else:
        return os.path.join("validation", str(epoch), "patch_overlay", file_name)


def output_pixel_pred(epoch: int, file_names: tuple[str], pred_BHW: torch.Tensor):
    """Output the per-pixel predictions as images."""

    pred_BHW = (pred_BHW.unsqueeze(1) * 255).byte().cpu()
    for i, pred_img in enumerate(pred_BHW):
        with tempfile.NamedTemporaryFile() as tmp_file:
            write_png(pred_img, tmp_file.name)
            ex.add_artifact(tmp_file.name, get_pixel_pred_dir(epoch, file_names[i]))


def get_pixel_pred_dir(epoch: int, file_name: str | None = None):
    if file_name is None:
        return os.path.join(observer.dir, "validation", str(epoch), "pixel_pred")
    else:
        return os.path.join("validation", str(epoch), "pixel_pred", file_name)


def output_submission_file(model: BaseModel, test_loader: DataLoader):
    """Given a model and data loader, output a submission file for the test set. It assumes that the
    data loader does not contain targets."""

    with open(os.path.join(observer.dir, "submission.csv"), "w") as f:
        f.write("id,prediction\n")

        for (input_BCHW, input_files) in test_loader:
            input_BCHW = input_BCHW.to(DEVICE)
            pred_BHW = model.predict(input_BCHW)
            pred_patches_BMNPP = patchify(pred_BHW)
            patchwise_pred_BMN = pred_patches_BMNPP.mean(dim=[-1, -2]) > FOREGROUND_THRESHOLD

            for i in range(patchwise_pred_BMN.shape[0]):
                for x in range(patchwise_pred_BMN.shape[1]):
                    for y in range(patchwise_pred_BMN.shape[2]):
                        image_id = int(input_files[i].split("_")[-1].split(".")[0])
                        f.write(f"{image_id:03d}_{x * 16}_{y * 16},{int(patchwise_pred_BMN[i, x, y])}\n")


@ex.automain
def main(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    is_pbar: bool,
    is_early_stopping: bool,
    early_stopping_config: dict,
    output_images_every: int,
    train_split: float,
):
    print(f"Device: {DEVICE}")

    transform = T.Compose([
        T.Resize((400, 400)),
    ])

    data = ImageSegmentationDataset("data/training/images", "data/training/groundtruth", normalize=True, transform=transform)
    train_data, valid_data = random_split(data, [train_split, 1 - train_split])
    test_data = ImageSegmentationDataset("data/test/images", normalize=(data.channel_means, data.channel_stds), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = create_model(model_name, { "pos_weight": data.pos_weight() })
    model.to(DEVICE)
    print(f"Model '{model_name}' created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_valid_score = -float("inf")
    no_improvement = 0
    # Create temporary file to save the best model while training
    model_tmp_file = tempfile.NamedTemporaryFile()

    pbar = get_iterator(range(epochs))
    for epoch in pbar:
        # Check for early stopping
        if is_early_stopping and no_improvement > early_stopping_config["patience"]:
            break

        # Training
        model.train()
        total_train_loss = 0
        for (input_BCHW, _, target_BHW, _) in get_iterator(train_loader, leave=False):
            model.zero_grad()

            # Forward pass
            input_BCHW, target_BHW = input_BCHW.to(DEVICE), target_BHW.to(DEVICE)
            pred_BHW = model.step(input_BCHW)
            loss = model.loss(pred_BHW, target_BHW.to(DEVICE))

            # Compute gradient and update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item() * input_BCHW.shape[0]

        # Predict validation images every `output_images_every`` epochs and save the images as artifacts
        if epoch % output_images_every == 0:
            os.makedirs(get_pixel_pred_dir(epoch))
            os.makedirs(get_patch_overlay_dir(epoch))
            image_count = 0

        # Validation
        model.eval()
        total_valid_score = 0
        for (input_BCHW, input_files, target_BHW, _) in get_iterator(valid_loader, leave=False):
            input_BCHW, target_BHW = input_BCHW.to(DEVICE), target_BHW.to(DEVICE)
            pred_BHW = model.predict(input_BCHW)
            score = eval_f1_score(pred_BHW, target_BHW)
            total_valid_score += score.item() * input_BCHW.shape[0]

            input_BCHW, pred_BHW = input_BCHW.cpu(), pred_BHW.cpu()

            if epoch % output_images_every == 0:
                output_pixel_pred(epoch, input_files, pred_BHW)
                output_mask_overlay(epoch, input_files, data.denormalize(input_BCHW), pred_BHW)
                image_count += input_BCHW.shape[0]

        train_loss = total_train_loss / len(train_data)
        valid_score = total_valid_score / len(valid_data)

        # Early stopping
        if is_early_stopping and valid_score - best_valid_score > early_stopping_config["min_delta"]:
            no_improvement = 0
        else:
            no_improvement += 1

        # Save best model based on validation loss
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            torch.save(model.state_dict(), model_tmp_file.name)

        # Log losses
        ex.log_scalar("train_loss", train_loss)
        ex.log_scalar("valid_score", valid_score)

        if is_pbar:
            pbar.set_description(f"train loss: {train_loss:.4f}, valid score: {valid_score:.4f}")

    # Save best model as an artifact, load model, and delete temporary file
    ex.add_artifact(model_tmp_file.name, "model.pth")
    model.load_state_dict(torch.load(model_tmp_file.name))
    model_tmp_file.close()

    # Test and output submission file
    model.eval()
    output_submission_file(model, test_loader)
