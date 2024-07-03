import os
import tempfile
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
import sacred
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm

from src.models.create_model import create_model
from src.dataset import ImageSegmentationDataset
from src.constants import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
from src.evaluation import (
    patch_f1_score,
    patch_accuracy,
    pixel_accuracy,
    get_pixel_pred_dir,
    get_patch_overlay_dir,
    output_pixel_pred,
    output_mask_overlay,
    output_submission_file,
)

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
    batch_size = 4
    lr = 1e-3
    is_pbar = True
    is_early_stopping = True
    early_stopping_config = {
        "patience": 50,
        "min_delta": 1e-4,
    }
    output_images_every = 10
    val_size = 10
    deterministic_flip = True
    early_stopping_key = "valid_patch_acc"


@ex.capture
def get_iterator(iterator, is_pbar, **kwargs):
    return tqdm(iterator, **kwargs) if is_pbar else iterator



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
    val_size: int,
    deterministic_flip: bool,
    early_stopping_key: str,
):
    print(f"Device: {DEVICE}")

    data = ImageSegmentationDataset(
        "data/training/images",
        "data/training/groundtruth",
        normalize=True,
        size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    train_data, valid_data = random_split(data, [len(data) - val_size, val_size])
    test_data = ImageSegmentationDataset(
        "data/test/images",
        normalize=(data.channel_means, data.channel_stds),
        size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = create_model(model_name, { "pos_weight": data.pos_weight() })
    model.to(DEVICE)
    print(f"Model '{model_name}' created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_score = -float("inf")
    no_improvement = 0
    # Create temporary file to save the best model while training
    model_tmp_file = tempfile.NamedTemporaryFile()

    metrics = {
        "train_loss": 0,
        "valid_f1": 0,
        "valid_patch_acc": 0,
        "valid_pixel_acc": 0,
    }

    pbar = get_iterator(range(epochs))
    for epoch in pbar:
        # Check for early stopping
        if no_improvement > early_stopping_config["patience"]:
            break

        # Reset metrics
        metrics = { k: 0 for k in metrics }

        # Training
        model.train()
        for (input_BCHW, _, target_BHW, _) in get_iterator(train_loader, leave=False):
            model.zero_grad()

            # TODO: Re-implement as in https://arxiv.org/pdf/2404.00498
            if deterministic_flip:
                match epoch % 4:
                    case 1:
                        input_BCHW = TF.hflip(input_BCHW)
                        target_BHW = TF.hflip(target_BHW)
                    case 2:
                        input_BCHW = TF.vflip(input_BCHW)
                        target_BHW = TF.vflip(target_BHW)
                    case 3:
                        input_BCHW = TF.hflip(input_BCHW)
                        input_BCHW = TF.vflip(input_BCHW)
                        target_BHW = TF.hflip(target_BHW)
                        target_BHW = TF.vflip(target_BHW)

            # Forward pass
            input_BCHW, target_BHW = input_BCHW.to(DEVICE), target_BHW.to(DEVICE)
            pred_BHW = model.step(input_BCHW)
            loss = model.loss(pred_BHW, target_BHW.to(DEVICE))

            # Compute gradient and update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics["train_loss"] += loss.item() * input_BCHW.shape[0]

        # Predict validation images every `output_images_every`` epochs and save the images as artifacts
        if epoch % output_images_every == 0:
            os.makedirs(get_pixel_pred_dir(observer.dir, epoch))
            os.makedirs(get_patch_overlay_dir(observer.dir, epoch))
            image_count = 0

        # Validation
        model.eval()
        with torch.no_grad():
            for (input_BCHW, input_files, target_BHW, _) in get_iterator(valid_loader, leave=False):
                input_BCHW, target_BHW = input_BCHW.to(DEVICE), target_BHW.to(DEVICE)
                pred_BHW = model.predict(input_BCHW)

                # Compute metrics and keep track
                metrics["valid_f1"] += patch_f1_score(pred_BHW, target_BHW).item() * input_BCHW.shape[0]
                metrics["valid_patch_acc"] += patch_accuracy(pred_BHW, target_BHW).item() * input_BCHW.shape[0]
                metrics["valid_pixel_acc"] += pixel_accuracy(pred_BHW, target_BHW).item() * input_BCHW.shape[0]

                # Output images
                if epoch % output_images_every == 0:
                    input_BCHW, pred_BHW = input_BCHW.cpu(), pred_BHW.cpu()
                    output_pixel_pred(ex, observer.dir, epoch, input_files, pred_BHW)
                    output_mask_overlay(ex, observer.dir, epoch, input_files, data.denormalize(input_BCHW), pred_BHW)
                    image_count += input_BCHW.shape[0]

        # Normalize metrics
        for k in metrics:
            if "valid" in k:
                metrics[k] /= len(valid_data)
            if "train" in k:
                metrics[k] /= len(train_data)

        # Early stopping
        if is_early_stopping:
            if metrics[early_stopping_key] - best_valid_score > early_stopping_config["min_delta"]:
                no_improvement = 0
            else:
                no_improvement += 1

        # Save best model based on validation loss
        if metrics[early_stopping_key] > best_valid_score:
            best_valid_score = metrics[early_stopping_key]
            torch.save(model.state_dict(), model_tmp_file.name)

        # Log metrics
        for k, v in metrics.items():
            ex.log_scalar(k, v)

        if is_pbar:
            pbar.set_description(", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    # Save best model as an artifact, load model, and delete temporary file
    ex.add_artifact(model_tmp_file.name, "model.pth")
    model.load_state_dict(torch.load(model_tmp_file.name))
    model_tmp_file.close()

    # Test and output submission file
    model.eval()
    with torch.no_grad():
        output_submission_file(ex, observer.dir, model, test_loader)
