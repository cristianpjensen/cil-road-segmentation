import os
import tempfile
import torch
import sacred
from collections import defaultdict
from tqdm import tqdm
from millify import millify
from torch.utils.data import DataLoader, random_split
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import Experiment
from sacred.observers import FileStorageObserver

from src.augmentation import alternating_transforms, compose_transforms
from src.models.base import BaseModel
from src.models.create_model import create_model
from src.dataset import ImageSegmentationDataset, denormalize
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
    final_data_dir = "data"
    pretrain_data_dir = None
    model_name = "dummy" # "dummy", "unet", "neighbor_unet", "unet++", "pix2pix"
    model_config = {
        # Configuration specific to U-nets
        "activation": "relu", # "relu", "gelu", "silu"
        "block": "conv", # "conv", "res18", "res50", "resv2"
        "blocks_per_layer": 1,
        "channels": [64, 128, 256, 512, 1024],
        "bottleneck_mhsa_layers": 0,
        "num_heads": 8,

        # Configuration specific to model_name="neighbor_unet"
        "neighbor_unet": {
            "neighbor_kernel_size": 3,
            "neighbor_loss_weight": 0.1,
        },

        # Configuration specific to model_name="unet++"
        "unetplusplus": {
            "deep_supervision": True,
        },

        # Configuration specific to model_name="pix2pix"
        "pix2pix": {
            "l1_weight": 100,
            "patch_discriminator": False,
        },
    }
    loss = "bce" # "bce", "dice", "bce+dice"
    epochs = 10000
    batch_size = 8
    lr = 1e-3
    is_pbar = True
    is_early_stopping = True
    early_stopping_config = {
        "patience": 100,
        "min_delta": 1e-4,
        "val_size": 16,
        "key": "valid_f1",
        "mode": "max",
    }
    pretraining_early_stopping_config = {
        "patience": 20,
        "min_delta": 1e-4,
        "val_size": 128,
        "key": "valid_f1",
        "mode": "max",
    }
    output_images_every = 10
    transforms = "hrv" # If contains "v", then vertical flip, if contains "h", then horizontal flip, and if contains "r", then rotates
    predict_patches = False


@ex.capture
def get_iterator(iterator, is_pbar, **kwargs):
    return tqdm(iterator, **kwargs) if is_pbar else iterator


def train(
    model: BaseModel,
    name: str,
    data: torch.utils.data.Dataset,
    valid_size: int,
    batch_size: int,
    transforms: list[str],
    seed: int,
    output_val_images_every: int=10,
    epochs: int=1000,
    patience: int=50,
    min_delta: float=1e-4,
    early_stopping_key: str="valid_patch_acc",
    early_stopping_mode: str="max",
    predict_patches: bool=False,
    is_pbar: bool=True,
):
    """Training loop with early stopping and data augmentation. Furthermore, we allow for predicting
    patches. Outputs the final model weights with file name `model_<name>.pt` in the experiment
    directory."""

    # Create temporary file to save the best model while training
    model_tmp_file = tempfile.NamedTemporaryFile()

    # Data loaders
    train_data, valid_data = random_split(data, [len(data) - valid_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    
    # Data augmentation
    transformations = compose_transforms(transforms)

    # Early stopping variables
    best_valid_score = -float("inf")
    no_improvement = 0

    pbar = get_iterator(range(epochs))
    for epoch in pbar:
        # Check for early stopping
        if no_improvement > patience:
            break

        # Reset metrics
        metrics = defaultdict(float)

        # Training
        model.train()
        for i, (input_BCHW, input_files, target_B1HW, _) in enumerate(get_iterator(train_loader, leave=False)):
            # `altflip` generalized to vertical and horizontal flips, and rotations
            # (https://arxiv.org/pdf/2404.00498). The transformation depends on the epoch and the
            # file name, thus it is different for each image, but every N epochs, all transformed
            # versions will have been seen, where N is the number of transformation combinations
            input_BCHW = alternating_transforms(input_BCHW, input_files, transformations, epoch, seed)
            target_B1HW = alternating_transforms(target_B1HW, input_files, transformations, epoch, seed)

            # Forward pass
            input_BCHW, target_B1HW = input_BCHW.to(DEVICE), target_B1HW.to(DEVICE)
            loss_log = model.training_step(input_BCHW, target_B1HW.squeeze(1))

            for k, v in loss_log.items():
                metrics["train_" + k] += v * input_BCHW.shape[0]

        # Predict validation images every `output_images_every`` epochs and save the images as artifacts
        if epoch % output_val_images_every == 0:
            os.makedirs(get_pixel_pred_dir(observer.dir, name, epoch))
            os.makedirs(get_patch_overlay_dir(observer.dir, name, epoch))
            image_count = 0

        # Validation
        model.eval()
        with torch.no_grad():
            for (input_BCHW, input_files, target_B1HW, _) in get_iterator(valid_loader, leave=False):
                input_BCHW, target_B1HW = input_BCHW.to(DEVICE), target_B1HW.to(DEVICE)
                target_BHW = target_B1HW.squeeze(1)
                pred_BHW = model.predict(input_BCHW)

                # Compute metrics and keep track
                metrics["valid_f1"] += patch_f1_score(pred_BHW, target_BHW, is_patches=predict_patches).item() * input_BCHW.shape[0]
                metrics["valid_patch_acc"] += patch_accuracy(pred_BHW, target_BHW, is_patches=predict_patches).item() * input_BCHW.shape[0]
                metrics["valid_pixel_acc"] += pixel_accuracy(pred_BHW, target_BHW).item() * input_BCHW.shape[0]

                # Output images
                if epoch % output_val_images_every == 0:
                    input_BCHW, pred_BHW = input_BCHW.cpu(), pred_BHW.cpu()
                    output_pixel_pred(ex, observer.dir, name, epoch, input_files, pred_BHW, is_patches=predict_patches)
                    output_mask_overlay(ex, observer.dir, name, epoch, input_files, denormalize(input_BCHW), pred_BHW, is_patches=predict_patches)
                    image_count += input_BCHW.shape[0]

        # Normalize metrics
        for k in metrics:
            if "valid" in k:
                metrics[k] /= len(valid_data)
            if "train" in k:
                metrics[k] /= len(train_data)

        if early_stopping_mode == "min":
            early_stopping_metric = -metrics[early_stopping_key]
        else:
            early_stopping_metric = metrics[early_stopping_key]

        # Early stopping
        if early_stopping_metric - best_valid_score > min_delta:
            no_improvement = 0
        else:
            no_improvement += 1

        # Save best model based on validation loss
        if early_stopping_metric > best_valid_score:
            best_valid_score = early_stopping_metric
            model.save(model_tmp_file.name)

        # Log metrics and update pbar with them
        for k, v in metrics.items():
            ex.log_scalar(k, v)

        if is_pbar:
            pbar.set_description(f"{name} -- no_improvement: {no_improvement}, " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    ex.add_artifact(model_tmp_file.name, f"model_{name}.pt")


@ex.automain
def main(
    seed: int,
    final_data_dir: str,
    pretrain_data_dir: str | None,
    model_name: str,
    model_config: dict,
    loss: str,
    epochs: int,
    batch_size: int,
    lr: float,
    is_pbar: bool,
    is_early_stopping: bool,
    early_stopping_config: dict,
    pretraining_early_stopping_config: dict,
    output_images_every: int,
    transforms: str,
    predict_patches: bool,
):
    # Config parsing
    if "".join(sorted(transforms)) not in ["", "h", "r", "v", "hr", "hv", "rv", "hrv"]:
        raise ValueError("Transforms should be a combination of 'v', 'h', and 'r'.")

    if loss not in ["bce", "dice", "bce+dice"]:
        raise ValueError("Loss should be 'bce', 'dice', or 'bce+dice'.")

    if model_name not in ["dummy", "unet", "neighbor_unet", "unet++", "pix2pix"]:
        raise ValueError("Model name should be 'dummy', 'unet', 'neighbor_unet', 'unet++', or 'pix2pix'.")
    
    if model_name == "pix2pix" and loss != "bce":
        raise ValueError("Changing loss for pix2pix is not supported.")

    if model_name == "unet++" and model_config["bottleneck_mhsa_layers"] > 0:
        raise ValueError("U-Net++ does not support bottleneck multi-head self-attention layers.")

    print(f"Device: {DEVICE}")

    model = create_model(
        model_name,
        {
            **model_config,
            "predict_patches": predict_patches,
            "lr": lr,
            "loss": loss,
        },
    )
    model.to_device(DEVICE)
    print(f"Model '{model_name}' created with {millify(model.num_params(), precision=1)} trainable parameters.")

    # Pretraining
    if pretrain_data_dir is not None:
        pretrain_data = ImageSegmentationDataset(
            os.path.join(pretrain_data_dir, "images"),
            os.path.join(pretrain_data_dir, "groundtruth"),
            target_is_patches=predict_patches,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        train(
            model,
            "pretraining",
            pretrain_data,
            pretraining_early_stopping_config["val_size"],
            batch_size,
            transforms,
            seed,
            output_val_images_every=output_images_every,
            epochs=epochs,
            patience=pretraining_early_stopping_config["patience"] if is_early_stopping else epochs + 1,
            min_delta=pretraining_early_stopping_config["min_delta"],
            early_stopping_key=pretraining_early_stopping_config["key"],
            early_stopping_mode=pretraining_early_stopping_config["mode"],
            predict_patches=predict_patches,
            is_pbar=is_pbar,
        )

        # Load best model from pretraining
        model.load(os.path.join(observer.dir, "model_pretraining.pt"))

        # Divide learning rate by 10 after pretraining (do not if no pretraining)
        model.set_optimizer_lr(lr / 10)

    final_data = ImageSegmentationDataset(
        os.path.join(final_data_dir, "training", "images"),
        os.path.join(final_data_dir, "training", "groundtruth"),
        target_is_patches=predict_patches,
        size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    # Finetuning
    train(
        model,
        "finetuning",
        final_data,
        early_stopping_config["val_size"],
        batch_size,
        transforms,
        seed,
        output_val_images_every=output_images_every,
        epochs=epochs,
        patience=early_stopping_config["patience"] if is_early_stopping else epochs + 1,
        min_delta=early_stopping_config["min_delta"],
        early_stopping_key=early_stopping_config["key"],
        early_stopping_mode=early_stopping_config["mode"],
        predict_patches=predict_patches,
        is_pbar=is_pbar,
    )

    # Load best model
    model.load(os.path.join(observer.dir, "model_finetuning.pt"))

    # Test and output submission file
    model.eval()
    with torch.no_grad():
        test_data = ImageSegmentationDataset(
            os.path.join(final_data_dir, "test", "images"),
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        output_submission_file(ex, observer.dir, model, test_loader, predict_patches)
