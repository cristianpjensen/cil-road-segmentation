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

from models.create_model import create_model
from dataset import ImageSegmentationDataset
from evaluation import eval_f1_score
from constants import DEVICE

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
    output_images_every = 50


@ex.capture
def get_iterator(iterator, is_pbar, **kwargs):
    return tqdm(iterator, **kwargs) if is_pbar else iterator


def save_data(loader, denormalize, path):
    os.makedirs(os.path.join(observer.dir, path, "input"))
    os.makedirs(os.path.join(observer.dir, path, "target"))
    image_count = 0
    for (input_, target) in loader:
        input_ = denormalize(input_).byte()
        target = (target.unsqueeze(1) * 255).byte()

        for i in range(input_.shape[0]):
            with tempfile.NamedTemporaryFile() as tmp_file:
                write_png(input_[i], tmp_file.name)
                ex.add_artifact(tmp_file.name, os.path.join(path, "input", f"{image_count+i:04d}.png"))

            with tempfile.NamedTemporaryFile() as tmp_file:
                write_png(target[i], tmp_file.name)
                ex.add_artifact(tmp_file.name, os.path.join(path, "target", f"{image_count+i:04d}.png"))

        image_count += input_.shape[0]


@ex.automain
def main(model_name, epochs, batch_size, lr, is_pbar, is_early_stopping, early_stopping_config, output_images_every):
    transform = T.Compose([
        T.Resize((400, 400)),
    ])

    data = ImageSegmentationDataset("data/training/images", "data/training/groundtruth", normalize=True, transform=transform)
    train_data, valid_data = random_split(data, [0.8, 0.2])
    test_data = ImageSegmentationDataset("data/test/images", normalize=(data.channel_means, data.channel_stds), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Save validation data for analysis later
    save_data(valid_loader, data.denormalize, "valid_data")

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
        for (input_, target) in get_iterator(train_loader, leave=False):
            model.zero_grad()

            # Forward pass
            input_, target = input_.to(DEVICE), target.to(DEVICE)
            pred = model.step(input_)
            loss = model.loss(pred, target.to(DEVICE))

            # Compute gradient and update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item() * input_.shape[0]

        # Predict validation images every `output_images_every`` epochs and save the images as artifacts
        if epoch % output_images_every == 0:
            os.makedirs(os.path.join(observer.dir, f"epoch_{epoch}", "valid_outputs"))
            image_count = 0

        # Validation
        model.eval()
        total_valid_score = 0
        for (input_, target) in get_iterator(valid_loader, leave=False):
            input_, target = input_.to(DEVICE), target.to(DEVICE)
            pred = model.predict(input_)
            score = eval_f1_score(pred, target)
            total_valid_score += score.item() * input_.shape[0]

            if epoch % output_images_every == 0:
                pred = (pred.unsqueeze(1) * 255).byte().cpu()
                for i, pred_img in enumerate(pred):
                    with tempfile.NamedTemporaryFile() as tmp_file:
                        write_png(pred_img, tmp_file.name)
                        ex.add_artifact(tmp_file.name, os.path.join(f"epoch_{epoch}", "valid_outputs", f"{image_count+i:04d}.png"))

                image_count += input_.shape[0]

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

    # Test
    model.eval()

    # TODO: Output submission file
