# Computational Intelligence Lab 2024: Road Segmentation Project

## Creating models

To create a new model, copy `models/dummy.py` and adjust the model to your needs. You can change initialization, where you initialize the architecture. You can also change the forward pass and loss computation. The `step` method is used for training and should not output the probabilities, but the logits. This is because it is more numerically stable. The `predict` method should output the probabilities and is used for inference. The loss function can also be changed in the `loss` method.

When the model is ready, make sure to add it to the match-case in `models/create_model.py`, such that it can be used in further scripts.

## Training models

To train a model, use the `train.py` script. You can specify the model, batch size, learning rate (ADAM optimizer), number of epochs, etc. See the `config` function for all available options. The training loop has been implemented with early stopping. Training and validation is run every epoch, which updates the progress bar. It reports the loss on the training data and the F1 score on the validation data (same metric as in the Kaggle competition). These values are also stored in the experiment directory `experiments/metrics.json`. The best model is saved, based on the validation performance.

Every `output_images_every` epoch, the script outputs the pixel-wise and patch-wise predictions on the validation data. These can be found under the experiment directory in `experiments/<experiment id>/validation`.

## Mask from submissions

The `mask_from_submission.py` script can be used to create a mask from a submission file for the first 5 test images. The masks are saved as `.png` files.

## Installation

### Packages

```
conda create --name cil-project python=3.11
conda activate cil-project
pip install torch torchvision
pip install -r requirements.txt
```

### Data

Download the data and put it in a `data/`-directory.

### Submissions

Make sure to create a `submissions/`-directory where all submissions will be stored.

## Variable naming

In general, inputs are named `input`, targets are named `target`, and predictions are named `pred`.

### Dimension suffixes

Tensors are named with a suffix that indicates the dimensions that they have. The suffixes are as
follows (add as new dimension types are needed):
 - `B`: Batch dimension
 - `C`: Channel dimension
 - `H`: Height dimension
 - `W`: Width dimension
 - `D`: `H x W` dimension (e.g. when the input is a 2D image, but the model expects a 1D input)
 - `N`: Total number of data points

This makes it much easier to work with the variables, because you always know how they are indexed,
which (hopefully) prevents bugs.