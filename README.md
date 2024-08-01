# Computational Intelligence Lab 2024: Road Segmentation Project

## Creating models

To create a new model, copy `src/models/dummy.py` and adjust the model to your needs. You can change initialization, where you initialize the architecture. You can also change the forward pass and backward pass. The `training_step` method should perform the backward pass and should output training losses. The `predict` method should be used for validation and testing.

When the model is ready, make sure to add it to the match-case in `src/models/create_model.py`, such that it can be used in further scripts.

## Training models

To train a model, use the `train.py` script. You can specify the model, batch size, learning rat , number of epochs, etc. This project uses SACRED, which means that hyperparameters can be set by 
```shell
python train.py with model="unet" batch_size=8 model_config.block="resv2"
```
See the `config()` function in `train.py` for a list of all configurations.

The training loop has been implemented with early stopping. Training and validation is run every epoch, which updates the progress bar. It reports the loss on the training data and the F1 score on the validation data. These values are also stored in the experiment directory `experiments/metrics.json`. The best model is saved, based on the validation performance.

Every `output_images_every` epoch, the script outputs the pixel-wise and patch-wise predictions on the validation data. These can be found under the experiment directory in `experiments/<experiment id>/validation`.

## Google Maps dataset

The Google Maps dataset can be fetched using the `scrape_google_maps_data.py` script and the `scraped_coords.txt` file. It is used by 
```shell
python scrape_google_maps_data.py scraped_coords.txt --output_dir scraped_data_scale18
```
This script requires a Google Maps API key that is stored in a `.env` file by
```
GOOGLE_API_KEY=<your key>
```
A ready-to-use version can be downloaded from [Google Drive](https://drive.google.com/file/d/1sVUp_ed1rV1ei5S8n715jCmu6DoliUeG/view?usp=drive_link).

You can pre-train on this data by setting the `pretrain_data_dir` variable in the `train.py` script.

## Mask from submissions

The `mask_from_submission.py` script can be used to create a mask from a submission file for the first 5 test images.

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