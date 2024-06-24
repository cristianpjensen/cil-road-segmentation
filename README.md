# Computational Intelligence Lab 2024: Road Segmentation Project

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