"""
Script to convert a submission file to a mask image for the first five test images. Adapted from the
provided code.

Usage: `python submission_to_mask.py <experiment_id>`

"""

import os
import sys
from PIL import Image
import numpy as np
from src.constants import PATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT


def main():
    # Read experiment from command-line
    experiment = sys.argv[1]
    for i in range(144, 149):
        reconstruct_from_labels(os.path.join("experiments", experiment, "submission.csv"), i)


def binary_to_uint8(img):
    return (img * 255).round().astype(np.uint8)


def reconstruct_from_labels(path, image_id):
    im = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.uint8)
    f = open(path)
    lines = f.readlines()
    image_id_str = f"{image_id:03d}_"
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(",")
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split("_")
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+PATCH_SIZE, IMAGE_WIDTH)
        ie = min(i+PATCH_SIZE, IMAGE_HEIGHT)
        if prediction == 0:
            adata = np.zeros((PATCH_SIZE, PATCH_SIZE))
        else:
            adata = np.ones((PATCH_SIZE, PATCH_SIZE))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save(f"submission_{image_id:03d}.png")


if __name__ == "__main__":
    main()
