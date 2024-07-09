import cv2
import numpy as np
import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

kaggle_api = KaggleApi()
kaggle_api.authenticate()

def main():

    # Set paths
    home_dir = os.path.expanduser('~')
    project_path= home_dir + "/cil-road-segmentation"
    data_path = project_path + "/data"

    # Make directories if they don't already exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(data_path + "/massachusetts", exist_ok=True)
    os.makedirs(data_path + "/massachusetts/images", exist_ok=True)
    os.makedirs(data_path + "/massachusetts/groundtruth", exist_ok=True)

    # Download dataset
    if not os.path.exists(data_path + '/massachusetts-roads-dataset.zip'):
        kaggle_api.dataset_download_files('balraj98/massachusetts-roads-dataset', path=data_path)

    # Unzip dataset
    if not os.path.exists(data_path + '/tiff'):
        z = zipfile.ZipFile(file=data_path + '/massachusetts-roads-dataset.zip')
        z.extractall(path=data_path)

    # Remove unnecessary files
    if os.path.exists(data_path + "/label_class_dict.csv"):
        os.remove(data_path + "/label_class_dict.csv")
    if os.path.exists(data_path + "/metadata.csv"):
        os.remove(data_path + "/metadata.csv")

    # Delete test and validation images
    shutil.rmtree(os.path.join(data_path, "tiff/test"))
    shutil.rmtree(os.path.join(data_path, "tiff/test_labels"))
    shutil.rmtree(os.path.join(data_path, "tiff/val"))
    shutil.rmtree(os.path.join(data_path, "tiff/val_labels"))

    # Remove images with too many white pixels
    for image_name in os.listdir(data_path + "/tiff/train"):
        image = cv2.imread(data_path + "/tiff/train/" + image_name, cv2.IMREAD_UNCHANGED)
        white_pixels = np.sum(np.all(image == [255, 255, 255], axis=-1))
        if white_pixels > 100:
            os.remove(os.path.join(data_path, "tiff/train/" + image_name))
            os.remove(os.path.join(data_path, "tiff/train_labels/" + image_name[:-1]))

    # Split images
    # Filter out images with less than 10% road pixels
    # Save images
    num_splits = 5
    for image_name in os.listdir(data_path + "/tiff/train"):
        if image_name == ".DS_Store":
            continue
        image = cv2.imread(data_path + "/tiff/train/" + image_name, cv2.IMREAD_UNCHANGED)
        groundtruth = cv2.imread(data_path + "/tiff/train_labels/" + image_name[:-1], cv2.IMREAD_UNCHANGED)

        h, w, _ = image.shape
        split_h = h // num_splits
        split_w = w // num_splits

        # Split images into 5x5 grid
        images = [image[i * split_h:(i + 1) * split_h, j * split_w:(j + 1) * split_w] for i in range(num_splits) for j in range(num_splits)]
        groundtruths = [groundtruth[i * split_h:(i + 1) * split_h, j * split_w:(j + 1) * split_w] for i in range(num_splits) for j in range(num_splits)]
        
        i = 0
        for image, groundtruth in zip(images, groundtruths):

            # Skip images with less than 10% road pixels
            if np.sum(groundtruth == 255) / groundtruth.size < 0.1:
                continue

            # Resize images
            image_resized = cv2.resize(image, (400, 400), interpolation=cv2.INTER_CUBIC)
            groundtruth_resized = cv2.resize(groundtruth, (400, 400), interpolation=cv2.INTER_NEAREST)

            # Save images
            cv2.imwrite(data_path + "/massachusetts/images/" + image_name.replace(".tiff","") + f"_{i}.png", image_resized)
            cv2.imwrite(data_path + "/massachusetts/groundtruth/" + image_name.replace(".tif","") + f"_{i}.png", groundtruth_resized)
            i += 1

if __name__ == '__main__':
    main()