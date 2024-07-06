import os
from PIL import Image
import argparse
import shutil


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", help="rgb satellite images")
    parser.add_argument("--image_mask_dir", help="image classification mask")

    args = parser.parse_args()

    return args


def resize_and_rename(base_dir, image_dirs, sat_img_idx, is_mask):
    base_name = "satimage_"
 
    for path in image_dirs:
        img = Image.open(os.path.join(base_dir, path))

        if is_mask:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img = img.resize((400, 400))

        os.remove(os.path.join(base_dir, path))
        img.save(os.path.join(base_dir, base_name + str(sat_img_idx) + ".png"))

        sat_img_idx += 1


def move_all_imgs(source, destination):
    path_list = os.listdir(source)

    for path in path_list:
        shutil.move(os.path.join(source, path), destination)


def main():
    args = get_args()

    orig_img_dir = "data/training/images/"
    orig_mask_dir = "data/training/groundtruth/"

    num_curr_satellite = len(os.listdir(orig_img_dir))
    num_curr_mask = len(os.listdir(orig_mask_dir))

    new_imgs_rgb = os.listdir(args.image_dir)
    new_imgs_mask = os.listdir(args.image_mask_dir)

    if len(new_imgs_rgb) != len(new_imgs_mask):
        raise Exception("Number of satellite images and mask images does not match")
    
    resize_and_rename(args.image_dir, new_imgs_rgb, num_curr_satellite, False)
    resize_and_rename(args.image_mask_dir, new_imgs_mask, num_curr_mask, True)

    move_all_imgs(args.image_dir, orig_img_dir)
    move_all_imgs(args.image_mask_dir, orig_mask_dir)


if __name__ == "__main__":
    main()

