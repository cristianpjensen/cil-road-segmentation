import os
import requests
import tempfile
import argparse
import torchvision
import torchvision.transforms.functional as TF
import torch
from dotenv import load_dotenv


load_dotenv()


def scrape_google_maps_image_and_segmentation(lat: float, lon: float, dir: str) -> bool:
    segmentation_url = get_segmentation_url(lat, lon)
    segmentation_img = fetch_image(segmentation_url)

    if segmentation_img is None:
        print(" Error: Failed to fetch segmentation image")
        return False

    # Round to 0 or 1 and save as 8-bit PNG
    segmentation_img = (segmentation_img.float() / 255).round()

    # Skip if segmentation image does not contain more than 5% road
    if segmentation_img.mean() < 0.05:
        print(" Error: Not enough road")
        return False

    segmentation_img = (segmentation_img * 255).byte()

    satellite_url = get_satellite_url(lat, lon)
    satellite_img = fetch_image(satellite_url)

    if satellite_img is None:
        print(" Error: Failed to fetch satellite image")
        return False

    torchvision.io.write_png(satellite_img, os.path.join(dir, "images", f"satimage_{lat:.3f},{lon:.3f}.png"))
    torchvision.io.write_png(segmentation_img, os.path.join(dir, "groundtruth", f"satimage_{lat:.3f},{lon:.3f}.png"))

    return True


def get_segmentation_url(lat: float, lon: float) -> str:
    key = os.environ["GOOGLE_API_KEY"]
    return f'https://maps.googleapis.com/maps/api/staticmap?key={key}&center={lat:.3f},{lon:.3f}&zoom=18&size=400x422&scale=1&style=feature:all|color:0x000000&style=feature:road|element:geometry.fill|color:0xFFFFFF&style=feature:road|element:geometry.stroke|color:0xFFFFFF|weight:0.5&style=feature:all|element:labels|visibility:off'


def get_satellite_url(lat: float, lon: float) -> str:
    key = os.environ["GOOGLE_API_KEY"]
    return f'https://maps.googleapis.com/maps/api/staticmap?key={key}&center={lat:.3f},{lon:.3f}&zoom=18&size=400x422&scale=1&maptype=satellite'


def fetch_image(url: str) -> torch.Tensor | None:
    try:
        img_data = requests.get(url).content
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(img_data)
            img = torchvision.io.read_image(tmp_file.name, torchvision.io.ImageReadMode.RGB)

        # Remove Google branding
        img = img[:, :-22, :]
        img = TF.resize(img, [400, 400], interpolation=TF.InterpolationMode.NEAREST)
        return img

    except RuntimeError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "coords_file",
        help=".txt file containing coordinates in <lat>, <lon> format.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scraped_data",
        help="Directory to save scraped data. Default: 'scraped_data'.",
    )
    parser.add_argument(
        "--output_coords_file",
        type=str,
        default="scraped_coords.txt",
        help="File to save coordinates of successfully scraped images. Default: 'scraped_coords.txt'.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "groundtruth"), exist_ok=True)

    with open(args.coords_file, "r") as f:
        lines = f.readlines()

    with open(args.output_coords_file, "w") as f:
        for i, line in enumerate(lines):
            lat, lon = map(float, line.strip().split(","))
            print(f"{i}: ({lat:.3f}, {lon:.3f})")
            if scrape_google_maps_image_and_segmentation(lat, lon, args.output_dir):
                f.write(f"{lat:.3f},{lon:.3f}\n")
 