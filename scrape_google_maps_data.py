import os
import requests
import tempfile
import argparse
import torchvision
import torchvision.transforms.functional as TF
import torch
from dotenv import load_dotenv


load_dotenv()


def scrape_google_maps_image_and_segmentation(lat: float, lon: float, dir: str):
    satellite_url = get_satellite_url(lat, lon)
    satellite_img = fetch_image(satellite_url)

    segmentation_url = get_segmentation_url(lat, lon)
    segmentation_img = fetch_image(segmentation_url)

    # Make sure both images can be saved
    if satellite_img is None or segmentation_img is None:
        return

    torchvision.io.write_png(satellite_img, os.path.join(dir, "images", f"satimage_{lat:.3f},{lon:.3f}.png"))

    # Round to 0 or 1 and save
    segmentation_img = (segmentation_img.float() / 255).round()
    segmentation_img = (segmentation_img * 255).byte()
    torchvision.io.write_png(segmentation_img, os.path.join(dir, "groundtruth", f"satimage_{lat:.3f},{lon:.3f}.png"))


def get_segmentation_url(lat: float, lon: float) -> str:
    key = os.environ["GOOGLE_API_KEY"]
    return f'https://maps.googleapis.com/maps/api/staticmap?key={key}&center={lat:.3f},{lon:.3f}&zoom=18&size=400x422&style=feature:all|color:0x000000&style=feature:road|element:geometry.fill|color:0xFFFFFF&style=feature:road|element:geometry.stroke|color:0xFFFFFF|weight:0.5&style=feature:all|element:labels|visibility:off'


def get_satellite_url(lat: float, lon: float) -> str:
    key = os.environ["GOOGLE_API_KEY"]
    return f'https://maps.googleapis.com/maps/api/staticmap?key={key}&center={lat:.3f},{lon:.3f}&zoom=18&size=400x422&maptype=satellite'


def fetch_image(url: str) -> torch.Tensor | None:
    try:
        img_data = requests.get(url).content
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(img_data)
            img = torchvision.io.read_image(tmp_file.name, torchvision.io.ImageReadMode.RGB)

        # Remove Google branding
        img = img[:, :-22, :]
        return TF.resize(img, [400, 400], interpolation=TF.InterpolationMode.NEAREST)

    except RuntimeError:
        print("Failed to fetch image from URL:", url)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coords_file", help=".txt file containing coordinates in <lat>, <lon> format.", type=str)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scraped_data",
        help="Directory to save scraped data. Default: 'scraped_data'.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "groundtruth"), exist_ok=True)

    with open(args.coords_file, "r") as f:
        for line in f:
            lat, lon = map(float, line.strip().split(","))
            print(f"Scraping: ({lat}, {lon})")
            scrape_google_maps_image_and_segmentation(lat, lon, args.output_dir)
 