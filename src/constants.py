import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOREGROUND_THRESHOLD = 0.25
PATCH_SIZE = 16
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

# Mean and std computed over the scraped Google maps dataset and the provided dataset
CHANNEL_MEANS = torch.tensor([0.4731, 0.4785, 0.4654]).view(1, 3, 1, 1)
CHANNEL_STDS = torch.tensor([0.2268, 0.2076, 0.1975]).view(1, 3, 1, 1)
