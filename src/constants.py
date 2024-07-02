import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOREGROUND_THRESHOLD = 0.25
PATCH_SIZE = 16
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400
