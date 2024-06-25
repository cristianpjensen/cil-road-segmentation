import os
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision.io import read_image
from torchvision.io import ImageReadMode

from constants import DEVICE


class ImageSegmentationDataset(Dataset):
    """Dataset for image segmentation tasks.

    Args:
        input_dir: Directory containing input images.
        target_dir: Directory containing target images.
        normalize: Normalize input images. If True, normalize using mean and std of input images.
            If a tuple, normalize using the provided mean and std.
        transform: Transform to apply to input images.
        target_transform: Transform to apply to target images.

    """

    def __init__(
            self,
            input_dir: str,
            target_dir=None,
            normalize: bool | tuple[torch.Tensor, torch.Tensor] = False,
            transform=lambda x: x,
            target_transform=lambda x: x,
        ):

        # Save data in memory for faster access (dataset is small)
        self.input_NCHW = self._get_sorted_images(input_dir, is_target=False).float()
        if target_dir is not None:
            self.target_NHW = self._get_sorted_images(target_dir, is_target=True) / 255
        else:
            self.target_NHW = None

        # Normalize input images
        match normalize:
            case True:
                self.channel_means = torch.mean(self.input_NCHW, dim=[0, 2, 3], keepdim=True)
                self.channel_stds = torch.std(self.input_NCHW, dim=[0, 2, 3], keepdim=True)
            
            case (channel_means, channel_stds):
                self.channel_means = channel_means
                self.channel_stds = channel_stds
        
        self.input_NCHW = self.normalize(self.input_NCHW)

        # Transform immediately to save on redundant computation
        self.input_NCHW = transform(self.input_NCHW)
        if self.target_NHW is not None:
            self.target_NHW = target_transform(self.target_NHW)
    
    def normalize(self, x):
        return (x - self.channel_means) / self.channel_stds

    def denormalize(self, x):
        return x * self.channel_stds + self.channel_means

    def _get_sorted_images(self, dir: str, is_target=False):
        image_tensors = []
        for path in sorted(glob(os.path.join(dir, "*"))):
            im = read_image(path, ImageReadMode.GRAY if is_target else ImageReadMode.RGB)
            if not is_target:
                im = im.unsqueeze(0)
            image_tensors.append(im)

        return torch.cat(image_tensors, dim=0).float()

    def __getitem__(self, item):
        input_CHW = self.input_NCHW[item]
        if self.target_NHW is not None:
            target_HW = self.target_NHW[item]
            return input_CHW, target_HW

        return input_CHW, None

    def __len__(self):
        return self.input_NCHW.shape[0]

    def pos_weight(self):
        """Positive / negative occurrence ratio."""
        return (1 - self.target_NHW).sum() / self.target_NHW.sum()
