import os
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision.transforms.functional as TF


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
        size: tuple[int, int]=(400, 400),
        transform=lambda x: x,
        target_transform=lambda x: x,
    ):
        self.transform = transform
        self.target_transform = target_transform

        # Save data in memory for faster access (dataset is small)
        self.file_names = self._get_image_files(input_dir)
        self.input_NCHW = self._get_images(input_dir, self.file_names, is_target=False).float() / 255
        if target_dir is not None:
            self.target_file_names = self._get_image_files(target_dir)
            self.target_NHW = self._get_images(target_dir, self.target_file_names, is_target=True).float() / 255
        else:
            self.target_file_names = None
            self.target_NHW = None

        # Set normalization parameters
        match normalize:
            case True:
                self.channel_means = torch.mean(self.input_NCHW, dim=[0, 2, 3], keepdim=True)
                self.channel_stds = torch.std(self.input_NCHW, dim=[0, 2, 3], keepdim=True)
            
            case (channel_means, channel_stds):
                self.channel_means = channel_means
                self.channel_stds = channel_stds

        # Normalize input images
        self.input_NCHW = self.normalize(self.input_NCHW)

        # Resize
        self.input_NCHW = TF.resize(self.input_NCHW, size)
        if self.target_NHW is not None:
            self.target_NHW = TF.resize(self.target_NHW, size)
    
    def normalize(self, x):
        return (x - self.channel_means) / self.channel_stds

    def denormalize(self, x):
        return x * self.channel_stds + self.channel_means

    def _get_image_files(self, dir: str):
        return sorted([path.split("/")[-1] for path in glob(os.path.join(dir, "*"))])

    def _get_images(self, dir: str, file_names: str, is_target=False):
        image_tensors = []
        for file_name in file_names:
            im = read_image(os.path.join(dir, file_name), ImageReadMode.GRAY if is_target else ImageReadMode.RGB)
            im = im.unsqueeze(0)
            image_tensors.append(im)

        return torch.cat(image_tensors, dim=0).float()

    def __getitem__(self, item):
        if self.target_NHW is not None:
            return (
                self.transform(self.input_NCHW[item]),
                self.file_names[item],
                self.target_transform(self.target_NHW[item]),
                self.target_file_names[item],
            )

        return self.transform(self.input_NCHW[item]), self.file_names[item]

    def __len__(self):
        return self.input_NCHW.shape[0]

    def pos_weight(self):
        """Negative / positive occurrence ratio."""
        return (1 - self.target_NHW).sum() / self.target_NHW.sum()
