import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: nn.Module=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn(),
        )

    def forward(self, x):
        return self.block(x)


class Res18Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: nn.Module=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.act = act_fn()

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.act(self.block(x) + self.conv_skip(x))


class Res50Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: nn.Module=nn.ReLU):
        super().__init__()

        bottleneck = in_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm2d(bottleneck),
            act_fn(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck),
            act_fn(),
            nn.Conv2d(bottleneck, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.act = act_fn()

    def forward(self, x):
        return self.act(self.block(x) + self.conv_skip(x))


class ResV2Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_fn: nn.Module=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            act_fn(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.conv_skip(x)
