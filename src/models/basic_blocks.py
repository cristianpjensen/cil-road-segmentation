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
        self.init_weights()

    def forward(self, x):
        return self.block(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.act = act_fn()
        self.init_weights()

    def forward(self, x):
        return self.act(self.block(x) + self.conv_skip(x))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
        self.init_weights()

    def forward(self, x):
        return self.act(self.block(x) + self.conv_skip(x))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
        self.init_weights()

    def forward(self, x):
        return self.block(x) + self.conv_skip(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    import torch
    from ..constants import IMAGE_HEIGHT, IMAGE_WIDTH

    x = torch.randn(4, 64, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f"Input mean, std: {x.mean():.3f}, {x.std():.3f}")
    for block in [ConvBlock, Res18Block, Res50Block, ResV2Block]:
        net = nn.Sequential(*[block(64, 64, nn.SiLU) for _ in range(3)])
        y = net(x)
        print(f"{block.__name__} mean, std: {y.mean():.3f}, {y.std():.3f}")
