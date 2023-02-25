import torch
import torch.nn as nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5) if dropout else nn.Sequential(),
        )
    
    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5) if dropout else nn.Sequential(),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, features: int = 64) -> None:
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        self.down2 = DownsampleBlock(features, features*2, dropout=False)
        self.down3 = DownsampleBlock(features*2, features*4, dropout=False)
        self.down4 = DownsampleBlock(features*4, features*8, dropout=False)
        self.down5 = DownsampleBlock(features*8, features*8, dropout=False)
        self.down6 = DownsampleBlock(features*8, features*8, dropout=False)
        self.down7 = DownsampleBlock(features*8, features*8, dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )

        self.up1 = UpsampleBlock(features*8, features*8, dropout=True)
        self.up2 = UpsampleBlock(features*8*2, features*8, dropout=True)
        self.up3 = UpsampleBlock(features*8*2, features*8, dropout=True)
        self.up4 = UpsampleBlock(features*8*2, features*8, dropout=False)
        self.up5 = UpsampleBlock(features*8*2, features*4, dropout=False)
        self.up6 = UpsampleBlock(features*4*2, features*2, dropout=False)
        self.up7 = UpsampleBlock(features*2*2, features, dropout=False)
        
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bn = self.bottleneck(d7)

        up1 = self.up1(bn)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        up8 = self.up8(torch.cat([up7, d1], dim=1))

        return up8


def test():
    x = torch.randn((2, 3, 256, 256))
    generator = Generator()

    assert generator(x).shape == (2, 3, 256, 256)
    print('Test passed successfully')

if __name__ == '__main__':
    test()
