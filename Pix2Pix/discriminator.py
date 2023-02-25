import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: list = [64, 128, 256, 512]) -> None:
        super().__init__()
        layers = []

        initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        layers.append(initial)
        
        for i in range(1, len(features)):
            layers.append(ConvBlock(
                features[i-1],
                features[i],
                stride=(1 if i == len(features) - 1 else 2)
            ))
        
        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        
        self.model = nn.Sequential(
            *layers,
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)


def test():
    x = torch.randn((2, 3, 256, 256))
    y = torch.randn((2, 3, 256, 256))

    discriminator = Discriminator()

    assert discriminator(x, y).shape == (2, 1, 26, 26)
    print('Test passed successfully')

if __name__ == '__main__':
    test()
