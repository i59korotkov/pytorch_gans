import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img: int, features_d: int, num_classes: int, image_size: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Conv2d(
                channels_img+1,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, 4, 2, 0),
        )
        self.embed = nn.Embedding(num_classes, image_size * image_size)
    
    def _block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        return self.model(x)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        channels_img: int,
        features_g: int,
        num_classes: int,
        image_size: int,
        embed_size: int
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        self.model = nn.Sequential(
            self._block(z_dim+embed_size , features_g*16, 4, 1, 0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)
    
    def _block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], self.embed_size, 1, 1)
        x = torch.cat([x, embedding], dim=1)
        return self.model(x)


def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))

    discriminator = Discriminator(in_channels, 8)
    init_weights(discriminator)
    assert discriminator(x).shape == (N, 1, 1, 1)

    generator = Generator(z_dim, in_channels, 8)
    init_weights(generator)
    z = torch.randn((N, z_dim, 1, 1))
    assert generator(z).shape == (N, in_channels, H, W)
    print('Tests passed')


if __name__ == '__main__':
    test()
