from models import Generator, Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
device = torch.device('cpu')
lr = 3e-4
z_dim = 64
image_dim = 28 * 28
batch_size = 64
num_epochs = 100

# Models
discriminator = Discriminator(image_dim).to(device)
generator = Generator(z_dim, image_dim).to(device)

# Data
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = datasets.MNIST(root='dataset/', transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
opt_gen = optim.Adam(generator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Tensorboard writers
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        # Get real data
        real = real.view(-1, 784).to(device)
        # Generate fake data
        noise = torch.randn((batch_size, z_dim)).to(device)
        fake = generator(noise)

        # Train discriminator
        disc_pred_real = discriminator(real).view(-1)
        disc_loss_real = criterion(disc_pred_real, torch.ones_like(disc_pred_real))

        disc_pred_fake = discriminator(fake).view(-1)
        disc_loss_fake = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))

        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        discriminator.zero_grad()
        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        # Train generator
        disc_pred_fake = discriminator(fake).view(-1)
        gen_loss = criterion(disc_pred_fake, torch.ones_like(disc_pred_fake))

        generator.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
        # Tensorbaord
        if batch_idx == 0:
            print(
                f'Epoch: [{epoch+1}/{num_epochs}] \ '
                f'Disc loss: {disc_loss:.4f}, Gen loss: {gen_loss:.4f}'
            )

            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                real = real.reshape(-1, 1, 28, 28)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image('MNIST generated images', img_grid_fake, global_step=step)
                writer_real.add_image('MNIST real images', img_grid_real, global_step=step)

                step += 1
