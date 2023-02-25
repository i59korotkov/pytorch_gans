import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from models import Discriminator, Generator, init_weights
from utils import gradient_penalty


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Data
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],
        [0.5 for _ in range(CHANNELS_IMG)],
    ),
])
dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
fixed_noise = torch.randn((32, Z_DIM, 1, 1)).to(device)

# Models
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
discriminator = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)
init_weights(generator)
init_weights(discriminator)
generator.train()
discriminator.train()

# Optimizer
opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# Tensorboard
writer_real = SummaryWriter('runs/real')
writer_fake = SummaryWriter('runs/fake')
step = 0

# Training
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels ) in enumerate(tqdm(loader)):
        # Get real data
        real = real.to(device)
        labels = labels.to(device)

        # Train discriminator
        for _ in range(CRITIC_ITERATIONS):
            # Generate fake data
            noise = torch.randn((real.shape[0], Z_DIM, 1, 1)).to(device)
            fake = generator(noise, labels)

            disc_pred_real = discriminator(real, labels).reshape(-1)
            disc_pred_fake = discriminator(fake, labels).reshape(-1)
            disc_gp = gradient_penalty(discriminator, labels, real, fake, device)

            disc_loss = -(torch.mean(disc_pred_real) - torch.mean(disc_pred_fake)) + LAMBDA_GP * disc_gp

            discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()

        # Train generator
        disc_pred_fake = discriminator(fake, labels).reshape(-1)
        gen_loss = -torch.mean(disc_pred_fake)

        generator.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # Tensorbaord
        if batch_idx % 100 == 0:
            print(
                f'Epoch: [{epoch+1}/{NUM_EPOCHS}] \ '
                f'Batch: [{batch_idx+1}/{len(loader)}]'
                f'Disc loss: {disc_loss:.4f}, Gen loss: {gen_loss:.4f}'
            )

            with torch.no_grad():
                fake = generator(noise, labels)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image('MNIST generated images', img_grid_fake, global_step=step)
                writer_real.add_image('MNIST real images', img_grid_real, global_step=step)

                step += 1
