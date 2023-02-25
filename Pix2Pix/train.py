import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_epoch(generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, l1_loss, bce_loss, loader):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train discriminator
        with torch.cuda.amp.autocast_mode.autocast():
            y_fake = generator(x)

            disc_real = discriminator(x, y)
            disc_fake = discriminator(x, y_fake)
            disc_real_loss = bce_loss(disc_real, torch.ones_like(disc_real))
            disc_fake_loss = bce_loss(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
        
        opt_disc.zero_grad()
        disc_scaler.scale(disc_loss).backward(retain_graph=True)
        disc_scaler.step(opt_disc)
        disc_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast_mode.autocast():
            disc_fake = discriminator(x, y_fake)
            gen_fake_loss = bce_loss(disc_fake, torch.ones_like(disc_fake))
            gen_l1_loss = l1_loss(y_fake, y) * config.L1_LAMBDA
            gen_loss = gen_fake_loss + gen_l1_loss
        
        opt_gen.zero_grad()
        gen_scaler.scale(gen_loss).backward(retain_graph=True)
        gen_scaler.step(opt_gen)
        gen_scaler.update()

def main():
    discriminator = Discriminator().to(config.DEVICE)
    generator = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, discriminator, opt_gen, config.LEARNING_RATE)
    
    train_dataset = MapDataset('data/maps/train')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = MapDataset('data/maps/val')
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    gen_scaler = torch.cuda.amp.grad_scaler.GradScaler()
    disc_scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(generator, discriminator, opt_gen, opt_disc, gen_scaler, disc_scaler, l1_loss, bce_loss, train_loader)

        if config.SAVE_MODEL:
            save_checkpoint(generator, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(discriminator, opt_gen, filename=config.CHECKPOINT_DISC)
        
        save_some_examples(generator, val_loader, epoch, folder='eval')

if __name__ == '__main__':
    main()
