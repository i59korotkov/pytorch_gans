import torch
import torch.nn as nn


def gradient_penalty(discriminator, labels, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape

    # Interpolate images
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * eps + fake * (1 - eps)
    
    # Discriminator score
    mixed_scores = discriminator(interpolated_images, labels).reshape(-1)

    # Calculate gradient
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)

    # Calculate penalty
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty
