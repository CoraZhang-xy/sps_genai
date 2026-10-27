import torch
from tqdm import tqdm


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    model.to(device)
    gen = model.generator
    discriminator = model.discriminator
    for epoch in range(epochs):
        model.train()
        progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_number, (real, _) in enumerate(progress):
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Standard GAN discriminator loss, as introduced in the class notebook.
            noise = torch.randn(batch_size, gen.z_dim, device=device)
            fake = gen(noise).detach()
            loss_discriminator = (
                criterion(discriminator(real), real_labels)
                + criterion(discriminator(fake), fake_labels)
            )
            discriminator.zero_grad()
            loss_discriminator.backward()
            optimizer["discriminator"].step()

            # Generator tries to have its images classified as real.
            noise = torch.randn(batch_size, gen.z_dim, device=device)
            fake = gen(noise)
            loss_gen = criterion(discriminator(fake), real_labels)
            gen.zero_grad()
            loss_gen.backward()
            optimizer["generator"].step()

            if batch_number % 100 == 0:
                progress.set_postfix({"D loss": f"{loss_discriminator.item():.4f}",
                                      "G loss": f"{loss_gen.item():.4f}"})
    return model
