import torch
from torchvision.utils import make_grid


def generate_images(model, device, num_samples=10):
    gen = model.generator
    was_training = gen.training
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, gen.z_dim, device=device)
        images = gen(noise).detach().cpu()
    gen.train(was_training)
    return images


def generate_samples(model, device, num_samples=10):
    import matplotlib.pyplot as plt

    images = generate_images(model, device, num_samples)
    grid = make_grid(images * 0.5 + 0.5, nrow=5)
    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    plt.close()
