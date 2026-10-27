from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from app.helper_lib.model import get_model
from app.helper_lib.trainer import train_gan
from app.helper_lib.generator import generate_images

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print("Using device:", device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(ROOT / "data", train=True, download=True,
                             transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    model = get_model("GAN").to(device)
    criterion = nn.BCELoss()
    # RMSprop and learning rate follow the supplied class practical.
    optimizer = {
        "generator": optim.RMSprop(model.generator.parameters(), lr=5e-5),
        "discriminator": optim.RMSprop(model.discriminator.parameters(), lr=5e-5),
    }
    model = train_gan(model, dataloader, criterion, optimizer, device, args.epochs)
    images = generate_images(model, device, 25)
    save_image(images * 0.5 + 0.5, ROOT / "gan_samples.png", nrow=5)
    model.generator.to("cpu")
    path = ROOT / "app" / "mnist_generator.pth"
    torch.save(model.generator.state_dict(), path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
