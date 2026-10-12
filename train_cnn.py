"""Train the specified 64x64 CNN on CIFAR-10 and save its weights for the API."""

import argparse
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from app.cnn_model import AssignmentCNN, IMAGE_TRANSFORM, MODEL_PATH


class ParquetCIFAR10(Dataset):
    """Read the UofT CIFAR-10 mirror when the course archive is slow to fetch."""

    def __init__(self, path):
        import pyarrow.parquet as pq

        rows = pq.read_table(path, columns=["img", "label"]).to_pylist()
        self.images = [row["img"]["bytes"] for row in rows]
        self.labels = [row["label"] for row in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with Image.open(BytesIO(self.images[index])) as image:
            tensor = IMAGE_TRANSFORM(image.convert("RGB"))
        return tensor, self.labels[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--parquet-dir", type=Path,
                        help="Optional folder with UofT train.parquet and test.parquet")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    if args.parquet_dir:
        train_dataset = ParquetCIFAR10(args.parquet_dir / "train.parquet")
        test_dataset = ParquetCIFAR10(args.parquet_dir / "test.parquet")
    else:
        train_dataset = CIFAR10(root="./data", train=True, download=True,
                                transform=IMAGE_TRANSFORM)
        test_dataset = CIFAR10(root="./data", train=False, download=True,
                               transform=IMAGE_TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"loss={running_loss / len(train_loader):.4f}, "
              f"train_accuracy={100 * correct / total:.2f}%", flush=True)

    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            correct += (outputs.argmax(dim=1).cpu() == labels).sum().item()
            total += labels.size(0)
    print(f"Test accuracy: {100 * correct / total:.2f}%", flush=True)
    torch.save(model.cpu().state_dict(), MODEL_PATH)
    print(f"Saved weights to {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
