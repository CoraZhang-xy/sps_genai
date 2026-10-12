from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MODEL_PATH = Path(__file__).resolve().parent / "cifar10_cnn.pth"

# The same preprocessing is used for CIFAR-10 training and uploaded images.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def load_classifier():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained CNN weights missing: {MODEL_PATH}")
    model = AssignmentCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_image(model: AssignmentCNN, image: Image.Image):
    input_tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        class_index = int(probabilities.argmax(dim=1).item())
    return {"class_id": class_index, "class_name": CLASSES[class_index],
            "confidence": round(float(probabilities[0, class_index]), 4)}
