from pathlib import Path
import torch
from app.helper_lib.model import get_model
from app.helper_lib.generator import generate_images

MODEL_PATH = Path(__file__).resolve().parent / "mnist_generator.pth"


def load_gan():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Run uv run python train_gan.py before GAN inference.")
    model = get_model("GAN")
    model.generator.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model
