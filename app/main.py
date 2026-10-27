import spacy
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Response
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from app.bigram_model import BigramModel
from app.cnn_model import load_classifier, predict_image

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)

nlp = spacy.load("en_core_web_lg")
classifier = load_classifier()


def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(word: str):
    embedding = calculate_embedding(word)
    return {"word": word, "embedding": embedding.tolist()}


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    try:
        contents = await file.read()
        with Image.open(BytesIO(contents)) as image:
            image.load()
            return predict_image(classifier, image)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file")


# Assignment 3: MNIST GAN image generation.
from app.gan_model import load_gan
from app.helper_lib.generator import generate_images
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

# Keep existing assignment endpoints available before GAN training is finished.
gan_model = None


@app.get("/gan/generate", response_class=Response,
         responses={200: {"content": {"image/png": {}}}})
def generate_digits(num_samples: int = Query(default=10, ge=1, le=100)):
    global gan_model
    if gan_model is None:
        try:
            gan_model = load_gan()
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
    images = generate_images(gan_model, "cpu", num_samples)
    grid = make_grid(images * 0.5 + 0.5, nrow=5)
    buffer = BytesIO()
    to_pil_image(grid).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
