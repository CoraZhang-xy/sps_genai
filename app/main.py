import spacy
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
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
