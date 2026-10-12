# FastAPI text generation, word embeddings, and CIFAR-10 image classification

This project follows the Module 3 class activity's `/generate` API and adds a
`GET /embedding` endpoint using the Module 2 spaCy method. It loads the
`en_core_web_lg` model and returns its 300-dimensional word vector as JSON.

Assignment 2 adds `POST /classify-image`. The CIFAR-10 classifier follows the
specified CNN architecture: RGB 64 × 64 input, 3 × 3 convolutions with 16 and
32 filters (stride 1, padding 1), ReLU and 2 × 2 max pooling after each, then
flatten, 100 hidden units with ReLU, and 10 output logits. CIFAR-10 images are
resized from 32 × 32 to 64 × 64. Both training and API prediction use the same
`IMAGE_TRANSFORM` in `app/cnn_model.py`.

## Train the classifier

The included `app/cifar10_cnn.pth` contains trained weights. To retrain using
the CIFAR-10 training set and evaluate on its test set:

```bash
uv sync --frozen
uv run python train_cnn.py --epochs 5 --batch-size 64
```

The script downloads CIFAR-10 to `data/`, prints training loss and accuracy for
each epoch and test accuracy, and writes new weights to `app/cifar10_cnn.pth`.
The API loads those weights at startup. Dataset files are not needed to run
inference and are excluded from the Docker build.
The included checkpoint was trained for 5 epochs (batch size 64) on the
50,000 CIFAR-10 training images using the UofT parquet mirror; it achieved
61.58% accuracy on the 10,000 test images.

If the course archive downloads slowly, the script also accepts the University
of Toronto CIFAR-10 parquet mirror. Place its `train.parquet` and `test.parquet`
files under `data/`, install the optional training reader with
`uv pip install --python .venv/bin/python pyarrow`, then run
`.venv/bin/python train_cnn.py --parquet-dir data --epochs 5 --batch-size 64`.
The default command above follows the classroom `torchvision.datasets.CIFAR10`
loader; both paths use the same preprocessing, training loop, and model.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --frozen
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs or query
http://127.0.0.1:8000/embedding?word=apple .

To classify an image, use the `POST /classify-image` file upload form in `/docs`,
or run:

```bash
curl -X POST -F "file=@your_picture.png" http://127.0.0.1:8000/classify-image
```

Example response: `{"class_id": 3, "class_name": "cat", "confidence": 0.72}`.
The number is an example only; predictions depend on the supplied image.

The original bigram endpoint is `POST /generate`, for example with body
`{"start_word":"the","length":10}`.

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Open http://127.0.0.1:8000/docs and run `POST /classify-image` with an image.
The existing `/generate` and `/embedding` endpoints remain available. Docker
uses CPU PyTorch and copies the trained model weights into the image, so the
server does not need to download CIFAR-10 or train when it starts.
The first build downloads the large spaCy English pipeline.
