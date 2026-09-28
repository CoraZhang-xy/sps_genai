# Module 3 FastAPI project with word embeddings

This project follows the Module 3 class activity's `/generate` API and adds a
`GET /embedding` endpoint using the Module 2 spaCy method. It loads the
`en_core_web_lg` model and returns its 300-dimensional word vector as JSON.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --frozen
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs or query
http://127.0.0.1:8000/embedding?word=apple .

The original bigram endpoint is `POST /generate`, for example with body
`{"start_word":"the","length":10}`.

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Open http://127.0.0.1:8000/docs and run `GET /embedding` with `word=apple`.
The first build downloads the large spaCy English pipeline.
