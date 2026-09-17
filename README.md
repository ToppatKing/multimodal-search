# Multi-Modal Image Search Engine

A full-stack, AI-powered image search engine that uses natural language to find images. It leverages OpenAI's CLIP (ViT-B/32) model for multi-modal embeddings, FAISS for lightning-fast vector similarity search, a FastAPI backend, and a Vanilla JS Masonry-grid frontend.

## Architecture
```text
Text Query -> CLIP Text Encoder -> 512-dim Vector -> FAISS Index (Cosine Sim) -> Top K Image URLs -> Frontend
```

## Features

- Semantic Search: Search for visual concepts ("a dog in the snow") rather than exact filenames or tags.

- Remote Streaming Indexer: Streams and indexes images directly from the Hugging Face conceptual_captions massive open dataset into RAM without requiring local storage.

- Masonry Layout: Responsive, CSS-column based frontend grid.

## Setup
**1. Install Python Dependencies:**

```bash
pip install -r requirements.txt
```
**2. Start the Backend**

```bash
cd backend
uvicorn main:app --reload --port 8000
```
**3. Start the Frontend**

Simply open `frontend/index.html` in your web browser or use a local development server (like VS Code Live Server).

**4. Index Images:**

Trigger the indexer via the frontend UI or by making a POST request to `http://127.0.0.1:8000/index_remote?max_images=500`.
