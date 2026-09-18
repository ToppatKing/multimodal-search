from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import requests
from io import BytesIO
from PIL import Image
from datasets import load_dataset

from encoder import CLIPEncoder
from vector_store import FAISSVectorStore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

encoder = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, vector_store
    print("Initializing Multi-Modal Search Backend...")
    encoder = CLIPEncoder()
    vector_store = FAISSVectorStore(dim=512, data_dir=DATA_DIR)
    yield

app = FastAPI(title="Multi-Modal Search Engine API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 20

@app.post("/search")
def search_images(req: SearchQuery):
    query_vector = encoder.encode_text(req.query)
    results = vector_store.search(query_vector, top_k=req.top_k)
    return {"results": results}

@app.post("/index_remote")
def index_remote_dataset(max_images: int = 100):
    """
    Streams images from the Google Conceptual Captions dataset on Hugging Face.
    Link: https://huggingface.co/datasets/google-research-datasets/conceptual_captions
    """
    print("Connecting to Hugging Face dataset stream (Conceptual Captions)...")
    
    # streaming=True prevents downloading the massive dataset locally
    dataset = load_dataset("google-research-datasets/conceptual_captions", split="train", streaming=True)
    
    indexed_count = 0
    failed_count = 0
    
    for row in dataset:
        if indexed_count >= max_images:
            break
            
        image_url = row['image_url']
        
        try:
            # Fetch image directly into RAM
            response = requests.get(image_url, timeout=3)
            if response.status_code != 200:
                continue
                
            image = Image.open(BytesIO(response.content))
            
            # Encode using the PIL image directly
            emb = encoder.encode_image(image)
            if emb is not None:
                meta = {
                    "url": image_url,
                    "source": "conceptual_captions",
                    "caption": row['caption'] # We can store the original caption too!
                }
                vector_store.add_image(emb, meta)
                indexed_count += 1
                print(f"Indexed {indexed_count}/{max_images}: {image_url}")
            
        except Exception as e:
            failed_count += 1
            continue
            
    return {
        "message": f"Successfully indexed {indexed_count} remote images.",
        "dead_links_skipped": failed_count
    }

@app.get("/stats")
def get_stats():
    return vector_store.get_stats()


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
