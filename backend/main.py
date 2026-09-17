from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

from encoder import CLIPEncoder
from vector_store import FAISSVectorStore

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Global instances
encoder = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models and vector database into memory
    global encoder, vector_store
    print("Initializing Multi-Modal Search Backend...")
    encoder = CLIPEncoder()
    vector_store = FAISSVectorStore(dim=512, data_dir=DATA_DIR)
    yield
    # Shutdown: Clean up resources if needed
    print("Shutting down...")

app = FastAPI(title="Multi-Modal Search Engine API", lifespan=lifespan)

# Allow our standalone HTML/JS frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class SearchQuery(BaseModel):
    query: str
    top_k: int = 20

# --- Endpoints ---
@app.post("/search")
def search_images(req: SearchQuery):
    """Takes a natural language text query and returns visually similar images."""
    query_vector = encoder.encode_text(req.query)
    results = vector_store.search(query_vector, top_k=req.top_k)
    return {"results": results}

@app.post("/index")
def trigger_indexing():
    """Scans the `images/` directory and indexes any new images."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    indexed_count = 0
    
    # Get existing filenames from metadata to avoid duplicate indexing
    existing_files = {meta.get("filename") for meta in vector_store.metadata.values()}
    
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
            
        if filename in existing_files:
            continue  # Skip already indexed images

        filepath = os.path.join(IMAGES_DIR, filename)
        print(f"Indexing new image: {filename}")
        
        emb = encoder.encode_image(filepath)
        if emb is not None:
            file_size = os.path.getsize(filepath)
            meta = {
                "filename": filename,
                "size_bytes": file_size,
                "tags": [] # Placeholder for future automatic tagging
            }
            vector_store.add_image(emb, meta)
            indexed_count += 1
            
    return {"message": f"Successfully indexed {indexed_count} new images."}

@app.get("/image")
def get_image(filename: str = Query(..., description="Exact filename of the image")):
    """Serves the image file to the frontend via a GET request."""
    filepath = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(filepath)

@app.get("/stats")
def get_stats():
    """Returns database size."""
    return vector_store.get_stats()
