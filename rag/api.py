from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from src.manifest import ProcessingManifest
from src.data_loader import discover_files, load_csv_documents, load_md_documents
from src.embedding import EmbeddingPipeline
from src.vector_store import VectorStore
from src.retrieving import RAGRetriever

app = FastAPI()

# Autoriser Angular (localhost:4200) à appeler cette API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialisation du RAG une seule fois au démarrage ---
manifest = ProcessingManifest("processed_files.json")
chunker = EmbeddingPipeline()
chromadb_store = VectorStore()
response_generator = RAGRetriever(chromadb_store, chunker)

# (Optionnel) réutilise ta logique d'ingestion existante ici si besoin,
# ou fais-la tourner à part comme un script d'indexation séparé.

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = response_generator.response(request.query)
    return ChatResponse(answer=answer)