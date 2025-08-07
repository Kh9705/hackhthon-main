import os
import tempfile
import httpx
from typing import List, Dict

# Dataclass to hold our cached data
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# --- Globals for lazy loading and caching ---
app = FastAPI(title="HackRx PDF RAG with Cache")
encoder = None
client = None

# This dictionary will be our in-memory cache.
# It will store { "pdf_url": CachedDocument_object }
document_cache: Dict[str, 'CachedDocument'] = {}

# --- Pydantic Models & Dataclasses ---
class QuestionPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOut(BaseModel):
    answers: List[str]

# A simple structure to hold the processed data in our cache
@dataclass
class CachedDocument:
    index: object # Using 'object' for FAISS index type hint simplicity
    chunks: List[str]

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=AnswerOut)
async def run_rag(payload: QuestionPayload):
    global encoder, client, document_cache

    # On the first request ever, import heavy libraries and load models.
    if client is None:
        try:
            print("First request: Initializing models...")
            from sentence_transformers import SentenceTransformer
            from openai import OpenAI
            
            encoder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print("Models initialized successfully.")
        except Exception as e:
            client = None # Reset on failure
            raise HTTPException(500, f"Model initialization failed: {e}")

    pdf_url = str(payload.documents)

    # ================== CACHE LOGIC STARTS HERE ==================
    # Check if the document is already in our cache
    if pdf_url in document_cache:
        print(f"Cache HIT for URL: {pdf_url}")
        cached_data = document_cache[pdf_url]
        index = cached_data.index
        chunks = cached_data.chunks
    else:
        # If not in cache, process it for the first time
        print(f"Cache MISS for URL: {pdf_url}. Processing document...")
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            import faiss
            import numpy as np

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                response = httpx.get(pdf_url)
                response.raise_for_status()
                tmp.write(response.content)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.unlink(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text("\n".join(d.page_content for d in docs))
            
            vectors = encoder.encode(chunks).astype("float32")
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)

            # Store the newly processed data in our cache
            document_cache[pdf_url] = CachedDocument(index=index, chunks=chunks)
            print("Document processed and stored in cache.")

        except Exception as e:
            raise HTTPException(500, f"Failed to process document: {e}")
    # =================== CACHE LOGIC ENDS HERE ===================

    answers = []
    for q in payload.questions:
        q_vec = encoder.encode([q]).astype("float32")
        _, I = index.search(q_vec, k=3)
        context = "\n\n".join(chunks[i] for i in I[0])

        prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        resp = client.chat.completions.create(
            model="moonshotai/kimi-k2:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )
        answers.append(resp.choices[0].message.content)

    return AnswerOut(answers=answers)

@app.get("/")
def root():
    return {"status": "ok"}