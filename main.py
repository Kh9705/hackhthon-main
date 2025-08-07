import os
import tempfile
import httpx
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# --- Globals for lazy loading ---
# We will load these only once on the very first request.
app = FastAPI(title="HackRx PDF RAG")
encoder = None
client = None

# --- Pydantic Models ---
class QuestionPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOut(BaseModel):
    answers: List[str]

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=AnswerOut)
async def run_rag(payload: QuestionPayload):
    global encoder, client

    # On the first request, import heavy libraries and load models.
    if client is None:
        try:
            print("First request: Initializing models...")
            # Lazy import heavy libraries
            from sentence_transformers import SentenceTransformer
            from openai import OpenAI

            # Use the smallest, most memory-efficient model
            encoder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print("Models initialized successfully.")
        except Exception as e:
            client = None # Reset on failure to allow retry
            raise HTTPException(500, f"Model initialization failed: {e}")

    # --- Document processing happens on every request ---
    try:
        # Lazy import document processing libraries
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import faiss
        import numpy as np

        # Download the PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = httpx.get(str(payload.documents))
            response.raise_for_status() # Raise an error if download fails
            tmp.write(response.content)
            tmp_path = tmp.name

        # Process the document into an index
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path) # Clean up temp file immediately

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text("\n".join(d.page_content for d in docs))

        vectors = encoder.encode(chunks).astype("float32")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

    except Exception as e:
        raise HTTPException(500, f"Failed to process document: {e}")
    # ----------------------------------------------------

    answers = []
    for q in payload.questions:
        q_vec = encoder.encode([q]).astype("float32")
        _, I = index.search(q_vec, k=3) # Use a smaller k for efficiency
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