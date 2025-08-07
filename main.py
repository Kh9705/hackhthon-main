import os
import tempfile
import httpx
import numpy as np
from typing import List

# Only import FastAPI and Pydantic at the top. They are fast.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# ================== LAZY IMPORT CHANGES ==================
# We will initialize our heavy objects as None.
# We will import the heavy libraries inside the function below.
app = FastAPI(title="HackRx PDF → Kimi K2 RAG")
encoder = None
client = None
text_splitter = None
# We no longer need a global index or chunks, as they will be created per-request.
# ==========================================================


class QuestionPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOut(BaseModel):
    answers: List[str]

# ---------- endpoint ----------
@app.post("/hackrx/run", response_model=AnswerOut)
async def run_rag(payload: QuestionPayload):
    global encoder, client, text_splitter

    # On the first request, import libraries and initialize models.
    # This part runs only once.
    if client is None:
        try:
            print("First request: Importing libraries and loading models...")
            
            # ========== LAZY IMPORTS ARE HERE ==========
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from sentence_transformers import SentenceTransformer
            import faiss
            from openai import OpenAI
            # ==========================================

            # Now we can initialize our clients and models
            encoder = SentenceTransformer("paraphrase-MiniLM-L3-v2") # Using the smallest model
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            print("Models are ready.")
        except Exception as e:
            # If something fails, reset state to allow retries
            client = None 
            encoder = None
            text_splitter = None
            raise HTTPException(500, f"Model setup failed: {e}")

    # --- Document Processing: This part runs for EVERY request ---
    try:
        print(f"Processing document from: {payload.documents}")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # Download the PDF content from the URL
            response = httpx.get(str(payload.documents))
            response.raise_for_status() # Raise an exception for bad status codes
            tmp.write(response.content)
            tmp.flush()
            
            # Load and split the PDF
            docs = PyPDFLoader(tmp.name).load()
            chunks = text_splitter.split_text("\n".join(d.page_content for d in docs))
            
            # Create a FAISS index for the current document
            vectors = encoder.encode(chunks, show_progress_bar=False).astype("float32")
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            
            # Clean up the temporary file
            os.unlink(tmp.name)
        print("Document processed and index created successfully.")
    except Exception as e:
        raise HTTPException(500, f"Failed to process document: {e}")
    # ----------------------------------------------------------------

    answers = []
    for q in payload.questions:
        q_vec = encoder.encode([q]).astype("float32")
        # Search the index we just created for this specific document
        _, I = index.search(q_vec, k=20) # Reduced k to 3 for max efficiency
        context = "\n\n".join(chunks[i] for i in I[0])

        prompt = f"""
Use the context to answer the question. If unsure, say "I don't know".

Context:
{context}

Question: {q}
Answer:
""".strip()

        resp = client.chat.completions.create(
            model="moonshotai/kimi-k2:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        answers.append(resp.choices[0].message.content)

    return AnswerOut(answers=answers)

# ---------- health ----------
@app.get("/")
def root():
    return {"status": "ok"}
