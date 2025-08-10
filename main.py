# main.py
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

# Local module imports
from config import ASYNC_CLIENTS, ENCODER
from processing import process_and_index_document
from llm import get_answer_from_llm, fetch_dynamic_data

# --- FastAPI App Initialization & In-Memory Cache ---
app = FastAPI(
    title="Modular RAG Service with Mistral AI",
    description="A RAG service supporting multiple file formats and agentic workflows.",
)
document_cache: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models for API validation ---
class QuestionPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOut(BaseModel):
    answers: List[str]

# --- API Endpoints ---
@app.get("/")
def health_check():
    """Provides a health check and lists currently cached documents."""
    return {"status": "ok", "cached_docs": list(document_cache.keys())}

@app.post("/hackrx/run", response_model=AnswerOut)
async def run_rag(payload: QuestionPayload):
    doc_url_str = str(payload.documents)
    
    # Process document on demand if not in cache
    if doc_url_str not in document_cache:
        process_and_index_document(doc_url_str, document_cache)

    # ... The rest of the endpoint logic, including the router agent ...
    # This logic remains the same as in your original file.
    # For brevity, it's not repeated here.
    
    # Example of how the rest of the logic would look:
    cached_data = document_cache[doc_url_str]
    client = ASYNC_CLIENTS[0]
    question = payload.questions[0]

    # This is a simplified version of your agent router logic
    if "follow instructions" in question or "find the" in question:
         # Agentic path
         full_text = "\n".join(cached_data["chunks"])
         # ... call agent logic ...
         return AnswerOut(answers=["Agent workflow is complex..."])
    else:
        # Simple Q&A path
        k = cached_data.get("k_value", 7)
        question_vector = ENCODER.encode([question]).astype("float32")
        _, indices = cached_data["index"].search(question_vector, k)
        context = "\n\n".join([cached_data["chunks"][i] for i in indices[0]])
        answer = await get_answer_from_llm(client, context, question)
        return AnswerOut(answers=[answer])