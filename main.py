# main.py
import httpx
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import os
import time
import torch
import asyncio
import io
import re
from pptx.enum.shapes import MSO_SHAPE_TYPE

from pptx import Presentation
import pandas as pd
from PIL import Image
import pytesseract
import zipfile
import magic

from urllib.parse import urlparse

from mistralai.async_client import MistralAsyncClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

import docx
import email
from email.policy import default
import extract_msg

load_dotenv()

API_KEYS = [
    key for key in os.environ if key.startswith("MISTRAL_API_KEY_")
]
if not API_KEYS:
    print("[WARNING] No MISTRAL_API_KEY_<n> environment variables set!")
    MISTRAL_CREDENTIALS = []
else:
    MISTRAL_CREDENTIALS = [os.getenv(key) for key in API_KEYS]
    print(f"[INFO] Loaded {len(MISTRAL_CREDENTIALS)} Mistral API keys.")

ASYNC_CLIENTS = [
    MistralAsyncClient(api_key=key) for key in MISTRAL_CREDENTIALS
]

SMALL_DOC_THRESHOLD = 20000
LARGE_DOC_THRESHOLD = 100000
CHUNK_CONFIG = {
    "small":  (500, 75, 10),
    "medium": (1000, 150, 7),
    "large":  (2000, 300, 5)
}

app = FastAPI(
    title="Optimized Multi-Format RAG Service with Mistral AI",
    description="RAG service for PDF, DOCX, and Email files using Mistral AI.",
)

class QuestionPayload(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOut(BaseModel):
    answers: List[str]

document_cache: Dict[str, Dict[str, Any]] = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device for embeddings: {device}")

print("[INFO] Loading SentenceTransformer model: all-MiniLM-L6-v2...")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("[INFO] Model loaded successfully.")

def _extract_text_from_eml(eml_bytes: bytes) -> str:
    try:
        return eml_bytes.decode('utf-8')
    except UnicodeDecodeError:
        print("[WARNING] Could not decode email as UTF-8, falling back to latin-1.")
        return eml_bytes.decode('latin-1', errors='ignore')

def process_and_index_document(doc_url: str):
    global document_cache
    print(f"[INFO] Processing document: {doc_url}")

    try:
        with httpx.Client() as http_client:
            response = http_client.get(doc_url, timeout=120.0)
            response.raise_for_status()
        doc_bytes = response.content
        print(f"[INFO] Document downloaded ({len(doc_bytes)} bytes).")

        # Detect file extension
        match = re.search(r"\.(pdf|docx|eml|msg|pptx|xlsx|png|jpeg|jpg|zip|bin)(?=\?|$)",
                          doc_url, re.IGNORECASE)
        file_extension = match.group(0).lower() if match else None

        if not file_extension:
            content_type = response.headers.get("content-type", "").lower()
            mime_map = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/vnd.ms-outlook": ".msg",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "image/png": ".png",
                "image/jpeg": ".jpeg",
                "application/zip": ".zip",
                "message/rfc822": ".eml"
            }
            file_extension = mime_map.get(content_type)

        if not file_extension:
            detected_type = magic.from_buffer(doc_bytes, mime=True)
            mime_map_reverse = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "application/vnd.ms-outlook": ".msg",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "image/png": ".png",
                "image/jpeg": ".jpeg",
                "application/zip": ".zip",
                "message/rfc822": ".eml"
            }
            file_extension = mime_map_reverse.get(detected_type)

        print(f"[INFO] Detected file extension: '{file_extension}'")

        full_text = ""
        if file_extension in [".pdf", ".docx", ".eml", ".msg"]:
            if file_extension == ".pdf":
                doc = fitz.open("pdf", doc_bytes)
                full_text = "".join(page.get_text() for page in doc)
                doc.close()
            elif file_extension == ".docx":
                doc_stream = io.BytesIO(doc_bytes)
                document = docx.Document(doc_stream)
                full_text = "\n".join([para.text for para in document.paragraphs])
            elif file_extension == ".eml":
                full_text = _extract_text_from_eml(doc_bytes)
            elif file_extension == ".msg":
                msg = extract_msg.Message(doc_bytes)
                full_text = msg.body

        elif file_extension == ".pptx":
            ppt_stream = io.BytesIO(doc_bytes)
            prs = Presentation(ppt_stream)
            for slide_number, slide in enumerate(prs.slides):
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        full_text += shape.text_frame.text + "\n"
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        try:
                            img = Image.open(io.BytesIO(shape.image.blob))
                            ocr_text = pytesseract.image_to_string(img)
                            if ocr_text.strip():
                                full_text += f"--- OCR Text from Image ---\n{ocr_text}\n--- End of OCR Text ---\n"
                        except Exception as ocr_error:
                            print(f"[WARNING] Could not process image on slide {slide_number + 1}: {ocr_error}")

        elif file_extension == ".xlsx":
            xls_stream = io.BytesIO(doc_bytes)
            sheets = pd.read_excel(xls_stream, sheet_name=None)
            for sheet_name, df in sheets.items():
                full_text += f"--- Sheet: {sheet_name} ---\n{df.to_string()}\n\n"

        elif file_extension in [".png", ".jpeg", ".jpg"]:
            img = Image.open(io.BytesIO(doc_bytes))
            full_text = pytesseract.image_to_string(img)

        elif file_extension == ".zip":
            zip_stream = io.BytesIO(doc_bytes)
            archive = zipfile.ZipFile(zip_stream)
            for filename in archive.namelist():
                if not filename.endswith('/'):
                    file_bytes = archive.read(filename)
                    inner_type = magic.from_buffer(file_bytes, mime=True)
                    if 'pdf' in inner_type:
                        doc = fitz.open("pdf", file_bytes)
                        full_text += "".join(page.get_text() for page in doc) + "\n\n"
                        doc.close()
                    else:
                        try:
                            full_text += file_bytes.decode('utf-8', errors='ignore') + "\n\n"
                        except Exception:
                            print(f"[WARNING] Could not extract text from '{filename}' in ZIP.")

        elif file_extension == ".bin":
            raise HTTPException(status_code=415, detail="Unsupported file type: .bin files cannot be processed for text.")

        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: '{file_extension}'.")

        if not full_text.strip():
            print("[WARNING] No text extracted.")
            document_cache[doc_url] = {"index": None, "chunks": [], "timestamp": time.time()}
            return

        doc_len = len(full_text)
        if doc_len < SMALL_DOC_THRESHOLD:
            size_category = "small"
        elif doc_len > LARGE_DOC_THRESHOLD:
            size_category = "large"
        else:
            size_category = "medium"

        chunk_size, chunk_overlap, k_value = CHUNK_CONFIG[size_category]
        print(f"[INFO] Document size is '{size_category}'. Using chunk_size={chunk_size}, k={k_value}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text)

        vectors = encoder.encode(
            chunks,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        document_cache[doc_url] = {
            "index": index,
            "chunks": chunks,
            "timestamp": time.time(),
            "k_value": k_value
        }
        print(f"[SUCCESS] Indexed: {doc_url}")

    except Exception as e:
        print(f"[ERROR] Document processing failed: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

async def get_answer_from_llm(client: MistralAsyncClient, messages: List[Dict[str, str]], question: str) -> str:
    try:
        response = await client.chat(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Mistral LLM call for question '{question}' failed: {e}")
        return "Error while generating answer."

@app.post("/hackrx/run", response_model=AnswerOut)
async def run_rag(payload: QuestionPayload):
    doc_url_str = str(payload.documents)
    questions = payload.questions
    print(f"[INFO] Request for {doc_url_str} with {len(questions)} questions.")

    if not ASYNC_CLIENTS:
        raise HTTPException(status_code=500, detail="No Mistral API keys configured.")

    if doc_url_str not in document_cache:
        print("[INFO] Document not in cache. Processing...")
        process_and_index_document(doc_url_str)
    else:
        print("[INFO] Using cached document index.")

    cached_data = document_cache[doc_url_str]
    index = cached_data["index"]
    chunks = cached_data["chunks"]

    if not chunks or index is None:
        return AnswerOut(
            answers=["I could not extract any text from the provided document."] * len(questions)
        )

    k = cached_data.get("k_value", CHUNK_CONFIG["medium"][2])
    tasks = []
    for i, question in enumerate(questions):
        question_vector = encoder.encode([question], convert_to_numpy=True).astype("float32")
        _, indices = index.search(question_vector, k)
        context = "\n\n---\n\n".join(chunks[i] for i in indices[0])

        prompt = f"""
You are an expert Q&A system. Use the context to answer accurately and concisely.
If the answer is not in the context, say you cannot answer.

## Context:
{context}

## Question:
{question}

## Answer:
"""
        messages = [{"role": "user", "content": prompt.strip()}]
        client_to_use = ASYNC_CLIENTS[i % len(ASYNC_CLIENTS)]
        tasks.append(get_answer_from_llm(client_to_use, messages, question))
        await asyncio.sleep(0.5)

    final_answers = await asyncio.gather(*tasks)
    return AnswerOut(answers=final_answers)

@app.get("/")
def health_check():
    return {"status": "ok", "cached_docs": list(document_cache.keys())}
