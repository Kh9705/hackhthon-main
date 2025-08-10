# llm.py
import httpx
from typing import List, Dict
from mistralai.async_client import MistralAsyncClient
from config import LLM_MODEL

async def get_answer_from_llm(client: MistralAsyncClient, context: str, question: str, override_messages: List[Dict[str, str]] = None) -> str:
    """Makes a single API call to the Mistral LLM."""
    if override_messages:
        messages = override_messages
    else:
        prompt = f"""
        You are an expert Q&A system. Use the provided context to answer the question accurately.
        If the answer is not in the context, state that you cannot answer.
        Context: {context}
        Question: {question}
        Answer:
        """
        messages = [{"role": "user", "content": prompt}]

    try:
        response = await client.chat(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Mistral LLM call failed: {e}")
        return "Error while generating answer."

async def fetch_dynamic_data(url: str) -> str:
    """Safely fetches data from a URL for the agent's tool-use."""
    print(f"[INFO] Agent Tool Use: Fetching data from {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.text
    except Exception as e:
        print(f"[WARNING] Agent Tool Use failed for {url}: {e}")
        return f"Error: Could not retrieve data. {e}"