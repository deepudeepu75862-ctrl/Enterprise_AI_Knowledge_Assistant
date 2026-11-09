from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Tuple
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class KnowledgeAssistant:
    """A small RAG pipeline: PDF extraction -> chunking -> embeddings -> retrieval -> answer."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks: List[Dict] = []
        self.vectors = None

    @staticmethod
    def _extract_pdf(filename: str, content: bytes) -> List[Dict]:
        reader = PdfReader(BytesIO(content))
        pages = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"source": filename, "page": page_no, "text": text})
        return pages

    @staticmethod
    def _chunk_pages(pages: List[Dict], chunk_size: int = 900, overlap: int = 140) -> List[Dict]:
        chunks = []
        for page in pages:
            text = " ".join(page["text"].split())
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                piece = text[start:end].strip()
                if piece:
                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "text": piece
                    })
                if end >= len(text):
                    break
                start = end - overlap
        return chunks

    def ingest_pdfs(self, documents: List[Tuple[str, bytes]]) -> int:
        pages = []
        for filename, content in documents:
            pages.extend(self._extract_pdf(filename, content))
        self.chunks = self._chunk_pages(pages)
        if not self.chunks:
            self.vectors = None
            return 0
        texts = [c["text"] for c in self.chunks]
        self.vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        return len(self.chunks)

    def retrieve(self, question: str, top_k: int = 4) -> List[Dict]:
        if self.vectors is None or not self.chunks:
            return []
        q_vector = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        q_vector = np.asarray(q_vector, dtype=np.float32)[0]
        scores = self.vectors @ q_vector
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            item = dict(self.chunks[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results

    @staticmethod
    def _fallback_answer(question: str, sources: List[Dict]) -> str:
        if not sources:
            return "I could not find relevant information in the uploaded documents."
        passages = "\n\n".join(
            f"[{i}] {s['text']}" for i, s in enumerate(sources, 1)
        )
        return (
            "Here are the most relevant passages I found in your documents. "
            "Use the numbered references below to verify the details.\n\n"
            + passages
            + "\n\nNote: This is a retrieval-based response, not a fully generated summary. "
              "Enable the Groq option and add an API key for a synthesized answer."
        )

    @staticmethod
    def _groq_answer(question: str, sources: List[Dict], api_key: str) -> str:
        from groq import Groq
        client = Groq(api_key=api_key)
        context = "\n\n".join(
            f"[{i}] Source: {s['source']}, page {s['page']}\n{s['text']}"
            for i, s in enumerate(sources, 1)
        )
        prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you cannot find it in the uploaded documents.
Do not invent facts. Cite supporting passages using [1], [2], etc.

CONTEXT:
{context}

QUESTION:
{question}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a careful enterprise document assistant. Ground every claim in the supplied context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def ask(self, question: str, top_k: int = 4, use_llm: bool = False, groq_api_key: str = "") -> Dict:
        sources = self.retrieve(question, top_k)
        if not sources:
            answer = "No document passages are indexed yet. Upload and process PDF files first."
        elif use_llm:
            if not groq_api_key.strip():
                raise ValueError("Enter your Groq API key or switch off the Groq LLM option.")
            answer = self._groq_answer(question, sources, groq_api_key.strip())
        else:
            answer = self._fallback_answer(question, sources)
        return {"answer": answer, "sources": sources}
