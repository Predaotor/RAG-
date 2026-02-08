"""Embedding model for multilingual (Georgian) text."""

from typing import List

from langchain_openai import OpenAIEmbeddings

from .config import OPENAI_API_KEY


class Embedder:
    """Embedder using OpenAI API - supports Georgian, no local ML dependencies."""

    def __init__(self, api_key: str | None = None):
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key or OPENAI_API_KEY,
        )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        return self._embeddings.embed_documents(texts)
