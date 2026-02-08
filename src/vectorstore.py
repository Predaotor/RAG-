"""FAISS vector store for document retrieval."""

from pathlib import Path
from typing import List, Optional

import faiss
from langchain_core.documents import Document

from .config import VECTORSTORE_PATH
from .embedder import Embedder


class VectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, persist_dir: Path | None = None):
        self.persist_dir = Path(persist_dir or VECTORSTORE_PATH)
        self.embedder = Embedder()
        self.index: faiss.IndexFlatL2 | None = None
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_documents(texts)

        import numpy as np
        embeddings_array = np.array(embeddings, dtype="float32")

        if self.index is None:
            dim = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.documents = []

        self.index.add(embeddings_array)
        self.documents.extend(documents)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if self.index is None or len(self.documents) == 0:
            return []

        query_embedding = self.embedder.embed_text(query)

        import numpy as np
        query_array = np.array([query_embedding], dtype="float32")

        distances, indices = self.index.search(query_array, min(k, len(self.documents)))

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def save(self) -> None:
        """Persist the vector store to disk."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(self.persist_dir / "index.faiss"))

        # Save document metadata for reconstruction
        import pickle
        with open(self.persist_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self) -> bool:
        """Load the vector store from disk."""
        index_path = self.persist_dir / "index.faiss"
        docs_path = self.persist_dir / "documents.pkl"

        if not index_path.exists() or not docs_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))

        import pickle
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

        return True
