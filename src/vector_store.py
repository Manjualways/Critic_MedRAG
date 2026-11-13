from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """ChromaDB semantic retrieval layer for CRITIC-MedRAG."""

    def __init__(
        self,
        persist_dir="chroma_db",
        collection_name="medical_documents",
        embedding_model="all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "CRITIC-MedRAG medical document chunks"
            },
        )

        self.embedder = SentenceTransformer(embedding_model)

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size=500,
        overlap=80
    ) -> List[str]:

        text = " ".join(text.split())

        if not text:
            return []

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        chunks = []
        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(text):
                break

            start = end - overlap

        return chunks

    def add_document(
        self,
        text: str,
        source: str
    ) -> int:

        chunks = self.chunk_text(text)

        if not chunks:
            return 0

        embeddings = self.embedder.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        ids = [
            f"{Path(source).stem}-{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": source,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k=3
    ) -> List[Dict]:

        if not query.strip():
            return []

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0].tolist()

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {
                "document": document,
                "metadata": metadata,
                "distance": distance
            }
            for document, metadata, distance
            in zip(
                documents,
                metadatas,
                distances
            )
        ]

    def count(self):
        return self.collection.count()