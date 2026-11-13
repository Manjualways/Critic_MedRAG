from .document_parser import read_document
from .vector_store import VectorStore


def main():

    source = "data/sample_medical_notes.txt"

    text = read_document(source)

    store = VectorStore()

    added = store.add_document(
        text,
        source
    )

    print(
        f"Chunks indexed in this run: {added}"
    )

    print(
        f"Total chunks in ChromaDB: {store.count()}"
    )

    query = "Which drug interacts with Metformin?"

    print(f"\nQuery: {query}")

    results = store.search(
        query,
        top_k=3
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {i}")

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print(
            f"Source: {result['metadata']['source']}"
        )

        print(
            f"Chunk: {result['document']}"
        )


if __name__ == "__main__":
    main()