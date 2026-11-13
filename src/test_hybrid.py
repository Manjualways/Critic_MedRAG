from .hybrid_retriever import HybridRetriever


def main():

    query = "Which drug interacts with Metformin?"

    # Entity identified from the query.
    entity_name = "Metformin"

    retriever = HybridRetriever()

    results = retriever.retrieve(
        query=query,
        entity_name=entity_name,
        top_k=3
    )

    print("\n==============================")
    print("FINAL HYBRID EVIDENCE")
    print("==============================")

    print("\nQUERY:")
    print(results["query"])

    print("\nGRAPH EVIDENCE:")

    for result in results["graph_results"]:

        print(
            " -> ".join(
                result["nodes"]
            )
        )

    print("\nVECTOR EVIDENCE:")

    for result in results["vector_results"]:

        print(
            result["document"]
        )


if __name__ == "__main__":
    main()