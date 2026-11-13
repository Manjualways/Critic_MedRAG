
from src.hybrid_retriever import HybridRetriever
from src.answer_generator import AnswerGenerator

query = "Which drug interacts with Metformin?"
entity_name = "Metformin"

retriever = HybridRetriever()

try:
    retrieval_result = retriever.retrieve(
        query=query,
        entity_name=entity_name,
        top_k=3,
    )

    generator = AnswerGenerator()

    answer = generator.generate(
        query=query,
        retrieval_result=retrieval_result,
    )

    print("\n=== GENERATED ANSWER ===")
    print(answer)

finally:
    retriever.vector_store  # Keep the existing retriever unchanged.
