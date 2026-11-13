
from pprint import pprint

from src.hybrid_retriever import HybridRetriever
from src.verification import verify_hybrid_answer

retriever = HybridRetriever()

query = "Which drug interacts with Metformin?"
retrieval_result = retriever.retrieve(
    query=query,
    entity_name="Metformin",
    top_k=3,
)

test_answers = [
    "Metformin interacts with Warfarin.",
    "Metformin interacts with Aspirin.",
    "Metformin interacts with Warfarin and causes kidney failure.",
]

for answer in test_answers:
    print("\n" + "=" * 60)
    print("ANSWER:", answer)

    result = verify_hybrid_answer(
        answer=answer,
        retrieval_result=retrieval_result,
    )

    pprint(result)
    print("OVERALL:", result["overall_status"])
    print("NEEDS RETRIEVAL:", result["needs_retrieval"])
