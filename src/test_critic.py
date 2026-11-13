
from src.hybrid_retriever import HybridRetriever
from src.verification import verify_hybrid_answer
from pprint import pprint


query = "Which drug interacts with Metformin?"
entity_name = "Metformin"

# 1. Retrieve actual evidence.
retriever = HybridRetriever()

try:
    retrieval_result = retriever.retrieve(
        query=query,
        entity_name=entity_name,
        top_k=3,
    )
finally:
    # Close database resources here if your retriever
    # exposes a close() method.
    pass

# 2. Temporary answer for testing the verifier.
# Replace this with your real LLM-generated answer
# once we connect the generation module.
answer = "Metformin interacts with Warfarin."

# 3. Verify against retrieved evidence.
result = verify_hybrid_answer(
    answer=answer,
    retrieval_result=retrieval_result,
)

pprint(result)

print("\nOverall status:", result["overall_status"])
print("Needs retrieval:", result["needs_retrieval"])
