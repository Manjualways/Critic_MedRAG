
from pprint import pprint
from src.verification import verify_answer


answer = (
    "Metformin interacts with Warfarin. "
    "Metformin always causes severe kidney failure."
)

text_evidence = [
    "The supplied synthetic test note states that "
    "Metformin interacts with Warfarin."
]

graph_evidence = [
    "Metformin -[INTERACTS_WITH]-> Warfarin"
]

result = verify_answer(
    answer=answer,
    text_evidence=text_evidence,
    graph_evidence=graph_evidence,
)

pprint(result)

print("\nOverall status:", result["overall_status"])
print("Needs retrieval:", result["needs_retrieval"])
