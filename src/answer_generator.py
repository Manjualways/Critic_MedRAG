
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class AnswerGenerator:
    """Generate answers grounded in hybrid retrieval evidence."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
        )

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        self.client = genai.Client(api_key=api_key)

    def generate(self, query: str, retrieval_result: dict) -> str:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Format Neo4j graph paths.
        graph_evidence = []

        for path in retrieval_result.get("graph_results", []):
            nodes = path.get("nodes", [])
            relationships = path.get("relationships", [])

            if not nodes:
                continue

            parts = [str(nodes[0])]

            for i, relation in enumerate(relationships):
                if i + 1 < len(nodes):
                    parts.append(
                        f"-[{relation}]-> {nodes[i + 1]}"
                    )

            graph_evidence.append(" ".join(parts))

        # Format ChromaDB text evidence.
        text_evidence = []

        for item in retrieval_result.get("vector_results", []):
            if isinstance(item, str):
                text_evidence.append(item)
            elif isinstance(item, dict) and item.get("document"):
                text_evidence.append(str(item["document"]))

        if not graph_evidence and not text_evidence:
            return (
                "I cannot answer this question because "
                "no evidence was retrieved."
            )

        prompt = f"""
You are the answer-generation component of CRITIC-MedRAG.

USER QUESTION:
{query}

NEO4J GRAPH EVIDENCE:
{chr(10).join(graph_evidence) or "No graph evidence retrieved."}

CHROMADB TEXT EVIDENCE:
{chr(10).join(text_evidence) or "No text evidence retrieved."}

Instructions:
1. Answer using only the evidence supplied above.
2. Do not invent facts or relationships.
3. If evidence is incomplete, clearly state the limitation.
4. Distinguish explicit evidence from reasonable interpretation.
5. Do not treat synthetic demonstration data as clinical proof.
6. Do not provide diagnoses or personalized medical advice.
7. Keep the answer concise and identify when evidence is insufficient.
8. Do not claim a fact is verified; CRITIC will check the answer.

Return only the answer text.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        answer = (response.text or "").strip()

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty generated answer."
            )

        return answer
