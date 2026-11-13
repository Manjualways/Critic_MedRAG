import os

from dotenv import load_dotenv
from google import genai

from src.hybrid_retriever import HybridRetriever
from src.answer_generator import AnswerGenerator
from src.verification import verify_hybrid_answer

load_dotenv()


class CriticPipeline:
    """
    End-to-end CRITIC-MedRAG pipeline:
    hybrid retrieval -> answer generation -> verification
    -> bounded self-correction.
    """

    def __init__(self, max_retries: int = 2, top_k: int = 3):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.max_retries = max_retries
        self.top_k = top_k

        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()

        api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv(
            "GEMINI_MODEL", "gemini-3.5-flash-lite"
        )

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        self.client = genai.Client(api_key=api_key)

    def rewrite_query(self, original_query: str, verification: dict) -> str:
        """Rewrite the question to target missing or unsupported evidence."""

        feedback = verification.get("summary", "")
        claims = verification.get("claims", [])

        claim_feedback = []
        for claim in claims:
            status = claim.get("status", "UNKNOWN")
            if status != "SUPPORTED":
                claim_feedback.append(
                    f"- {claim.get('claim', '')} "
                    f"(status: {status}; "
                    f"reason: {claim.get('explanation', '')})"
                )

        prompt = f"""
You are rewriting a question for a medical knowledge retrieval system.

Original question:
{original_query}

Verification summary:
{feedback}

Claims needing more evidence:
{chr(10).join(claim_feedback) or "The answer needs stronger evidence."}

Rules:
- Preserve the user's original intent.
- Focus the rewritten question on the missing evidence.
- Do not answer the question.
- Do not introduce new medical assumptions.
- Return only one concise search question.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        rewritten = (response.text or "").strip().strip('"')

        if not rewritten:
            return original_query

        return rewritten

    def run(
        self,
        query: str,
        entity_name: str | None = None,
    ) -> dict:
        """Run retrieval, generation, verification, and bounded retries."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        original_query = query.strip()
        current_query = original_query
        history = []
        final_answer = ""
        final_verification = {}
        final_retrieval = {}

        for attempt in range(self.max_retries + 1):
            print(
                f"\n[CRITIC-MedRAG] Attempt "
                f"{attempt + 1}/{self.max_retries + 1}"
            )
            print(f"Retrieval query: {current_query}")

            # 1. Retrieve evidence from the existing hybrid retriever.
            retrieval_result = self.retriever.retrieve(
                query=current_query,
                entity_name=entity_name,
                top_k=self.top_k,
            )

            # 2. Generate an answer grounded in retrieved evidence.
            answer = self.generator.generate(
                query=original_query,
                retrieval_result=retrieval_result,
            )

            verification = verify_hybrid_answer(
                answer,
                retrieval_result,
            )

            answer_lower = answer.lower()

            limitation_phrases = [
                "insufficient to answer",
                "evidence is insufficient",
                "evidence is incomplete",
                "no mention of",
                "cannot answer this question",
                "not enough evidence",
            ]

            has_limitation = any(
                phrase in answer_lower
                for phrase in limitation_phrases
            )

            if (
                has_limitation
                and verification.get("overall_status") == "SUPPORTED"
            ):
                verification["overall_status"] = "INSUFFICIENT_EVIDENCE"
                verification["needs_retrieval"] = True
                verification["summary"] = (
                    "The answer acknowledges a limitation or missing evidence. "
                    "The original question may not have been answered."
                )

            status = verification.get(
                "overall_status", "INSUFFICIENT_EVIDENCE"
            )
            history.append(
                {
                    "attempt": attempt + 1,
                    "retrieval_query": current_query,
                    "answer": answer,
                    "verification": verification,
                }
            )

            final_answer = answer
            final_verification = verification
            final_retrieval = retrieval_result

            print(f"Verification status: {status}")

            # Stop when all claims are supported.
            if status == "SUPPORTED":
                break

            # No more retries available.
            if attempt >= self.max_retries:
                print("Retry limit reached.")
                break

            # 4. Rewrite the query for a targeted retrieval attempt.
            current_query = self.rewrite_query(
                original_query,
                verification,
            )

            print(f"Rewritten query: {current_query}")

        return {
            "question": original_query,
            "answer": final_answer,
            "verification": final_verification,
            "retrieval": final_retrieval,
            "history": history,
            "attempts": len(history),
            "final_status": final_verification.get(
                "overall_status", "INSUFFICIENT_EVIDENCE"
            ),
        }