
import json
import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)


class ClaimReview(BaseModel):
    claim: str
    status: Literal[
        "SUPPORTED",
        "UNSUPPORTED",
        "INSUFFICIENT_EVIDENCE",
    ]
    evidence: list[str] = Field(default_factory=list)
    explanation: str


class VerificationResult(BaseModel):
    overall_status: Literal[
        "SUPPORTED",
        "UNSUPPORTED",
        "INSUFFICIENT_EVIDENCE",
    ]
    claims: list[ClaimReview]
    needs_retrieval: bool
    summary: str


def verify_answer(
    answer: str,
    text_evidence: list[str],
    graph_evidence: list[str] | None = None,
) -> dict:
    """Verify individual claims against supplied evidence."""

    if not answer or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    graph_evidence = graph_evidence or []

    # Give every source a stable ID.
    evidence_catalog = {}

    for item in text_evidence:
        if item and item.strip():
            evidence_catalog[f"T{len(evidence_catalog) + 1}"] = {
                "source_type": "retrieved_text",
                "content": item.strip(),
            }

    graph_index = 1
    for item in graph_evidence:
        if item and item.strip():
            evidence_catalog[f"G{graph_index}"] = {
                "source_type": "knowledge_graph",
                "content": item.strip(),
            }
            graph_index += 1

    if not evidence_catalog:
        return VerificationResult(
            overall_status="INSUFFICIENT_EVIDENCE",
            claims=[],
            needs_retrieval=True,
            summary="No evidence was supplied for verification.",
        ).model_dump()

    prompt = f"""
You are CRITIC, an evidence-grounding verifier.

ANSWER TO VERIFY:
{answer}

AVAILABLE EVIDENCE:
{json.dumps(evidence_catalog, ensure_ascii=False, indent=2)}

For every material factual claim in the answer:

1. Split compound sentences into separate factual claims.
2. Evaluate each claim using ONLY the supplied evidence.
3. Return the IDs of evidence that directly supports or
   explicitly contradicts the claim.
4. Never invent evidence IDs, quotations, or sources.

STATUS DEFINITIONS:

SUPPORTED:
The cited evidence directly supports the entire claim.
A partially supported claim must not pass as SUPPORTED.

UNSUPPORTED:
The cited evidence explicitly contradicts the claim.
Do not use this status merely because a claim is absent
from the retrieved evidence.

INSUFFICIENT_EVIDENCE:
Evidence is absent, incomplete, ambiguous, or does not
establish whether the claim is true or false.

ADDITIONAL RULES:

- A graph path supports only the relationships it explicitly
  represents. Do not infer extra relationships.
- A source mentioning one drug does not prove that another
  drug has no interaction with it.
- Do not use general medical knowledge to fill evidence gaps.
- Do not treat synthetic demonstration data as clinical proof.
- Do not provide a diagnosis or medical advice.
- If only part of a claim is supported, do not mark the
  entire claim SUPPORTED.
- Return one JSON object with these fields:
  claims: array of objects containing claim, status,
  evidence_ids, explanation.
  summary: brief explanation of the verification result.
- Each status must be SUPPORTED, UNSUPPORTED, or
  INSUFFICIENT_EVIDENCE.
- evidence_ids must contain only IDs from AVAILABLE EVIDENCE.
- Return JSON only, without Markdown fences.

IMPORTANT VERIFICATION RULES:
1. Evaluate whether the answer directly answers the user's question.
2. A statement saying that evidence is missing may itself be supported,
   but it does not mean the original question was answered.
3. If the answer explicitly says evidence is insufficient to answer
   the question, the overall status must be INSUFFICIENT_EVIDENCE.
4. Do not mark the overall answer SUPPORTED merely because some
   statements within it are supported.
5. Overall SUPPORTED requires that the substantive answer to the
   original question is supported by the retrieved evidence.
6. Do not infer an answer from unrelated evidence.
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    if not response.text:
        raise RuntimeError(
            "The verification model returned an empty response."
        )

    try:
        data = json.loads(response.text)
        raw_claims = data["claims"]

        if not isinstance(raw_claims, list):
            raise ValueError("Claims must be a list.")

        reviews = []

        for item in raw_claims:
            claim = str(item["claim"]).strip()
            status = item["status"]
            explanation = str(item.get("explanation", ""))

            if status not in {
                "SUPPORTED",
                "UNSUPPORTED",
                "INSUFFICIENT_EVIDENCE",
            }:
                raise ValueError("Invalid claim status.")

            # Only accept IDs that actually exist in our catalog.
            evidence_ids = item.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                evidence_ids = []

            valid_ids = list(dict.fromkeys(
                evidence_id
                for evidence_id in evidence_ids
                if isinstance(evidence_id, str)
                and evidence_id in evidence_catalog
            ))

            cited_evidence = [
                evidence_catalog[evidence_id]["content"]
                for evidence_id in valid_ids
            ]

            # A claim cannot pass or be contradicted without
            # an identifiable evidence source.
            if status in {"SUPPORTED", "UNSUPPORTED"} and not cited_evidence:
                status = "INSUFFICIENT_EVIDENCE"
                explanation = (
                    "No valid evidence reference was supplied; "
                    "the claim cannot be verified."
                )

            reviews.append(
                ClaimReview(
                    claim=claim,
                    status=status,
                    evidence=cited_evidence,
                    explanation=explanation,
                )
            )

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            "CRITIC returned invalid structured output."
        ) from exc

    if not reviews:
        raise ValueError(
            "CRITIC returned no claims; verification cannot pass."
        )

    statuses = {review.status for review in reviews}

    if statuses == {"SUPPORTED"}:
        overall = "SUPPORTED"
    elif "UNSUPPORTED" in statuses:
        overall = "UNSUPPORTED"
    else:
        overall = "INSUFFICIENT_EVIDENCE"

    result = VerificationResult(
        overall_status=overall,
        claims=reviews,
        needs_retrieval=(overall != "SUPPORTED"),
        summary=str(data.get("summary", "")),
    ).model_dump()

    answer_lower = answer.lower()

    limitation_phrases = [
        "evidence is insufficient",
        "insufficient evidence",
        "evidence is incomplete",
        "cannot answer this question",
        "not enough evidence",
        "does not provide enough information",
    ]

    has_limitation = any(
        phrase in answer_lower for phrase in limitation_phrases
    )

    if has_limitation and result["overall_status"] == "SUPPORTED":
        result["overall_status"] = "INSUFFICIENT_EVIDENCE"
        result["needs_retrieval"] = True
        result["summary"] = (
            "The answer acknowledges insufficient evidence; "
            "the original question is not fully answered."
        )

    return result


def verify_hybrid_answer(
    answer: str,
    retrieval_result: dict,
) -> dict:
    """Adapt HybridRetriever results for CRITIC."""

    text_evidence = []
    graph_evidence = []

    for result in retrieval_result.get("graph_results", []):
        nodes = result.get("nodes", [])
        relationships = result.get("relationships", [])

        if not nodes:
            continue

        parts = [str(nodes[0])]

        for index, relation in enumerate(relationships):
            if index + 1 < len(nodes):
                parts.append(
                    f"-[{relation}]-> {nodes[index + 1]}"
                )

        graph_evidence.append(" ".join(parts))

    for result in retrieval_result.get("vector_results", []):
        if isinstance(result, str):
            text_evidence.append(result)
        elif isinstance(result, dict):
            document = result.get("document")
            if document:
                text_evidence.append(str(document))

    return verify_answer(
        answer=answer,
        text_evidence=text_evidence,
        graph_evidence=graph_evidence,
    )
