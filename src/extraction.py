from typing import List
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from .config import settings
from .graph_schema import normalize_label, normalize_relation

class Entity(BaseModel):
    name: str = Field(min_length=1)
    type: str

class Triple(BaseModel):
    subject: str = Field(min_length=1)
    subject_type: str
    relation: str
    object: str = Field(min_length=1)
    object_type: str

class ExtractionResult(BaseModel):
    entities: List[Entity] = []
    triples: List[Triple] = []

SYSTEM_PROMPT = """
You extract a small medical knowledge graph from text.

Allowed entity types:
Disease, Symptom, Drug, LabTest, Doctor, Patient

Allowed relationship types:
TREATED_BY, INTERACTS_WITH, COMORBID_WITH, INDICATES, REQUIRES_TEST

Rules:
- Extract only relationships explicitly supported by the supplied text.
- Do not infer treatment, diagnosis, interaction, or causality that is not stated.
- Normalize entity names.
- Return JSON only with keys: entities and triples.
"""

def extract_with_llm(text: str) -> ExtractionResult:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )

    payload = json.loads(response.choices[0].message.content)
    result = ExtractionResult.model_validate(payload)

    for entity in result.entities:
        entity.type = normalize_label(entity.type)

    for triple in result.triples:
        triple.subject_type = normalize_label(triple.subject_type)
        triple.object_type = normalize_label(triple.object_type)
        triple.relation = normalize_relation(triple.relation)

    return result

def demo_extraction(text: str) -> ExtractionResult:
    entities = [
        Entity(name="Diabetes mellitus", type="Disease"),
        Entity(name="Metformin", type="Drug"),
        Entity(name="Warfarin", type="Drug"),
        Entity(name="Fatigue", type="Symptom"),
        Entity(name="Anemia", type="Disease"),
        Entity(name="CBC", type="LabTest"),
    ]
    triples = [
        Triple(
            subject="Diabetes mellitus", subject_type="Disease",
            relation="TREATED_BY", object="Metformin", object_type="Drug"
        ),
        Triple(
            subject="Metformin", subject_type="Drug",
            relation="INTERACTS_WITH", object="Warfarin", object_type="Drug"
        ),
        Triple(
            subject="Fatigue", subject_type="Symptom",
            relation="INDICATES", object="Anemia", object_type="Disease"
        ),
        Triple(
            subject="Anemia", subject_type="Disease",
            relation="REQUIRES_TEST", object="CBC", object_type="LabTest"
        ),
    ]
    return ExtractionResult(entities=entities, triples=triples)
