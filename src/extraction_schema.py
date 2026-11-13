from pydantic import BaseModel, Field, field_validator
from typing import List


ALLOWED_ENTITY_TYPES = {
    "Disease",
    "Drug",
    "Symptom",
    "LabTest",
    "Procedure",
    "Condition",
}

ALLOWED_RELATION_TYPES = {
    "TREATED_BY",
    "HAS_SYMPTOM",
    "REQUIRES_TEST",
    "INTERACTS_WITH",
    "ASSOCIATED_WITH",
    "INDICATES",
    "MONITORED_BY",
}


class Entity(BaseModel):
    name: str = Field(min_length=1)
    type: str

    @field_validator("type")
    @classmethod
    def validate_entity_type(cls, value):
        if value not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {value}")
        return value

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        return " ".join(value.split()).strip()


class Relationship(BaseModel):
    source: str = Field(min_length=1)
    relation: str
    target: str = Field(min_length=1)

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value):
        if value not in ALLOWED_RELATION_TYPES:
            raise ValueError(f"Invalid relationship: {value}")
        return value

    @field_validator("source", "target")
    @classmethod
    def clean_entity_name(cls, value):
        return " ".join(value.split()).strip()


class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]