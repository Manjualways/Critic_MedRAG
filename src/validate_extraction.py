from src.extraction_schema import ExtractionResult
from src.entity_normalizer import normalize_entity_name


def validate_extraction(data):

    # --------------------------------------------------
    # Normalize relationship field name
    # --------------------------------------------------

    for relationship in data.get("relationships", []):

        # Gemini may return "relationship"
        # instead of our expected "relation"

        if "relation" not in relationship:

            if "relationship" in relationship:

                relationship["relation"] = (
                    relationship.pop("relationship")
                )

            elif "type" in relationship:

                relationship["relation"] = (
                    relationship.pop("type")
                )

    # --------------------------------------------------
    # Normalize entity names
    # --------------------------------------------------

    for entity in data.get("entities", []):

        entity["name"] = normalize_entity_name(
            entity["name"]
        )

    # --------------------------------------------------
    # Normalize relationship entity names
    # --------------------------------------------------

    for relationship in data.get("relationships", []):

        relationship["source"] = normalize_entity_name(
            relationship["source"]
        )

        relationship["target"] = normalize_entity_name(
            relationship["target"]
        )

    # --------------------------------------------------
    # Validate structure
    # --------------------------------------------------

    result = ExtractionResult(
        **data
    )

    # --------------------------------------------------
    # Remove duplicate entities
    # --------------------------------------------------

    unique_entities = {}

    for entity in result.entities:

        key = (
            entity.name.lower(),
            entity.type
        )

        if key not in unique_entities:

            unique_entities[key] = entity

    result.entities = list(
        unique_entities.values()
    )

    # --------------------------------------------------
    # Valid entity names
    # --------------------------------------------------

    entity_names = {
        entity.name.lower()
        for entity in result.entities
    }

    # --------------------------------------------------
    # Validate relationships
    # --------------------------------------------------

    valid_relationships = []

    for relationship in result.relationships:

        source_exists = (
            relationship.source.lower()
            in entity_names
        )

        target_exists = (
            relationship.target.lower()
            in entity_names
        )

        if source_exists and target_exists:

            valid_relationships.append(
                relationship
            )

    # --------------------------------------------------
    # Remove duplicate relationships
    # --------------------------------------------------

    unique_relationships = {}

    for relationship in valid_relationships:

        key = (
            relationship.source.lower(),
            relationship.relation,
            relationship.target.lower()
        )

        if key not in unique_relationships:

            unique_relationships[key] = relationship

    result.relationships = list(
        unique_relationships.values()
    )

    return result