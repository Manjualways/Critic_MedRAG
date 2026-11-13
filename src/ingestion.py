from .graph_schema import NODE_TYPES, RELATIONSHIP_TYPES
from .neo4j_client import Neo4jClient
from .extraction import ExtractionResult

def validate_result(result: ExtractionResult):
    for entity in result.entities:
        if entity.type not in NODE_TYPES:
            raise ValueError(f"Unsupported entity type: {entity.type}")

    for triple in result.triples:
        if triple.subject_type not in NODE_TYPES:
            raise ValueError(f"Unsupported subject type: {triple.subject_type}")
        if triple.object_type not in NODE_TYPES:
            raise ValueError(f"Unsupported object type: {triple.object_type}")
        if triple.relation not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unsupported relationship: {triple.relation}")

def load_to_neo4j(result: ExtractionResult, source: str = ""):
    validate_result(result)

    client = Neo4jClient()
    try:
        client.verify()
        client.create_constraints()

        for triple in result.triples:
            client.upsert_triple(
                triple.subject,
                triple.subject_type,
                triple.relation,
                triple.object,
                triple.object_type,
                source=source,
            )

        return client.graph_summary(), client.sample_paths()
    finally:
        client.close()
