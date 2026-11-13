from src.llm_extractor import MedicalExtractor
from src.validate_extraction import validate_extraction
from src.neo4j_loader import Neo4jLoader


medical_text = """
Diabetes mellitus is associated with elevated blood glucose.
Metformin is commonly used for type 2 diabetes.
Metformin interacts with Warfarin.
HbA1c and blood glucose tests can be used to assess
glucose-related status.
"""


print("\n==============================")
print("STEP 1: LLM EXTRACTION")
print("==============================")

extractor = MedicalExtractor()

raw_result = extractor.extract(
    medical_text
)

print(raw_result)


print("\n==============================")
print("STEP 2: VALIDATION")
print("==============================")

validated_result = validate_extraction(
    raw_result
)

print(
    validated_result.model_dump_json(
        indent=2
    )
)


print("\n==============================")
print("STEP 3: NEO4J INSERTION")
print("==============================")

loader = Neo4jLoader()

try:

    loader.insert_entities(
        validated_result.entities
    )

    loader.insert_relationships(
        validated_result.relationships
    )

    print("Entities inserted:", len(
        validated_result.entities
    ))

    print("Relationships inserted:", len(
        validated_result.relationships
    ))

    print("\nNeo4j insertion successful!")

finally:

    loader.close()