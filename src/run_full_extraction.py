from pathlib import Path

from src.llm_extractor import MedicalExtractor
from src.validate_extraction import validate_extraction
from src.neo4j_loader import Neo4jLoader


DATA_FILE = Path("data/sample_medical_notes.txt")


def main():

    print("\n==============================")
    print("CRITIC-MedRAG OBJECTIVE 2")
    print("==============================")

    medical_text = DATA_FILE.read_text(
        encoding="utf-8"
    )

    print(
        f"\nLoaded {len(medical_text)} characters."
    )

    extractor = MedicalExtractor()

    print("\nExtracting entities and relationships...")

    raw_result = extractor.extract(
        medical_text
    )

    print("\nLLM extraction completed.")

    print("\nValidating extraction...")

    validated_result = validate_extraction(
        raw_result
    )

    print(
        f"Entities: {len(validated_result.entities)}"
    )

    print(
        f"Relationships: "
        f"{len(validated_result.relationships)}"
    )

    loader = Neo4jLoader()

    try:

        print("\nInserting into Neo4j...")

        loader.insert_entities(
            validated_result.entities
        )

        loader.insert_relationships(
            validated_result.relationships
        )

        print("\nNeo4j insertion successful!")

    finally:

        loader.close()


if __name__ == "__main__":
    main()