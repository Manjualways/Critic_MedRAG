import argparse
from pathlib import Path

from .document_parser import read_document
from .extraction import extract_with_llm, demo_extraction
from .ingestion import load_to_neo4j

def main():
    parser = argparse.ArgumentParser(description="CRITIC-MedRAG Week 2 ingestion")
    parser.add_argument(
        "--file",
        default="data/sample_medical_notes.txt",
        help="Path to .txt/.md/.pdf medical document",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM extraction instead of deterministic demo extraction",
    )
    args = parser.parse_args()

    text = read_document(args.file)
    print(f"Loaded {len(text)} characters from {args.file}")

    result = extract_with_llm(text) if args.llm else demo_extraction(text)

    print(f"Entities: {len(result.entities)}")
    print(f"Triples: {len(result.triples)}")

    for triple in result.triples:
        print(f"  ({triple.subject}) -[{triple.relation}]-> ({triple.object})")

    summary, paths = load_to_neo4j(result, source=Path(args.file).name)

    print("\nNeo4j graph summary:")
    print(summary)

    print("\nSample graph relationships:")
    for row in paths:
        print(f"  {row['subject']} -[{row['relation']}]-> {row['object']}")

if __name__ == "__main__":
    main()
