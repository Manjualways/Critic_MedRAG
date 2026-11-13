from src.critic_pipeline import CriticPipeline


def main():
    pipeline = CriticPipeline(
        max_retries=2,
        top_k=3,
    )

    result = pipeline.run(
    query="What treatment is recommended for condition XYZ?",
    entity_name="Condition XYZ",
)

    print("\n" + "=" * 50)
    print("CRITIC-MedRAG FINAL RESULT")
    print("=" * 50)

    print("\nQuestion:")
    print(result["question"])

    print("\nFinal answer:")
    print(result["answer"])

    print("\nVerification status:")
    print(result["final_status"])

    print("\nTotal attempts:")
    print(result["attempts"])

    print("\nAttempt history:")
    for item in result["history"]:
        print(f"\nAttempt {item['attempt']}")
        print(f"Query: {item['retrieval_query']}")
        print(f"Answer: {item['answer']}")
        print(
            "Status:",
            item["verification"].get(
                "overall_status", "UNKNOWN"
            ),
        )


if __name__ == "__main__":
    main()