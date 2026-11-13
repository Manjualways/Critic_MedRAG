from src.neo4j_loader import Neo4jLoader


loader = Neo4jLoader()

try:
    with loader.driver.session(
        database=loader.database
    ) as session:

        result = session.run(
            "RETURN 1 AS test"
        )

        print("Neo4j connection successful!")
        print(result.single()["test"])

finally:
    loader.close()