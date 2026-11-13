import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class Neo4jLoader:

    def __init__(self):

        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE")

        if not self.uri:
            raise ValueError("NEO4J_URI is missing")

        if not self.password:
            raise ValueError("NEO4J_PASSWORD is missing")

        if not self.database:
            raise ValueError("NEO4J_DATABASE is missing")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def close(self):
        self.driver.close()

    def insert_entities(self, entities):

        with self.driver.session(
            database=self.database
        ) as session:

            for entity in entities:

                # Entity type becomes Neo4j label
                label = entity.type

                query = f"""
                MERGE (n:{label} {{name: $name}})
                """

                session.run(
                    query,
                    name=entity.name
                )

    def insert_relationships(self, relationships):

        with self.driver.session(
            database=self.database
        ) as session:

            for relationship in relationships:

                query = f"""
                MATCH (source {{name: $source}})
                MATCH (target {{name: $target}})
                MERGE (source)-[:{relationship.relation}]->(target)
                """

                session.run(
                    query,
                    source=relationship.source,
                    target=relationship.target
                )