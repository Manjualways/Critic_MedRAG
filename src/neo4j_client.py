from neo4j import GraphDatabase
from .config import settings

class Neo4jClient:
    def __init__(self):
        if not settings.neo4j_uri or not settings.neo4j_password:
            raise ValueError(
                "Neo4j is not configured. Set NEO4J_URI and NEO4J_PASSWORD in .env."
            )
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.database = settings.neo4j_database

    def verify(self):
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def create_constraints(self):
        queries = [
            "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT labtest_name_unique IF NOT EXISTS FOR (n:LabTest) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT doctor_name_unique IF NOT EXISTS FOR (n:Doctor) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT patient_name_unique IF NOT EXISTS FOR (n:Patient) REQUIRE n.name IS UNIQUE",
        ]
        for query in queries:
            self.driver.execute_query(query, database_=self.database)

    def upsert_triple(self, subject, subject_type, relation, object_, object_type, source=None):
        # Labels/types are validated before this method is called.
        query = f"""
        MERGE (s:{subject_type} {{name: $subject}})
        MERGE (o:{object_type} {{name: $object}})
        MERGE (s)-[r:{relation}]->(o)
        SET r.source = $source
        RETURN s.name AS subject, type(r) AS relation, o.name AS object
        """
        records, _, _ = self.driver.execute_query(
            query,
            subject=subject,
            object=object_,
            source=source or "",
            database_=self.database,
        )
        return [record.data() for record in records]

    def graph_summary(self):
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships
        """
        records, _, _ = self.driver.execute_query(query, database_=self.database)
        return records[0].data()

    def sample_paths(self, limit=20):
        query = """
        MATCH (a)-[r]->(b)
        RETURN labels(a) AS from_labels, a.name AS subject,
               type(r) AS relation, labels(b) AS to_labels, b.name AS object
        ORDER BY subject, relation
        LIMIT $limit
        """
        records, _, _ = self.driver.execute_query(
            query, limit=limit, database_=self.database
        )
        return [record.data() for record in records]
