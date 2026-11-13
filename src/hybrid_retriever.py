from .neo4j_client import Neo4jClient
from .vector_store import VectorStore


class HybridRetriever:

    def __init__(self):
        self.vector_store = VectorStore()

    def graph_search(self, entity_name, max_hops=3):

        client = Neo4jClient()

        try:
            client.verify()

            query = """
            MATCH path = (start)-[*1..3]->(target)
            WHERE toLower(start.name) = toLower($entity_name)
            RETURN
                [node IN nodes(path) | node.name] AS nodes,
                [rel IN relationships(path) | type(rel)] AS relationships
            LIMIT 10
            """

            records, _, _ = client.driver.execute_query(
                query,
                entity_name=entity_name,
                database_=client.database
            )

            return [
                record.data()
                for record in records
            ]

        finally:
            client.close()

    def vector_search(self, query, top_k=3):

        return self.vector_store.search(
            query,
            top_k=top_k
        )

    def retrieve(self, query, entity_name, top_k=3):

        print("\n==============================")
        print("QUERY-AWARE HYBRID RETRIEVAL")
        print("==============================")

        print(f"\nQuery: {query}")
        print(f"Graph entity: {entity_name}")

        # -------------------------
        # Neo4j retrieval
        # -------------------------

        print("\n[1] Neo4j Graph Retrieval")

        graph_results = self.graph_search(
            entity_name
        )

        for result in graph_results:

            print(
                f"PATH: {result['nodes']}"
            )

            print(
                f"RELATIONSHIPS: "
                f"{result['relationships']}"
            )

        # -------------------------
        # ChromaDB retrieval
        # -------------------------

        print("\n[2] ChromaDB Vector Retrieval")

        vector_results = self.vector_search(
            query,
            top_k
        )

        for result in vector_results:

            print(
                f"TEXT: {result['document']}"
            )

        return {
            "query": query,
            "entity": entity_name,
            "graph_results": graph_results,
            "vector_results": vector_results
        }