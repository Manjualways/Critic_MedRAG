import sys
from pathlib import Path
import graphviz
import streamlit as st
import pandas as pd

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm_extractor import MedicalExtractor
from src.validate_extraction import validate_extraction
from src.neo4j_loader import Neo4jLoader


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CRITIC-MedRAG",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧠 CRITIC-MedRAG")
st.subheader("Automated Medical Knowledge Extraction")

st.markdown(
    """
    **Medical Text → Gemini LLM → Entity & Relationship Extraction
    → Validation → Neo4j Knowledge Graph**
    """
)

st.divider()


# --------------------------------------------------
# Load medical document
# --------------------------------------------------

DATA_FILE = PROJECT_ROOT / "data" / "sample_medical_notes.txt"

try:

    medical_text = DATA_FILE.read_text(
        encoding="utf-8"
    )

except Exception as e:

    st.error(f"Unable to load medical document: {e}")
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("⚙️ Configuration")

st.sidebar.success(" Pipeline Ready")

st.sidebar.markdown(
    """
    **Pipeline**

    1. Medical Text
    2. Gemini LLM
    3. Entity Extraction
    4. Relationship Extraction
    5. Validation
    6. Neo4j
    """
)


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.header("1️⃣ Medical Document")

st.text_area(
    "Input medical text",
    value=medical_text,
    height=250,
    key="medical_text"
)


# --------------------------------------------------
# Run extraction
# --------------------------------------------------

run_extraction = st.button(
    "🚀 Run Extraction",
    type="primary",
    use_container_width=True
)


if run_extraction:

    try:

        # --------------------------------------------------
        # Step 1 — LLM Extraction
        # --------------------------------------------------

        with st.spinner(
            "Gemini is extracting entities and relationships..."
        ):

            extractor = MedicalExtractor()

            raw_result = extractor.extract(
                st.session_state.medical_text
            )

        st.success(
            "✅ LLM extraction completed"
        )


        # --------------------------------------------------
        # Step 2 — Validation
        # --------------------------------------------------

        with st.spinner(
            "Validating and normalizing extracted information..."
        ):

            validated_result = validate_extraction(
                raw_result
            )

        st.success(
            "✅ Validation completed"
        )


        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        st.session_state["validated_result"] = (
            validated_result
        )


        # --------------------------------------------------
        # Step 3 — Insert into Neo4j
        # --------------------------------------------------

        with st.spinner(
            "Updating Neo4j knowledge graph..."
        ):

            loader = Neo4jLoader()

            try:

                loader.insert_entities(
                    validated_result.entities
                )

                loader.insert_relationships(
                    validated_result.relationships
                )

            finally:

                loader.close()


        st.success(
            "✅ Neo4j knowledge graph updated"
        )


    except Exception as e:

        st.error(
            f"❌ Pipeline failed: {e}"
        )


# --------------------------------------------------
# Display results
# --------------------------------------------------

if "validated_result" in st.session_state:

    result = st.session_state[
        "validated_result"
    ]

    st.divider()

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    st.header("2️⃣ Extraction Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Entities",
        len(result.entities)
    )

    col2.metric(
        "Relationships",
        len(result.relationships)
    )

    col3.metric(
        "Pipeline Status",
        "SUCCESS"
    )


    # --------------------------------------------------
    # Entities
    # --------------------------------------------------

    st.header("3️⃣ Extracted Entities")

    entity_data = []

    for entity in result.entities:

        entity_data.append(
            {
                "Entity": entity.name,
                "Type": entity.type
            }
        )

    entity_df = pd.DataFrame(
        entity_data
    )

    st.dataframe(
        entity_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------
    # Entity distribution
    # --------------------------------------------------

    st.subheader(
        "Entity Type Distribution"
    )

    entity_counts = (
        entity_df["Type"]
        .value_counts()
        .reset_index()
    )

    entity_counts.columns = [
        "Entity Type",
        "Count"
    ]

    st.bar_chart(
        entity_counts.set_index(
            "Entity Type"
        )
    )


    # --------------------------------------------------
    # Relationships
    # --------------------------------------------------

    st.header(
        "4️⃣ Extracted Relationships"
    )

    relationship_data = []

    for relationship in result.relationships:

        relationship_data.append(
            {
                "Source": relationship.source,
                "Relationship": relationship.relation,
                "Target": relationship.target
            }
        )

    relationship_df = pd.DataFrame(
        relationship_data
    )

    st.dataframe(
        relationship_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------
    # Relationship distribution
    # --------------------------------------------------

    st.subheader(
        "Relationship Type Distribution"
    )

    relationship_counts = (
        relationship_df["Relationship"]
        .value_counts()
        .reset_index()
    )

    relationship_counts.columns = [
        "Relationship",
        "Count"
    ]

    st.bar_chart(
        relationship_counts.set_index(
            "Relationship"
        )
    )
from src.hybrid_retriever import HybridRetriever
from src.verification import verify_hybrid_answer


def build_graphviz(graph_results):
    """Create a graph diagram from retrieved Neo4j paths."""
    lines = [
        "digraph CRITIC {",
        'rankdir="LR";',
        'node [shape=box, style="rounded,filled", fillcolor="#E8F1FC"];',
        'edge [fontsize=10];',
    ]

    for path_index, result in enumerate(graph_results):
        nodes = result.get("nodes", [])
        relationships = result.get("relationships", [])

        for i, node in enumerate(nodes):
            node_id = f"p{path_index}_n{i}"
            safe_name = str(node).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{node_id} [label="{safe_name}"];')

        for i, relationship in enumerate(relationships):
            if i + 1 >= len(nodes):
                continue

            source = f"p{path_index}_n{i}"
            target = f"p{path_index}_n{i + 1}"
            safe_relation = str(relationship).replace('"', '\\"')

            lines.append(
                f'{source} -> {target} [label="{safe_relation}"];'
            )

    lines.append("}")
    return "\n".join(lines)


st.title("CRITIC-MedRAG")
st.caption(
    "Hybrid retrieval • Evidence-grounded verification • Explainability"
)

st.info(
    "Research prototype using synthetic medical demonstration data. "
    "Results are not clinical validation or medical advice."
)

st.subheader("Ask a question")

with st.form("critic_verification_form"):
    query = st.text_input(
        "Medical query",
        value="Which drug interacts with Metformin?",
    )

    entity_name = st.text_input(
        "Graph entity",
        value="Metformin",
    )

    answer = st.text_area(
        "Answer to verify",
        value="Metformin interacts with Warfarin.",
        help="Enter an answer to test against retrieved evidence.",
    )

    top_k = st.slider(
        "Number of vector results",
        min_value=1,
        max_value=10,
        value=3,
    )

    submitted = st.form_submit_button(
        "Retrieve and verify",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not query.strip() or not entity_name.strip() or not answer.strip():
        st.warning("Enter a query, graph entity, and answer.")
    else:
        try:
            with st.spinner("Retrieving evidence and running CRITIC..."):
                retriever = HybridRetriever()

                retrieval_result = retriever.retrieve(
                    query=query,
                    entity_name=entity_name,
                    top_k=top_k,
                )

                verification = verify_hybrid_answer(
                    answer=answer,
                    retrieval_result=retrieval_result,
                )

                st.session_state["critic_dashboard_result"] = {
                    "query": query,
                    "answer": answer,
                    "retrieval": retrieval_result,
                    "verification": verification,
                }

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")


# Keep results visible after Streamlit reruns.
data = st.session_state.get("critic_dashboard_result")

if data:
    retrieval = data["retrieval"]
    verification = data["verification"]

    st.divider()
    st.subheader("Verification overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Graph paths",
        len(retrieval.get("graph_results", [])),
    )

    col2.metric(
        "Text results",
        len(retrieval.get("vector_results", [])),
    )

    col3.metric(
        "Claims checked",
        len(verification.get("claims", [])),
    )

    st.markdown("### Answer")
    st.write(data["answer"])

    status = verification["overall_status"]
    needs_retrieval = verification["needs_retrieval"]

    if status == "SUPPORTED":
        st.success("CRITIC status: SUPPORTED")
    elif status == "UNSUPPORTED":
        st.error("CRITIC status: UNSUPPORTED")
    else:
        st.warning("CRITIC status: INSUFFICIENT EVIDENCE")

    if needs_retrieval:
        st.warning(
            "Additional retrieval is recommended. "
            "Automatic query rewriting is not triggered by this dashboard yet."
        )
    else:
        st.success("No additional retrieval requested.")

    st.caption(verification.get("summary", ""))

    st.divider()
    st.subheader("Claim-level verification")

    for index, claim in enumerate(verification.get("claims", []), start=1):
        with st.expander(
            f"Claim {index}: {claim['status']} — {claim['claim']}",
            expanded=True,
        ):
            st.write("**Explanation**")
            st.write(claim.get("explanation", ""))

            st.write("**Evidence references and content**")
            evidence = claim.get("evidence", [])

            if evidence:
                for evidence_index, item in enumerate(evidence, start=1):
                    st.markdown(f"**Evidence {evidence_index}**")
                    st.write(item)
            else:
                st.info("No supporting or contradicting evidence was cited.")

    st.divider()
    st.subheader("Neo4j graph explanation")

    graph_results = retrieval.get("graph_results", [])

    if graph_results:
        st.graphviz_chart(
            build_graphviz(graph_results),
            use_container_width=True,
        )

        for index, path in enumerate(graph_results, start=1):
            with st.expander(f"Graph path {index}"):
                nodes = path.get("nodes", [])
                relationships = path.get("relationships", [])

                for i, node in enumerate(nodes):
                    st.write(f"**{node}**")
                    if i < len(relationships):
                        st.caption(f"— {relationships[i]} →")
    else:
        st.info("No graph paths were retrieved.")

    st.divider()
    st.subheader("ChromaDB text evidence")

    vector_results = retrieval.get("vector_results", [])

    if vector_results:
        for index, item in enumerate(vector_results, start=1):
            with st.expander(f"Retrieved document {index}"):
                if isinstance(item, dict):
                    st.write(item.get("document", str(item)))
                    if item.get("metadata"):
                        st.json(item["metadata"])
                else:
                    st.write(str(item))
    else:
        st.info("No text evidence was retrieved.")

# The graph requires a validated extraction from the session.
if "validated_result" not in st.session_state:

    st.info(
        "Run the extraction to view the knowledge graph."
    )

    st.stop()

# --------------------------------------------------
# Knowledge Graph Visualization
# --------------------------------------------------

st.header("5️⃣ Knowledge Graph")

st.info(
    "Select an entity to explore its connected medical "
    "knowledge graph."
)


# --------------------------------------------------
# Prepare graph data
# --------------------------------------------------

all_entities = sorted(
    {
        entity.name
        for entity in result.entities
    }
)

selected_entity = st.selectbox(
    "🔎 Select an entity",
    all_entities
)


# --------------------------------------------------
# Find connected relationships
# --------------------------------------------------

connected_relationships = []

for relationship in result.relationships:

    if (
        relationship.source.lower()
        == selected_entity.lower()
        or
        relationship.target.lower()
        == selected_entity.lower()
    ):

        connected_relationships.append(
            relationship
        )


# --------------------------------------------------
# Graph statistics
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Selected Entity",
    selected_entity
)

col2.metric(
    "Connections",
    len(connected_relationships)
)

col3.metric(
    "Total Relationships",
    len(result.relationships)
)


# --------------------------------------------------
# Build Graphviz graph
# --------------------------------------------------

import graphviz


graph = graphviz.Digraph()

graph.attr(
    rankdir="LR",
    bgcolor="transparent",
    nodesep="0.7",
    ranksep="1.0"
)


# --------------------------------------------------
# Entity type lookup
# --------------------------------------------------

entity_type_map = {
    entity.name.lower(): entity.type
    for entity in result.entities
}


# --------------------------------------------------
# Node style
# --------------------------------------------------

def get_node_style(entity_name):

    entity_type = entity_type_map.get(
        entity_name.lower(),
        "Condition"
    )

    if entity_type == "Disease":

        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#DCEBFF"
        }

    elif entity_type == "Drug":

        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#DFF5E1"
        }

    elif entity_type == "Symptom":

        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#FFF0CC"
        }

    elif entity_type == "LabTest":

        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#E8DFFF"
        }

    elif entity_type == "Procedure":

        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#FFE1D6"
        }

    else:

        return {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#EEEEEE"
        }


# --------------------------------------------------
# Add selected entity
# --------------------------------------------------

style = get_node_style(
    selected_entity
)

entity_type = entity_type_map.get(
    selected_entity.lower(),
    "Unknown"
)

graph.node(
    selected_entity,
    label=f"{selected_entity}\n[{entity_type}]",
    **style
)


# --------------------------------------------------
# Add connected entities and relationships
# --------------------------------------------------

added_nodes = {
    selected_entity.lower()
}

for relationship in connected_relationships:

    source = relationship.source
    target = relationship.target
    relation = relationship.relation

    # Add source node
    if source.lower() not in added_nodes:

        source_style = get_node_style(
            source
        )

        source_type = entity_type_map.get(
            source.lower(),
            "Unknown"
        )

        graph.node(
            source,
            label=f"{source}\n[{source_type}]",
            **source_style
        )

        added_nodes.add(
            source.lower()
        )


    # Add target node
    if target.lower() not in added_nodes:

        target_style = get_node_style(
            target
        )

        target_type = entity_type_map.get(
            target.lower(),
            "Unknown"
        )

        graph.node(
            target,
            label=f"{target}\n[{target_type}]",
            **target_style
        )

        added_nodes.add(
            target.lower()
        )


    # Add relationship
    graph.edge(
        source,
        target,
        label=relation
    )


# --------------------------------------------------
# Display graph
# --------------------------------------------------

st.graphviz_chart(
    graph,
    use_container_width=True
)


# --------------------------------------------------
# Connected relationship table
# --------------------------------------------------

st.subheader(
    "🔗 Connected Relationships"
)

if connected_relationships:

    graph_table = []

    for relationship in connected_relationships:

        graph_table.append(
            {
                "Source": relationship.source,
                "Relationship": relationship.relation,
                "Target": relationship.target
            }
        )

    graph_df = pd.DataFrame(
        graph_table
    )

    st.dataframe(
        graph_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No relationships found for this entity."
    )

    # --------------------------------------------------
    # Raw JSON
    # --------------------------------------------------

    st.header(
        "6️⃣ Structured Extraction JSON"
    )

    st.json(
        result.model_dump()
    )


    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    st.header(
        "7️⃣ Objective 2 Status"
    )

    st.success(
        """
        Objective 2 pipeline completed successfully.

        ✓ Medical text processed

        ✓ LLM-based entity extraction

        ✓ LLM-based relationship extraction

        ✓ Entity validation

        ✓ Entity normalization

        ✓ Relationship validation

        ✓ Neo4j knowledge graph construction
        """
    )
    


