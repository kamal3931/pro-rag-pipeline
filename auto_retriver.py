import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# 1. Setup
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")
openai_key = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(model="gpt-4o", api_key=openai_key)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_key)


def run_auto_query(query_text, section_filter=None):
    client = QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(collection_name="sandisk_report", client=client)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # 2. Industry-Ready Metadata Filtering
    filters = None
    if section_filter:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="section", value=section_filter)]
        )

    # 3. Create the Query Engine with Filtering Capabilities
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters  # This is the "Auto-Retriever" magic
    )

    response = query_engine.query(query_text)

    print(f"\n🔍 QUERY: {query_text}")
    print(f"🤖 ANSWER: {response}")
    return response


if __name__ == "__main__":
    # Example 1: General Query
    run_auto_query("What are the requirements for director independence?")  # [cite: 522]

    # Example 2: Targetted Metadata Query (The "Level Up")
    # This forces the AI to look only at the Compensation section [cite: 1024]
    run_auto_query("Describe the 2H Fiscal 2025 short-term incentive program.", section_filter="Executive Compensation")