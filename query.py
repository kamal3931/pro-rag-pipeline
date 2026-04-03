import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Setup Environment
base_dir = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=base_dir / ".env")

# Retrieve and verify key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# 2. Configure global settings with EXPLICIT api_key
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=openai_key)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_key)


def run_query(query_text):
    # 3. Connect to existing Qdrant DB
    client = QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(collection_name="sandisk_report", client=client)

    # 4. Create index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # 5. Create the Query Engine
    query_engine = index.as_query_engine(similarity_top_k=5)

    response = query_engine.query(query_text)

    print(f"\n🔍 QUERY: {query_text}")
    print(f"🤖 ANSWER: {response}")

    # Pagination/Source check
    if hasattr(response, 'source_nodes'):
        print("\n📄 SOURCES:")
        for node in response.source_nodes:
            page = node.metadata.get('page_label', 'N/A')
            print(f"- Score: {node.score:.4f} | Page: {page}")


if __name__ == "__main__":
    run_query("  What was the 'Transaction Completion Award' given to CEO David Goeckeler?")