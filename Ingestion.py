import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- ABSOLUTE PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

DB_PATH = str(BASE_DIR / "qdrant_db")
COLLECTION_NAME = "sandisk_report"

# Verify Keys
openai_key = os.getenv("OPENAI_API_KEY")
llama_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Global Settings
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=openai_key)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_key)


def ingest_document(file_path):
    # 1. Parse
    parser = LlamaParse(api_key=llama_key, result_type="markdown")
    print(f"🚀 Parsing {file_path}...")
    documents = parser.load_data(file_path)

    # 2. Connect to Qdrant (FORCE ABSOLUTE PATH)
    client = QdrantClient(path=DB_PATH)

    # 3. Setup Vector Store
    vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME, client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Index and Save
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print(f"✅ SUCCESS: {len(documents)} pages indexed in {DB_PATH}")
    return index


if __name__ == "__main__":
    # Ensure PDF is in the same folder as the script
    pdf_path = Path(__file__).resolve().parent / "SNDK - Annual report.pdf"
    if pdf_path.exists():
        ingest_document(str(pdf_path))
    else:
        print(f"❌ Could not find PDF at: {pdf_path}")