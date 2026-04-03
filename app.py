import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 1. SET UP PATHS (Absolute Pathing) ---
# This ensures we always find the root 'qdrant_db' folder
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

DB_PATH = str(BASE_DIR / "qdrant_db")
COLLECTION_NAME = "sandisk_report"

# --- 2. INITIALIZE MODELS (OpenAI) ---
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("❌ OPENAI_API_KEY not found in .env file!")
    st.stop()

# Global LLM and Embedding Settings
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=openai_key)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_key)


# --- 3. CACHED DATABASE FUNCTIONS (Prevents Locking Errors) ---
@st.cache_resource
def get_qdrant_client():
    """Creates a single, persistent connection to the Qdrant database."""
    return QdrantClient(path=DB_PATH)


@st.cache_resource
def get_index():
    """Loads the Vector Store Index once and caches it for performance."""
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME, client=client)
    return VectorStoreIndex.from_vector_store(vector_store)


# --- 4. UI CONFIGURATION ---
st.set_page_config(page_title="SanDisk AI Analyst", page_icon="📈", layout="wide")
st.title("📈 SanDisk 2025 Proxy Statement Analyst")
st.markdown(f"**Database Folder:** `{DB_PATH}`")
st.markdown("---")

# --- 5. SIDEBAR: STATUS & CONTROLS ---
with st.sidebar:
    st.header("Pipeline Status")

    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME in names:
            st.success(f"✅ Collection '{COLLECTION_NAME}' is Online")
        else:
            st.error(f"❌ '{COLLECTION_NAME}' NOT FOUND")
            st.info("💡 Run 'python src/Ingestion.py' from the root folder first.")
            st.stop()

    except Exception as e:
        st.error("🔄 Database Lock Detected")
        st.info("Run this in your terminal to unlock: \n `ps aux | grep python | awk '{print $2}' | xargs kill -9`")
        st.stop()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 6. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about SanDisk's 2025 performance or executive pay..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document vectors..."):
            try:
                # 1. Use the cached index
                index = get_index()

                # 2. Query the engine (Top 3 most relevant chunks)
                query_engine = index.as_query_engine(similarity_top_k=3)
                response = query_engine.query(prompt)

                # 3. Format and Display Response
                answer = str(response)

                # Attach sources if available
                if hasattr(response, 'source_nodes'):
                    answer += "\n\n---\n**Verified Sources:**"
                    for node in response.source_nodes:
                        page = node.metadata.get('page_label', 'N/A')
                        score = node.score if node.score else 0.0
                        answer += f"\n- Page {page} (Relevancy Score: {score:.2f})"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")