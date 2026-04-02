# pro-rag-pipeline
# 📈 SanDisk 2025 Proxy Analyst (RAG Pipeline)

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline built to analyze complex financial documents. It uses **LlamaParse** for high-fidelity PDF table extraction and **Qdrant** for vector storage.

## 🚀 Features
- **Advanced Table Parsing:** Leverages LlamaCloud's Vision models to interpret complex financial tables.
- **Vector Search:** Uses Qdrant for fast, semantic retrieval of document context.
- **Streamlit UI:** A clean, professional interface for interacting with the financial data.

## 🛠️ Tech Stack
- **Framework:** LlamaIndex
- **Parser:** LlamaParse (LlamaCloud)
- **Vector DB:** Qdrant (Local Mode)
- **LLM:** OpenAI GPT-4o-mini
- **Frontend:** Streamlit

## 📦 Installation
1. Clone the repo: `git clone <your-repo-url>`
2. Create venv: `python -m venv venv && source venv/bin/activate`
3. Install deps: `pip install -r requirements.txt`
4. Set up `.env` with `OPENAI_API_KEY` and `LLAMA_CLOUD_API_KEY`.

## 📖 Usage
1. Ingest the data: `python src/Ingestion.py`
2. Launch the app: `streamlit run src/app.py`
