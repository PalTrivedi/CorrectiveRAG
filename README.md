# CRAG (Corrective RAG) — Game of Thrones QA

This project is a **Corrective RAG** system that answers questions about *A Song of Ice and Fire* using:
- **Pinecone** for vector search over book chunks
- **Hugging Face Inference API** for embeddings
- **NVIDIA gpt-oss-120b** for query planning and answering
- **Optional web search** fallback when local sources are insufficient
- **Streamlit** UI for the frontend

## Screenshots

## Repo Structure

```
backend/
  main.py        # FastAPI API server
  agent.py       # retrieval + corrective RAG logic
  embed.py       # document ingestion + Pinecone upsert
frontend/
  frontend.py    # Streamlit UI
requirements.txt
```

## Setup

1. Create and activate a virtual environment
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Environment Variables

Required for core functionality:
- `NVIDIA_API_KEY`
- `PINECONE_API_KEY`
- `HF_TOKEN` **or** `HUGGINGFACE_API_KEY`

Optional / recommended:
- `CRAG_API_URL` (frontend → backend URL; default `http://127.0.0.1:8000/query`)
- `WEB_SEARCH_ENABLED` (`true` / `false`)
- `OLLAMA_API_KEY` (required if web search enabled)

Tuning / advanced:
- `RETRIEVAL_TOP_K`
- `WEB_SCORE_THRESHOLD`
- `WEB_MIN_CHUNKS`
- `WEB_MIN_TERM_MATCH`
- `SKIP_QUERY_PLAN` (`true` to skip query planning)
- `LOG_LEVEL`, `CRAG_LOG_LEVEL`
- `CRAG_TIMEOUT_SECONDS`
- `HUGGINGFACE_EMBED_MODEL`
- `NVIDIA_JSON_MODEL`
- `NVIDIA_API_URL`
- `NVIDIA_REQUEST_DELAY_SECONDS`
- `NVIDIA_MAX_RETRIES`
- `NVIDIA_RPM_LIMIT`
- `EMBED_DELAY_SECONDS`
- `UPSERT_BATCH_SIZE`
- `EMBED_CHECKPOINT_FILE`
- `PINECONE_METRIC`, `PINECONE_CLOUD`, `PINECONE_REGION`
- `SINGLE_BOOK_TITLE`, `SINGLE_BOOK_NUMBER`, `SINGLE_BOOK_SLUG`

## Ingest / Embed Books

From the repo root:
```
python backend/embed.py --source Book_Name.pdf --index-name crag
```

## Run Backend (FastAPI)

From repo root:
```
uvicorn backend.main:app --reload
```

Or from `backend/`:
```
uvicorn main:app --reload
```

Health check:
```
GET http://127.0.0.1:8000/health
```

## Run Frontend (Streamlit)

From repo root:
```
streamlit run frontend/frontend.py
```

Set `CRAG_API_URL` if the backend is not running locally.

## Corrective RAG Flow

1. Retrieve from Pinecone
2. Generate an answer
3. LLM checks if the answer is sufficient
4. If insufficient, fall back to web search and answer again

---
