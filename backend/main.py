import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .agent import get_answer
except ImportError:
    # Allows running as a script from inside backend/ (e.g., uvicorn main:app)
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from agent import get_answer  # type: ignore


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("crag.api")


app = FastAPI(
    title="Corrective RAG API — Game of Thrones",
    version="2.0.0",
    description=(
        "CRAG backend using NVIDIA gpt-oss-120b for planning and answering, "
        "Ollama mxbai-embed-large for embeddings, and Pinecone for vector storage."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    relevance_score: float
    used_web_search: bool
    answer: str
    sources: list[dict]


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Corrective RAG API is running. POST to /query."}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest) -> QueryResponse:
    try:
        start = time.perf_counter()
        logger.info("Received query: %s", payload.query)
        result = get_answer(payload.query)
        elapsed = time.perf_counter() - start
        logger.info("Completed in %.2fs", elapsed)
        return QueryResponse(**result)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
