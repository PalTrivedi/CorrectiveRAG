import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pinecone import Pinecone
import requests


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("crag.agent")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DEFAULTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEFAULT_INDEX_NAME = "crag"
DEFAULT_EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_NVIDIA_MODEL = "openai/gpt-oss-120b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MAX_RETRIES = 3
DEFAULT_NVIDIA_RPM_LIMIT = 40
DEFAULT_TOP_K = 5
DEFAULT_WEB_SCORE_THRESHOLD = 0.35
DEFAULT_WEB_MIN_CHUNKS = 1
DEFAULT_WEB_MIN_TERM_MATCH = 0.2


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA CLASSES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class RetrievedChunk:
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RetrievedSource:
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    query: str
    rewritten_query: str
    relevance_score: float
    used_web_search: bool
    local_sources: list[RetrievedSource]
    web_sources: list[RetrievedSource]


@dataclass
class QueryPlan:
    rewritten_query: str
    filters: dict[str, Any]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {raw_text}")
    return json.loads(raw_text[start : end + 1])


def _answer_needs_web(
    query: str, answer: str, sources: list[RetrievedSource]
) -> bool:
    if not sources:
        return True

    source_text = "\n\n".join(
        f"[SOURCE-{i}] {s.title}\n{_truncate_text(s.content, 600)}"
        for i, s in enumerate(sources, 1)
    )

    prompt = f"""You are checking whether the answer fully addresses the question using ONLY the provided sources.
Return strict JSON:
{{"sufficient": true/false, "reason": "short"}}

Rules:
- If the answer says there isn't enough information, return false.
- If the answer is generic or doesn't address the core of the question, return false.
- Be conservative: only return true if the sources clearly support the answer.

Question: {query}
Answer: {answer}
Sources:
{source_text}
"""

    try:
        raw = _invoke_nvidia_json(prompt)
        parsed = _safe_json_loads(raw)
        return not bool(parsed.get("sufficient", False))
    except Exception as exc:
        logger.warning("Answer check failed: %s", exc)
        return False


def _normalize_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        cleaned = _normalize_string(item)
        if cleaned:
            result.append(cleaned)
    return result


def _normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate_text(value: str, limit: int = 1400) -> str:
    return " ".join(value.split())[:limit]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NVIDIA API  (used for BOTH planning AND answering)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_nvidia_request_times: list[float] = []


def _throttle_nvidia() -> None:
    """
    Enforces the 40 RPM limit.
    Each query uses 1-2 NVIDIA calls:
      - Query plan: 1 call (skippable)
      - Answer generation: 1 call
    So max ~20-40 user queries per minute.
    """
    rpm_limit = int(os.getenv("NVIDIA_RPM_LIMIT", str(DEFAULT_NVIDIA_RPM_LIMIT)))
    if rpm_limit <= 0:
        return

    now = time.monotonic()
    # Clean old timestamps outside the 60s window
    _nvidia_request_times[:] = [t for t in _nvidia_request_times if t > now - 60.0]

    if len(_nvidia_request_times) >= rpm_limit:
        oldest = _nvidia_request_times[0]
        sleep_for = 60.0 - (now - oldest) + 0.1  # +0.1s safety buffer
        if sleep_for > 0:
            logger.warning(
                "NVIDIA RPM limit reached (%d/%d). Sleeping %.1fs.",
                len(_nvidia_request_times),
                rpm_limit,
                sleep_for,
            )
            time.sleep(sleep_for)

    _nvidia_request_times.append(time.monotonic())


def _remaining_nvidia_calls() -> int:
    """How many NVIDIA calls are left in the current 60s window."""
    rpm_limit = int(os.getenv("NVIDIA_RPM_LIMIT", str(DEFAULT_NVIDIA_RPM_LIMIT)))
    now = time.monotonic()
    recent = sum(1 for t in _nvidia_request_times if t > now - 60.0)
    return max(0, rpm_limit - recent)


def _invoke_nvidia(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> str:
    """
    Single NVIDIA API call with throttling and retries.
    Used for both JSON planning and free-text answer generation.
    """
    api_key = _require_env("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_API_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    model = os.getenv("NVIDIA_JSON_MODEL", DEFAULT_NVIDIA_MODEL)
    max_retries = int(os.getenv("NVIDIA_MAX_RETRIES", str(DEFAULT_NVIDIA_MAX_RETRIES)))

    endpoint = (
        base_url
        if base_url.endswith("/chat/completions")
        else f"{base_url}/chat/completions"
    )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    last_error: Exception | None = None

    for attempt in range(max_retries):
        _throttle_nvidia()

        if attempt > 0:
            backoff = 1.0 * (attempt + 1)
            logger.warning(
                "NVIDIA retry %d/%d after %.1fs backoff.",
                attempt + 1,
                max_retries,
                backoff,
            )
            time.sleep(backoff)

        try:
            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug(
                "NVIDIA call OK. Remaining RPM: %d", _remaining_nvidia_calls()
            )
            return content

        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)

            # Rate limited â€” wait and retry
            if status == 429:
                retry_after = float(
                    exc.response.headers.get("Retry-After", 5)
                )
                logger.warning(
                    "NVIDIA 429 rate limited. Waiting %.1fs.", retry_after
                )
                time.sleep(retry_after)
                continue

            logger.warning("NVIDIA HTTP %s: %s", status, exc)
            if attempt == max_retries - 1:
                raise

        except Exception as exc:
            last_error = exc
            logger.warning("NVIDIA attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NVIDIA invocation failed.")


def _invoke_nvidia_json(prompt: str) -> str:
    """Convenience wrapper for calls that expect JSON output."""
    return _invoke_nvidia(prompt, temperature=0.0)


def _invoke_nvidia_answer(prompt: str, system_prompt: str | None = None) -> str:
    """Convenience wrapper for answer generation."""
    return _invoke_nvidia(
        prompt,
        temperature=0.15,
        max_tokens=2048,
        system_prompt=system_prompt,
    )


def _hf_embed(texts: list[str]) -> list[list[float]]:
    model = os.getenv("HUGGINGFACE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("Missing Hugging Face token: HF_TOKEN or HUGGINGFACE_API_KEY")
    client = InferenceClient(provider="hf-inference", api_key=token)

    vectors: list[list[float]] = []
    for text in texts:
        result = client.feature_extraction(text, model=model)
        vectors.append([float(x) for x in list(result)])
    return vectors


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PINECONE FILTER BUILDER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _build_pinecone_filter(plan: dict[str, Any]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []

    for field in ("chapter_number", "book_number"):
        val = _normalize_int(plan.get(field))
        if val is not None:
            parts.append({field: {"$eq": val}})

    for field in ("book_title", "pov_character", "location", "region"):
        val = _normalize_string(plan.get(field))
        if val:
            parts.append({field: {"$eq": val}})

    for field in ("characters", "houses"):
        vals = _normalize_string_list(plan.get(field))
        if vals:
            parts.append({field: {"$in": vals}})

    if not parts:
        return {}
    return parts[0] if len(parts) == 1 else {"$and": parts}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# WEB SEARCH
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _search_web(query: str, num_results: int = 5) -> list[RetrievedSource]:
    try:
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            logger.warning(
                "Web search skipped because OLLAMA_API_KEY is missing."
            )
            return []
        resp = requests.post(
            os.getenv(
                "OLLAMA_WEB_SEARCH_URL", "https://ollama.com/api/web_search"
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "max_results": min(num_results, 10)},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("results", [])
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return []

    return [
        RetrievedSource(
            source_type="web",
            title=item.get("title", "Untitled"),
            content=_truncate_text(
                item.get("snippet") or item.get("content") or "", 800
            ),
            metadata={"url": item.get("url", "")},
        )
        for item in items[:num_results]
    ]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PINECONE CLIENT  (Hugging Face Inference API embeddings)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PineconeClient:
    def __init__(self, index_name: str = DEFAULT_INDEX_NAME) -> None:

        pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
        self.index = pc.Index(index_name)

    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector = _hf_embed([query])[0]
        response = self.index.query(
            vector=vector,
            top_k=k,
            include_metadata=True,
            filter=metadata_filter or None,
        )
        matches = getattr(response, "matches", None) or response.get(
            "matches", []
        )
        chunks: list[RetrievedChunk] = []

        for match in matches:
            meta = dict(
                getattr(match, "metadata", None)
                or match.get("metadata", {})
            )
            score = float(
                getattr(match, "score", None) or match.get("score", 0.0)
            )
            content = meta.get("chunk_text") or meta.get("text", "")
            chunks.append(
                RetrievedChunk(content=content, score=score, metadata=meta)
            )

        return chunks


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# RETRIEVAL AGENT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class RetrievalAgent:
    def __init__(self) -> None:
        self.top_k = int(os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_TOP_K)))
        self.vector_store = PineconeClient()
        self.skip_plan = os.getenv("SKIP_QUERY_PLAN", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self.web_score_threshold = float(
            os.getenv(
                "WEB_SCORE_THRESHOLD", str(DEFAULT_WEB_SCORE_THRESHOLD)
            )
        )
        self.web_min_chunks = int(
            os.getenv("WEB_MIN_CHUNKS", str(DEFAULT_WEB_MIN_CHUNKS))
        )
        self.web_min_term_match = float(
            os.getenv("WEB_MIN_TERM_MATCH", str(DEFAULT_WEB_MIN_TERM_MATCH))
        )

    def _keyword_overlap_ratio(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> float:
        if not chunks:
            return 0.0
        tokens = re.findall(r"[a-z0-9']+", query.lower())
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "about",
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "do",
            "does",
            "did",
            "it",
            "this",
            "that",
            "these",
            "those",
            "tell",
            "describe",
        }
        terms = [t for t in tokens if t not in stopwords and len(t) > 2]
        if not terms:
            return 0.0

        max_ratio = 0.0
        for chunk in chunks:
            haystack = chunk.content.lower()
            matched = sum(1 for term in terms if term in haystack)
            ratio = matched / len(terms)
            if ratio > max_ratio:
                max_ratio = ratio
        return max_ratio

    def _build_query_plan(self, query: str) -> QueryPlan:
        """
        Uses 1 NVIDIA API call to rewrite query and extract filters.
        Set SKIP_QUERY_PLAN=true to skip this (saves 1 API call per query).
        """
        if self.skip_plan:
            logger.info(
                "Query planning skipped (SKIP_QUERY_PLAN=true). RPM remaining: %d",
                _remaining_nvidia_calls(),
            )
            return QueryPlan(rewritten_query=query, filters={})

        logger.info(
            "Building query plan via NVIDIA. RPM remaining: %d",
            _remaining_nvidia_calls(),
        )

        prompt = f"""You are planning retrieval for a RAG system over Game of Thrones book chunks.
Extract metadata filters ONLY if the user explicitly mentions them.

Return strict JSON:
{{
  "rewritten_query": "improved semantic search query",
  "book_title": null,
  "book_number": null,
  "chapter_number": null,
  "pov_character": null,
  "characters": [],
  "location": null,
  "region": null,
  "houses": []
}}

IMPORTANT: Be conservative with filters. Only add a filter if the user EXPLICITLY
mentions a specific book, chapter, character, or location. Over-filtering causes
missed results. When in doubt, leave as null/[].

User query: {query}"""

        try:
            raw = _invoke_nvidia_json(prompt)
            parsed = _safe_json_loads(raw)
            return QueryPlan(
                rewritten_query=parsed.get("rewritten_query") or query,
                filters=_build_pinecone_filter(parsed),
            )
        except Exception as exc:
            logger.warning(
                "Query planning failed (%s). Using raw query.", exc
            )
            return QueryPlan(rewritten_query=query, filters={})

    def _should_use_web_search(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[bool, float]:
        """
        Pure math on Pinecone scores. NO API call needed.
        """
        if not chunks:
            logger.info("No local chunks. Web search eligible.")
            return True, 0.0

        scores = [c.score for c in chunks]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        good_chunks = sum(
            1 for s in scores if s >= self.web_score_threshold
        )
        overlap_ratio = self._keyword_overlap_ratio(
            self.last_query if hasattr(self, "last_query") else "", chunks
        )

        logger.info(
            "Scores: top=%.4f avg=%.4f good=%d/%d threshold=%.2f overlap=%.2f",
            top_score,
            avg_score,
            good_chunks,
            len(scores),
            self.web_score_threshold,
            overlap_ratio,
        )

        if (
            top_score >= 0.65
            and good_chunks >= self.web_min_chunks
            and overlap_ratio >= self.web_min_term_match
        ):
            logger.info(
                "Strong local (%.4f, %d good, overlap %.2f). Skip web.",
                top_score,
                good_chunks,
                overlap_ratio,
            )
            return False, top_score

        logger.info(
            "Local not strong enough (%.4f, %d good, overlap %.2f). Web needed.",
            top_score,
            good_chunks,
            overlap_ratio,
        )
        return True, top_score

    def retrieve(self, query: str, allow_web: bool = True) -> RetrievalResult:
        start = time.perf_counter()
        logger.info("Retrieval start: %s", query[:80])

        # Step 1: Query plan (0 or 1 NVIDIA call)
        plan = self._build_query_plan(query)
        search_query = plan.rewritten_query
        self.last_query = search_query
        logger.info("Search: %s | Filters: %s", search_query[:60], plan.filters)

        # Step 2: Vector search (Ollama embedding + Pinecone)
        chunks = self.vector_store.search(
            search_query, k=self.top_k, metadata_filter=plan.filters
        )

        # Fallback without filters
        if len(chunks) < 2 and plan.filters:
            logger.info("Few results with filters. Retrying unfiltered.")
            chunks = self.vector_store.search(search_query, k=self.top_k)

        # Step 3: Web decision (pure math, no API call)
        relevance_score = max([c.score for c in chunks], default=0.0)
        use_web = False
        if allow_web:
            use_web, relevance_score = self._should_use_web_search(chunks)

        # Step 4: Web search (only if needed AND enabled)
        web_sources: list[RetrievedSource] = []
        if use_web:
            web_sources = _search_web(search_query)
            if not web_sources:
                logger.info("Web returned nothing.")
                use_web = False

        elapsed = time.perf_counter() - start
        logger.info(
            "Retrieval done %.2fs | chunks=%d | web=%s | score=%.4f",
            elapsed,
            len(chunks),
            use_web,
            relevance_score,
        )

        local_sources = [
            RetrievedSource(
                source_type="local",
                title=c.metadata.get("title", f"Chunk {i}"),
                content=c.content,
                metadata={**c.metadata, "pinecone_score": c.score},
            )
            for i, c in enumerate(chunks, 1)
        ]

        return RetrievalResult(
            query=query,
            rewritten_query=search_query,
            relevance_score=relevance_score,
            used_web_search=use_web,
            local_sources=local_sources,
            web_sources=web_sources,
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORRECTION AGENT  (uses NVIDIA for answering)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class CorrectionAgent:
    """
    Uses NVIDIA gpt-oss-120b for answer generation.
    This is the second (and final) NVIDIA call per query.
    """

    SYSTEM_PROMPT = """You are a knowledgeable Maester from the Citadel, an expert on the world of Ice and Fire (Game of Thrones / A Song of Ice and Fire).

Your role:
- Answer questions using ONLY the provided sources.
- Do NOT include source tags like [SOURCE-1] or [WEB-1] in the answer.
- If sources don't contain enough information, say so honestly in plain language.
- Be concise but thorough. Use specific details from the sources.
- Do NOT make up information beyond what the sources contain.
- Write in clear, direct prose. No roleplay or dramatic narration.
- Keep responses humanlike and well-formed.
- Avoid bullet lists and avoid using special symbols like '*' or '-' unless absolutely required."""

    def answer(self, query: str, retrieval: RetrievalResult) -> str:
        source_blocks: list[str] = []

        for i, src in enumerate(retrieval.local_sources, 1):
            meta = src.metadata
            header_parts = [f"score={meta.get('pinecone_score', 0):.3f}"]
            if meta.get("chapter_number"):
                header_parts.append(f"ch={meta['chapter_number']}")
            if meta.get("pov_character"):
                header_parts.append(f"pov={meta['pov_character']}")
            if meta.get("location"):
                header_parts.append(f"loc={meta['location']}")
            if meta.get("houses"):
                header_parts.append(
                    "houses=" + ", ".join(meta["houses"][:3])
                )
            summary = meta.get("chunkSummary", "")
            summary_line = f"\nSummary: {summary}" if summary else ""

            source_blocks.append(
                f"[SOURCE-{i}] ({', '.join(header_parts)}){summary_line}\n{_truncate_text(src.content, 1200)}"
            )

        for i, src in enumerate(retrieval.web_sources, 1):
            url = src.metadata.get("url", "")
            source_blocks.append(
                f"[WEB-{i}] {src.title} ({url})\n{_truncate_text(src.content, 600)}"
            )

        context = (
            "\n\n".join(source_blocks)
            if source_blocks
            else "No sources found."
        )

        prompt = f"""Question: {query}

Retrieval info:
- Rewritten query: {retrieval.rewritten_query}
- Top relevance score: {retrieval.relevance_score:.4f}
- Web search used: {retrieval.used_web_search}
- Local sources: {len(retrieval.local_sources)}
- Web sources: {len(retrieval.web_sources)}

Sources:
{context}

Provide a complete answer without citations."""

        logger.info(
            "Generating answer via NVIDIA. RPM remaining: %d",
            _remaining_nvidia_calls(),
        )

        answer = _invoke_nvidia_answer(prompt, system_prompt=self.SYSTEM_PROMPT)
        answer = re.sub(r"\[(SOURCE|WEB)-\d+\]", "", answer)
        answer = re.sub(r"\s+\n", "\n", answer).strip()
        answer = re.sub(r" {2,}", " ", answer)
        return answer


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SINGLETON AGENTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_retrieval_agent: RetrievalAgent | None = None
_correction_agent: CorrectionAgent | None = None


def _get_retrieval_agent() -> RetrievalAgent:
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = RetrievalAgent()
    return _retrieval_agent


def _get_correction_agent() -> CorrectionAgent:
    global _correction_agent
    if _correction_agent is None:
        _correction_agent = CorrectionAgent()
    return _correction_agent


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PUBLIC ENTRY POINT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_answer(query: str) -> dict[str, Any]:
    total_start = time.perf_counter()
    logger.info(
        "=== Query: %s | NVIDIA RPM remaining: %d",
        query[:80],
        _remaining_nvidia_calls(),
    )

    retrieval = _get_retrieval_agent().retrieve(query, allow_web=False)
    answer = ""
    if retrieval.local_sources:
        answer = _get_correction_agent().answer(query, retrieval)

    needs_web = (not retrieval.local_sources) or _answer_needs_web(
        query, answer, retrieval.local_sources
    )
    if needs_web:
        logger.info("Local sources insufficient. Falling back to web search.")
        web_sources = _search_web(retrieval.rewritten_query)
        if web_sources:
            retrieval = RetrievalResult(
                query=retrieval.query,
                rewritten_query=retrieval.rewritten_query,
                relevance_score=retrieval.relevance_score,
                used_web_search=True,
                local_sources=retrieval.local_sources,
                web_sources=web_sources,
            )
            answer = _get_correction_agent().answer(query, retrieval)
        else:
            logger.info("Web search yielded no sources. Returning local answer.")
            if not answer:
                answer = "I couldn't find enough information in the available sources to answer that."

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "=== Done %.2fs | local=%d | web=%d | RPM remaining: %d",
        total_elapsed,
        len(retrieval.local_sources),
        len(retrieval.web_sources),
        _remaining_nvidia_calls(),
    )


    sources = [
        {"type": s.source_type, "title": s.title, "metadata": s.metadata}
        for s in retrieval.local_sources + retrieval.web_sources
    ]

    return {
        "query": query,
        "rewritten_query": retrieval.rewritten_query,
        "relevance_score": retrieval.relevance_score,
        "used_web_search": retrieval.used_web_search,
        "answer": answer,
        "sources": sources,
    }
