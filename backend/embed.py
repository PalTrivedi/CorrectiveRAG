import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
import requests


load_dotenv()


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".py", ".html", ".csv"}
DEFAULT_INDEX_NAME = "crag"
DEFAULT_EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_NVIDIA_MODEL = "openai/gpt-oss-120b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_DELAY_SECONDS = 0.2
DEFAULT_NVIDIA_MAX_RETRIES = 3
DEFAULT_NVIDIA_RPM_LIMIT = 40
DEFAULT_DIMENSION = 1024
DEFAULT_SOURCE_PATH = "Rag.pdf"
DEFAULT_CHECKPOINT_FILE = ".embed_checkpoint.json"
DEFAULT_HF_MAX_RETRIES = 5
DEFAULT_HF_RETRY_DELAY_SECONDS = 2.0
DEFAULT_HF_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_SINGLE_BOOK_TITLE = os.getenv("SINGLE_BOOK_TITLE")
DEFAULT_SINGLE_BOOK_NUMBER = os.getenv("SINGLE_BOOK_NUMBER")
DEFAULT_SINGLE_BOOK_SLUG = os.getenv("SINGLE_BOOK_SLUG")


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


_nvidia_request_times: list[float] = []


def _throttle_nvidia(rpm_limit: int) -> None:
    if rpm_limit <= 0:
        return
    now = time.monotonic()
    _nvidia_request_times[:] = [
        t for t in _nvidia_request_times if t > now - 60.0
    ]
    if len(_nvidia_request_times) >= rpm_limit:
        oldest = _nvidia_request_times[0]
        sleep_for = 60.0 - (now - oldest) + 0.1
        if sleep_for > 0:
            logger.warning(
                "NVIDIA RPM limit (%d). Sleeping %.1fs.", rpm_limit, sleep_for
            )
            time.sleep(sleep_for)
    _nvidia_request_times.append(time.monotonic())


def invoke_nvidia_json(prompt: str) -> str:
    api_key = require_env("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_API_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    model = os.getenv("NVIDIA_JSON_MODEL", DEFAULT_NVIDIA_MODEL)
    delay_seconds = float(
        os.getenv(
            "NVIDIA_REQUEST_DELAY_SECONDS", str(DEFAULT_NVIDIA_DELAY_SECONDS)
        )
    )
    max_retries = int(
        os.getenv("NVIDIA_MAX_RETRIES", str(DEFAULT_NVIDIA_MAX_RETRIES))
    )
    rpm_limit = int(
        os.getenv("NVIDIA_RPM_LIMIT", str(DEFAULT_NVIDIA_RPM_LIMIT))
    )
    last_error: Exception | None = None

    endpoint = (
        base_url
        if base_url.endswith("/chat/completions")
        else f"{base_url}/chat/completions"
    )

    for attempt in range(max_retries):
        _throttle_nvidia(rpm_limit)

        if attempt > 0:
            sleep_for = delay_seconds * (attempt + 1)
            logger.warning(
                "Retrying NVIDIA (attempt %s/%s) after %.1fs",
                attempt + 1,
                max_retries,
                sleep_for,
            )
            time.sleep(sleep_for)
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=60,
            )

            if response.status_code == 429:
                retry_after = float(
                    response.headers.get("Retry-After", 5)
                )
                logger.warning("NVIDIA 429. Waiting %.1fs.", retry_after)
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            logger.warning("NVIDIA request failed: %s", exc)
            if attempt == max_retries - 1:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NVIDIA JSON invocation failed.")


def _hf_embed(texts: list[str]) -> list[list[float]]:
    model = os.getenv("HUGGINGFACE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("Missing Hugging Face token: HF_TOKEN or HUGGINGFACE_API_KEY")
    max_retries = int(os.getenv("HF_MAX_RETRIES", str(DEFAULT_HF_MAX_RETRIES)))
    delay_seconds = float(
        os.getenv("HF_RETRY_DELAY_SECONDS", str(DEFAULT_HF_RETRY_DELAY_SECONDS))
    )
    timeout_seconds = float(
        os.getenv(
            "HF_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_HF_REQUEST_TIMEOUT_SECONDS),
        )
    )
    client = InferenceClient(
        provider="hf-inference", api_key=token, timeout=timeout_seconds
    )

    vectors: list[list[float]] = []
    for text in texts:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            if attempt > 0:
                sleep_for = delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying Hugging Face embed (attempt %s/%s) after %.1fs",
                    attempt + 1,
                    max_retries,
                    sleep_for,
                )
                time.sleep(sleep_for)
            try:
                result = client.feature_extraction(text, model=model)
                vectors.append([float(x) for x in list(result)])
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Hugging Face embed failed: %s", exc)
        if last_error is not None:
            raise last_error
    return vectors


def safe_json_loads(raw_text: str) -> dict:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {raw_text}")
    return json.loads(raw_text[start : end + 1])


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = " ".join(item.split()).strip()
            if cleaned:
                items.append(cleaned)
    return items


def normalize_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def normalize_book_title(value: object) -> str | None:
    cleaned = normalize_optional_str(value)
    if not cleaned:
        return None
    if cleaned.strip().lower() in {"0", "unknown", "n/a", "na"}:
        return None
    return cleaned


def _get_single_book_overrides() -> tuple[str | None, int | None, str | None]:
    title = normalize_book_title(DEFAULT_SINGLE_BOOK_TITLE)
    number = normalize_optional_int(DEFAULT_SINGLE_BOOK_NUMBER)
    slug = normalize_optional_str(DEFAULT_SINGLE_BOOK_SLUG)
    return title, number, slug


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def infer_book_abbreviation(book_title: str | None, source: str) -> str:
    title = (book_title or "").strip().lower()
    override_title, _, override_slug = _get_single_book_overrides()
    if not title and override_slug:
        return slugify(override_slug)
    known = {
        "a game of thrones": "agot",
        "a clash of kings": "acok",
        "a storm of swords": "asos",
        "a feast for crows": "affc",
        "a dance with dragons": "adwd",
        "the winds of winter": "twow",
        "a dream of spring": "ados",
    }
    if title in known:
        return known[title]
    if override_title and override_title.strip().lower() in known:
        return known[override_title.strip().lower()]
    if override_slug:
        return slugify(override_slug)
    return slugify(Path(source).stem)[:16]


def build_chunk_id(metadata: dict, fallback_index: int) -> str:
    book_slug = infer_book_abbreviation(
        metadata.get("book_title"), metadata.get("source", "document")
    )
    chapter_number = metadata.get("chapter_number")
    chunk_index = metadata.get("chunk_index", fallback_index)
    if chapter_number is None:
        return f"{book_slug}-chunk-{chunk_index}"
    return f"{book_slug}-ch{chapter_number}-chunk-{chunk_index}"


def build_fallback_metadata(document: Document) -> dict:
    chunk_text = document.page_content
    base_metadata = document.metadata
    override_title, override_number, _ = _get_single_book_overrides()
    return {
        "book_title": override_title
        or normalize_book_title(
            Path(str(base_metadata.get("source", ""))).stem.replace("_", " ")
        )
        or "0",
        "book_number": override_number if override_number is not None else 0,
        "chapter_number": 0,
        "pov_character": "0",
        "characters": [],
        "location": "0",
        "region": "0",
        "houses": [],
        "chunkSummary": _truncate_words(chunk_text, 28),
        "chunk_index": int(base_metadata.get("chunk", 0)),
        "chunk_text": chunk_text,
        "word_count": len(chunk_text.split()),
        "has_dialogue": '"' in chunk_text or "'" in chunk_text,
        "is_prologue_epilogue": False,
    }


def _truncate_words(value: str, limit: int) -> str:
    words = value.split()
    if len(words) <= limit:
        return " ".join(words) or "Summary unavailable."
    return " ".join(words[:limit]) + "..."


def load_checkpoint(index_name: str, source_path: Path) -> int:
    checkpoint_path = Path(
        os.getenv("EMBED_CHECKPOINT_FILE", DEFAULT_CHECKPOINT_FILE)
    )
    if not checkpoint_path.exists():
        return 0
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Ignoring unreadable checkpoint: %s", checkpoint_path)
        return 0

    if payload.get("index_name") != index_name or payload.get(
        "source"
    ) != str(source_path):
        return 0
    return int(payload.get("last_completed_chunk", 0))


def save_checkpoint(
    index_name: str, source_path: Path, chunk_number: int
) -> None:
    checkpoint_path = Path(
        os.getenv("EMBED_CHECKPOINT_FILE", DEFAULT_CHECKPOINT_FILE)
    )
    payload = {
        "index_name": index_name,
        "source": str(source_path),
        "last_completed_chunk": chunk_number,
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_checkpoint(index_name: str, source_path: Path) -> None:
    checkpoint_path = Path(
        os.getenv("EMBED_CHECKPOINT_FILE", DEFAULT_CHECKPOINT_FILE)
    )
    if not checkpoint_path.exists():
        return
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        checkpoint_path.unlink(missing_ok=True)
        return
    if payload.get("index_name") == index_name and payload.get(
        "source"
    ) == str(source_path):
        checkpoint_path.unlink(missing_ok=True)


def generate_chunk_metadata(document: Document) -> dict:
    chunk_text = document.page_content
    base_metadata = document.metadata
    override_title, override_number, _ = _get_single_book_overrides()

    prompt = f"""
You are generating Pinecone metadata for a chunk from a fantasy novel.
Return strict JSON only.

Schema:
{{
  "book_title": "string or null",
  "book_number": 1,
  "chapter_number": 14,
  "pov_character": "string or null",
  "characters": ["string"],
  "location": "string or null",
  "region": "string or null",
  "houses": ["string"],
  "chunkSummary": "one or two sentence summary",
  "has_dialogue": true,
  "is_prologue_epilogue": false
}}

Rules:
- Use only evidence from the chunk.
- If unknown, return null for scalar fields and [] for lists.
- book_number and chapter_number must be integers or null.
- characters and houses must contain only distinct strings.
- chunkSummary must be concise and factual.

Source file: {base_metadata.get("source")}
Document title: {base_metadata.get("title")}
Chunk index: {base_metadata.get("chunk")}

Chunk text:
{chunk_text}
"""
    raw_response = invoke_nvidia_json(prompt)
    try:
        parsed = safe_json_loads(raw_response)
    except Exception:
        logger.warning(
            "Invalid JSON for chunk %s. Falling back. Preview: %r",
            base_metadata.get("chunk"),
            (raw_response or "")[:200],
        )
        return build_fallback_metadata(document)

    metadata = {
        "book_title": normalize_book_title(parsed.get("book_title"))
        or override_title
        or "0",
        "book_number": normalize_optional_int(parsed.get("book_number"))
        if normalize_optional_int(parsed.get("book_number")) is not None
        else (override_number if override_number is not None else 0),
        "chapter_number": normalize_optional_int(
            parsed.get("chapter_number")
        )
        or 0,
        "pov_character": normalize_optional_str(parsed.get("pov_character"))
        or "0",
        "characters": normalize_string_list(parsed.get("characters")),
        "location": normalize_optional_str(parsed.get("location")) or "0",
        "region": normalize_optional_str(parsed.get("region")) or "0",
        "houses": normalize_string_list(parsed.get("houses")),
        "chunkSummary": normalize_optional_str(parsed.get("chunkSummary"))
        or "Summary unavailable.",
        "chunk_index": int(base_metadata.get("chunk", 0)),
        "chunk_text": chunk_text,
        "word_count": len(chunk_text.split()),
        "has_dialogue": bool(parsed.get("has_dialogue", False)),
        "is_prologue_epilogue": bool(
            parsed.get("is_prologue_epilogue", False)
        ),
    }
    return metadata


def read_pdf(path: Path) -> str:
    logger.info("Reading PDF: %s", path)
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    logger.info(
        "PDF done: %s (%s pages, %s chars)",
        path.name,
        len(reader.pages),
        len(text),
    )
    return text


def read_text_file(path: Path) -> str:
    logger.info("Reading: %s", path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info("Done: %s (%s chars)", path.name, len(text))
    return text


def load_documents(source_path: Path) -> list[Document]:
    documents: list[Document] = []
    logger.info("Scanning: %s", source_path)

    if source_path.is_file():
        candidate_files = [source_path]
    else:
        candidate_files = [
            f
            for f in source_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    logger.info("Found %s file(s)", len(candidate_files))

    for file_path in candidate_files:
        content = (
            read_pdf(file_path)
            if file_path.suffix.lower() == ".pdf"
            else read_text_file(file_path)
        )
        if not content.strip():
            logger.warning("Skipping empty: %s", file_path)
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "title": file_path.name,
                    "extension": file_path.suffix.lower(),
                },
            )
        )
        logger.info("Loaded: %s", file_path.name)

    logger.info("Total documents: %s", len(documents))
    return documents


def chunk_documents(documents: Iterable[Document]) -> list[Document]:
    documents = list(documents)
    logger.info("Chunking %s document(s)", len(documents))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = index
    logger.info("Generated %s chunk(s)", len(chunks))
    return chunks


def ensure_index(index_name: str, dimension: int) -> None:
    logger.info("Checking index '%s' (dim=%s)", index_name, dimension)
    pc = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    index_list = pc.list_indexes()
    existing = (
        set(index_list.names())
        if hasattr(index_list, "names")
        else {item["name"] for item in index_list}
    )
    if index_name in existing:
        logger.info("Index '%s' exists.", index_name)
        return

    logger.info("Creating index '%s'", index_name)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=os.getenv("PINECONE_METRIC", "cosine"),
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
    logger.info("Created index '%s'", index_name)


def upsert_documents(
    index_name: str, documents: list[Document], source_path: Path
) -> None:
    logger.info("Embedding %s chunk(s)", len(documents))
    embed_delay = float(os.getenv("EMBED_DELAY_SECONDS", "0"))
    pc = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    last_completed = load_checkpoint(index_name, source_path)
    if last_completed:
        logger.info("Resuming from chunk %s", last_completed + 1)

    batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "32"))
    vectors: list[dict] = []

    last_successful_idx = last_completed
    try:
        for idx, document in enumerate(documents, start=1):
            if idx <= last_completed:
                continue

            logger.info("Metadata for chunk %s/%s", idx, len(documents))
            generated_metadata = generate_chunk_metadata(document)
            metadata = {
                "source": document.metadata.get("source"),
                "title": document.metadata.get("title"),
                "extension": document.metadata.get("extension"),
                "chunk": int(document.metadata.get("chunk", idx)),
                **generated_metadata,
            }
            metadata["text"] = metadata["chunk_text"]

            vector_id = build_chunk_id(metadata, idx)
            logger.info("Embedding chunk %s/%s -> '%s'", idx, len(documents), vector_id)
            vectors.append(
                {
                    "id": vector_id,
                    "values": _hf_embed([document.page_content])[0],
                    "metadata": metadata,
                }
            )
            last_successful_idx = idx
            if embed_delay > 0:
                time.sleep(embed_delay)

            if len(vectors) >= batch_size:
                logger.info("Upserting batch ending at %s", idx)
                index.upsert(vectors=vectors)
                save_checkpoint(index_name, source_path, idx)
                vectors.clear()
    except Exception:
        if last_successful_idx:
            save_checkpoint(index_name, source_path, last_successful_idx)
            logger.error(
                "Embedding failed. Saved checkpoint at chunk %s.",
                last_successful_idx,
            )
        raise

    if vectors:
        logger.info("Upserting final batch (%s vectors)", len(vectors))
        index.upsert(vectors=vectors)
        save_checkpoint(index_name, source_path, len(documents))

    clear_checkpoint(index_name, source_path)
    logger.info("Finished upserting into '%s'", index_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed documents into Pinecone for CRAG."
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE_PATH, help="PDF or folder"
    )
    parser.add_argument(
        "--index-name", default=DEFAULT_INDEX_NAME, help="Pinecone index"
    )
    parser.add_argument(
        "--dimension", type=int, default=DEFAULT_DIMENSION, help="Embed dim"
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Not found: {source_path}")

    logger.info("Starting embedding job")
    logger.info("Source: %s", source_path)
    logger.info("Index: %s", args.index_name)
    logger.info(
        "Embed model: %s (Hugging Face Inference API)",
        os.getenv("HUGGINGFACE_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )
    logger.info(
        "Metadata model: %s (NVIDIA)",
        os.getenv("NVIDIA_JSON_MODEL", DEFAULT_NVIDIA_MODEL),
    )

    documents = load_documents(source_path)
    if not documents:
        raise ValueError(f"No documents in {source_path}")

    chunks = chunk_documents(documents)
    ensure_index(args.index_name, args.dimension)
    upsert_documents(args.index_name, chunks, source_path)

    logger.info("Embedding job done")
    print(
        f"Indexed {len(chunks)} chunks from '{source_path}' -> '{args.index_name}'."
    )


if __name__ == "__main__":
    main()
