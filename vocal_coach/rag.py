"""ChromaDB-backed retrieval-augmented generation store for Sprint-3 coaching.

The store holds two collections:

``playbooks``
    One entry per highlight detector type (~60-100 entries).  Each entry
    contains pedagogy context, coaching voice guidance, few-shot summary
    examples, and a practice tip.  Keyed by ``detector`` metadata field.

``pedagogy``
    General vocal science passages (~300-500 tokens each) drawn from
    open-access sources.  Tagged by topic (breath_support, vibrato, pitch,
    dynamics, phrasing, registration, ...).

Typical usage::

    from vocal_coach.rag import RAGStore

    # Build / rebuild the database (run once, or after editing sources)
    RAGStore.build(sources_dir=Path("data/rag/sources"),
                   playbooks_dir=Path("data/rag/playbooks"),
                   db_path=Path("data/rag/chroma_db"))

    # Query at inference time
    store = RAGStore(db_path=Path("data/rag/chroma_db"))
    passages = store.query_detector("scoop_habit", k=3)
    topic_passages = store.query_topics(["breath_support", "pitch"], k=5)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default paths (relative to repo root).  Override when calling build() or
# constructing RAGStore.
DEFAULT_DB_PATH = Path("data/rag/chroma_db")
DEFAULT_SOURCES_DIR = Path("data/rag/sources")
DEFAULT_PLAYBOOKS_DIR = Path("data/rag/playbooks")

PLAYBOOKS_COLLECTION = "playbooks"
PEDAGOGY_COLLECTION = "pedagogy"

# Token-budget for pedagogy chunks.
_CHUNK_WORDS = 400
_CHUNK_OVERLAP_WORDS = 60


# ---------------------------------------------------------------------------
# RAGStore
# ---------------------------------------------------------------------------


class RAGStore:
    """Wrapper around a ChromaDB persistent client."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path or DEFAULT_DB_PATH)
        self._client = None
        self._playbooks = None
        self._pedagogy = None

    # ------------------------------------------------------------------
    # Lazy connection
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore
        except ImportError:
            raise ImportError(
                "'chromadb' package not installed. "
                "Run: pip install chromadb"
            )
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))
        self._playbooks = self._client.get_or_create_collection(
            name=PLAYBOOKS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._pedagogy = self._client.get_or_create_collection(
            name=PEDAGOGY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def query_detector(self, detector_type: str, k: int = 3) -> list[str]:
        """Return up to *k* playbook passages relevant to *detector_type*.

        Combines a metadata ``where`` filter for exact-match entries plus a
        semantic search so nearby detector types can augment sparse results.
        """
        self._ensure_connected()
        results: list[str] = []

        # 1. Exact metadata match first.
        try:
            exact = self._playbooks.get(
                where={"detector": detector_type},
                include=["documents"],
            )
            if exact and exact.get("documents"):
                results.extend(exact["documents"])
        except Exception as exc:
            logger.debug("[rag] exact playbook lookup failed for %s: %s", detector_type, exc)

        # 2. Semantic neighbours for remaining slots.
        remaining = k - len(results)
        if remaining > 0:
            try:
                sem = self._playbooks.query(
                    query_texts=[detector_type.replace("_", " ")],
                    n_results=min(remaining + 2, self._playbooks.count() or 1),
                    include=["documents", "metadatas"],
                )
                seen = set(results)
                for doc in (sem.get("documents") or [[]])[0]:
                    if doc not in seen:
                        results.append(doc)
                        seen.add(doc)
                        if len(results) >= k:
                            break
            except Exception as exc:
                logger.debug("[rag] semantic playbook query failed: %s", exc)

        return results[:k]

    def query_topics(self, topics: list[str], k: int = 5) -> list[str]:
        """Return up to *k* pedagogy passages relevant to *topics*."""
        self._ensure_connected()
        if not topics:
            return []
        query_text = " ".join(topics)
        try:
            count = self._pedagogy.count()
            if count == 0:
                return []
            sem = self._pedagogy.query(
                query_texts=[query_text],
                n_results=min(k, count),
                include=["documents"],
            )
            return (sem.get("documents") or [[]])[0]
        except Exception as exc:
            logger.debug("[rag] pedagogy query failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Build (one-time or on update)
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        sources_dir: Optional[Path] = None,
        playbooks_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> "RAGStore":
        """(Re)build the ChromaDB store from source files.

        Safe to re-run: existing documents are upserted, not duplicated.
        """
        sources_dir = Path(sources_dir or DEFAULT_SOURCES_DIR)
        playbooks_dir = Path(playbooks_dir or DEFAULT_PLAYBOOKS_DIR)
        db_path = Path(db_path or DEFAULT_DB_PATH)

        store = cls(db_path=db_path)
        store._ensure_connected()

        n_playbooks = _ingest_playbooks(store._playbooks, playbooks_dir)
        n_pedagogy = _ingest_pedagogy(store._pedagogy, sources_dir)

        logger.info(
            "[rag] build complete: %d playbook entries, %d pedagogy chunks",
            n_playbooks,
            n_pedagogy,
        )
        return store


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def _ingest_playbooks(collection, playbooks_dir: Path) -> int:
    """Load all *.yaml playbook files and upsert into *collection*."""
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("[rag] pyyaml not installed; skipping playbook ingestion")
        return 0

    total = 0
    yaml_files = sorted(playbooks_dir.glob("*.yaml")) if playbooks_dir.is_dir() else []
    if not yaml_files:
        logger.warning("[rag] no playbook YAML files found in %s", playbooks_dir)
        return 0

    for yaml_path in yaml_files:
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("[rag] failed to load %s: %s", yaml_path, exc)
            continue

        entries = data.get("entries") or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            detector = entry.get("detector", "")
            if not detector:
                continue

            # Flatten the entry into a single text document.
            doc = _playbook_entry_to_text(entry)
            doc_id = f"playbook__{detector}"
            try:
                collection.upsert(
                    ids=[doc_id],
                    documents=[doc],
                    metadatas=[{"detector": detector, "source": yaml_path.name}],
                )
                total += 1
            except Exception as exc:
                logger.warning("[rag] upsert failed for %s: %s", doc_id, exc)

    return total


def _playbook_entry_to_text(entry: dict) -> str:
    """Serialise a playbook entry dict into a single retrieval document."""
    detector = entry.get("detector", "")
    parts = [f"Highlight type: {detector}"]

    if entry.get("pedagogy"):
        parts.append(f"Pedagogy: {entry['pedagogy']}")
    if entry.get("coaching_voice"):
        parts.append(f"Coaching voice: {entry['coaching_voice']}")
    if entry.get("few_shot_summaries"):
        examples = "\n".join(
            f"  - {s}" for s in entry["few_shot_summaries"]
        )
        parts.append(f"Example summaries:\n{examples}")
    if entry.get("practice_tip"):
        parts.append(f"Practice tip: {entry['practice_tip']}")

    return "\n".join(parts)


def _ingest_pedagogy(collection, sources_dir: Path) -> int:
    """Load all *.md source files, chunk them, and upsert into *collection*."""
    total = 0
    md_files = sorted(sources_dir.glob("*.md")) if sources_dir.is_dir() else []
    if not md_files:
        logger.warning("[rag] no pedagogy .md files found in %s", sources_dir)
        return 0

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("[rag] failed to read %s: %s", md_path, exc)
            continue

        # Extract topic tags from the frontmatter comment or first heading.
        topics = _extract_topics(text, md_path.stem)
        chunks = _chunk_text(text)

        for i, chunk in enumerate(chunks):
            doc_id = f"pedagogy__{md_path.stem}__{i:03d}"
            try:
                collection.upsert(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": md_path.name,
                        "topics": ",".join(topics),
                        "chunk_index": i,
                    }],
                )
                total += 1
            except Exception as exc:
                logger.warning("[rag] upsert failed for %s: %s", doc_id, exc)

    return total


def _extract_topics(text: str, stem: str) -> list[str]:
    """Heuristic: look for a 'topics:' line in the first 10 lines."""
    for line in text.splitlines()[:10]:
        m = re.match(r"topics\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            return [t.strip() for t in m.group(1).split(",") if t.strip()]
    # Fall back to filename stem split on underscores.
    return [p for p in stem.split("_") if p]


def _chunk_text(text: str, chunk_words: int = _CHUNK_WORDS, overlap: int = _CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split *text* into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


__all__ = ["RAGStore", "DEFAULT_DB_PATH", "DEFAULT_SOURCES_DIR", "DEFAULT_PLAYBOOKS_DIR"]
