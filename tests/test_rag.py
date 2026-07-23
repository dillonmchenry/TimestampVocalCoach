"""Tests for vocal_coach.rag (ChromaDB RAG store).

These tests use an in-memory-style temporary directory so no persistent
ChromaDB state is created on disk.  They require the ``chromadb`` package
to be installed.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip entire module if chromadb is not installed.
chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from vocal_coach.rag import RAGStore, _chunk_text, _playbook_entry_to_text  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests (no disk I/O)
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_text(self) -> None:
        assert _chunk_text("") == []

    def test_short_text_is_single_chunk(self) -> None:
        text = " ".join(["word"] * 100)
        chunks = _chunk_text(text, chunk_words=400)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_with_overlap(self) -> None:
        text = " ".join([f"w{i}" for i in range(1000)])
        chunks = _chunk_text(text, chunk_words=100, overlap=20)
        # Should produce multiple chunks
        assert len(chunks) > 1
        # Each chunk should be <= 100 words
        for c in chunks:
            assert len(c.split()) <= 100

    def test_overlap_preserves_words(self) -> None:
        words = [f"w{i}" for i in range(200)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_words=50, overlap=10)
        # The start of chunk N should overlap with the end of chunk N-1
        if len(chunks) >= 2:
            end_of_first = set(chunks[0].split()[-10:])
            start_of_second = set(chunks[1].split()[:10])
            assert end_of_first & start_of_second  # non-empty overlap


class TestPlaybookEntryToText:
    def test_minimal_entry(self) -> None:
        entry = {"detector": "scoop_habit"}
        text = _playbook_entry_to_text(entry)
        assert "scoop_habit" in text

    def test_full_entry(self) -> None:
        entry = {
            "detector": "scoop_habit",
            "pedagogy": "Scooping is...",
            "coaching_voice": "Lead with what.",
            "few_shot_summaries": ["Example 1", "Example 2"],
            "practice_tip": "Do this.",
        }
        text = _playbook_entry_to_text(entry)
        assert "scoop_habit" in text
        assert "Scooping is..." in text
        assert "Lead with what." in text
        assert "Example 1" in text
        assert "Do this." in text


# ---------------------------------------------------------------------------
# Integration tests (small in-memory ChromaDB via tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture()
def rag_store(tmp_path: Path) -> RAGStore:
    """RAGStore backed by a temp directory."""
    db_path = tmp_path / "chroma_db"
    store = RAGStore(db_path=db_path)
    store._ensure_connected()
    return store


class TestRAGStoreConnects:
    def test_connects_and_creates_collections(self, rag_store: RAGStore) -> None:
        assert rag_store._playbooks is not None
        assert rag_store._pedagogy is not None

    def test_empty_store_returns_empty_on_query(self, rag_store: RAGStore) -> None:
        results = rag_store.query_detector("scoop_habit", k=3)
        assert results == []

    def test_empty_pedagogy_returns_empty(self, rag_store: RAGStore) -> None:
        results = rag_store.query_topics(["breath", "pitch"], k=5)
        assert results == []


class TestRAGStoreBuildAndQuery:
    def test_build_from_sources(self, tmp_path: Path) -> None:
        """Build the store from real source files and query it."""
        sources_dir = ROOT / "data" / "rag" / "sources"
        playbooks_dir = ROOT / "data" / "rag" / "playbooks"

        if not sources_dir.is_dir() or not playbooks_dir.is_dir():
            pytest.skip("data/rag/ source files not present")

        db_path = tmp_path / "chroma_db"
        store = RAGStore.build(
            sources_dir=sources_dir,
            playbooks_dir=playbooks_dir,
            db_path=db_path,
        )

        # Playbooks should have entries.
        assert store._playbooks.count() > 0
        # Pedagogy should have entries.
        assert store._pedagogy.count() > 0

    def test_query_detector_returns_results(self, tmp_path: Path) -> None:
        sources_dir = ROOT / "data" / "rag" / "sources"
        playbooks_dir = ROOT / "data" / "rag" / "playbooks"

        if not sources_dir.is_dir() or not playbooks_dir.is_dir():
            pytest.skip("data/rag/ source files not present")

        db_path = tmp_path / "chroma_db"
        store = RAGStore.build(
            sources_dir=sources_dir,
            playbooks_dir=playbooks_dir,
            db_path=db_path,
        )

        results = store.query_detector("scoop_habit", k=3)
        assert len(results) > 0
        # The top result should mention scoop_habit
        assert "scoop_habit" in results[0].lower() or "scoop" in results[0].lower()

    def test_query_topics_returns_results(self, tmp_path: Path) -> None:
        sources_dir = ROOT / "data" / "rag" / "sources"
        playbooks_dir = ROOT / "data" / "rag" / "playbooks"

        if not sources_dir.is_dir() or not playbooks_dir.is_dir():
            pytest.skip("data/rag/ source files not present")

        db_path = tmp_path / "chroma_db"
        store = RAGStore.build(
            sources_dir=sources_dir,
            playbooks_dir=playbooks_dir,
            db_path=db_path,
        )

        results = store.query_topics(["breath_support", "pitch"], k=5)
        assert len(results) > 0


class TestRAGStoreSmallInMemory:
    """Faster tests using a manually populated store."""

    def test_upsert_and_exact_query(self, rag_store: RAGStore) -> None:
        rag_store._playbooks.upsert(
            ids=["playbook__scoop_habit"],
            documents=["Highlight type: scoop_habit\nPedagogy: scooping test"],
            metadatas=[{"detector": "scoop_habit", "source": "test.yaml"}],
        )
        results = rag_store.query_detector("scoop_habit", k=3)
        assert any("scoop_habit" in r for r in results)

    def test_idempotent_upsert(self, rag_store: RAGStore) -> None:
        doc = "Highlight type: steady_sustain\nPedagogy: stable test"
        for _ in range(3):
            rag_store._playbooks.upsert(
                ids=["playbook__steady_sustain"],
                documents=[doc],
                metadatas=[{"detector": "steady_sustain", "source": "test.yaml"}],
            )
        assert rag_store._playbooks.count() == 1
