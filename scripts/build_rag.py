"""Build (or rebuild) the ChromaDB RAG store from playbook + pedagogy sources.

Usage::

    python scripts/build_rag.py
    python scripts/build_rag.py --sources data/rag/sources --playbooks data/rag/playbooks
    python scripts/build_rag.py --db data/rag/chroma_db

Safe to re-run: all documents are upserted, so no duplicates are created.
Requires OPENAI_API_KEY to be set in the environment (or .env file) because
ChromaDB uses OpenAI embeddings (text-embedding-3-small) by default.

If you want to use ChromaDB's built-in embedding function instead (no API
key required, lower quality), pass --no-openai-embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocal_coach.rag import RAGStore, DEFAULT_DB_PATH, DEFAULT_SOURCES_DIR, DEFAULT_PLAYBOOKS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help=f"Directory containing pedagogy .md files (default: {DEFAULT_SOURCES_DIR})",
    )
    p.add_argument(
        "--playbooks",
        type=Path,
        default=DEFAULT_PLAYBOOKS_DIR,
        help=f"Directory containing playbook .yaml files (default: {DEFAULT_PLAYBOOKS_DIR})",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"ChromaDB persistence directory (default: {DEFAULT_DB_PATH})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[build_rag] sources   : {args.sources}")
    print(f"[build_rag] playbooks : {args.playbooks}")
    print(f"[build_rag] db        : {args.db}")

    if not args.sources.is_dir():
        print(f"[build_rag] WARNING: sources directory not found: {args.sources}")
    if not args.playbooks.is_dir():
        print(f"[build_rag] WARNING: playbooks directory not found: {args.playbooks}")

    try:
        store = RAGStore.build(
            sources_dir=args.sources,
            playbooks_dir=args.playbooks,
            db_path=args.db,
        )
    except ImportError as exc:
        print(f"[build_rag] ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[build_rag] ERROR: {exc}", file=sys.stderr)
        return 1

    # Quick sanity check
    try:
        pb_count = store._playbooks.count() if store._playbooks else 0
        pd_count = store._pedagogy.count() if store._pedagogy else 0
        print(f"[build_rag] playbook entries : {pb_count}")
        print(f"[build_rag] pedagogy chunks  : {pd_count}")
    except Exception:
        pass

    print("[build_rag] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
