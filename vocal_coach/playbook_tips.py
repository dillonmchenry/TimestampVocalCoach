"""Lightweight loader for practice_tip fields from RAG playbook YAML files.

Provides a direct key-value lookup (detector → tip string) without depending
on ChromaDB, suitable for attaching tips to CoachingMoment objects in
select_highlights().
"""

from pathlib import Path

import yaml


def load_practice_tips(playbooks_dir: Path) -> dict[str, str]:
    """Return {detector_type: practice_tip} from all playbook YAML files.

    Strips leading/trailing whitespace from tip text (YAML block scalars
    often include a trailing newline). Returns an empty dict if the directory
    does not exist or contains no eligible YAML files.
    """
    tips: dict[str, str] = {}
    for yaml_path in sorted(playbooks_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            detector = entry.get("detector")
            tip = entry.get("practice_tip")
            if detector and tip:
                tips[str(detector)] = str(tip).strip()
    return tips
