"""Append-only audit log for pre-match winner predictions."""
import json
import os
from datetime import datetime, timezone

DEFAULT_LOG_PATH = os.path.join("data", "pre_match_predictions.jsonl")


def log_pre_match_prediction(record, log_path=DEFAULT_LOG_PATH):
    """Persist a prediction once per provider fixture ID."""
    if not isinstance(record, dict):
        return False
    source = str(record.get("source") or "").strip()
    match_id = str(record.get("source_match_id") or "").strip()
    if not source or not match_id or not record.get("prediction"):
        return False
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    keys = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    keys.add((row.get("source"), str(row.get("source_match_id") or "")))
                except (json.JSONDecodeError, AttributeError):
                    continue
    if (source, match_id) in keys:
        return False
    row = dict(record)
    row["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")
    return True
