"""
Log retention (report §21): telemetry, routing and feedback logs are deleted
after EDUROUTE_LOG_RETENTION_DAYS days (default 30). The app enforces it at
most once a day; it can also be run by hand or from a scheduled task:

    venv\\Scripts\\python -m src.utils.retention
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings


def _timestamp(line):
    try:
        value = json.loads(line).get("timestamp")
        stamp = datetime.fromisoformat(value) if value else None
        return stamp if stamp is None or stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError, TypeError):
        return None


def purge_logs(days=None, log_dir=None, now=None):
    """Remove log records older than `days`. Returns {file: records removed}."""
    days = settings.LOG_RETENTION_DAYS if days is None else days
    if days <= 0:
        return {}
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    removed = {}
    for path in Path(log_dir or settings.LOG_DIR).glob("*.jsonl"):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if not line.strip()
                or (_timestamp(line) is None or _timestamp(line) >= cutoff)]
        if len(kept) != len(lines):
            # Write to a temporary file and swap, so a crash never truncates a log.
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.writelines(kept)
            os.replace(tmp, path)
            removed[path.name] = len(lines) - len(kept)
    return removed


if __name__ == "__main__":
    print(purge_logs() or "Nothing older than the retention period.")
