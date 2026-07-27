"""Registro de auditoría local del agente."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIRECTORY = Path("logs")
AUDIT_FILE = LOG_DIRECTORY / "agent_audit.jsonl"


def write_audit_event(event: dict[str, Any]) -> None:
    """Añade un evento JSON a un archivo JSONL local."""

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    audit_event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        **event,
    }

    with AUDIT_FILE.open(
        mode="a",
        encoding="utf-8",
    ) as audit_log:
        audit_log.write(
            json.dumps(
                audit_event,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def read_recent_events(limit: int = 10) -> list[dict[str, Any]]:
    """Obtiene los últimos eventos del registro de auditoría."""

    if not AUDIT_FILE.exists():
        return []

    lines = AUDIT_FILE.read_text(
        encoding="utf-8"
    ).splitlines()

    recent_events: list[dict[str, Any]] = []

    for line in lines[-limit:]:
        try:
            recent_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return recent_events