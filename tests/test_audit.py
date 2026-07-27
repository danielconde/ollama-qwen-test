"""Pruebas unitarias del registro de auditoría."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import audit


def configure_temporary_audit_file(
    tmp_path: Path,
    monkeypatch: Any,
) -> Path:
    """Redirige la auditoría a un directorio temporal."""

    temporary_log_directory = tmp_path / "logs"
    temporary_audit_file = (
        temporary_log_directory / "agent_audit.jsonl"
    )

    monkeypatch.setattr(
        audit,
        "LOG_DIRECTORY",
        temporary_log_directory,
    )

    monkeypatch.setattr(
        audit,
        "AUDIT_FILE",
        temporary_audit_file,
    )

    return temporary_audit_file


def test_write_audit_event(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Debe crear el directorio y escribir un evento."""

    audit_file = configure_temporary_audit_file(
        tmp_path,
        monkeypatch,
    )

    audit.write_audit_event(
        {
            "event_type": "test_event",
            "status": "ok",
        }
    )

    assert audit_file.exists()

    lines = audit_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    event = json.loads(lines[0])

    assert event["event_type"] == "test_event"
    assert event["status"] == "ok"
    assert "timestamp_utc" in event


def test_write_multiple_audit_events(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Los eventos deben añadirse sin sobrescribir."""

    audit_file = configure_temporary_audit_file(
        tmp_path,
        monkeypatch,
    )

    audit.write_audit_event(
        {
            "event_type": "first_event",
        }
    )

    audit.write_audit_event(
        {
            "event_type": "second_event",
        }
    )

    lines = audit_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    first_event = json.loads(lines[0])
    second_event = json.loads(lines[1])

    assert first_event["event_type"] == "first_event"
    assert second_event["event_type"] == "second_event"


def test_read_recent_events(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Debe devolver únicamente los últimos eventos."""

    configure_temporary_audit_file(
        tmp_path,
        monkeypatch,
    )

    for event_number in range(5):
        audit.write_audit_event(
            {
                "event_type": "numbered_event",
                "event_number": event_number,
            }
        )

    recent_events = audit.read_recent_events(limit=2)

    assert len(recent_events) == 2
    assert recent_events[0]["event_number"] == 3
    assert recent_events[1]["event_number"] == 4


def test_read_events_when_file_does_not_exist(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """La ausencia del fichero debe devolver una lista vacía."""

    configure_temporary_audit_file(
        tmp_path,
        monkeypatch,
    )

    recent_events = audit.read_recent_events(limit=10)

    assert recent_events == []


def test_invalid_json_line_is_ignored(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Una línea corrupta no debe impedir leer eventos válidos."""

    audit_file = configure_temporary_audit_file(
        tmp_path,
        monkeypatch,
    )

    audit_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_file.write_text(
        '{"event_type": "valid"}\n'
        'this is not json\n'
        '{"event_type": "also_valid"}\n',
        encoding="utf-8",
    )

    events = audit.read_recent_events(limit=10)

    assert len(events) == 2
    assert events[0]["event_type"] == "valid"
    assert events[1]["event_type"] == "also_valid"