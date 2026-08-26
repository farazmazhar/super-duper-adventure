"""Chunking — one chunk per free-text record (feedback + tickets).

Pure module (records -> chunks, no I/O, no network). Every record is already a
short, self-contained unit (feedback median ~76 chars, tickets ~128 chars), so
classic document chunking (overlapping windows, token splitting) would fragment
meaning with zero benefit. Metadata is carried alongside the text for filtering
+ citations; only the text is embedded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import duckdb

FEEDBACK_SQL = """
SELECT feedback_id, customer_id, created_at, feedback_text, feedback_source, rating
FROM main.fact_feedback
WHERE feedback_text IS NOT NULL
  AND trim(feedback_text) <> ''
ORDER BY feedback_id
"""

TICKET_SQL = """
SELECT ticket_id, customer_id, created_at, subject, message, category, priority, status, satisfaction_score
FROM main.fact_ticket
WHERE subject IS NOT NULL
  AND trim(subject) <> ''
ORDER BY ticket_id
"""


@dataclass
class Chunk:
    """A single embeddable unit: text + record metadata."""

    record_type: str
    record_id: str
    customer_id: str | None
    created_at: datetime | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def source_hash(self) -> str:
        """Stable hash of the embeddable source — the cache key."""
        import hashlib

        return hashlib.sha256(
            f"{self.record_type}|{self.record_id}|{self.text}".encode("utf-8")
        ).hexdigest()


def _feedback_chunks(con: duckdb.DuckDBPyConnection) -> list[Chunk]:
    rows = con.execute(FEEDBACK_SQL).fetchall()
    return [
        Chunk(
            record_type="feedback",
            record_id=feedback_id,
            customer_id=customer_id,
            created_at=created_at,
            text=feedback_text,
            metadata={
                "feedback_id": feedback_id,
                "customer_id": customer_id,
                "created_at": created_at.isoformat() if created_at else None,
                "rating": rating,
                "source": feedback_source,
            },
        )
        for feedback_id, customer_id, created_at, feedback_text, feedback_source, rating in rows
    ]


def _ticket_chunks(con: duckdb.DuckDBPyConnection) -> list[Chunk]:
    rows = con.execute(TICKET_SQL).fetchall()
    chunks: list[Chunk] = []
    for (
        ticket_id,
        customer_id,
        created_at,
        subject,
        message,
        category,
        priority,
        status,
        satisfaction_score,
    ) in rows:
        text = subject if message is None else f"{subject} | {message}"
        chunks.append(
            Chunk(
                record_type="ticket",
                record_id=ticket_id,
                customer_id=customer_id,
                created_at=created_at,
                text=text,
                metadata={
                    "ticket_id": ticket_id,
                    "customer_id": customer_id,
                    "created_at": created_at.isoformat() if created_at else None,
                    "category": category,
                    "priority": priority,
                    "status": status,
                    "satisfaction": satisfaction_score,
                },
            )
        )
    return chunks


def build_chunks(con: duckdb.DuckDBPyConnection) -> list[Chunk]:
    """Build one chunk per free-text record from main.fact_feedback / fact_ticket."""
    return _feedback_chunks(con) + _ticket_chunks(con)
