"""Chunker tests: one chunk per record, metadata shape, null handling."""

from __future__ import annotations

import duckdb

from apps.embedding.chunker import build_chunks


def test_one_chunk_per_feedback_record(con: duckdb.DuckDBPyConnection) -> None:
    chunks = build_chunks(con)
    feedback_chunks = [c for c in chunks if c.record_type == "feedback"]
    # 4 feedback rows, but FDB-0003 has NULL text -> excluded
    assert len(feedback_chunks) == 3
    ids = {c.record_id for c in feedback_chunks}
    assert ids == {"FDB-0001", "FDB-0002", "FDB-0004"}


def test_one_chunk_per_ticket_record(con: duckdb.DuckDBPyConnection) -> None:
    chunks = build_chunks(con)
    ticket_chunks = [c for c in chunks if c.record_type == "ticket"]
    assert len(ticket_chunks) == 3
    assert {c.record_id for c in ticket_chunks} == {"TCK-0001", "TCK-0002", "TCK-0003"}


def test_feedback_chunk_metadata(con: duckdb.DuckDBPyConnection) -> None:
    chunk = next(c for c in build_chunks(con) if c.record_id == "FDB-0001")
    assert chunk.text == "Love the search feature"
    assert chunk.metadata["rating"] == 5
    assert chunk.metadata["source"] == "support_chat"
    assert chunk.metadata["customer_id"] == "CUST-0001"
    assert chunk.metadata["feedback_id"] == "FDB-0001"


def test_ticket_text_joins_subject_and_message(con: duckdb.DuckDBPyConnection) -> None:
    chunk = next(c for c in build_chunks(con) if c.record_id == "TCK-0001")
    assert chunk.text == "Bug | It crashes on export"
    assert chunk.metadata["category"] == "bug"
    assert chunk.metadata["priority"] == "high"
    assert chunk.metadata["satisfaction"] == 4.0
    # null message -> subject only
    chunk2 = next(c for c in build_chunks(con) if c.record_id == "TCK-0002")
    assert chunk2.text == "Question"


def test_source_hash_is_stable(con: duckdb.DuckDBPyConnection) -> None:
    chunks = build_chunks(con)
    first = [c.source_hash() for c in chunks]
    second = [c.source_hash() for c in build_chunks(con)]
    assert first == second
    # different text -> different hash
    assert first[0] != chunks[1].source_hash()
