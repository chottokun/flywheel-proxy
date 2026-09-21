import asyncio
import os

import aiosqlite
import pytest

from app.storage import Storage


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_flywheel.db")

@pytest.mark.asyncio
async def test_storage_start_stop(db_path):
    storage = Storage(db_path=db_path)
    await storage.start()

    assert os.path.exists(db_path)

    # Check that tables are created
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests';") as cursor:
            row = await cursor.fetchone()
            assert row is not None

    await storage.stop()

@pytest.mark.asyncio
async def test_storage_record_flush_on_stop(db_path):
    storage = Storage(db_path=db_path, batch_size=50, flush_interval=1.0)
    await storage.start()

    await storage.record(
        trace_id="t1", route="ruri", latency_ms=10.0, entropy=0.5, margin=0.1,
        escalated=False, finish_reason="stop", messages_json="[]", response_text="hi"
    )

    await storage.stop()

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT trace_id, route FROM requests;") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0] == ("t1", "ruri")

@pytest.mark.asyncio
async def test_storage_record_flush_on_batch_size(db_path):
    storage = Storage(db_path=db_path, batch_size=3, flush_interval=10.0)
    await storage.start()

    # Send 3 records, which should trigger a batch flush
    for i in range(3):
        await storage.record(
            trace_id=f"t{i}", route="ruri", latency_ms=10.0, entropy=0.5, margin=0.1,
            escalated=False, finish_reason="stop", messages_json="[]", response_text="hi"
        )

    # Give the worker a moment to process
    await asyncio.sleep(0.1)

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT trace_id FROM requests;") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 3

    await storage.stop()

@pytest.mark.asyncio
async def test_storage_record_flush_on_interval(db_path):
    storage = Storage(db_path=db_path, batch_size=50, flush_interval=0.2)
    await storage.start()

    # Send 1 record
    await storage.record(
        trace_id="t_interval", route="ruri", latency_ms=10.0, entropy=0.5, margin=0.1,
        escalated=False, finish_reason="stop", messages_json="[]", response_text="hi"
    )

    # Initially not flushed
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT trace_id FROM requests;") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 0

    # Wait for the interval to pass
    await asyncio.sleep(0.3)

    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT trace_id FROM requests;") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "t_interval"

    await storage.stop()
