import asyncio
import logging
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

class Storage:
    def __init__(self, db_path: str = "flywheel.db", batch_size: int = 50, flush_interval: float = 1.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA busy_timeout=10000;")
            await db.execute('''
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    route TEXT,
                    latency_ms REAL,
                    entropy REAL,
                    margin REAL,
                    escalated BOOLEAN,
                    finish_reason TEXT,
                    messages_json TEXT,
                    response_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.commit()

    async def start(self) -> None:
        await self.init_db()
        self._stop_event.clear()
        self.worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._stop_event.set()
        if self.worker_task:
            await self.worker_task
            self.worker_task = None

    async def record(self, trace_id: str, route: str, latency_ms: float, entropy: float, margin: float, escalated: bool, finish_reason: str, messages_json: str, response_text: str) -> None:
        await self.queue.put((
            trace_id, route, latency_ms, entropy, margin, escalated, finish_reason, messages_json, response_text
        ))

    async def _worker(self) -> None:
        batch: list[tuple[Any, ...]] = []
        loop = asyncio.get_running_loop()
        last_flush = loop.time()

        while not self._stop_event.is_set() or not self.queue.empty():
            now = loop.time()
            time_to_wait = max(0.01, self.flush_interval - (now - last_flush))

            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=time_to_wait)
                batch.append(item)
                self.queue.task_done()
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            now = loop.time()
            is_timeout = (now - last_flush) >= self.flush_interval
            is_full = len(batch) >= self.batch_size
            is_stopping = self._stop_event.is_set() and len(batch) > 0 and self.queue.empty()

            if is_full or (is_timeout and len(batch) > 0) or is_stopping:
                await self._flush(batch)
                batch.clear()
                last_flush = loop.time()

        if batch:
            await self._flush(batch)

    async def _flush(self, batch: list[tuple[Any, ...]]) -> None:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL;")
                await db.execute("PRAGMA busy_timeout=10000;")
                await db.executemany('''
                    INSERT INTO requests (
                        trace_id, route, latency_ms, entropy, margin, escalated, finish_reason, messages_json, response_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
