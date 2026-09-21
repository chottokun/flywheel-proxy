import json
import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener
from typing import Any

_log_queue: queue.Queue = queue.Queue(-1)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {"message": record.getMessage()}

        payload["timestamp"] = self.formatTime(record, self.datefmt)
        payload["level"] = record.levelname
        payload["logger"] = record.name
        return json.dumps(payload, ensure_ascii=False)


_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(JsonFormatter())
_listener = QueueListener(_log_queue, _stream_handler)


def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 重複追加を防ぐ
    if not any(isinstance(h, QueueHandler) for h in root_logger.handlers):
        root_logger.addHandler(QueueHandler(_log_queue))
    if not _listener._thread:
        _listener.start()


def shutdown_logging():
    if _listener._thread:
        _listener.stop()


def log_event(logger_name: str, event_type: str, data: dict[str, Any], level: int = logging.INFO):
    logger = logging.getLogger(logger_name)
    event_payload = {"log_type": event_type, **data}
    logger.log(level, event_payload)
