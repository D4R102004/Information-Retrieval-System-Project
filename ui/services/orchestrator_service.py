"""Adapter layer for `MainOrchestator` integration."""

from __future__ import annotations

import logging
import queue
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class OrchestratorService:
    """Lazy adapter that forwards UI calls to the backend orchestrator."""

    def __init__(self) -> None:
        self._orchestrator = None

    def _get_orchestrator(self):
        """Instantiate the orchestrator only when the UI needs it."""
        if self._orchestrator is None:
            from main_orchestator import MainOrchestator

            self._orchestrator = MainOrchestator()
        return self._orchestrator

    def retrieve_documents(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self._get_orchestrator().retrieve_documents(question, **kwargs)

    def stream_retrieve_documents(self, question: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        """Stream retrieval progress events and the final retrieval payload."""

        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        done_event = threading.Event()
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, Exception] = {}

        def _enqueue_progress(record: logging.LogRecord) -> None:
            event_queue.put(
                {
                    "kind": "log",
                    "logger": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        class _QueueLogHandler(logging.Handler):
            def __init__(self, callback: Callable[[logging.LogRecord], None]) -> None:
                super().__init__()
                self._callback = callback

            def emit(self, record: logging.LogRecord) -> None:
                self._callback(record)

        handler = _QueueLogHandler(_enqueue_progress)
        handler.setLevel(logging.INFO)

        watched_loggers = [
            logging.getLogger("main_orchestator"),
            logging.getLogger("sri.crawler.caller"),
        ]

        previous_levels: dict[str, int] = {}
        for target_logger in watched_loggers:
            previous_levels[target_logger.name] = target_logger.level
            target_logger.setLevel(logging.INFO)
            target_logger.addHandler(handler)

        def _run_retrieval() -> None:
            try:
                result_holder["value"] = self._get_orchestrator().retrieve_documents(question, 
                                                                                     **kwargs
                                                                                     )
            except Exception as exc:
                error_holder["error"] = exc
            finally:
                done_event.set()

        worker = threading.Thread(target=_run_retrieval, daemon=True)
        worker.start()

        yield {
            "kind": "status",
            "stage": "start",
            "message": "Starting retrieval workflow",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            while not done_event.is_set() or not event_queue.empty():
                try:
                    yield event_queue.get(timeout=0.15)
                except queue.Empty:
                    continue
        finally:
            for target_logger in watched_loggers:
                target_logger.removeHandler(handler)
                previous = previous_levels.get(target_logger.name)
                if previous is not None:
                    target_logger.setLevel(previous)

        if "error" in error_holder:
            raise error_holder["error"]

        yield {
            "kind": "result",
            "payload": result_holder.get("value", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def augment_response(self, question: str, documents: list[dict[str, Any]]) -> Any:
        return self._get_orchestrator().augment_response(question, documents)

    def clear_all_indices(self) -> dict[str, Any]:
        return self._get_orchestrator().clear_all_indices()
    
    def check_database_health(self) -> Dict[str, Any]:
        return self._get_orchestrator().check_database_health()

    def load_documents_from_crawlers(self, **kwargs: Any) -> dict[str, Any]:
        return self._get_orchestrator().load_documents_from_crawlers(**kwargs)

    def evaluate_test(self, test_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._get_orchestrator().evaluate_test(test_spec)

    def count_raw_documents(self, folder: str = "*") -> int:
        return self._get_orchestrator().crawler_caller.count_raw_documents(folder.lower())
    
    def count_initial_corpus_documents(self) -> int:
        return self._get_orchestrator().crawler_caller.count_initial_corpus_documents()
    
    def get_last_crawled_date(self, source: str) -> str:
        return self._get_orchestrator().crawler_caller.get_last_crawled(source.lower())
    
    def get_setting(self, key: str) -> Any:
        return self._get_orchestrator().get_setting(key)
            
    def sync_backend(self, state: dict[str, Any]):
        return self._get_orchestrator().sync_backend(state)

_ORCHESTRATOR_SERVICE: OrchestratorService | None = None


def get_orchestrator_service() -> OrchestratorService:
    """Return a shared orchestrator service instance."""
    global _ORCHESTRATOR_SERVICE
    if _ORCHESTRATOR_SERVICE is None:
        _ORCHESTRATOR_SERVICE = OrchestratorService()
    return _ORCHESTRATOR_SERVICE
