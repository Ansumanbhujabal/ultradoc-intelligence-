"""Request tracing with per-stage spans."""

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from app.observability.logger import get_logger

logger = get_logger("tracer")


class Span:
    def __init__(self, stage: str):
        self.stage = stage
        self.status = "in_progress"
        self.latency_ms = 0
        self.metadata: dict = {}
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "stage": self.stage,
            "status": self.status,
            "latency_ms": self.latency_ms,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        if self.error:
            result["error"] = self.error
        return result


class Tracer:
    """Traces a single request through the pipeline."""

    _recent_traces: list[dict] = []
    MAX_TRACES = 50

    def __init__(self, endpoint: str, doc_id: str = ""):
        self.trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        self.endpoint = endpoint
        self.doc_id = doc_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.spans: list[Span] = []
        self._start_time = time.perf_counter()

    @contextmanager
    def span(self, stage: str):
        s = Span(stage)
        start = time.perf_counter()
        try:
            yield s
            s.status = "success"
        except Exception as e:
            s.status = "error"
            s.error = str(e)
            raise
        finally:
            s.latency_ms = round((time.perf_counter() - start) * 1000)
            self.spans.append(s)

    def skip(self, stage: str, note: str = ""):
        s = Span(stage)
        s.status = "skipped"
        if note:
            s.metadata["note"] = note
        self.spans.append(s)

    def to_dict(self) -> dict:
        total_latency = round((time.perf_counter() - self._start_time) * 1000)
        total_tokens = sum(
            s.metadata.get("tokens_in", 0) + s.metadata.get("tokens_out", 0)
            for s in self.spans
        )
        return {
            "trace_id": self.trace_id,
            "doc_id": self.doc_id,
            "endpoint": self.endpoint,
            "timestamp": self.timestamp,
            "total_latency_ms": total_latency,
            "total_tokens": total_tokens,
            "spans": [s.to_dict() for s in self.spans],
        }

    def finish(self):
        trace_dict = self.to_dict()
        Tracer._recent_traces.append(trace_dict)
        if len(Tracer._recent_traces) > Tracer.MAX_TRACES:
            Tracer._recent_traces.pop(0)
        logger.info("trace_complete", extra={"extra_data": trace_dict})
        return trace_dict

    @classmethod
    def get_recent_traces(cls) -> list[dict]:
        return list(reversed(cls._recent_traces))
