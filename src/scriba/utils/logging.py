from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOGGER_NAME = "scriba"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Never let tokens or secrets reach log files.
_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)(token|password|secret|api[_-]?key)(['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]+"),
]


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(lambda m: (m.group(1) + m.group(2) + "***") if m.lastindex else "***", redacted)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    if any(getattr(h, "_scriba_console", False) for h in root.handlers):
        for h in root.handlers:
            if getattr(h, "_scriba_console", False):
                h.setLevel(level.upper())
        return
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(show_path=False, rich_tracebacks=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.setLevel(level.upper())
    handler.addFilter(RedactingFilter())
    handler._scriba_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.propagate = False


@contextmanager
def job_log_file(path: Path) -> Iterator[logging.Handler]:
    """Attach a per-job processing.log handler for the duration of the job."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    handler.addFilter(RedactingFilter())
    root = logging.getLogger(LOGGER_NAME)
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        handler.close()
