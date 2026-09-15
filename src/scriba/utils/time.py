from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _split(seconds: float) -> tuple[int, int, int, int]:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return hours, minutes, secs, ms


def format_srt_time(seconds: float) -> str:
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_clock(seconds: float) -> str:
    """hh:mm:ss (no milliseconds)."""
    h, m, s, _ = _split(float(int(seconds)))
    return f"{h:02d}:{m:02d}:{s:02d}"
