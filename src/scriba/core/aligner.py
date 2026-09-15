"""Map forced-aligner tokens back onto the punctuated transcript and build segments.

The aligner emits "cleaned" tokens (letters, digits and apostrophes only, CJK split
per character). We walk the original text in order and attach timings to each
original, punctuated word so exports keep the model's punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from scriba.core.engine import AlignedToken
from scriba.models import Segment, Word

_SENTENCE_END = re.compile(r"[.!?…。！？]['\"”»)]*$")
_CLAUSE_END = re.compile(r"[,;:，；：]$")


def _is_kept(ch: str) -> bool:
    return ch == "'" or unicodedata.category(ch)[0] in ("L", "N")


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF
    )


def clean(text: str) -> str:
    return "".join(ch for ch in text if _is_kept(ch)).lower()


def split_display_words(text: str) -> list[str]:
    """Whitespace tokens, with CJK characters split out (punctuation stays attached)."""
    words: list[str] = []
    for raw in text.split():
        buf = ""
        for ch in raw:
            if _is_cjk(ch):
                if buf:
                    words.append(buf)
                    buf = ""
                words.append(ch)
            elif not _is_kept(ch) and not buf and words and _is_cjk(words[-1][-1:] or " "):
                words[-1] += ch  # punctuation after a CJK char
            else:
                buf += ch
        if buf:
            words.append(buf)
    return words


def align_words(text: str, tokens: Sequence[AlignedToken]) -> list[Word]:
    """Attach aligner timings to the punctuated words of `text`.

    Handles tokens glued across aligner chunk boundaries ("fine.Inizio") by consuming
    several aligner tokens for one display word, and falls back to resynchronisation
    when text and tokens diverge.
    """
    display = split_display_words(text)
    toks = [t for t in tokens if clean(t.text)]
    words: list[Word] = []
    ti = 0
    pending: list[str] = []  # punctuation-only words waiting for a timed neighbour

    for dw in display:
        target = clean(dw)
        if not target:
            if words:
                words[-1].text += dw if _attach_left(dw) else " " + dw
            else:
                pending.append(dw)
            continue
        if ti >= len(toks):
            break

        start_tok = ti
        acc = clean(toks[ti].text)
        end_tok = ti
        while acc != target and len(acc) < len(target) and end_tok + 1 < len(toks):
            candidate = acc + clean(toks[end_tok + 1].text)
            if not target.startswith(candidate):
                break
            acc = candidate
            end_tok += 1

        if acc != target:
            # Try to resync: look ahead a few tokens for an exact match.
            match = next(
                (j for j in range(ti, min(ti + 6, len(toks))) if clean(toks[j].text) == target),
                None,
            )
            if match is not None:
                start_tok = end_tok = match
            # else: accept the positional token as best effort
        word_text = (" ".join(pending) + " " + dw).strip() if pending else dw
        pending = []
        words.append(Word(text=word_text, start=toks[start_tok].start, end=max(toks[end_tok].end, toks[start_tok].start)))
        ti = end_tok + 1

    _enforce_monotonic(words)
    return words


def _attach_left(word: str) -> bool:
    return bool(re.fullmatch(r"[.,;:!?…)\]}»”'\"%]+", word))


def _enforce_monotonic(words: list[Word]) -> None:
    last_end = 0.0
    for w in words:
        if w.start < last_end:
            w.start = last_end
        if w.end < w.start:
            w.end = w.start
        last_end = w.end


def join_words(words: Sequence[Word]) -> str:
    text = ""
    for w in words:
        if not text:
            text = w.text
        elif _is_cjk(text[-1]) and _is_cjk(w.text[:1] or " "):
            text += w.text
        else:
            text += " " + w.text
    return text


def build_segments(
    words: Sequence[Word],
    max_seconds: float = 12.0,
    max_chars: int = 160,
    pause_split: float = 0.8,
) -> list[Segment]:
    """Group words into subtitle-friendly segments.

    Breaks on: speaker change, long pause, sentence end (when the segment is not tiny),
    clause end near the limits, or hard duration/length limits.
    """
    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            segments.append(
                Segment(
                    text=join_words(current),
                    start=current[0].start,
                    end=current[-1].end,
                    speaker=current[0].speaker,
                    words=list(current),
                )
            )
            current.clear()

    for w in words:
        if current:
            prev = current[-1]
            seg_len = sum(len(x.text) + 1 for x in current)
            duration = w.end - current[0].start
            if (
                w.speaker != prev.speaker
                or (w.start - prev.end) >= pause_split
                or duration > max_seconds
                or seg_len + len(w.text) > max_chars
            ):
                flush()
        current.append(w)
        seg_len = sum(len(x.text) + 1 for x in current)
        duration = current[-1].end - current[0].start
        if _SENTENCE_END.search(w.text) and (seg_len >= 40 or duration >= 3.0):
            flush()
        elif _CLAUSE_END.search(w.text) and (seg_len >= max_chars * 0.7 or duration >= max_seconds * 0.7):
            flush()
    flush()
    return segments
