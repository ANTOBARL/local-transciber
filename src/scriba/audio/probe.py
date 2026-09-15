"""Media inspection via PyAV (no ffprobe binary required)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from scriba.errors import InvalidAudioError


class AudioInfo(BaseModel):
    path: str
    filename: str
    size_bytes: int
    duration: float
    container: str | None = None
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None
    has_video: bool = False

    def summary(self) -> dict[str, str]:
        from scriba.utils.time import format_clock

        return {
            "File": self.filename,
            "Size": f"{self.size_bytes / 1024**2:.1f} MB",
            "Duration": format_clock(self.duration),
            "Codec": self.codec or "?",
            "Sample rate": f"{self.sample_rate} Hz" if self.sample_rate else "?",
            "Channels": str(self.channels or "?"),
        }


def probe_audio(path: str | Path) -> AudioInfo:
    path = Path(path)
    if not path.is_file():
        raise InvalidAudioError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise InvalidAudioError(f"File is empty: {path}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise InvalidAudioError("PyAV is not installed (pip install av)") from exc

    try:
        with av.open(str(path)) as container:
            audio_streams = container.streams.audio
            if not audio_streams:
                raise InvalidAudioError(f"No audio stream found in {path.name}")
            stream = audio_streams[0]

            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            if not duration and container.duration:
                duration = container.duration / 1_000_000  # av.time_base (microseconds)
            if not duration:
                duration = _decode_duration(container, stream)

            ctx = stream.codec_context
            return AudioInfo(
                path=str(path.resolve()),
                filename=path.name,
                size_bytes=path.stat().st_size,
                duration=round(float(duration or 0.0), 3),
                container=container.format.name if container.format else None,
                codec=ctx.name if ctx else None,
                sample_rate=ctx.sample_rate if ctx else None,
                channels=getattr(ctx, "channels", None) or (ctx.layout.nb_channels if ctx and ctx.layout else None),
                bit_rate=container.bit_rate,
                has_video=bool(container.streams.video),
            )
    except InvalidAudioError:
        raise
    except Exception as exc:
        raise InvalidAudioError(f"Cannot read media file {path.name}: {exc}") from exc


def _decode_duration(container, stream) -> float:
    """Last resort for containers without duration metadata: decode and count samples."""
    samples = 0
    rate = stream.codec_context.sample_rate or 1
    for frame in container.decode(stream):
        samples += frame.samples
    return samples / rate
