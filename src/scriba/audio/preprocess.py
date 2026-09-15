"""Audio decoding / resampling.

No denoise, compression, EQ or loudness normalization: the signal is only
decoded, downmixed and resampled.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from scriba.errors import FFmpegError, InvalidAudioError
from scriba.utils.logging import get_logger

log = get_logger("audio")


def find_ffmpeg(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() or shutil.which(explicit) else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def load_waveform(
    path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_path: str | None = None,
) -> np.ndarray:
    """Decode any media file into float32 PCM in [-1, 1], shape (samples,) or (samples, channels)."""
    ffmpeg = find_ffmpeg(ffmpeg_path)
    if ffmpeg_path and ffmpeg is None:
        raise FFmpegError(f"FFmpeg not found at configured path: {ffmpeg_path}")
    if ffmpeg:
        audio = _decode_ffmpeg(ffmpeg, Path(path), sample_rate, channels)
    else:
        log.warning("FFmpeg not found, decoding with PyAV")
        audio = _decode_pyav(Path(path), sample_rate, channels)
    if audio.size == 0:
        raise InvalidAudioError(f"Decoded audio is empty: {Path(path).name}")
    return audio


def _decode_ffmpeg(ffmpeg: str, path: Path, sample_rate: int, channels: int) -> np.ndarray:
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", str(channels), "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    log.debug("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except OSError as exc:
        raise FFmpegError(f"Cannot execute FFmpeg: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise FFmpegError(f"FFmpeg failed on {path.name}: {stderr[-800:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels > 1:
        audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels)
    return audio


def _decode_pyav(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    import av

    layout = "mono" if channels == 1 else "stereo"
    resampler = av.AudioResampler(format="flt", layout=layout, rate=sample_rate)
    parts: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            for frame in container.decode(stream):
                frame.pts = None
                for out in resampler.resample(frame):
                    parts.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                parts.append(out.to_ndarray().reshape(-1))
    except Exception as exc:
        raise InvalidAudioError(f"Cannot decode {path.name}: {exc}") from exc
    audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    if channels > 1:
        audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels)
    return audio


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write 16-bit PCM WAV using only the standard library."""
    path = Path(path)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path
