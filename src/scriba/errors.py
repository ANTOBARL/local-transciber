"""Explicit error types, each with a user-facing hint."""


class ScribaError(Exception):
    hint: str = ""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        if hint is not None:
            self.hint = hint

    def user_message(self) -> str:
        return f"{self}\n→ {self.hint}" if self.hint else str(self)


class InvalidAudioError(ScribaError):
    hint = "Check that the file exists, is not corrupted and contains an audio stream."


class FFmpegError(ScribaError):
    hint = "Install FFmpeg (PATH) or `pip install imageio-ffmpeg`, or set audio.ffmpeg_path."


class DeviceUnavailableError(ScribaError):
    hint = "Run `scriba doctor`. Set asr.device=cpu to run without a GPU (slow)."


class OutOfMemoryError(ScribaError):
    hint = (
        "Lower asr.max_inference_batch_size, asr.gpu_memory_utilization (vLLM) "
        "or use a smaller dtype/model."
    )


class ModelLoadError(ScribaError):
    hint = "Check the model id/path, network access for the first download, and `scriba doctor`."


class BackendUnavailableError(ScribaError):
    hint = "vLLM is not supported on native Windows: use backend=transformers or run under Linux/WSL2."


class AlignmentError(ScribaError):
    hint = "Retry with timestamps disabled, or lower asr.max_inference_batch_size."


class DiarizationError(ScribaError):
    hint = "Install pyannote.audio and set diarization.hf_token (accept the model terms on Hugging Face)."


class OutputDirError(ScribaError):
    hint = "Choose a writable output directory."
