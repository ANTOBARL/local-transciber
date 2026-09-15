"""Scriba command line interface."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from scriba import APP_NAME, __version__
from scriba.config import ScribaSettings, dotted_to_nested, load_settings, parse_set_option
from scriba.errors import ScribaError

app = typer.Typer(
    name="scriba",
    help=f"{APP_NAME} — local-first long-form transcription with Qwen3-ASR.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML config file.")]
SetOpt = Annotated[
    Optional[list[str]],
    typer.Option("--set", help="Override any setting: --set asr.max_new_tokens=2048 (repeatable)."),
]


def _settings(config: Path | None, flat: dict[str, Any], sets: list[str] | None) -> ScribaSettings:
    overrides = {k: v for k, v in flat.items() if v is not None}
    for item in sets or []:
        key, value = parse_set_option(item)
        overrides[key] = value
    return load_settings(config, dotted_to_nested(overrides))


def _fail(exc: Exception) -> None:
    message = exc.user_message() if isinstance(exc, ScribaError) else f"{type(exc).__name__}: {exc}"
    console.print(f"[bold red]Error:[/] {message}")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------- transcribe
def _transcribe_options(
    config, language, output, timestamps, context, context_file, backend, model, aligner_model,
    device, diarize, num_speakers, min_speakers, max_speakers, formats, batch_size, gpu_memory,
    max_new_tokens, keep_audio, sets,
) -> ScribaSettings:
    if context_file is not None:
        from scriba.utils.paths import read_text_file

        file_ctx = read_text_file(context_file)
        context = f"{context}\n\n{file_ctx}".strip() if context else file_ctx
    flat: dict[str, Any] = {
        "asr.language": language,
        "app.output_root": str(output) if output else None,
        "asr.return_timestamps": timestamps,
        "asr.context": context,
        "asr.backend": backend,
        "asr.model": model,
        "asr.forced_aligner.model": aligner_model,
        "asr.device": device,
        "diarization.enabled": diarize,
        "diarization.num_speakers": num_speakers,
        "diarization.min_speakers": min_speakers,
        "diarization.max_speakers": max_speakers,
        "asr.max_inference_batch_size": batch_size,
        "asr.gpu_memory_utilization": gpu_memory,
        "asr.max_new_tokens": max_new_tokens,
        "app.keep_normalized_audio": keep_audio,
    }
    if formats:
        wanted = {f.strip().lower().replace("md", "markdown") for f in formats.split(",") if f.strip()}
        for fmt in ("json", "txt", "markdown", "srt", "vtt"):
            flat[f"export.{fmt}"] = fmt in wanted
    return _settings(config, flat, sets)


@app.command()
def transcribe(
    audio: Annotated[list[Path], typer.Argument(help="Audio/video file(s).", exists=True, dir_okay=False)],
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="Language or 'auto'.")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output root directory.")] = None,
    timestamps: Annotated[Optional[bool], typer.Option("--timestamps/--no-timestamps")] = None,
    context: Annotated[Optional[str], typer.Option("--context", help="Context / vocabulary prompt.")] = None,
    context_file: Annotated[Optional[Path], typer.Option("--context-file", exists=True, dir_okay=False)] = None,
    backend: Annotated[Optional[str], typer.Option("--backend", help="auto | transformers | vllm")] = None,
    model: Annotated[Optional[str], typer.Option("--model")] = None,
    aligner_model: Annotated[Optional[str], typer.Option("--aligner-model")] = None,
    device: Annotated[Optional[str], typer.Option("--device", help="auto | cuda | cuda:N | cpu")] = None,
    diarize: Annotated[Optional[bool], typer.Option("--diarize/--no-diarize")] = None,
    num_speakers: Annotated[Optional[int], typer.Option("--num-speakers")] = None,
    min_speakers: Annotated[Optional[int], typer.Option("--min-speakers")] = None,
    max_speakers: Annotated[Optional[int], typer.Option("--max-speakers")] = None,
    formats: Annotated[Optional[str], typer.Option("--formats", "-f", help="e.g. json,txt,md,srt,vtt")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size")] = None,
    gpu_memory: Annotated[Optional[float], typer.Option("--gpu-memory", help="vLLM gpu_memory_utilization")] = None,
    max_new_tokens: Annotated[Optional[int], typer.Option("--max-new-tokens")] = None,
    keep_audio: Annotated[Optional[bool], typer.Option("--keep-audio/--no-keep-audio")] = None,
    config: ConfigOpt = None,
    sets: SetOpt = None,
) -> None:
    """Transcribe one or more files. The model is loaded once for all of them."""
    from scriba.core.pipeline import TranscriptionService
    from scriba.utils.logging import setup_logging
    from scriba.utils.time import format_clock

    try:
        settings = _transcribe_options(
            config, language, output, timestamps, context, context_file, backend, model, aligner_model,
            device, diarize, num_speakers, min_speakers, max_speakers, formats, batch_size, gpu_memory,
            max_new_tokens, keep_audio, sets,
        )
    except Exception as exc:
        _fail(exc)
    setup_logging(settings.app.log_level)

    service = TranscriptionService()
    failures = 0
    for path in audio:
        console.rule(f"[bold]{path.name}")
        try:
            result = service.transcribe(path, settings)
        except Exception as exc:
            failures += 1
            message = exc.user_message() if isinstance(exc, ScribaError) else f"{type(exc).__name__}: {exc}"
            console.print(f"[bold red]Failed:[/] {message}")
            continue
        table = Table(show_header=False, box=None)
        table.add_row("Job", result.job_id)
        table.add_row("Output", str(result.job_dir))
        table.add_row("Audio duration", format_clock(result.audio.duration))
        table.add_row("Processing time", format_clock(result.processing_seconds))
        if result.rtf:
            table.add_row("Real-time factor", f"{result.rtf:.3f}")
        table.add_row("Files", ", ".join(p.name for p in result.files.values()))
        console.print(table)
        for w in result.warnings:
            console.print(f"[yellow]Warning:[/] {w}")
    raise typer.Exit(1 if failures else 0)


# ---------------------------------------------------------------------------- doctor
@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check environment, GPU, dependencies and output directory."""
    from scriba.audio.preprocess import find_ffmpeg
    from scriba.utils import device as dev
    from scriba.utils.paths import ensure_writable_dir

    ok_all = True

    def check(label: str, ok: bool, detail: str = "", required: bool = True) -> None:
        nonlocal ok_all
        mark = "[green]✓[/]" if ok else ("[red]✗[/]" if required else "[yellow]![/]")
        if not ok and required:
            ok_all = False
        console.print(f"  {mark} {label}" + (f" [dim]— {detail}[/]" if detail else ""))

    console.print(f"\n[bold]{APP_NAME} System Check[/] [dim]v{__version__}[/]\n")
    try:
        settings = load_settings(config)
    except Exception as exc:
        _fail(exc)

    console.print("[bold]Python[/]")
    check("Python version", sys.version_info >= (3, 10), platform.python_version())

    console.print("[bold]Compute[/]")
    torch_version = dev.package_version("torch")
    check("PyTorch", torch_version is not None, torch_version or "not installed")
    gpus = dev.list_gpus()
    if torch_version:
        import torch

        check("CUDA available", bool(gpus), f"CUDA {torch.version.cuda}" if gpus else "running on CPU (slow)",
              required=False)
    for g in gpus:
        check(f"GPU {g.index}", True, f"{g.name} · {g.total_memory_gb} GB · compute {g.capability[0]}.{g.capability[1]}"
              f" · bf16 {'yes' if g.bf16 else 'no'}")
    try:
        device = dev.resolve_device(settings.asr.device)
        check("Configured device", True, f"{settings.asr.device} → {device} ({dev.resolve_dtype_name(settings.asr.dtype, device)})")
    except ScribaError as exc:
        check("Configured device", False, str(exc))

    console.print("[bold]Inference[/]")
    qwen_version = dev.package_version("qwen-asr")
    check("qwen-asr", qwen_version is not None, qwen_version or "pip install qwen-asr")
    if qwen_version:
        try:
            from scriba.compat import ensure_nagisa_importable

            ensure_nagisa_importable()
            import qwen_asr  # noqa: F401

            check("qwen-asr import", True)
        except Exception as exc:
            hint = ""
            if any(ord(c) > 127 for c in str(exc)):
                hint = " (site-packages path contains non-ASCII characters: install into the env, not --user)"
            check("qwen-asr import", False, f"{type(exc).__name__}: {str(exc)[:160]}{hint}")
    check("transformers", dev.package_version("transformers") is not None, dev.package_version("transformers") or "missing")
    vllm_ok, vllm_reason = dev.vllm_supported()
    check("vLLM", vllm_ok, dev.package_version("vllm") or vllm_reason, required=False)
    try:
        from scriba.core.engine import resolve_backend

        check("Backend", True, f"{settings.asr.backend} → {resolve_backend(settings.asr.backend)}")
    except ScribaError as exc:
        check("Backend", False, str(exc))
    check("flash-attn", dev.package_version("flash-attn") is not None,
          dev.package_version("flash-attn") or "optional", required=False)

    console.print("[bold]Audio[/]")
    ffmpeg = find_ffmpeg(settings.audio.ffmpeg_path)
    check("FFmpeg", ffmpeg is not None, ffmpeg or "not found — PyAV fallback will be used", required=False)
    check("PyAV (probe/decoder)", dev.package_version("av") is not None, dev.package_version("av") or "pip install av")

    console.print("[bold]Optional[/]")
    check("Gradio (UI)", dev.package_version("gradio") is not None, dev.package_version("gradio") or "pip install gradio",
          required=False)
    check("pyannote.audio (diarization)", dev.package_version("pyannote.audio") is not None,
          dev.package_version("pyannote.audio") or "pip install pyannote.audio", required=False)

    console.print("[bold]Storage & models[/]")
    try:
        path = ensure_writable_dir(settings.app.output_root)
        check("Output directory writable", True, str(path))
    except ScribaError as exc:
        check("Output directory writable", False, str(exc))
    for label, repo in (("ASR model", settings.asr.model), ("Aligner model", settings.asr.forced_aligner.model)):
        status = _model_status(repo)
        check(label, status != "unavailable", f"{repo} — {status}", required=False)

    console.print("\n[bold green]System ready.[/]\n" if ok_all else "\n[bold red]Some required checks failed.[/]\n")
    raise typer.Exit(0 if ok_all else 1)


def _model_status(repo: str) -> str:
    if Path(repo).expanduser().is_dir():
        return "local directory"
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo, "config.json")
        if isinstance(cached, str):
            return "cached locally"
        return "will be downloaded on first use"
    except ImportError:
        return "unavailable"


# ---------------------------------------------------------------------------- info / models
@app.command()
def info(config: ConfigOpt = None, sets: SetOpt = None) -> None:
    """Show version and the effective configuration."""
    import yaml

    try:
        settings = _settings(config, {}, sets)
    except Exception as exc:
        _fail(exc)
    console.print(f"[bold]{APP_NAME}[/] v{__version__}\n")
    console.print(yaml.safe_dump(settings.to_public_dict(), sort_keys=False, allow_unicode=True))


@app.command()
def models(config: ConfigOpt = None) -> None:
    """List configured models and their local availability."""
    settings = load_settings(config)
    table = Table("Role", "Model", "Status")
    table.add_row("ASR", settings.asr.model, _model_status(settings.asr.model))
    table.add_row("Forced aligner", settings.asr.forced_aligner.model, _model_status(settings.asr.forced_aligner.model))
    table.add_row("Diarization", settings.diarization.model, _model_status(settings.diarization.model))
    console.print(table)
    console.print("[dim]Other Qwen3-ASR checkpoints: Qwen/Qwen3-ASR-0.6B (lighter), or any local path.[/]")


# ---------------------------------------------------------------------------- benchmark
@app.command()
def benchmark(
    audio: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: ConfigOpt = None,
    sets: SetOpt = None,
) -> None:
    """Transcribe a file and report RTF, peak VRAM and peak RAM."""
    from scriba.core.pipeline import TranscriptionService
    from scriba.utils.logging import setup_logging
    from scriba.utils.monitor import ResourceMonitor
    from scriba.utils.time import format_clock

    try:
        settings = _settings(config, {}, sets)
    except Exception as exc:
        _fail(exc)
    setup_logging(settings.app.log_level)
    service = TranscriptionService()
    try:
        with ResourceMonitor() as monitor:
            result = service.transcribe(audio, settings)
    except Exception as exc:
        _fail(exc)

    out_size = sum(p.stat().st_size for p in result.files.values())
    engine_info = service.engine.info
    table = Table(show_header=False, box=None)
    rows = [
        ("Audio duration", format_clock(result.audio.duration)),
        ("Processing time", format_clock(result.processing_seconds)),
        ("Real-time factor", f"{result.rtf:.4f}" if result.rtf else "n/a"),
        ("Peak VRAM", f"{monitor.peak_vram_mb:.0f} MB (device-wide)" if monitor.peak_vram_mb else "n/a"),
        ("Peak RAM", f"{monitor.peak_ram_mb:.0f} MB" if monitor.peak_ram_mb else "n/a (pip install psutil)"),
        ("Model", settings.asr.model),
        ("Backend", engine_info.backend if engine_info else settings.asr.backend),
        ("Device", engine_info.device_name if engine_info else "?"),
        ("Batch size", str(settings.asr.max_inference_batch_size)),
        ("Timestamping", "enabled" if settings.asr.timestamps_active else "disabled"),
        ("Diarization", "enabled" if settings.diarization.enabled else "disabled"),
        ("Output size", f"{out_size / 1024:.1f} KB"),
        ("Output", str(result.job_dir)),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


# ---------------------------------------------------------------------------- rename
@app.command()
def rename(
    job_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Job output directory.")],
    mapping: Annotated[list[str], typer.Argument(help="SPEAKER_00=Luca SPEAKER_01=Clara ...")],
) -> None:
    """Rename speakers of a finished job and regenerate exports (no ASR re-run)."""
    from scriba.core.pipeline import apply_speaker_names

    names = {}
    for item in mapping:
        if "=" not in item:
            _fail(ValueError(f"Invalid mapping '{item}', expected SPEAKER_00=Name"))
        k, v = item.split("=", 1)
        names[k.strip()] = v.strip()
    try:
        files = apply_speaker_names(job_dir, names)
    except Exception as exc:
        _fail(exc)
    console.print(f"[green]Updated:[/] {', '.join(p.name for p in files.values())}")


# ---------------------------------------------------------------------------- ui
@app.command()
def ui(
    host: Annotated[str, typer.Option(help="Bind address (localhost only by default).")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
    preload: Annotated[bool, typer.Option("--preload/--lazy", help="Load the model at startup.")] = False,
    lang: Annotated[str, typer.Option("--lang", help="Default UI language: it | en")] = "it",
    config: ConfigOpt = None,
    sets: SetOpt = None,
) -> None:
    """Launch the local web UI."""
    try:
        settings = _settings(config, {}, sets)
    except Exception as exc:
        _fail(exc)
    from scriba.ui import launch

    launch(settings, host=host, port=port, preload=preload, lang=lang)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address. Use 0.0.0.0 inside containers.")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    preload: Annotated[bool, typer.Option("--preload/--lazy", help="Load the model at startup.")] = False,
    frontend: Annotated[Optional[Path], typer.Option("--frontend", help="Built React app (frontend/dist).")] = None,
    lang: Annotated[str, typer.Option("--lang", help="Default UI language: it | en")] = "it",
    config: ConfigOpt = None,
    sets: SetOpt = None,
) -> None:
    """Start the HTTP API and the React web app."""
    try:
        settings = _settings(config, {}, sets)
    except Exception as exc:
        _fail(exc)
    from scriba.api import serve as run_server

    run_server(settings, host=host, port=port, preload=preload, frontend_dir=frontend, lang=lang)


@app.callback(invoke_without_command=True)
def _main(version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if version:
        console.print(f"{APP_NAME} {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
