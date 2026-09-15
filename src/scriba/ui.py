"""Local Gradio UI (Italian / English). Contains no ASR logic: everything goes through TranscriptionService."""

from __future__ import annotations

import html
import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from scriba import APP_NAME, __version__
from scriba.config import ScribaSettings, SettingsProvider
from scriba.errors import ScribaError
from scriba.i18n import DEFAULT_LANG, UI_LANGUAGES, language_choices, t
from scriba.languages import SUPPORTED_LANGUAGES
from scriba.models import JobStatus, Transcript
from scriba.utils.logging import get_logger, setup_logging
from scriba.utils.paths import MEDIA_EXTENSIONS
from scriba.utils.time import format_clock

log = get_logger("ui")

from scriba.webform import BACKEND_CHOICES, DTYPE_CHOICES, TranscriptionForm  # noqa: E402

EXPORT_CHOICES = [("JSON", "json"), ("TXT", "txt"), ("Markdown", "markdown"), ("SRT", "srt"), ("VTT", "vtt"), ("Word", "docx")]
SYSTEM_FONTS = ["ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
MONO_FONTS = ["ui-monospace", "Cascadia Code", "Consolas", "Menlo", "monospace"]

CSS = """
.gradio-container {max-width: 1320px !important; margin: 0 auto !important}
#scriba-topbar {align-items: center; margin-bottom: .25rem}
#scriba-title h1 {margin: 0; font-size: 2rem; letter-spacing: -.02em}
#scriba-title p {margin: .15rem 0 0; opacity: .65; font-size: .95rem}
#scriba-lang {flex-grow: 0 !important; min-width: 330px !important; margin-left: auto}
#scriba-lang .wrap {gap: .6rem; justify-content: flex-end}
#scriba-lang label {
  font-size: 1.2rem !important; font-weight: 700; padding: .7rem 1.25rem !important;
  border-radius: 999px !important; border-width: 2px !important; cursor: pointer;
}
#scriba-lang label.selected {background: var(--color-accent) !important; color: #fff !important;
  border-color: var(--color-accent) !important}
#scriba-lang input[type=radio] {display: none}

.scriba-card {border-radius: 14px !important; overflow: hidden}
.scriba-card > .styler, .scriba-card .form {background: transparent !important; border: 0 !important}
#scriba-status-card {padding: .9rem 1rem !important}
#scriba-upload {min-height: 0}
.scriba-confirm {padding: .9rem 1rem !important; border-radius: 12px !important;
  border: 1px solid rgba(34,197,94,.5) !important}
.scriba-optimized {display: flex; flex-direction: column; gap: .1rem; padding: .8rem 1rem; border-radius: 12px;
  background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.5); margin-bottom: .4rem}
.scriba-optimized strong {font-size: 1.05rem; color: #22c55e}
.scriba-optimized-detail {opacity: .75; font-size: .85rem; margin-bottom: .5rem}
.scriba-tiles {display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem}
.scriba-tile {display: flex; flex-direction: column; gap: .2rem; padding: .55rem .7rem; border-radius: 10px;
  background: var(--background-fill-secondary)}
.scriba-tile-label {font-size: .75rem; opacity: .7}
.scriba-tile-value {font-size: 1.3rem; font-weight: 700; font-variant-numeric: tabular-nums}
.scriba-tile-value small {font-size: .75rem; font-weight: 500; opacity: .7; margin-left: .2rem}
.scriba-meter {height: 6px; border-radius: 99px; background: var(--border-color-primary); overflow: hidden}
.scriba-meter > div {height: 100%; background: var(--color-accent); border-radius: 99px}
.scriba-facts {margin: .6rem 0 .2rem; padding-left: 1.1rem; font-size: .9rem}
.scriba-footnote {font-size: .75rem; opacity: .6}
@media (max-width: 560px) {.scriba-tiles {grid-template-columns: 1fr}}
#scriba-apply {font-weight: 800; font-size: 1.1rem; min-height: 3.2rem; margin-top: .8rem}
#scriba-transcribe {font-weight: 800; letter-spacing: .1em; font-size: 1.1rem; min-height: 3.2rem}
#scriba-context-btn {align-self: flex-end; max-width: 13rem}
.scriba-hint {opacity: .65; font-size: .9rem}

.scriba-status {display: flex; align-items: center; gap: .6rem; font-size: 1rem; min-height: 2rem}
.scriba-dot {width: .7rem; height: .7rem; border-radius: 50%; background: #64748b; flex: none}
.scriba-status.running .scriba-dot {background: #f59e0b; animation: scriba-pulse 1.2s infinite}
.scriba-status.done .scriba-dot {background: #22c55e}
.scriba-status.error .scriba-dot {background: #ef4444}
@keyframes scriba-pulse {50% {opacity: .35}}
.scriba-metrics {display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin-top: .6rem}
.scriba-metric {background: var(--background-fill-secondary); border-radius: 10px; padding: .55rem .8rem}
.scriba-metric .v {font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums}
.scriba-metric .l {font-size: .8rem; opacity: .65}
.scriba-toggle {border: 2px solid var(--border-color-primary) !important; border-radius: 12px !important;
  padding: .95rem 1.1rem !important; margin: .3rem .6rem !important; background: var(--input-background-fill) !important;
  cursor: pointer; transition: border-color .15s, background .15s; width: auto !important}
.scriba-toggle:hover {border-color: var(--color-accent) !important}
.scriba-toggle:has(input:checked) {border-color: var(--color-accent) !important;
  background: color-mix(in srgb, var(--color-accent) 16%, transparent) !important}
.scriba-toggle .checkbox-container {display: flex; flex-direction: row-reverse; justify-content: space-between;
  align-items: center; gap: 1rem; cursor: pointer}
.scriba-toggle .label-text {font-size: 1.12rem !important; font-weight: 700}
.scriba-toggle .info-text {font-size: .85rem; opacity: .75; margin-top: .15rem; padding-right: 4rem}
.scriba-toggle input[type=checkbox] {appearance: none; -webkit-appearance: none; width: 3.1rem !important;
  height: 1.75rem !important; border-radius: 99px !important; background: var(--border-color-primary) !important;
  position: relative; flex: none; cursor: pointer; border: 0 !important; transition: background .15s; margin: 0}
.scriba-toggle input[type=checkbox]::after {content: ""; position: absolute; top: .2rem; left: .2rem; width: 1.35rem;
  height: 1.35rem; border-radius: 50%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.35); transition: transform .15s}
.gradio-container .scriba-toggle .checkbox-container input[type=checkbox]:not(:checked) {
  background-color: var(--border-color-primary) !important; background-image: none !important}
.gradio-container .scriba-toggle .checkbox-container input[type=checkbox]:checked {
  background-color: var(--color-accent) !important; background-image: none !important}
.scriba-toggle input[type=checkbox]:checked::after {transform: translateX(1.35rem)}
.scriba-alert {display: flex; gap: .85rem; align-items: flex-start; padding: 1rem 1.1rem; border-radius: 14px;
  border: 2px solid; margin-bottom: .6rem}
.scriba-alert.error {background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.6)}
.scriba-alert.error .scriba-alert-icon {color: #ef4444}
.scriba-alert.warn {background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.55); padding: .65rem .85rem;
  border-width: 1px}
.scriba-alert.warn .scriba-alert-icon {color: #f59e0b}
.scriba-alert strong {display: block; font-size: 1.05rem}
.scriba-alert p {margin: .15rem 0}
.scriba-alert p:empty {display: none}
.scriba-alert details {margin-top: .4rem; font-size: .85rem}
.scriba-alert summary {cursor: pointer; opacity: .75}
.scriba-alert code {display: block; margin-top: .35rem; padding: .5rem .65rem; border-radius: 8px; white-space: pre-wrap;
  word-break: break-word; background: var(--background-fill-secondary)}
.scriba-progress {margin-top: .7rem}
.scriba-opt {width: 100%; border-collapse: collapse; margin-top: .8rem; font-variant-numeric: tabular-nums}
.scriba-opt th {text-align: left; font-size: .8rem; opacity: .7; padding: .3rem .4rem}
.scriba-opt td {padding: .35rem .4rem; border-top: 1px solid var(--border-color-primary)}
.scriba-opt tr.best td {color: var(--color-accent); font-weight: 600}
.scriba-progress-head {display: flex; justify-content: space-between; gap: 1rem; font-size: .9rem; margin-bottom: .35rem;
  flex-wrap: wrap}
.scriba-progress-head span:last-child {opacity: .75}
.scriba-bar {height: 10px; border-radius: 99px; background: var(--background-fill-secondary); overflow: hidden}
.scriba-bar > div {height: 100%; border-radius: 99px; background: var(--color-accent); transition: width .9s linear}
#scriba-preview textarea {font-family: ui-monospace, Consolas, monospace; font-size: .92rem; line-height: 1.5}
@media (max-width: 720px) {
  #scriba-lang {min-width: 100% !important} #scriba-lang .wrap {justify-content: center}
  .scriba-metrics {grid-template-columns: 1fr}
}
"""


def render_preview(transcript: Transcript) -> str:
    from scriba.exporters.txt_exporter import render_txt

    return render_txt(transcript)


def status_html(message: str, state: str = "idle") -> str:
    return f'<div class="scriba-status {state}"><span class="scriba-dot"></span><span>{html.escape(message)}</span></div>'


_ICON_ERROR = ('<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" '
               'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
               '<path d="m15 9-6 6M9 9l6 6"/></svg>')
_ICON_WARN = ('<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 '
              '4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/></svg>')


def alert_html(title: str, message: str, detail: str | None = None, kind: str = "error",
               detail_label: str = "Details") -> str:
    """Prominent error/warning box (same look as the React frontend)."""
    icon = _ICON_ERROR if kind == "error" else _ICON_WARN
    extra = ""
    if detail and detail != message:
        extra = f"<details><summary>{html.escape(detail_label)}</summary><code>{html.escape(detail)}</code></details>"
    return (f'<div class="scriba-alert {kind}" role="alert"><span class="scriba-alert-icon">{icon}</span>'
            f'<div><strong>{html.escape(title)}</strong><p>{html.escape(message)}</p>{extra}</div></div>')


def progress_html(snapshot: dict[str, Any] | None, lang: str) -> str:
    if not snapshot:
        return ""
    from scriba.core.progress import format_eta

    aligning = snapshot["phase"] == "align"
    asr_done, asr_total = snapshot["asr"]
    align_done, align_total = snapshot["align"]

    def phase(label: str, done: int, total: int, active: bool) -> str:
        mark = "✓ " if total and done >= total else ""
        text = f"{mark}{html.escape(label)} {done}/{total}" if total else html.escape(label)
        return f"<b>{text}</b>" if active else f'<span style="opacity:.7">{text}</span>'

    # Each phase keeps its own counter, so finishing ASR never looks like a reset.
    parts = [phase(t(lang, "progress_asr"), asr_done, asr_total, not aligning)]
    if aligning or align_total:
        parts.append(phase(t(lang, "progress_align"), align_done, align_total or asr_total, aligning))
    pct = snapshot["percent"]
    return (
        '<div class="scriba-progress">'
        f'<div class="scriba-progress-head"><span>{" · ".join(parts)} · {pct}%</span>'
        f'<span>{html.escape(format_eta(snapshot["eta_seconds"], lang))}</span></div>'
        f'<div class="scriba-bar"><div style="width:{snapshot["fraction"] * 100:.1f}%"></div></div>'
        "</div>"
    )


def optimize_desc_md(lang: str) -> str:
    return f"{t(lang, 'optimize_desc')} **{t(lang, 'optimize_duration')}**"


def format_minutes(minutes: float) -> str:
    """1 min 15 s, 4 min, 45 s (kept in sync with frontend/src/OptimizedInfographic.tsx)."""
    total = round(minutes * 60)
    if total < 60:
        return f"{total} s"
    m, s = divmod(total, 60)
    return f"{m} min {s:02d} s" if m < 10 and s >= 5 else f"{round(total / 60)} min"


def optimized_html(lang: str) -> str:
    """Compact infographic shown once tuned parameters have been applied."""
    from scriba.core.optimizer import optimization_status

    status = optimization_status()
    if not status["optimized"]:
        return ""
    details = " · ".join(x for x in (t(lang, "optimized_detail").format(batch=status["batch"]),
                                     status.get("device"), status.get("applied_at")) if x)
    head = (f'<div class="scriba-optimized"><strong>{html.escape(t(lang, "optimized_title"))}</strong>'
            f'<span class="scriba-optimized-detail">{html.escape(details)}</span>')
    if status.get("speed") is None:
        return head + "</div>"

    tiles = [
        (t(lang, "info_speed"), f'{status["speed"]:.0f}<small>{html.escape(t(lang, "info_speed_unit"))}</small>', ""),
        (t(lang, "info_hour"), f'≈ {format_minutes(status["minutes_per_audio_hour"])}', ""),
    ]
    if status.get("gpu_total_mb"):
        pct = min(100.0, status["peak_vram_mb"] / status["gpu_total_mb"] * 100)
        tiles.append((t(lang, "info_vram"),
                      f'{status["peak_vram_mb"] / 1024:.1f}<small> / {status["gpu_total_mb"] / 1024:.0f} GB</small>',
                      f'<div class="scriba-meter"><div style="width:{pct:.0f}%"></div></div>'))
    tiles_html = "".join(
        f'<div class="scriba-tile"><span class="scriba-tile-label">{html.escape(label)}</span>'
        f'<span class="scriba-tile-value">{value}</span>{extra}</div>'
        for label, value, extra in tiles
    )
    facts = [t(lang, "info_fact_long").format(time=format_minutes(status["minutes_per_audio_hour"] * 3))]
    if (status.get("speedup_vs_smallest") or 0) > 1:
        facts.append(t(lang, "info_fact_speedup").format(x=status["speedup_vs_smallest"]))
    facts_html = "".join(f"<li>{html.escape(f)}</li>" for f in facts)
    source = (t(lang, "info_source_real").format(n=status.get("real_jobs", 0)) if status.get("speed_source") == "real"
              else t(lang, "info_source_benchmark"))
    return (head + f'<div class="scriba-tiles">{tiles_html}</div><ul class="scriba-facts">{facts_html}</ul>'
            f'<span class="scriba-footnote">{html.escape(source)} {html.escape(t(lang, "info_footnote"))}</span></div>')


def optimizer_html(snapshot: dict[str, Any], lang: str, result: dict[str, Any] | None = None) -> str:
    """Live progress and result table of the inference optimizer."""
    stage = snapshot["stage"]
    if result is None:
        label = t(lang, f"optimize_stage_{stage}") if stage != "trial" else \
            f"{t(lang, 'optimize_stage_trial')} {snapshot['current_batch']}"
        head = (f'<div class="scriba-progress"><div class="scriba-progress-head"><span><b>{html.escape(label)}</b>'
                f' · {snapshot["done"]}/{snapshot["total"]}</span><span>{format_clock(snapshot["elapsed_seconds"])}'
                f'</span></div><div class="scriba-bar"><div style="width:{snapshot["fraction"] * 100:.0f}%"></div>'
                f'</div></div>')
    else:
        msg = t(lang, "optimize_done").format(best=result["best_batch"], previous=result["previous_batch"])
        head = f'<div class="scriba-status done"><span class="scriba-dot"></span><span>{html.escape(msg)}</span></div>'
    rows = []
    best = result["best_batch"] if result else None
    for trial in (result["trials"] if result else snapshot["trials"]):
        is_best = trial["batch"] == best
        speed = f'{trial["speed"]:.1f}×' if trial["speed"] else "—"
        vram = f'{trial["peak_vram_mb"]:,.0f} MB' if trial["peak_vram_mb"] is not None else "—"
        outcome = t(lang, f'optimize_result_{trial["status"]}') + (f' · <b>{t(lang, "optimize_best")}</b>' if is_best else "")
        rows.append(f'<tr{" class=best" if is_best else ""}><td>{trial["batch"]}</td><td>{speed}</td><td>{vram}</td>'
                    f'<td>{outcome}</td></tr>')
    table = ""
    if rows:
        table = (f'<table class="scriba-opt"><thead><tr><th>{t(lang, "optimize_col_batch")}</th>'
                 f'<th>{t(lang, "optimize_col_speed")}</th><th>{t(lang, "optimize_col_vram")}</th>'
                 f'<th>{t(lang, "optimize_col_result")}</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    return head + table


def metrics_html(lang: str, elapsed: str = "—", duration: str = "—", rtf: str = "—") -> str:
    items = [(elapsed, t(lang, "elapsed")), (duration, t(lang, "audio_duration")), (rtf, t(lang, "rtf"))]
    cells = "".join(f'<div class="scriba-metric"><div class="v">{v}</div><div class="l">{l}</div></div>'
                    for v, l in items)
    return f'<div class="scriba-metrics">{cells}</div>'


def _download_copies(files: dict[str, Path], job_id: str) -> list[str]:
    """Copy outputs into the temp dir so Gradio can serve them from any output location."""
    target = Path(tempfile.gettempdir()) / "scriba_downloads" / job_id
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for path in files.values():
        dst = target / path.name
        shutil.copy2(path, dst)
        out.append(str(dst))
    return out


def _file_info_markdown(path: str | None, lang: str) -> str:
    if not path:
        return ""
    from scriba.audio.probe import probe_audio

    try:
        info = probe_audio(path)
    except ScribaError as exc:
        return alert_html(t(lang, "error_title"), str(exc))
    parts = [
        f"**{html.escape(info.filename)}**",
        f"{info.size_bytes / 1024**2:.1f} MB",
        format_clock(info.duration),
        info.codec or "?",
        f"{info.sample_rate} Hz" if info.sample_rate else "?",
        f"{info.channels or '?'} ch",
    ]
    return " · ".join(parts)


def make_theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue="indigo", neutral_hue="slate", radius_size="md",
        font=SYSTEM_FONTS, font_mono=MONO_FONTS,  # no Google Fonts: nothing leaves the machine
    ).set(
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color="*body_text_color_subdued",
        block_label_text_color_dark="*body_text_color_subdued",
        block_label_border_width="0px",
        block_label_padding="0 0 .25rem 0",
        block_label_text_weight="600",
        block_title_text_weight="600",
        block_shadow="none",
        button_primary_shadow="none",
    )


def build_app(settings: ScribaSettings | SettingsProvider, service: Any = None, default_lang: str = DEFAULT_LANG):
    import gradio as gr

    from scriba.core.pipeline import TranscriptionService, apply_speaker_names

    # A provider re-reads .env on change, so optimizer results apply without a restart.
    current = settings.get if isinstance(settings, SettingsProvider) else (lambda: settings)
    base = current()
    service = service or TranscriptionService()
    lock = threading.Lock()
    a, asr, dia, ex = base.app, base.asr, base.diarization, base.export
    L = default_lang

    with gr.Blocks(title=APP_NAME, analytics_enabled=False) as demo:
        job_dir_state = gr.State(None)
        lang_state = gr.State(L)
        saved_lang = gr.BrowserState(L, storage_key="scriba_ui_lang") if hasattr(gr, "BrowserState") else None

        # ---------------------------------------------------------------- top bar
        with gr.Row(elem_id="scriba-topbar"):
            title = gr.Markdown(f"# {APP_NAME}\n{t(L, 'subtitle')}", elem_id="scriba-title")
            lang_radio = gr.Radio(
                choices=[(label, code) for code, label in UI_LANGUAGES.items()], value=L,
                show_label=False, container=False, elem_id="scriba-lang",
            )

        with gr.Row(equal_height=False):
            # ------------------------------------------------------------ left: input
            with gr.Column(scale=5, min_width=380):
                with gr.Group(elem_classes="scriba-card"):
                    audio_in = gr.File(label=t(L, "audio_file"), type="filepath", file_types=sorted(MEDIA_EXTENSIONS),
                                       height=150, elem_id="scriba-upload")
                    audio_info = gr.Markdown(elem_classes="scriba-hint")

                with gr.Group(elem_classes="scriba-card"):
                    language = gr.Dropdown(label=t(L, "language"), choices=language_choices(L, SUPPORTED_LANGUAGES),
                                           value=asr.language or "auto")
                    context = gr.Textbox(label=t(L, "context"), lines=4, max_lines=12, value=asr.context,
                                         placeholder=t(L, "context_placeholder"))
                    context_btn = gr.UploadButton(t(L, "load_context"), file_types=[".txt", ".md"], size="sm",
                                                  variant="secondary", elem_id="scriba-context-btn")
                    timestamps = gr.Checkbox(label=t(L, "timestamps"), info=t(L, "timestamps_hint"),
                                             value=asr.return_timestamps, elem_classes="scriba-toggle")
                    diarize = gr.Checkbox(label=t(L, "diarize"), info=t(L, "diarize_hint"),
                                          value=dia.enabled, elem_classes="scriba-toggle")
                    with gr.Row(visible=dia.enabled) as speaker_opts:
                        num_speakers = gr.Number(label=t(L, "num_speakers"), value=dia.num_speakers or 0,
                                                 precision=0, minimum=0)
                        min_speakers = gr.Number(label=t(L, "min_speakers"), value=dia.min_speakers or 0,
                                                 precision=0, minimum=0)
                        max_speakers = gr.Number(label=t(L, "max_speakers"), value=dia.max_speakers or 0,
                                                 precision=0, minimum=0)
                    formats = gr.CheckboxGroup(label=t(L, "formats"), choices=EXPORT_CHOICES,
                                               value=ex.enabled_formats())

                with gr.Row():
                    run_btn = gr.Button(t(L, "transcribe"), variant="primary", size="lg", elem_id="scriba-transcribe",
                                        scale=4)
                    stop_btn = gr.Button(t(L, "stop"), variant="stop", size="lg", scale=1)

                with gr.Accordion(t(L, "settings"), open=False) as acc_settings:
                    with gr.Tabs():
                        with gr.Tab(t(L, "tab_output")) as tab_output:
                            output_dir = gr.Textbox(label=t(L, "output_dir"), value=str(a.output_root))
                            subfolder = gr.Checkbox(label=t(L, "subfolder"), value=a.create_job_subfolder)
                            keep_audio = gr.Checkbox(label=t(L, "keep_audio"), value=a.keep_normalized_audio)
                            hf_token = gr.Textbox(label=t(L, "hf_token"), type="password", value="",
                                                  placeholder=t(L, "hf_token_placeholder"))
                        with gr.Tab(t(L, "tab_model")) as tab_model:
                            model = gr.Textbox(label=t(L, "asr_model"), value=asr.model)
                            aligner_model = gr.Textbox(label=t(L, "aligner_model"), value=asr.forced_aligner.model)
                            with gr.Row():
                                backend = gr.Dropdown(label=t(L, "backend"), choices=BACKEND_CHOICES,
                                                      value=asr.backend)
                                device = gr.Textbox(label=t(L, "device"), value=asr.device)
                                dtype = gr.Dropdown(label=t(L, "dtype"), choices=DTYPE_CHOICES, value=asr.dtype)
                            backend_kwargs = gr.Code(label=t(L, "backend_kwargs"), language="json", lines=3,
                                                     value=json.dumps(asr.backend_kwargs, indent=2))
                            with gr.Row():
                                unload_btn = gr.Button(t(L, "unload"), variant="secondary", size="sm")
                                model_state = gr.Markdown(elem_classes="scriba-hint")
                        with gr.Tab(t(L, "tab_performance")) as tab_perf:
                            gpu_mem = gr.Slider(label=t(L, "gpu_mem"), minimum=0.1, maximum=0.98, step=0.01,
                                                value=asr.gpu_memory_utilization)
                            with gr.Row():
                                batch = gr.Number(label=t(L, "batch"), value=asr.max_inference_batch_size,
                                                  precision=0)
                                max_tokens = gr.Number(label=t(L, "max_tokens"), value=asr.max_new_tokens,
                                                       precision=0)
                                align_batch = gr.Number(label=t(L, "align_batch"), value=asr.align_batch_size,
                                                        precision=0, minimum=0)
                            with gr.Row():
                                sample_rate = gr.Number(label=t(L, "sample_rate"), value=base.audio.sample_rate,
                                                        precision=0)
                                channels = gr.Number(label=t(L, "channels"), value=base.audio.channels, precision=0)
                            normalize = gr.Checkbox(label=t(L, "normalize"), value=base.audio.normalize)
                        with gr.Tab(t(L, "tab_optimize")) as tab_optimize:
                            optimized_box = gr.HTML(optimized_html(L))
                            optimize_desc = gr.Markdown(optimize_desc_md(L), elem_classes="scriba-hint")
                            optimize_btn = gr.Button(t(L, "optimize_btn"), variant="primary")
                            with gr.Group(visible=False, elem_classes="scriba-confirm") as confirm_box:
                                confirm_text = gr.Markdown(t(L, "optimize_confirm"))
                                with gr.Row():
                                    confirm_no = gr.Button(t(L, "no"), variant="secondary")
                                    confirm_yes = gr.Button(t(L, "yes"), variant="primary")
                            optimize_out = gr.HTML()
                            optimize_state = gr.State(None)
                            apply_btn = gr.Button(t(L, "optimize_apply"), variant="primary", size="lg", visible=False,
                                                  elem_id="scriba-apply")
                            apply_out = gr.HTML()

            # ------------------------------------------------------------ right: results
            with gr.Column(scale=6, min_width=380):
                with gr.Group(elem_classes="scriba-card", elem_id="scriba-status-card"):
                    status = gr.HTML(status_html(t(L, "ready")))
                    progress = gr.HTML("")
                    metrics = gr.HTML(metrics_html(L))
                warnings_box = gr.HTML()

                with gr.Tabs():
                    with gr.Tab(t(L, "tab_transcript")) as tab_transcript:
                        preview = gr.Textbox(show_label=False, lines=22, max_lines=22, interactive=False,
                                             placeholder=t(L, "preview_empty"), elem_id="scriba-preview")
                    with gr.Tab(t(L, "tab_files")) as tab_files:
                        downloads = gr.File(label=t(L, "download"), file_count="multiple", interactive=False)
                        output_folder = gr.Textbox(label=t(L, "output_folder"), interactive=False)
                    with gr.Tab(t(L, "tab_speakers")) as tab_speakers:
                        speakers_hint = gr.Markdown(t(L, "speakers_empty"), elem_classes="scriba-hint")
                        with gr.Column(visible=False) as rename_group:
                            speaker_table = gr.Dataframe(headers=[t(L, "speaker_col"), t(L, "name_col")],
                                                         datatype=["str", "str"], type="array", interactive=True,
                                                         column_count=(2, "fixed"))
                            rename_btn = gr.Button(t(L, "apply_names"), variant="primary")

        # ---------------------------------------------------------------- language switching
        def relabel(lang: str) -> list[Any]:
            return [
                lang,
                f"# {APP_NAME}\n{t(lang, 'subtitle')}",
                gr.File(label=t(lang, "audio_file")),
                gr.Dropdown(label=t(lang, "language"), choices=language_choices(lang, SUPPORTED_LANGUAGES)),
                gr.Textbox(label=t(lang, "context"), placeholder=t(lang, "context_placeholder")),
                gr.UploadButton(label=t(lang, "load_context")),
                gr.Checkbox(label=t(lang, "timestamps"), info=t(lang, "timestamps_hint")),
                gr.Checkbox(label=t(lang, "diarize"), info=t(lang, "diarize_hint")),
                gr.Number(label=t(lang, "num_speakers")),
                gr.Number(label=t(lang, "min_speakers")),
                gr.Number(label=t(lang, "max_speakers")),
                gr.CheckboxGroup(label=t(lang, "formats")),
                gr.Button(value=t(lang, "transcribe")),
                gr.Button(value=t(lang, "stop")),
                gr.Accordion(label=t(lang, "settings")),
                gr.Tab(label=t(lang, "tab_output")),
                gr.Textbox(label=t(lang, "output_dir")),
                gr.Checkbox(label=t(lang, "subfolder")),
                gr.Checkbox(label=t(lang, "keep_audio")),
                gr.Textbox(label=t(lang, "hf_token"), placeholder=t(lang, "hf_token_placeholder")),
                gr.Tab(label=t(lang, "tab_model")),
                gr.Textbox(label=t(lang, "asr_model")),
                gr.Textbox(label=t(lang, "aligner_model")),
                gr.Dropdown(label=t(lang, "backend")),
                gr.Textbox(label=t(lang, "device")),
                gr.Dropdown(label=t(lang, "dtype")),
                gr.Code(label=t(lang, "backend_kwargs")),
                gr.Button(value=t(lang, "unload")),
                gr.Tab(label=t(lang, "tab_performance")),
                gr.Slider(label=t(lang, "gpu_mem")),
                gr.Number(label=t(lang, "batch")),
                gr.Number(label=t(lang, "max_tokens")),
                gr.Number(label=t(lang, "align_batch")),
                gr.Number(label=t(lang, "sample_rate")),
                gr.Number(label=t(lang, "channels")),
                gr.Checkbox(label=t(lang, "normalize")),
                gr.Tab(label=t(lang, "tab_optimize")),
                optimize_desc_md(lang),
                gr.Button(value=t(lang, "optimize_btn")),
                gr.Button(value=t(lang, "optimize_apply")),
                gr.Tab(label=t(lang, "tab_transcript")),
                gr.Textbox(placeholder=t(lang, "preview_empty")),
                gr.Tab(label=t(lang, "tab_files")),
                gr.File(label=t(lang, "download")),
                gr.Textbox(label=t(lang, "output_folder")),
                gr.Tab(label=t(lang, "tab_speakers")),
                t(lang, "speakers_empty"),
                gr.Dataframe(headers=[t(lang, "speaker_col"), t(lang, "name_col")]),
                gr.Button(value=t(lang, "apply_names")),
                t(lang, "optimize_confirm"),
                gr.Button(value=t(lang, "no")),
                gr.Button(value=t(lang, "yes")),
                optimized_html(lang),
            ]

        relabel_outputs = [
            lang_state, title, audio_in, language, context, context_btn, timestamps, diarize, num_speakers,
            min_speakers, max_speakers, formats, run_btn, stop_btn, acc_settings, tab_output, output_dir, subfolder,
            keep_audio, hf_token, tab_model, model, aligner_model, backend, device, dtype, backend_kwargs,
            unload_btn, tab_perf, gpu_mem, batch, max_tokens, align_batch, sample_rate, channels, normalize,
            tab_optimize, optimize_desc, optimize_btn, apply_btn, tab_transcript,
            preview, tab_files, downloads, output_folder, tab_speakers, speakers_hint, speaker_table, rename_btn, confirm_text, confirm_no, confirm_yes, optimized_box,
        ]

        def switch_language(lang: str, current_status: str, current_metrics: str):
            lang = lang if lang in UI_LANGUAGES else DEFAULT_LANG
            updates = relabel(lang)
            # Only reset the idle status/metrics; never overwrite a running job's display.
            idle = "scriba-status idle" in (current_status or "")
            updates += [status_html(t(lang, "ready")) if idle else gr.skip(),
                        metrics_html(lang) if idle else gr.skip()]
            if saved_lang is not None:
                updates.append(lang)
            return updates

        switch_outputs = relabel_outputs + [status, metrics] + ([saved_lang] if saved_lang is not None else [])
        lang_radio.change(switch_language, inputs=[lang_radio, status, metrics], outputs=switch_outputs, queue=False)
        if saved_lang is not None:
            demo.load(lambda s: s if s in UI_LANGUAGES else L, inputs=saved_lang, outputs=lang_radio, queue=False)

        # ---------------------------------------------------------------- input helpers
        audio_in.change(_file_info_markdown, inputs=[audio_in, lang_state], outputs=audio_info)
        diarize.change(lambda on: gr.Row(visible=on), inputs=diarize, outputs=speaker_opts)

        def load_context(file: Any, current: str) -> str:
            path = getattr(file, "name", file)
            if not path:
                return current
            from scriba.utils.paths import read_text_file

            return read_text_file(path)

        context_btn.upload(load_context, inputs=[context_btn, context], outputs=context)

        def build_settings(values: dict[str, Any]) -> ScribaSettings:
            values = {**values, "formats": values["formats"] or [], "output_dir": values["output_dir"] or str(a.output_root)}
            return TranscriptionForm(**values).to_settings(current())

        inputs = dict(
            output_dir=output_dir, subfolder=subfolder, language=language, context=context, timestamps=timestamps,
            diarize=diarize, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
            hf_token=hf_token, formats=formats, model=model, aligner_model=aligner_model, backend=backend,
            device=device, dtype=dtype, gpu_mem=gpu_mem, batch=batch, max_tokens=max_tokens, align_batch=align_batch, normalize=normalize,
            sample_rate=sample_rate, channels=channels, keep_audio=keep_audio, backend_kwargs=backend_kwargs,
        )
        input_names = list(inputs)
        output_names = ["status", "metrics", "preview", "downloads", "output_folder", "warnings",
                        "speakers_hint", "rename_group", "speaker_table", "job_dir", "progress"]
        outputs = [status, metrics, preview, downloads, output_folder, warnings_box, speakers_hint, rename_group,
                   speaker_table, job_dir_state, progress]

        # ---------------------------------------------------------------- transcription
        def run(audio_path: str | None, lang: str, *vals: Any):
            def emit(msg: str, state: str, **updates: Any):
                row: list[Any] = [status_html(msg, state)] + [gr.skip()] * (len(outputs) - 1)
                for name, value in updates.items():
                    row[output_names.index(name)] = value
                return row

            if not audio_path:
                yield emit(t(lang, "failed"), "error", warnings=alert_html(t(lang, "error_title"), t(lang, "upload_first")))
                return
            try:
                settings = build_settings(dict(zip(input_names, vals)))
            except Exception as exc:
                msg = exc.user_message() if isinstance(exc, ScribaError) else str(exc)
                yield emit(t(lang, "failed"), "error",
                           warnings=alert_html(t(lang, "error_title"), f"{t(lang, 'invalid_settings')} {msg}"))
                return

            events: queue.Queue = queue.Queue()
            box: dict[str, Any] = {}

            def worker() -> None:
                with lock:
                    try:
                        box["result"] = service.transcribe(audio_path, settings,
                                                           progress=lambda s, m: events.put((s, m)))
                    except Exception as exc:  # reported below
                        box["error"] = exc
                events.put(None)

            started = time.perf_counter()
            threading.Thread(target=worker, daemon=True).start()
            message = t(lang, "queued")
            current: JobStatus | None = None
            yield emit(message, "running", metrics=metrics_html(lang, "00:00:00"), preview="", downloads=None,
                       output_folder="", warnings="", rename_group=gr.Column(visible=False),
                       speakers_hint=gr.Markdown(visible=True), progress="")

            def tick_updates() -> dict[str, Any]:
                snap = service.progress.snapshot() if current == JobStatus.TRANSCRIBING and service.progress else None
                return {"metrics": metrics_html(lang, format_clock(time.perf_counter() - started)),
                        "progress": progress_html(snap, lang)}

            while True:
                try:
                    item = events.get(timeout=1.0)
                except queue.Empty:
                    yield emit(message, "running", **tick_updates())
                    continue
                if item is None:
                    break
                current, raw_message = item
                if current == JobStatus.FAILED:
                    continue
                message = t(lang, f"status_{current.value}") if isinstance(current, JobStatus) else raw_message
                yield emit(message, "running", **tick_updates())

            if "error" in box:
                exc = box["error"]
                msg = exc.user_message() if isinstance(exc, ScribaError) else f"{type(exc).__name__}: {exc}"
                yield emit(t(lang, "failed"), "error", progress="",
                           warnings=alert_html(t(lang, "error_title"), t(lang, "error_job"), msg,
                                               detail_label=t(lang, "error_details")),
                           metrics=metrics_html(lang, format_clock(time.perf_counter() - started)))
                return

            result = box["result"]
            transcript = result.transcript
            has_speakers = bool(transcript.speakers)
            yield emit(
                t(lang, "done") if transcript.completed else t(lang, "stopped").format(
                    processed=format_clock(transcript.processed_seconds or 0), duration=format_clock(transcript.duration)),
                "done" if transcript.completed else "error",
                progress="",
                metrics=metrics_html(lang, format_clock(result.processing_seconds), format_clock(result.audio.duration),
                                     f"{result.rtf:.3f}" if result.rtf else "—"),
                preview=render_preview(transcript),
                downloads=_download_copies(result.files, result.job_id),
                output_folder=str(result.job_dir),
                warnings="".join(alert_html(t(lang, w[5:]) if w.startswith("warn:") else w, "", kind="warn")
                                 for w in result.warnings),
                speakers_hint=gr.Markdown(visible=not has_speakers),
                rename_group=gr.Column(visible=has_speakers),
                speaker_table=[[s, ""] for s in transcript.speakers] if has_speakers else [],
                job_dir=str(result.job_dir),
            )

        run_btn.click(run, inputs=[audio_in, lang_state, *inputs.values()], outputs=outputs,
                      concurrency_limit=1, concurrency_id="gpu")

        def stop(lang: str):
            # Not on the GPU queue: it must run while a transcription holds it.
            return status_html(t(lang, "stopping"), "running") if service.request_cancel() else gr.skip()

        stop_btn.click(stop, inputs=lang_state, outputs=status, queue=False)

        def do_rename(job_dir: str | None, table: Any, lang: str):
            if not job_dir:
                return gr.skip(), gr.skip(), alert_html(t(lang, "error_title"), t(lang, "no_job"))
            rows = table.values.tolist() if hasattr(table, "values") else (table or [])
            names = {str(r[0]): str(r[1]) for r in rows if len(r) >= 2 and r[0]}
            try:
                files = apply_speaker_names(job_dir, names)
            except Exception as exc:
                return gr.skip(), gr.skip(), alert_html(t(lang, "error_title"), t(lang, "rename_failed"), str(exc),
                                                        detail_label=t(lang, "error_details"))
            from scriba.exporters import load_transcript

            transcript = load_transcript(Path(job_dir) / "transcript.json")
            return render_preview(transcript), _download_copies(files, transcript.id), ""

        rename_btn.click(do_rename, inputs=[job_dir_state, speaker_table, lang_state],
                         outputs=[preview, downloads, warnings_box], concurrency_limit=1, concurrency_id="gpu")

        def unload(lang: str) -> str:
            with lock:
                service.unload()
            return t(lang, "unloaded")

        unload_btn.click(unload, inputs=lang_state, outputs=model_state, concurrency_limit=1, concurrency_id="gpu")

        # ---------------------------------------------------------------- optimization
        def run_optimization(lang: str, *vals: Any):
            from scriba.core.optimizer import CANDIDATES, InferenceOptimizer, OptimizerProgress

            try:
                opt_settings = build_settings(dict(zip(input_names, vals)))
            except Exception as exc:
                msg = exc.user_message() if isinstance(exc, ScribaError) else str(exc)
                yield alert_html(t(lang, "error_title"), f"{t(lang, 'invalid_settings')} {msg}"), gr.skip(), gr.skip()
                return
            progress_state = OptimizerProgress(len(CANDIDATES))
            box: dict[str, Any] = {}

            def worker() -> None:
                with lock:
                    try:
                        box["result"] = InferenceOptimizer(service).run(opt_settings, write_env=False,
                                                                        progress=progress_state)
                    except Exception as exc:
                        box["error"] = exc
                box["finished"] = True

            threading.Thread(target=worker, daemon=True).start()
            yield "", gr.Button(visible=False), None
            while "finished" not in box:
                yield optimizer_html(progress_state.snapshot(), lang), gr.skip(), gr.skip()
                time.sleep(1.0)
            if "error" in box:
                exc = box["error"]
                msg = exc.user_message() if isinstance(exc, ScribaError) else f"{type(exc).__name__}: {exc}"
                yield alert_html(t(lang, "error_title"), t(lang, "optimize_failed"), msg,
                                 detail_label=t(lang, "error_details")), gr.skip(), gr.skip()
                return
            result = box["result"]
            yield optimizer_html(progress_state.snapshot(), lang, result.to_dict()), gr.Button(visible=True), result

        def ask_or_run():
            from scriba.core.optimizer import optimization_status

            already = optimization_status()["optimized"]
            return gr.Group(visible=already), not already

        def run_if_not_optimized(go: bool, lang: str, *vals: Any):
            if not go:  # already optimized: wait for the Yes/No answer
                yield gr.skip(), gr.skip(), gr.skip()
                return
            yield from run_optimization(lang, *vals)

        proceed = gr.State(False)
        optimize_btn.click(ask_or_run, outputs=[confirm_box, proceed], queue=False).then(
            run_if_not_optimized,
            inputs=[proceed, lang_state, *inputs.values()], outputs=[optimize_out, apply_btn, optimize_state],
            concurrency_limit=1, concurrency_id="gpu")
        confirm_no.click(lambda: gr.Group(visible=False), outputs=confirm_box, queue=False)
        confirm_yes.click(lambda: gr.Group(visible=False), outputs=confirm_box, queue=False).then(
            run_optimization, inputs=[lang_state, *inputs.values()], outputs=[optimize_out, apply_btn, optimize_state],
            concurrency_limit=1, concurrency_id="gpu")

        def apply_optimization(result, lang: str):
            from scriba.core.optimizer import apply_result

            if result is None:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()
            try:
                path = apply_result(result)
            except Exception as exc:
                return (alert_html(t(lang, "error_title"), t(lang, "optimize_failed"), str(exc)), gr.skip(), gr.skip(),
                        gr.skip())
            message = t(lang, "optimize_applied").format(path=path)
            done = f'<div class="scriba-status done"><span class="scriba-dot"></span><span>{html.escape(message)}</span></div>'
            return done, gr.Number(value=result.best_batch), gr.Button(visible=False), optimized_html(lang)

        apply_btn.click(apply_optimization, inputs=[optimize_state, lang_state],
                        outputs=[apply_out, batch, apply_btn, optimized_box])
        # New sessions pick up tuned values written to .env by the optimizer.
        demo.load(lambda: gr.Number(value=current().asr.max_inference_batch_size), outputs=batch, queue=False)

    return demo, service


def launch(settings: ScribaSettings | SettingsProvider, host: str = "127.0.0.1", port: int = 7860,
           preload: bool = False, lang: str = DEFAULT_LANG) -> None:
    import os

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    resolved = settings.get() if isinstance(settings, SettingsProvider) else settings
    setup_logging(resolved.app.log_level)
    try:
        import gradio  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Gradio is not installed: pip install gradio") from exc

    demo, service = build_app(settings, default_lang=lang)
    if preload:
        log.info("Preloading model...")
        service.engine.ensure_loaded(resolved.asr)

    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    log.info("%s UI on http://%s:%s", APP_NAME, shown, port)
    demo.queue(default_concurrency_limit=1, max_size=32).launch(
        server_name=host, server_port=port, share=False, inbrowser=False, theme=make_theme(), css=CSS,
    )
