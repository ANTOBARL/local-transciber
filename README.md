# Scriba

Local-first transcription for long recordings, built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
Audio, transcripts and metadata never leave your machine.

- Long-form audio and video in all common formats (WAV, MP3, M4A, FLAC, OGG, Opus, MP4, MKV, …)
- Windowed processing with on-disk checkpoints: bounded GPU memory, stop at any time, resume after a restart
- Word-level timestamps via Qwen3-ForcedAligner
- Speaker diarization with pyannote (online or fully offline), with post-hoc speaker renaming
- Context prompting for names, acronyms and domain vocabulary
- Exports: JSON (canonical), TXT, Markdown, SRT, VTT, Word (.docx)
- Built-in inference optimizer that benchmarks the GPU and tunes batch sizes
- CLI, Gradio UI and React web app (with a multi-file queue), all backed by the same pipeline
- Bilingual interface (Italian / English)
- Hardware-agnostic: device, precision and backend are resolved at runtime
- Runs in a Python environment or as a GPU Docker container

## Requirements

| | |
|---|---|
| Python | 3.10+ (3.12 recommended) |
| GPU | NVIDIA with CUDA recommended; CPU works but is slow |
| FFmpeg | Optional — `imageio-ffmpeg` or the built-in PyAV decoder are used as fallback |
| vLLM backend | Linux or WSL2 only; `transformers` is used elsewhere |
| Diarization | Access to `pyannote/speaker-diarization-community-1` (token or local copy) |

## Installation

### Python environment

Install a CUDA build of PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -e ".[all]"             # qwen-asr, Gradio, FFmpeg binary, psutil, pytest
pip install -e ".[diarization]"     # optional: pyannote.audio
pip install -e ".[vllm]"            # optional: Linux/WSL2 only
scriba doctor
```

> **Windows:** install into the environment's own `site-packages`, not with `pip --user`, when the
> user profile path contains non-ASCII characters. Scriba includes a workaround for the DyNet
> path issue, but a clean environment avoids package shadowing.

### Docker

Requires an NVIDIA GPU exposed to Docker (Docker Desktop with WSL2, or the NVIDIA Container Toolkit).

```bash
docker compose up -d --build
```

The web app is served at <http://localhost:8000>.

| Host path / volume | Container | Content |
|---|---|---|
| `./outputs` | `/data/outputs` | Transcripts, one folder per job |
| `./config` | `/data/config` | `.env` (token, tuned parameters), local models, speed history |
| `scriba-models` | `/models` | Hugging Face cache (Qwen models) |

| Build arg | Default | Description |
|---|---|---|
| `INSTALL_DIARIZATION` | `true` | Install pyannote.audio |
| `INSTALL_VLLM` | `false` | Install the vLLM backend |
| `TORCH_INDEX` | `.../whl/cu128` | PyTorch wheel index (CUDA version) |

## Usage

### Web interfaces

```bash
scriba ui       # Gradio  — http://localhost:7860
scriba serve    # React   — http://localhost:8000 (API docs at /api/docs)
```

The React app accepts several files at once and processes them one after another; select a file in the
queue to see its progress and results. A running transcription can be stopped: what was processed so far
is exported with a clear interruption note.

`scriba serve` uses the built React app in `frontend/dist`. To develop the frontend:

```bash
cd frontend
npm install
npm run dev     # proxies /api to scriba serve on port 8000
```

### CLI

```bash
scriba transcribe meeting.m4a
scriba transcribe meeting.m4a --language Italian --context-file glossary.txt --output D:/Transcripts
scriba transcribe a.mp3 b.mp4 --formats json,txt,srt,docx --num-speakers 3
scriba transcribe talk.wav --no-diarize --set asr.max_inference_batch_size=16
scriba rename outputs/meeting_20260915_142301 SPEAKER_00=Luca SPEAKER_01=Clara
scriba optimize                     # benchmark and save the best batch size
scriba benchmark long_recording.m4a
scriba doctor | scriba info | scriba models
```

Any setting can be overridden with `--set section.key=value`. `Ctrl+C` during `transcribe` stops after the
current window and exports the partial transcript; a second `Ctrl+C` exits immediately.

## Long recordings

Audio is split at low-energy points into chunks of at most `asr.chunk_seconds` (30 s by default). Longer
chunks (up to the aligner's 180 s limit) are allowed, but on real recordings they make Qwen3-ASR silently drop
speech or loop on a phrase; on a 3 h 42 min seminar, 30 s chunks recovered about 27% more text in half the
time. Chunks are processed in windows: ASR → alignment → saved to `segments/` → memory released. As a result:

- peak VRAM depends on the window size, not on the recording length;
- alignment uses its own, smaller batch (`asr.align_batch_size`, automatic by default);
- on a CUDA out-of-memory error the batch is halved and only that window is retried;
- running the same file again with the same settings resumes from the first unsaved window.

### Quality checks

Every chunk is checked after ASR:

- **repetition loops** (the same phrase repeated many times) and **dropped speech** (far fewer words per
  second than the rest of the recording) trigger a re-decode of that chunk in shorter pieces;
- pieces that still loop keep one occurrence of the phrase;
- pieces where the model recites the context prompt are transcribed again without it;
- **background chatter** (the model is unsure, `asr.min_confidence`, and hears no language when asked) produces
  no text: the exports show a `[Brusio di fondo – nessun parlato comprensibile]` section with its time range;
- the affected time ranges are listed as *Parts to check* in the web apps, in `job.json` (`quality_issues`)
  and in `transcript.json` (`metadata.quality_issues`).

## Optimization

Settings → Optimization (or `scriba optimize`) runs a bundled Italian benchmark at increasing batch sizes,
measures speed and peak VRAM, and proposes the fastest setting that safely fits in free memory. Applying it
writes a managed block in the `.env` file, which running servers reload automatically. Speed estimates are
replaced by the average measured on real transcriptions as soon as one completes.

## Diarization

Diarization is enabled by default. Without access to the model it is skipped with a warning and the
transcript is still exported.

1. Accept the model conditions at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Either add a read token to `.env` (`config/.env` for Docker):

   ```dotenv
   SCRIBA_DIARIZATION__HF_TOKEN=hf_...
   ```

3. Or download the model once for offline use; Scriba picks it up automatically, no token needed:

   ```bash
   HF_TOKEN=hf_... python -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/speaker-diarization-community-1', local_dir='config/models/pyannote--speaker-diarization-community-1')"
   ```

Tokens are redacted from logs and job metadata. Never put a real token in `.env.example`.

## Configuration

Defaults live in [`src/scriba/default.yaml`](src/scriba/default.yaml). Precedence, lowest to highest:

1. Built-in defaults
2. `--config file.yaml`
3. `.env` file (project root, or `SCRIBA_ENV_FILE`) — reloaded on change
4. Environment variables (`SCRIBA_<SECTION>__<KEY>`, e.g. `SCRIBA_ASR__BACKEND=vllm`)
5. CLI flags, `--set`, or UI fields

| Key | Default | Notes |
|---|---|---|
| `asr.model` | `Qwen/Qwen3-ASR-1.7B` | Hub id or local path |
| `asr.backend` | `auto` | `vllm` when available, otherwise `transformers` |
| `asr.device` / `asr.dtype` | `auto` | `cuda:0` + `bfloat16` on capable GPUs |
| `asr.language` | `Italian` | `auto` for detection |
| `asr.max_inference_batch_size` | `32` | Chunks per window; tune with the optimizer |
| `asr.align_batch_size` | `0` | `0` = half of the ASR batch |
| `asr.chunk_seconds` | `30` | Audio piece length sent to the model (5–180) |
| `asr.min_confidence` | `-0.25` | Mean token log-probability below which a piece may be chatter (`null` = off) |
| `asr.glossary` | `""` | Proper names, one per line, fixed after transcription (`--glossary-file`) |
| `asr.gpu_memory_utilization` | `0.70` | vLLM only |
| `asr.forced_aligner.enabled` | `true` | Required for timestamps, SRT and VTT |
| `diarization.enabled` | `true` | |
| `diarization.min_turn_seconds` | `0.5` | Shorter speaker turns are ignored |
| `diarization.min_speaker_run_seconds` | `1.0` | Shorter speaker changes inside a sentence are undone |
| `diarization.sentence_majority` | `0.7` | A sentence goes to the speaker holding this share of its words (`0` = off) |
| `export.docx` | `false` | Word export |
| `app.output_root` | `./outputs` | |

## Output

Each job gets its own directory named after the audio file:

```text
outputs/meeting_20260915_142301/
├── job.json            status, timings, real-time factor, device
├── config.json         effective configuration (secrets redacted)
├── source.json         input metadata and SHA-256
├── processing.log
├── transcript.json     canonical transcript
├── transcript.{txt,md,srt,vtt,docx}
├── diarization.json, speakers.json   when diarization runs
└── segments/           window checkpoints, removed when the job completes
```

All human-readable formats are rendered from `transcript.json`; renaming speakers regenerates them
without re-running ASR. A stopped job has `completed: false` in the JSON and an interruption note in
every text format.

## Architecture

```text
CLI · Gradio UI · React app (FastAPI, FIFO job queue)
              │
      TranscriptionService          one model instance, one job at a time
              │
  preprocess → windows [Qwen3-ASR → ForcedAligner → checkpoint] → pyannote (optional) → exporters
```

The internal data model (`scriba.models`) is independent of qwen-asr, vLLM and pyannote, so upstream
changes do not affect the transcript format or the exporters. Progress and ETA are computed from measured
work per window, without adding work to the inference path.

## Development

```bash
pytest
```

Tests use fake engines and require neither a GPU nor model downloads.

## License

[MIT](LICENSE)
