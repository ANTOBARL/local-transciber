# Scriba

Local-first transcription for long recordings, built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
Audio, transcripts and metadata never leave your machine.

- Long-form audio and video in all common formats (WAV, MP3, M4A, FLAC, OGG, Opus, MP4, MKV, …)
- Word-level timestamps via Qwen3-ForcedAligner
- Speaker diarization with pyannote, with post-hoc speaker renaming
- Context prompting for names, acronyms and domain vocabulary
- Exports: JSON (canonical), TXT, Markdown, SRT, VTT
- CLI, Gradio UI and React web app, all backed by the same pipeline
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
| Diarization | A Hugging Face token with access to `pyannote/speaker-diarization-community-1` |

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

The web app is served at <http://localhost:8000>. Models are cached in the `scriba-models` volume and
transcripts are written to `./outputs`.

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
scriba transcribe a.mp3 b.mp4 --formats json,txt,srt --num-speakers 3
scriba transcribe talk.wav --no-diarize --set asr.max_inference_batch_size=64
scriba rename outputs/20260915_142301_a83f91 SPEAKER_00=Luca SPEAKER_01=Clara
scriba benchmark long_recording.m4a
scriba doctor | scriba info | scriba models
```

Any setting can be overridden with `--set section.key=value`.

## Diarization

Diarization is enabled by default. If pyannote or the token is missing, the transcript is still produced
and a warning is recorded.

1. Accept the model conditions at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Add it to `.env` in the project root:

```dotenv
SCRIBA_DIARIZATION__HF_TOKEN=hf_...
```

The token is only used to download the model; it is redacted from logs and job metadata.

## Configuration

Defaults live in [`src/scriba/default.yaml`](src/scriba/default.yaml). Precedence, lowest to highest:

1. Built-in defaults
2. `--config file.yaml`
3. Environment variables (`SCRIBA_<SECTION>__<KEY>`, e.g. `SCRIBA_ASR__BACKEND=vllm`)
4. CLI flags, `--set`, or UI fields

| Key | Default | Notes |
|---|---|---|
| `asr.model` | `Qwen/Qwen3-ASR-1.7B` | Hub id or local path |
| `asr.backend` | `auto` | `vllm` when available, otherwise `transformers` |
| `asr.device` / `asr.dtype` | `auto` | `cuda:0` + `bfloat16` on capable GPUs |
| `asr.language` | `Italian` | `auto` for detection |
| `asr.max_inference_batch_size` | `32` | Main speed/memory trade-off |
| `asr.gpu_memory_utilization` | `0.70` | vLLM only |
| `asr.forced_aligner.enabled` | `true` | Required for timestamps, SRT and VTT |
| `diarization.enabled` | `true` | |
| `app.output_root` | `./outputs` | |

## Output

Each job gets its own directory:

```text
outputs/20260915_142301_a83f91/
├── job.json            status, timings, real-time factor, device
├── config.json         effective configuration (secrets redacted)
├── source.json         input metadata and SHA-256
├── processing.log
├── transcript.json     canonical transcript
├── transcript.{txt,md,srt,vtt}
└── diarization.json, speakers.json   when diarization runs
```

All human-readable formats are rendered from `transcript.json`; renaming speakers regenerates them
without re-running ASR.

## Architecture

```text
CLI · Gradio UI · React app (FastAPI)
              │
      TranscriptionService          one model instance, one job at a time
              │
  preprocess → Qwen3-ASR + ForcedAligner → pyannote (optional) → exporters
```

The internal data model (`scriba.models`) is independent of qwen-asr, vLLM and pyannote, so upstream
changes do not affect the transcript format or the exporters. Progress is reported per processed
batch without adding work to the inference path.

## Development

```bash
pytest
```

Tests use a fake engine and require neither a GPU nor model downloads.

## License

[MIT](LICENSE)
