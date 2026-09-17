export type Form = {
  language: string;
  context: string;
  glossary: string;
  timestamps: boolean;
  diarize: boolean;
  num_speakers: number;
  min_speakers: number;
  max_speakers: number;
  formats: string[];
  output_dir: string;
  subfolder: boolean;
  keep_audio: boolean;
  hf_token: string;
  model: string;
  aligner_model: string;
  backend: string;
  device: string;
  dtype: string;
  backend_kwargs: string;
  gpu_mem: number;
  batch: number;
  align_batch: number;
  max_tokens: number;
  chunk_seconds: number | null;
  sample_rate: number;
  channels: number;
  normalize: boolean;
};

export type OptimizationStatus = {
  optimized: boolean;
  batch?: number;
  applied_at?: string | null;
  device?: string | null;
  speed?: number;
  speed_source?: "real" | "benchmark";
  real_jobs?: number;
  peak_vram_mb?: number;
  minutes_per_audio_hour?: number;
  speedup_vs_smallest?: number;
  gpu_total_mb?: number | null;
};

export type Config = {
  app: string;
  version: string;
  default_lang: string;
  defaults: Form;
  languages: string[];
  export_formats: string[];
  dtypes: string[];
  backends: string[];
  media_extensions: string[];
  optimization: OptimizationStatus;
};

export type I18n = {
  ui_languages: Record<string, string>;
  texts: Record<string, Record<string, string>>;
  language_names: Record<string, Record<string, string>>;
};

export type AudioInfo = {
  filename: string;
  size_bytes: number;
  duration: number;
  codec: string | null;
  sample_rate: number | null;
  channels: number | null;
};

export type Progress = {
  phase: "asr" | "align";
  fraction: number;
  percent: number;
  eta_seconds: number | null;
  elapsed_seconds: number;
  asr: [number, number];
  align: [number, number];
  estimated_from_history: boolean;
};

export type Trial = {
  batch: number;
  status: "ok" | "too_slow_gain" | "exceeds_vram" | "oom" | "skipped" | "error";
  speed: number | null;
  peak_vram_mb: number | null;
  note: string | null;
};

export type OptimizerSnapshot = {
  stage: "loading" | "warmup" | "trial" | "saving" | "done";
  current_batch: number | null;
  done: number;
  total: number;
  fraction: number;
  elapsed_seconds: number;
  trials: Trial[];
};

export type Optimization = {
  best_batch: number;
  previous_batch: number;
  trials: Trial[];
  device: string;
  env_path: string | null;
};

export type QualityIssue = {
  kind: "repetition" | "sparse";
  action: "redecoded" | "collapsed" | "kept";
  start: number;
  end: number;
  words_removed: number;
};

export type TaskStatus = {
  id: string;
  kind?: "transcribe" | "optimize";
  optimizer?: OptimizerSnapshot | null;
  optimization?: Optimization;
  defaults?: Form;
  optimization_status?: OptimizationStatus;
  state: "queued" | "running" | "done" | "error" | "cancelled";
  status: string | null;
  error: string | null;
  queue_position: number;
  elapsed: number;
  progress: Progress | null;
  cancel_requested?: boolean;
  completed?: boolean;
  processed_seconds?: number | null;
  audio_duration?: number;
  rtf?: number | null;
  preview?: string;
  files?: { format: string; name: string; url: string }[];
  output_folder?: string;
  warnings?: string[];
  quality?: QualityIssue[];
  speakers?: string[];
  speaker_names?: Record<string, string>;
};

/** Error raised by API calls. `status` is 0 when the server could not be reached at all. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
  get unreachable() {
    return this.status === 0;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    throw new ApiError((e as Error).message || "Network error", 0); // e.g. "Failed to fetch"
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    // 502/503/504 come from proxies when the backend is down: treat as unreachable.
    throw new ApiError(detail, [502, 503, 504].includes(res.status) ? 0 : res.status);
  }
  return res.json() as Promise<T>;
}

const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  config: () => request<Config>("/api/config"),
  i18n: () => request<I18n>("/api/i18n"),
  task: (id: string) => request<TaskStatus>(`/api/tasks/${id}`),
  start: (upload_id: string, form: Form) => request<{ task_id: string }>("/api/tasks", post({ upload_id, form })),
  cancel: (id: string) => request<TaskStatus>(`/api/tasks/${id}/cancel`, { method: "POST" }),
  applyOptimization: (id: string) => request<TaskStatus>(`/api/tasks/${id}/apply`, { method: "POST" }),
  optimize: (form: Form) => request<{ task_id: string }>("/api/optimize", post({ form })),
  rename: (id: string, names: Record<string, string>) => request<TaskStatus>(`/api/tasks/${id}/speakers`, post({ names })),
  unload: () => request<{ model_loaded: boolean }>("/api/model/unload", { method: "POST" }),

  upload(file: File, onProgress: (pct: number) => void): Promise<{ upload_id: string; info: AudioInfo }> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const data = new FormData();
      data.append("file", file);
      xhr.open("POST", "/api/uploads");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
      xhr.onload = () => {
        let body: { detail?: string } | null = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* non-JSON */
        }
        if (xhr.status >= 200 && xhr.status < 300 && body) resolve(body as { upload_id: string; info: AudioInfo });
        else reject(new ApiError(body?.detail || `HTTP ${xhr.status} ${xhr.statusText}`, xhr.status));
      };
      xhr.onerror = () => reject(new ApiError("Network error", 0));
      xhr.send(data);
    });
  },
};

export function clock(seconds?: number | null): string {
  if (seconds == null || Number.isNaN(seconds)) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}
