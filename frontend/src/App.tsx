import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  AudioLines,
  CheckCircle2,
  ChevronDown,
  Clock,
  Download,
  FileAudio,
  FileText,
  FolderOpen,
  Gauge,
  Hourglass,
  Languages,
  Loader2,
  PowerOff,
  Settings2,
  Timer,
  Upload,
  UserPen,
  Users,
  WifiOff,
  X,
  XCircle,
  type LucideIcon,
} from "lucide-react";
import { ApiError, api, clock, type AudioInfo, type Config, type Form, type I18n, type TaskStatus } from "./api";
import { ProgressBar } from "./progress";

const EXPORT_LABELS: Record<string, string> = { json: "JSON", txt: "TXT", markdown: "Markdown", srt: "SRT", vtt: "VTT" };
const LANG_KEY = "scriba_ui_lang";

type SettingsTab = "output" | "model" | "performance";
type ResultTab = "transcript" | "files" | "speakers";
type Translate = (key: string) => string;

/** Everything that went wrong, shown in a prominent banner. */
type UiError = { kind: "error" | "connection"; message: string; detail?: string; hint?: string };

function readLang(fallback: string): string {
  try {
    return localStorage.getItem(LANG_KEY) || fallback;
  } catch {
    return fallback;
  }
}

// ======================================================================== root

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [i18n, setI18n] = useState<I18n | null>(null);
  const [lang, setLang] = useState("it");
  const [form, setForm] = useState<Form | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const boot = useCallback(() => {
    setBootError(null);
    Promise.all([api.config(), api.i18n()])
      .then(([cfg, tr]) => {
        setConfig(cfg);
        setI18n(tr);
        setForm((f) => f ?? cfg.defaults);
        setLang(readLang(cfg.default_lang));
      })
      .catch((e) => setBootError((e as Error).message));
  }, []);
  useEffect(boot, [boot]);

  const t = useCallback<Translate>(
    (key) => i18n?.texts[lang]?.[key] ?? i18n?.texts.en?.[key] ?? key,
    [i18n, lang],
  );

  useEffect(() => {
    document.documentElement.lang = lang;
    try {
      localStorage.setItem(LANG_KEY, lang);
    } catch {
      /* private mode */
    }
  }, [lang]);

  if (bootError) {
    // Translations are not available yet: show both languages.
    return (
      <div className="boot">
        <div className="alert error" role="alert">
          <XCircle className="alert-icon" />
          <div className="alert-body">
            <strong>Qualcosa è andato storto · Something went wrong</strong>
            <p>Impossibile contattare il server di Scriba. · Cannot reach the Scriba server.</p>
            <code>{bootError}</code>
          </div>
          <button className="btn secondary sm" onClick={boot}>
            Riprova · Retry
          </button>
        </div>
      </div>
    );
  }
  if (!config || !i18n || !form)
    return (
      <div className="boot">
        <Loader2 className="spin" />
      </div>
    );

  return <Workspace config={config} i18n={i18n} lang={lang} setLang={setLang} form={form} setForm={setForm} t={t} />;
}

// ======================================================================== workspace

type WorkspaceProps = {
  config: Config;
  i18n: I18n;
  lang: string;
  setLang: (l: string) => void;
  form: Form;
  setForm: React.Dispatch<React.SetStateAction<Form | null>>;
  t: Translate;
};

function Workspace({ config, i18n, lang, setLang, form, setForm, t }: WorkspaceProps) {
  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => (f ? { ...f, [key]: value } : f));
  const [error, setError] = useState<UiError | null>(null);

  const reportError = useCallback(
    (context: string, e: unknown) => {
      const err = e as Error;
      if (err instanceof ApiError && err.unreachable) {
        setError({ kind: "error", message: `${t(context)} ${t("error_server")}`, hint: t("error_connection_hint"), detail: err.message });
      } else {
        setError({ kind: "error", message: t(context), detail: err?.message ?? String(e) });
      }
    },
    [t],
  );

  // ------------------------------------------------------------------ upload
  const [upload, setUpload] = useState<{ id: string; info: AudioInfo } | null>(null);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const contextInput = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    setUpload(null);
    setError(null);
    setUploadPct(0);
    try {
      const res = await api.upload(file, setUploadPct);
      setUpload({ id: res.upload_id, info: res.info });
    } catch (e) {
      reportError("error_upload", e);
    } finally {
      setUploadPct(null);
    }
  };

  const loadContext = async (file: File | undefined) => {
    if (file) set("context", (await file.text()).trim());
  };

  // ------------------------------------------------------------------ task polling (resilient)
  const [taskId, setTaskId] = useState<string | null>(null);
  const [task, setTask] = useState<TaskStatus | null>(null);
  const [connectionLost, setConnectionLost] = useState(false);
  const [tick, setTick] = useState(0);
  const running = task?.state === "queued" || task?.state === "running";

  useEffect(() => {
    if (!taskId) return;
    let stop = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const s = await api.task(taskId);
        if (stop) return;
        failures = 0;
        setConnectionLost(false);
        setTask(s);
        if (s.state === "error") setError({ kind: "error", message: t("error_job"), detail: s.error ?? undefined });
        if (s.state === "done" || s.state === "error") return;
      } catch (e) {
        if (stop) return;
        const err = e as ApiError;
        if (err instanceof ApiError && err.status === 404) {
          // The server answers but no longer knows the task: it was restarted.
          setConnectionLost(false);
          setTask((prev) => (prev ? { ...prev, state: "error", error: t("error_task_lost"), progress: null } : prev));
          setError({ kind: "error", message: t("error_task_lost") });
          return;
        }
        failures += 1;
        setConnectionLost(true);
      }
      // Keep retrying while disconnected, backing off up to 5 s.
      timer = setTimeout(poll, failures ? Math.min(5000, 1000 * (1 + failures)) : 1000);
    };
    poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [taskId, t]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  const startedAt = useRef<number>(0);
  const transcribe = async () => {
    if (!upload) {
      setError({ kind: "error", message: t("upload_first") });
      return;
    }
    setError(null);
    setTask(null);
    setResultTab("transcript");
    try {
      startedAt.current = Date.now();
      const { task_id } = await api.start(upload.id, form);
      setTaskId(task_id);
    } catch (e) {
      reportError("error_start", e);
    }
  };

  const status = useMemo(() => {
    if (connectionLost) return { text: t("error_connection"), state: "warn" as const };
    if (!task) return { text: t("ready"), state: "idle" as const };
    if (task.state === "queued")
      return { text: `${t("queued")}${task.queue_position > 1 ? ` (#${task.queue_position})` : ""}`, state: "running" as const };
    if (task.state === "running") return { text: t(`status_${task.status ?? "preprocessing"}`), state: "running" as const };
    if (task.state === "error") return { text: t("failed"), state: "error" as const };
    return { text: t("done"), state: "done" as const };
  }, [connectionLost, task, t]);

  void tick; // re-render every second while running
  const liveElapsed = !task
    ? undefined
    : task.state === "running"
      ? Math.max(task.elapsed, (Date.now() - startedAt.current) / 1000)
      : task.elapsed;

  // ------------------------------------------------------------------ results
  const [resultTab, setResultTab] = useState<ResultTab>("transcript");
  const [names, setNames] = useState<Record<string, string>>({});
  const [renameMsg, setRenameMsg] = useState<string | null>(null);
  useEffect(() => {
    if (task?.state === "done") setNames(task.speaker_names ?? {});
  }, [task?.state, task?.speaker_names]);

  const applyNames = async () => {
    if (!taskId) return;
    setRenameMsg(null);
    try {
      setTask(await api.rename(taskId, names));
      setRenameMsg(t("names_applied"));
    } catch (e) {
      reportError("rename_failed", e);
    }
  };

  // ------------------------------------------------------------------ settings
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("output");
  const [unloadMsg, setUnloadMsg] = useState<string | null>(null);
  const languageNames = i18n.language_names[lang] ?? {};

  const numberField = (key: keyof Form, label: string, step = 1, min?: number, max?: number) => (
    <label className="field">
      <span className="label">{label}</span>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={form[key] as number}
        onChange={(e) => set(key, (step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value || "0", 10)) as never)}
      />
    </label>
  );
  const textField = (key: keyof Form, label: string, extra: React.InputHTMLAttributes<HTMLInputElement> = {}) => (
    <label className="field">
      <span className="label">{label}</span>
      <input type="text" value={form[key] as string} onChange={(e) => set(key, e.target.value as never)} {...extra} />
    </label>
  );
  const check = (key: keyof Form, label: string) => (
    <label className="check">
      <input type="checkbox" checked={form[key] as boolean} onChange={(e) => set(key, e.target.checked as never)} />
      <span>{label}</span>
    </label>
  );

  const info = upload?.info;

  return (
    <div className="app">
      {/* ------------------------------------------------------------ top bar */}
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <AudioLines />
          </span>
          <div className="title">
            <h1>{config.app}</h1>
            <p>
              {t("subtitle")} · v{config.version}
            </p>
          </div>
        </div>
        <div className="lang-switch" role="radiogroup" aria-label="Language">
          <Languages className="lang-icon" aria-hidden />
          {Object.entries(i18n.ui_languages).map(([code, label]) => (
            <button
              key={code}
              role="radio"
              aria-checked={lang === code}
              className={lang === code ? "selected" : ""}
              onClick={() => setLang(code)}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      {connectionLost && (
        <Alert kind="connection" icon={WifiOff} title={t("error_connection")} message={t("error_connection_hint")} spinner />
      )}
      {error && (
        <Alert
          kind="error"
          icon={XCircle}
          title={t("error_title")}
          message={error.message}
          hint={error.hint}
          detail={error.detail}
          detailLabel={t("error_details")}
          onDismiss={() => setError(null)}
          dismissLabel={t("error_dismiss")}
        />
      )}

      <main className="columns">
        {/* ---------------------------------------------------------- left column */}
        <section className="col col-left">
          <div
            className={`card dropzone ${dragOver ? "over" : ""}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFile(e.dataTransfer.files[0]);
            }}
          >
            <input
              ref={fileInput}
              type="file"
              hidden
              accept={config.media_extensions.join(",")}
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <div className="dropzone-body">
              {uploadPct !== null ? (
                <>
                  <Loader2 className="spin muted" />
                  <div className="progress">
                    <div style={{ width: `${uploadPct}%` }} />
                  </div>
                  <span className="muted">{uploadPct}%</span>
                </>
              ) : info ? (
                <div className="file-info">
                  <FileAudio className="file-icon" />
                  <div>
                    <strong>{info.filename}</strong>
                    <span>
                      {(info.size_bytes / 1024 ** 2).toFixed(1)} MB · {clock(info.duration)} · {info.codec ?? "?"} ·{" "}
                      {info.sample_rate ?? "?"} Hz · {info.channels ?? "?"} ch
                    </span>
                  </div>
                </div>
              ) : (
                <>
                  <Upload className="drop-icon" />
                  <span className="drop-title">{t("audio_file")}</span>
                  <span className="muted small">{config.media_extensions.slice(0, 8).join("  ")} …</span>
                </>
              )}
            </div>
          </div>

          <div className="card stack">
            <label className="field">
              <span className="label">{t("language")}</span>
              <select value={form.language} onChange={(e) => set("language", e.target.value)}>
                <option value="auto">{t("auto_detect")}</option>
                {config.languages.map((l) => (
                  <option key={l} value={l}>
                    {languageNames[l] ?? l}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="label">{t("context")}</span>
              <textarea
                rows={4}
                value={form.context}
                placeholder={t("context_placeholder")}
                onChange={(e) => set("context", e.target.value)}
              />
            </label>
            <div className="row end">
              <input ref={contextInput} type="file" hidden accept=".txt,.md" onChange={(e) => loadContext(e.target.files?.[0])} />
              <button className="btn secondary sm" onClick={() => contextInput.current?.click()}>
                <FileText size={15} /> {t("load_context")}
              </button>
            </div>

            <div className="toggles">
              <ToggleCard
                icon={Clock}
                title={t("timestamps")}
                hint={t("timestamps_hint")}
                checked={form.timestamps}
                onChange={(v) => set("timestamps", v)}
              />
              <ToggleCard
                icon={Users}
                title={t("diarize")}
                hint={t("diarize_hint")}
                checked={form.diarize}
                onChange={(v) => set("diarize", v)}
              />
            </div>
            {form.diarize && (
              <div className="row three">
                {numberField("num_speakers", t("num_speakers"), 1, 0)}
                {numberField("min_speakers", t("min_speakers"), 1, 0)}
                {numberField("max_speakers", t("max_speakers"), 1, 0)}
              </div>
            )}

            <div className="field">
              <span className="label">{t("formats")}</span>
              <div className="chips">
                {config.export_formats.map((fmt) => (
                  <label key={fmt} className={`chip ${form.formats.includes(fmt) ? "on" : ""}`}>
                    <input
                      type="checkbox"
                      checked={form.formats.includes(fmt)}
                      onChange={(e) =>
                        set("formats", e.target.checked ? [...form.formats, fmt] : form.formats.filter((f) => f !== fmt))
                      }
                    />
                    {EXPORT_LABELS[fmt] ?? fmt}
                  </label>
                ))}
              </div>
            </div>
          </div>

          <button className="btn primary lg" onClick={transcribe} disabled={running || uploadPct !== null}>
            {running ? <Loader2 className="spin" size={20} /> : <AudioLines size={20} />}
            {t("transcribe")}
          </button>

          <div className={`card accordion ${settingsOpen ? "open" : ""}`}>
            <button className="accordion-head" onClick={() => setSettingsOpen((o) => !o)} aria-expanded={settingsOpen}>
              <span className="with-icon">
                <Settings2 size={17} /> {t("settings")}
              </span>
              <ChevronDown className="caret" size={18} />
            </button>
            {settingsOpen && (
              <div className="accordion-body">
                <div className="tabs">
                  {(["output", "model", "performance"] as SettingsTab[]).map((tab) => (
                    <button key={tab} className={settingsTab === tab ? "active" : ""} onClick={() => setSettingsTab(tab)}>
                      {t(`tab_${tab}`)}
                    </button>
                  ))}
                </div>
                {settingsTab === "output" && (
                  <div className="stack">
                    {textField("output_dir", t("output_dir"))}
                    {check("subfolder", t("subfolder"))}
                    {check("keep_audio", t("keep_audio"))}
                    {textField("hf_token", t("hf_token"), { type: "password", placeholder: t("hf_token_placeholder") })}
                  </div>
                )}
                {settingsTab === "model" && (
                  <div className="stack">
                    {textField("model", t("asr_model"))}
                    {textField("aligner_model", t("aligner_model"))}
                    <div className="row three">
                      <label className="field">
                        <span className="label">{t("backend")}</span>
                        <select value={form.backend} onChange={(e) => set("backend", e.target.value)}>
                          {config.backends.map((b) => (
                            <option key={b}>{b}</option>
                          ))}
                        </select>
                      </label>
                      {textField("device", t("device"))}
                      <label className="field">
                        <span className="label">{t("dtype")}</span>
                        <select value={form.dtype} onChange={(e) => set("dtype", e.target.value)}>
                          {config.dtypes.map((d) => (
                            <option key={d}>{d}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <label className="field">
                      <span className="label">{t("backend_kwargs")}</span>
                      <textarea
                        className="mono"
                        rows={3}
                        value={form.backend_kwargs}
                        onChange={(e) => set("backend_kwargs", e.target.value)}
                      />
                    </label>
                    <div className="row">
                      <button
                        className="btn secondary sm"
                        onClick={async () => {
                          try {
                            await api.unload();
                            setUnloadMsg(t("unloaded"));
                          } catch (e) {
                            reportError("failed", e);
                          }
                        }}
                      >
                        <PowerOff size={15} /> {t("unload")}
                      </button>
                      {unloadMsg && <span className="muted">{unloadMsg}</span>}
                    </div>
                  </div>
                )}
                {settingsTab === "performance" && (
                  <div className="stack">
                    <label className="field">
                      <span className="label">
                        {t("gpu_mem")} <b>{form.gpu_mem.toFixed(2)}</b>
                      </span>
                      <input
                        type="range"
                        min={0.1}
                        max={0.98}
                        step={0.01}
                        value={form.gpu_mem}
                        onChange={(e) => set("gpu_mem", parseFloat(e.target.value))}
                      />
                    </label>
                    <div className="row two">
                      {numberField("batch", t("batch"), 1, -1)}
                      {numberField("max_tokens", t("max_tokens"), 1, 1)}
                    </div>
                    <div className="row two">
                      {numberField("sample_rate", t("sample_rate"), 1, 8000)}
                      {numberField("channels", t("channels"), 1, 1, 2)}
                    </div>
                    {check("normalize", t("normalize"))}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* ---------------------------------------------------------- right column */}
        <section className="col col-right">
          <div className="card status-card">
            <div className={`status ${status.state}`}>
              <StatusIcon state={status.state} />
              <span>{status.text}</span>
            </div>
            {task?.state === "running" && task.progress && <ProgressBar progress={task.progress} lang={lang} t={t} />}
            <div className="metrics">
              <Metric icon={Timer} value={clock(liveElapsed)} label={t("elapsed")} />
              <Metric icon={Hourglass} value={clock(task?.audio_duration ?? info?.duration)} label={t("audio_duration")} />
              <Metric icon={Gauge} value={task?.rtf != null ? task.rtf.toFixed(3) : "—"} label={t("rtf")} />
            </div>
          </div>

          {task?.warnings?.length ? (
            <div className="alert warn compact">
              <AlertTriangle className="alert-icon" />
              <div className="alert-body">
                {task.warnings.map((w) => (
                  <div key={w}>{w}</div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="tabs results-tabs">
            {(
              [
                ["transcript", FileText],
                ["files", Download],
                ["speakers", Users],
              ] as [ResultTab, LucideIcon][]
            ).map(([tab, Icon]) => (
              <button key={tab} className={resultTab === tab ? "active" : ""} onClick={() => setResultTab(tab)}>
                <Icon size={15} /> {t(`tab_${tab}`)}
              </button>
            ))}
          </div>

          <div className="card result-panel">
            {resultTab === "transcript" && (
              <textarea className="preview mono" readOnly value={task?.preview ?? ""} placeholder={t("preview_empty")} />
            )}
            {resultTab === "files" && (
              <div className="stack">
                <span className="label">{t("download")}</span>
                {task?.files?.length ? (
                  <ul className="files">
                    {task.files.map((f) => (
                      <li key={f.name}>
                        <a href={f.url} download={f.name}>
                          <span className="badge">{EXPORT_LABELS[f.format] ?? f.format}</span>
                          <span className="grow">{f.name}</span>
                          <Download size={16} className="muted" />
                        </a>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <span className="muted">—</span>
                )}
                <label className="field">
                  <span className="label with-icon">
                    <FolderOpen size={14} /> {t("output_folder")}
                  </span>
                  <input type="text" readOnly value={task?.output_folder ?? ""} />
                </label>
              </div>
            )}
            {resultTab === "speakers" && (
              <div className="stack">
                {task?.speakers?.length ? (
                  <>
                    <table className="speakers">
                      <thead>
                        <tr>
                          <th>{t("speaker_col")}</th>
                          <th>{t("name_col")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {task.speakers.map((s) => (
                          <tr key={s}>
                            <td className="mono">{s}</td>
                            <td>
                              <input
                                type="text"
                                value={names[s] ?? ""}
                                placeholder={s}
                                onChange={(e) => setNames((n) => ({ ...n, [s]: e.target.value }))}
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="row">
                      <button className="btn primary" onClick={applyNames}>
                        <UserPen size={16} /> {t("apply_names")}
                      </button>
                      {renameMsg && (
                        <span className="ok with-icon">
                          <CheckCircle2 size={16} /> {renameMsg}
                        </span>
                      )}
                    </div>
                  </>
                ) : (
                  <span className="muted">{t("speakers_empty")}</span>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

// ======================================================================== components

function Alert(props: {
  kind: "error" | "connection";
  icon: LucideIcon;
  title: string;
  message: string;
  hint?: string;
  detail?: string;
  detailLabel?: string;
  spinner?: boolean;
  onDismiss?: () => void;
  dismissLabel?: string;
}) {
  const Icon = props.icon;
  return (
    <div className={`alert ${props.kind === "error" ? "error" : "warn"}`} role="alert" aria-live="assertive">
      <Icon className="alert-icon" />
      <div className="alert-body">
        <strong>{props.title}</strong>
        <p>{props.message}</p>
        {props.hint && <p className="alert-hint">{props.hint}</p>}
        {props.detail && props.detail !== props.message && (
          <details>
            <summary>{props.detailLabel}</summary>
            <code>{props.detail}</code>
          </details>
        )}
      </div>
      {props.spinner && <Loader2 className="spin alert-spinner" />}
      {props.onDismiss && (
        <button className="icon-btn" onClick={props.onDismiss} aria-label={props.dismissLabel} title={props.dismissLabel}>
          <X size={18} />
        </button>
      )}
    </div>
  );
}

function StatusIcon({ state }: { state: string }) {
  if (state === "running") return <Loader2 className="spin status-icon" />;
  if (state === "done") return <CheckCircle2 className="status-icon" />;
  if (state === "error") return <XCircle className="status-icon" />;
  if (state === "warn") return <WifiOff className="status-icon" />;
  return <span className="dot" />;
}

function ToggleCard(props: { icon: LucideIcon; title: string; hint: string; checked: boolean; onChange: (v: boolean) => void }) {
  const Icon = props.icon;
  return (
    <label className={`toggle-card ${props.checked ? "on" : ""}`}>
      <span className="toggle-icon" aria-hidden>
        <Icon />
      </span>
      <span className="toggle-text">
        <span className="toggle-title">{props.title}</span>
        <span className="toggle-hint">{props.hint}</span>
      </span>
      <input type="checkbox" role="switch" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
      <span className="switch" aria-hidden />
    </label>
  );
}

function Metric({ icon: Icon, value, label }: { icon: LucideIcon; value: string; label: string }) {
  return (
    <div className="metric">
      <div className="v">{value}</div>
      <div className="l with-icon">
        <Icon size={13} /> {label}
      </div>
    </div>
  );
}
