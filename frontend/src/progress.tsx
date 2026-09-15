import type { Progress } from "./api";

/** Human-friendly remaining time — kept in sync with scriba/core/progress.py::format_eta. */
export function formatEta(seconds: number | null, lang: string): string {
  const it = lang === "it";
  if (seconds == null) return it ? "stima in corso…" : "estimating…";
  if (seconds < 10) return it ? "quasi terminato" : "almost done";
  const s = Math.round(seconds);
  if (s < 60) return it ? `circa ${s} s rimanenti` : `about ${s} s left`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const sec = Math.floor((s % 60) / 10) * 10;
    const body = m < 10 && sec ? `${m} min ${String(sec).padStart(2, "0")} s` : `${m} min`;
    return it ? `circa ${body} rimanenti` : `about ${body} left`;
  }
  const h = Math.floor(s / 3600);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  return it ? `circa ${h} h ${mm} min rimanenti` : `about ${h} h ${mm} min left`;
}

export function ProgressBar({ progress, lang, t }: { progress: Progress; lang: string; t: (k: string) => string }) {
  const align = progress.phase === "align";
  const [done, total] = align ? progress.align : progress.asr;
  return (
    <div className="progress-block" role="progressbar" aria-valuenow={progress.percent} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-head">
        <span>
          <b>{t(align ? "progress_align" : "progress_asr")}</b> · {progress.percent}%
          {total ? ` · ${done}/${total} ${t("progress_segments")}` : ""}
        </span>
        <span className="muted">{formatEta(progress.eta_seconds, lang)}</span>
      </div>
      <div className="bar">
        <div style={{ width: `${(progress.fraction * 100).toFixed(1)}%` }} />
      </div>
    </div>
  );
}
