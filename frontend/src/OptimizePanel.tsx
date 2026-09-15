import { useEffect, useState } from "react";
import { CheckCircle2, Gauge, Loader2, Save, ShieldCheck } from "lucide-react";
import { ConfirmDialog } from "./ConfirmDialog";
import { OptimizedInfographic } from "./OptimizedInfographic";
import { ApiError, api, clock, type Form, type Optimization, type OptimizationStatus, type OptimizerSnapshot } from "./api";

type Props = {
  form: Form;
  t: (key: string) => string;
  onError: (context: string, e: unknown) => void;
  onApplied: (defaults: Form) => void;
  initialStatus: OptimizationStatus;
};

/** Settings → Optimization: benchmark batch sizes on the bundled recording, then let the user apply the result. */
export function OptimizePanel({ form, t, onError, onApplied, initialStatus }: Props) {
  const [status, setStatus] = useState<OptimizationStatus>(initialStatus);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [snapshot, setSnapshot] = useState<OptimizerSnapshot | null>(null);
  const [result, setResult] = useState<Optimization | null>(null);
  const [queued, setQueued] = useState(false);
  const [applying, setApplying] = useState(false);
  const [appliedPath, setAppliedPath] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!taskId || !running) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const s = await api.task(taskId);
        if (stop) return;
        setQueued(s.state === "queued");
        if (s.optimizer) setSnapshot(s.optimizer);
        if (s.state === "error") {
          onError("optimize_failed", new Error(s.error ?? ""));
          setRunning(false);
          return;
        }
        if (s.state === "done" && s.optimization) {
          setResult(s.optimization);
          setRunning(false);
          return;
        }
      } catch (e) {
        if (stop) return;
        if (e instanceof ApiError && e.status === 404) {
          onError("error_task_lost", e);
          setRunning(false);
          return;
        }
      }
      timer = setTimeout(poll, 1000);
    };
    poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [taskId, running, onError]);

  const start = async () => {
    setResult(null);
    setSnapshot(null);
    setAppliedPath(null);
    try {
      setTaskId((await api.optimize(form)).task_id);
      setRunning(true);
    } catch (e) {
      onError("optimize_failed", e);
    }
  };

  const apply = async () => {
    if (!taskId) return;
    setApplying(true);
    try {
      const s = await api.applyOptimization(taskId);
      setAppliedPath(s.optimization?.env_path ?? "");
      if (s.defaults) onApplied(s.defaults);
      if (s.optimization_status) setStatus(s.optimization_status);
    } catch (e) {
      onError("optimize_failed", e);
    } finally {
      setApplying(false);
    }
  };

  const trials = result?.trials ?? snapshot?.trials ?? [];
  const stageLabel =
    snapshot?.stage === "trial" ? `${t("optimize_stage_trial")} ${snapshot.current_batch}` : t(`optimize_stage_${snapshot?.stage ?? "loading"}`);

  return (
    <div className="stack">
      {status.optimized && <OptimizedInfographic status={status} t={t} />}
      <p className="muted optimize-desc">
        {t("optimize_desc")} <b className="optimize-duration">{t("optimize_duration")}</b>
      </p>
      <button className="btn secondary" onClick={() => (status.optimized ? setConfirming(true) : start())} disabled={running}>
        {running ? <Loader2 className="spin" size={17} /> : <Gauge size={17} />}
        {t("optimize_btn")}
      </button>

      {confirming && (
        <ConfirmDialog
          icon={ShieldCheck}
          title={t("optimized_title")}
          message={t("optimize_confirm")}
          confirmLabel={t("yes")}
          cancelLabel={t("no")}
          onConfirm={() => {
            setConfirming(false);
            start();
          }}
          onCancel={() => setConfirming(false)}
        />
      )}

      {running && (
        <div className="progress-block">
          <div className="progress-head">
            <span>
              <b>{queued ? t("queued") : stageLabel}</b>
              {snapshot ? ` · ${snapshot.done}/${snapshot.total}` : ""}
            </span>
            <span className="muted">{clock(snapshot?.elapsed_seconds)}</span>
          </div>
          <div className="bar">
            <div style={{ width: `${((snapshot?.fraction ?? 0) * 100).toFixed(0)}%` }} />
          </div>
        </div>
      )}

      {trials.length > 0 && (
        <table className="opt-table">
          <thead>
            <tr>
              <th>{t("optimize_col_batch")}</th>
              <th>{t("optimize_col_speed")}</th>
              <th>{t("optimize_col_vram")}</th>
              <th>{t("optimize_col_result")}</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial) => {
              const best = result?.best_batch === trial.batch;
              return (
                <tr key={trial.batch} className={best ? "best" : ""} title={trial.note ?? undefined}>
                  <td>{trial.batch}</td>
                  <td>{trial.speed ? `${trial.speed.toFixed(1)}×` : "—"}</td>
                  <td>{trial.peak_vram_mb != null ? `${Math.round(trial.peak_vram_mb).toLocaleString()} MB` : "—"}</td>
                  <td>
                    {t(`optimize_result_${trial.status}`)}
                    {best && <b> · {t("optimize_best")}</b>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {result && appliedPath === null && (
        <>
          <p className="optimize-summary">
            {t("optimize_done").replace("{best}", String(result.best_batch)).replace("{previous}", String(result.previous_batch))}
          </p>
          <button className="btn primary lg apply-btn" onClick={apply} disabled={applying}>
            {applying ? <Loader2 className="spin" size={20} /> : <Save size={20} />}
            {t("optimize_apply")}
          </button>
        </>
      )}

      {appliedPath !== null && (
        <div className="status done">
          <CheckCircle2 className="status-icon" />
          <span>{t("optimize_applied").replace("{path}", appliedPath)}</span>
        </div>
      )}
    </div>
  );
}
