import { AlertTriangle, CheckCircle2, CircleDashed, Clock3, FileAudio, Loader2, Square, X, XCircle } from "lucide-react";
import { clock } from "./api";
import { itemStatus, type QueueItem, type QueueStatus } from "./queue";

type Props = {
  items: QueueItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onCancel: (id: string) => void;
  t: (key: string) => string;
};

const ICONS: Record<QueueStatus, typeof Clock3> = {
  uploading: Loader2,
  upload_error: XCircle,
  ready: CircleDashed,
  queued: Clock3,
  running: Loader2,
  done: CheckCircle2,
  stopped: AlertTriangle,
  error: XCircle,
};

/** Files waiting, running and finished — click one to see its results. */
export function QueueList({ items, selectedId, onSelect, onRemove, onCancel, t }: Props) {
  const done = items.filter((it) => ["done", "stopped", "error"].includes(itemStatus(it))).length;

  return (
    <div className="queue">
      <div className="queue-head">
        <span className="label">{t("queue_title")}</span>
        <span className="muted small">
          {done}/{items.length}
        </span>
      </div>
      <ul>
        {items.map((item, position) => {
          const status = itemStatus(item);
          const Icon = ICONS[status];
          const progress = item.task?.progress;
          const detail =
            status === "uploading"
              ? `${item.uploadPct ?? 0}%`
              : status === "running" && progress
                ? `${progress.percent}%`
                : status === "queued" && item.task?.queue_position
                  ? `#${item.task.queue_position}`
                  : "";
          const removable = ["ready", "upload_error", "done", "stopped", "error"].includes(status);
          return (
            <li key={item.id} className={`queue-item ${status} ${item.id === selectedId ? "selected" : ""}`}>
              <button className="queue-main" onClick={() => onSelect(item.id)} title={item.name}>
                <FileAudio size={18} className="queue-file" aria-hidden />
                <span className="queue-text">
                  <span className="queue-name">
                    {position + 1}. {item.name}
                  </span>
                  <span className="queue-meta">
                    {item.info ? clock(item.info.duration) : `${(item.size / 1024 ** 2).toFixed(1)} MB`}
                  </span>
                </span>
                <span className={`queue-status ${status}`}>
                  <Icon size={14} className={status === "running" || status === "uploading" ? "spin" : ""} aria-hidden />
                  {t(`q_${status}`)}
                  {detail ? ` ${detail}` : ""}
                </span>
              </button>
              {(status === "running" || status === "queued") && (
                <button className="icon-btn" onClick={() => onCancel(item.id)} title={t("stop")} aria-label={t("stop")}>
                  <Square size={15} />
                </button>
              )}
              {removable && (
                <button className="icon-btn" onClick={() => onRemove(item.id)} title={t("q_remove")} aria-label={t("q_remove")}>
                  <X size={16} />
                </button>
              )}
              {status === "running" && progress && (
                <div className="queue-bar">
                  <div style={{ width: `${(progress.fraction * 100).toFixed(1)}%` }} />
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
