import { ChevronLeft, ChevronRight, FileAudio } from "lucide-react";
import { itemStatus, type QueueItem } from "./queue";

type Props = {
  items: QueueItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  t: (key: string) => string;
};

/** Chooses which recording the status, transcript, files and speakers panels refer to. */
export function RecordingPicker({ items, selectedId, onSelect, t }: Props) {
  if (items.length < 2) return null;
  const index = items.findIndex((it) => it.id === selectedId);
  const go = (step: number) => {
    const next = items[index + step];
    if (next) onSelect(next.id);
  };

  return (
    <div className="card picker">
      <FileAudio className="picker-icon" aria-hidden />
      <label className="picker-field">
        <span className="label">{t("recording")}</span>
        <select value={selectedId ?? ""} onChange={(e) => onSelect(e.target.value)}>
          {index < 0 && (
            <option value="" disabled>
              —
            </option>
          )}
          {items.map((item, i) => (
            <option key={item.id} value={item.id}>
              {i + 1}. {item.name} · {t(`q_${itemStatus(item)}`)}
            </option>
          ))}
        </select>
      </label>
      <div className="picker-nav">
        <button className="icon-btn" disabled={index <= 0} onClick={() => go(-1)} title={t("prev_recording")} aria-label={t("prev_recording")}>
          <ChevronLeft size={18} />
        </button>
        <span className="muted small">
          {index < 0 ? "–" : index + 1}/{items.length}
        </span>
        <button
          className="icon-btn"
          disabled={index < 0 || index >= items.length - 1}
          onClick={() => go(1)}
          title={t("next_recording")}
          aria-label={t("next_recording")}
        >
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
}
