import { Clock3, Cpu, Rocket, ShieldCheck, Zap } from "lucide-react";
import type { OptimizationStatus } from "./api";

/** "1 min 15 s", "4 min", "45 s" */
export function formatMinutes(minutes: number): string {
  const total = Math.round(minutes * 60);
  if (total < 60) return `${total} s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m < 10 && s >= 5 ? `${m} min ${String(s).padStart(2, "0")} s` : `${Math.round(total / 60)} min`;
}

type Props = { status: OptimizationStatus; t: (key: string) => string };

/** Compact summary shown in Settings → Optimization once tuned parameters are applied. */
export function OptimizedInfographic({ status, t }: Props) {
  const details = [t("optimized_detail").replace("{batch}", String(status.batch)), status.device, status.applied_at]
    .filter(Boolean)
    .join(" · ");
  const hasData = status.speed != null && status.minutes_per_audio_hour != null;
  const vramPct =
    status.peak_vram_mb != null && status.gpu_total_mb ? Math.min(100, (status.peak_vram_mb / status.gpu_total_mb) * 100) : null;
  const gb = (mb: number) => (mb / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 });

  return (
    <section className="optimized-box" aria-label={t("optimized_title")}>
      <header className="optimized-head">
        <ShieldCheck className="optimized-icon" aria-hidden />
        <div>
          <strong>{t("optimized_title")}</strong>
          <span className="muted small">{details}</span>
        </div>
      </header>

      {hasData && (
        <>
          <div className="info-tiles">
            <div className="info-tile">
              <span className="info-label">
                <Zap size={14} aria-hidden /> {t("info_speed")}
              </span>
              <span className="info-value">
                {status.speed!.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                <small>{t("info_speed_unit")}</small>
              </span>
            </div>
            <div className="info-tile">
              <span className="info-label">
                <Clock3 size={14} aria-hidden /> {t("info_hour")}
              </span>
              <span className="info-value">≈ {formatMinutes(status.minutes_per_audio_hour!)}</span>
            </div>
            {vramPct != null && (
              <div className="info-tile">
                <span className="info-label">
                  <Cpu size={14} aria-hidden /> {t("info_vram")}
                </span>
                <span className="info-value">
                  {gb(status.peak_vram_mb!)}
                  <small> / {gb(status.gpu_total_mb!)} GB</small>
                </span>
                <div
                  className="meter"
                  role="meter"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(vramPct)}
                  aria-label={t("info_vram")}
                >
                  <div style={{ width: `${vramPct.toFixed(0)}%` }} />
                </div>
              </div>
            )}
          </div>

          <ul className="info-facts">
            <li>
              <Rocket size={14} aria-hidden />
              {t("info_fact_long").replace("{time}", formatMinutes(status.minutes_per_audio_hour! * 3))}
            </li>
            {status.speedup_vs_smallest != null && status.speedup_vs_smallest > 1 && (
              <li>
                <Zap size={14} aria-hidden />
                {t("info_fact_speedup").replace("{x}", status.speedup_vs_smallest.toLocaleString())}
              </li>
            )}
          </ul>
          <p className="info-footnote">
            {status.speed_source === "real"
              ? t("info_source_real").replace("{n}", String(status.real_jobs ?? 0))
              : t("info_source_benchmark")}{" "}
            {t("info_footnote")}
          </p>
        </>
      )}
    </section>
  );
}
