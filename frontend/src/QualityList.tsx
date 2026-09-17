import { clock, type QualityIssue } from "./api";

/** Time ranges where the transcript may be wrong, so they can be checked against the audio. */
export function QualityList({ issues, t }: { issues: QualityIssue[]; t: (key: string) => string }) {
  return (
    <details className="quality" open>
      <summary>
        {t("quality_title")} ({issues.length})
      </summary>
      <ul>
        {issues.map((issue, i) => (
          <li key={i}>
            <span className="mono">
              {clock(issue.start)}–{clock(issue.end)}
            </span>{" "}
            {t(`quality_${issue.kind}_${issue.action}`).replace("{words}", String(issue.words_removed))}
          </li>
        ))}
      </ul>
    </details>
  );
}
