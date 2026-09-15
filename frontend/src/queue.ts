import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type AudioInfo, type Form, type TaskStatus } from "./api";

export type QueueItem = {
  id: string;
  name: string;
  size: number;
  info?: AudioInfo;
  uploadId?: string;
  uploadPct: number | null; // null = not uploading
  uploadError?: string;
  taskId?: string;
  task?: TaskStatus | null;
  startedAt?: number;
  lost?: boolean; // the server no longer knows the task (restarted)
};

export type QueueStatus = "uploading" | "upload_error" | "ready" | "queued" | "running" | "done" | "stopped" | "error";

const TERMINAL = new Set(["done", "error", "cancelled"]);

export function itemStatus(item: QueueItem): QueueStatus {
  if (item.uploadPct !== null) return "uploading";
  if (item.uploadError) return "upload_error";
  if (item.lost) return "error";
  const task = item.task;
  if (!item.taskId) return "ready";
  if (!task || task.state === "queued") return "queued";
  if (task.state === "running") return "running";
  if (task.state === "error") return "error";
  if (task.state === "cancelled" || task.completed === false) return "stopped";
  return "done";
}

export const isActive = (item: QueueItem) => ["queued", "running"].includes(itemStatus(item));

type Options = {
  onUploadError: (e: unknown) => void;
  onStartError: (e: unknown) => void;
  onJobError: (item: QueueItem) => void;
  onTaskLost: (item: QueueItem) => void;
};

/**
 * Several files, processed one after another by the server's FIFO queue (one GPU job at a time).
 * Uploads run sequentially too, so a large batch of files never floods the connection.
 */
export function useQueue({ onUploadError, onStartError, onJobError, onTaskLost }: Options) {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [connectionLost, setConnectionLost] = useState(false);
  const uploadChain = useRef<Promise<void>>(Promise.resolve());
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const callbacks = useRef({ onUploadError, onStartError, onJobError, onTaskLost });
  callbacks.current = { onUploadError, onStartError, onJobError, onTaskLost };

  const update = useCallback((id: string, patch: Partial<QueueItem> | ((item: QueueItem) => Partial<QueueItem>)) => {
    setItems((list) => list.map((it) => (it.id === id ? { ...it, ...(typeof patch === "function" ? patch(it) : patch) } : it)));
  }, []);

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const added: QueueItem[] = Array.from(files).map((file) => ({
        id: crypto.randomUUID(),
        name: file.name,
        size: file.size,
        uploadPct: 0,
      }));
      setItems((list) => [...list, ...added]);
      added.forEach((item, i) => {
        const file = Array.from(files)[i];
        uploadChain.current = uploadChain.current.then(async () => {
          if (!itemsRef.current.some((it) => it.id === item.id)) return; // removed before its turn
          try {
            const res = await api.upload(file, (pct) => update(item.id, { uploadPct: pct }));
            update(item.id, { uploadId: res.upload_id, info: res.info, uploadPct: null });
          } catch (e) {
            update(item.id, { uploadError: (e as Error).message, uploadPct: null });
            callbacks.current.onUploadError(e);
          }
        });
      });
      return added.map((it) => it.id);
    },
    [update],
  );

  const remove = useCallback((id: string) => setItems((list) => list.filter((it) => it.id !== id)), []);

  /** Submit every uploaded file that has not been started yet, in list order. */
  const startAll = useCallback(
    async (form: Form) => {
      const ready = itemsRef.current.filter((it) => itemStatus(it) === "ready");
      const started: string[] = [];
      for (const item of ready) {
        try {
          const { task_id } = await api.start(item.uploadId!, form);
          update(item.id, { taskId: task_id, task: null, startedAt: Date.now(), lost: false });
          started.push(item.id);
        } catch (e) {
          callbacks.current.onStartError(e);
          break;
        }
      }
      return started;
    },
    [update],
  );

  const cancel = useCallback(
    async (id: string) => {
      const item = itemsRef.current.find((it) => it.id === id);
      if (!item?.taskId) return;
      const task = await api.cancel(item.taskId);
      update(id, { task });
    },
    [update],
  );

  // One lightweight poll loop for all active jobs.
  const hasActive = items.some((it) => it.taskId && !it.lost && !TERMINAL.has(it.task?.state ?? ""));
  useEffect(() => {
    if (!hasActive) return;
    let stop = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const active = itemsRef.current.filter((it) => it.taskId && !it.lost && !TERMINAL.has(it.task?.state ?? ""));
      let unreachable = false;
      await Promise.all(
        active.map(async (item) => {
          try {
            const task = await api.task(item.taskId!);
            if (stop) return;
            update(item.id, { task });
            if (task.state === "error") callbacks.current.onJobError({ ...item, task });
          } catch (e) {
            if (e instanceof ApiError && e.status === 404) {
              update(item.id, { lost: true });
              callbacks.current.onTaskLost(item);
            } else {
              unreachable = true;
            }
          }
        }),
      );
      if (stop) return;
      failures = unreachable ? failures + 1 : 0;
      setConnectionLost(unreachable);
      timer = setTimeout(poll, failures ? Math.min(5000, 1000 * (1 + failures)) : 1000);
    };
    poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [hasActive, update]);

  return { items, addFiles, remove, startAll, cancel, update, connectionLost };
}
