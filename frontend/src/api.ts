export type Segment = {
  schema_version: number; id: string; start_ms: number; end_ms: number; zh_text: string;
  words: { t: string; s: number; e: number }[]; subtitle_vi: string; dub_vi: string;
  emotion: string; voice_id: string | null; speaker_id: string | null;
  action: 'DUB' | 'KEEP' | 'NEEDS_REVIEW'; needs_review: boolean;
  confidence: Record<string, number | null>;
};
export type Artifact = { id: string; kind: string; bytes: number };
export type Issue = { id: string; code: string; resolved: number; details: { message?: string } };
export type Episode = {
  id: string; ordinal: number; filename: string; total_bytes: number; uploaded_bytes: number;
  status: string; progress: number; queue_requested: number; duration_ms: number | null;
  next_stage: string | null; issues: Issue[]; artifacts: Artifact[];
};
export type Term = { zh: string; vi: string; confidence: number; locked_by_user: boolean | number };
export type Series = { id: string; title: string; priority: number; episodes: Episode[]; glossary: Term[] };
export type System = {
  worker_state: string; gpu: { available: boolean; name: string | null; free_mb: number | null };
  ram: { free_bytes: number }; disk: { free_bytes: number }; ffprobe_available: boolean;
  cost: { projected_vnd: number; policy: string; cloud_rate_vnd_per_hour: number };
  models: { name: string; ready: boolean }[]; limits: { chunk_bytes: number };
};
export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options, credentials: 'same-origin',
    headers: { 'X-Autodub-Request': '1', ...(options.body && typeof options.body === 'string' ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = error.detail;
    throw new ApiError(response.status, typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(e => e.msg).join('; ') : `HTTP ${response.status}`);
  }
  return response.json();
}
export function post<T>(path: string, payload?: unknown): Promise<T> {
  return api<T>(path, { method: 'POST', body: payload === undefined ? undefined : JSON.stringify(payload) });
}
export const bytes = (n: number) => {
  const unit = n >= 1024 ** 3 ? ['GB', 1024 ** 3] as const : n >= 1024 ** 2 ? ['MB', 1024 ** 2] as const : ['KB', 1024] as const;
  return `${(n / unit[1]).toFixed(1)} ${unit[0]}`;
};
export const money = (n: number) => `${Math.round(n).toLocaleString('vi-VN')} ₫`;
export const duration = (n: number | null) => n ? `${Math.floor(n / 60000)}:${String(Math.floor(n / 1000) % 60).padStart(2, '0')}` : '—';
