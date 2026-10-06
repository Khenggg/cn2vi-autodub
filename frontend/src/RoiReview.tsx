import { useRef, useState, type PointerEvent } from 'react';
import { api, post, type Episode, type Segment } from './api';

type Box = { x: number; y: number; w: number; h: number; scope: 'episode' | 'series' };
export function RoiReview({ episode, onClose, onSaved }: {
  episode: Episode; onClose: () => void; onSaved: () => Promise<void>;
}) {
  const [box, setBox] = useState<Box>({ x: 0.15, y: 0.75, w: 0.7, h: 0.18, scope: 'episode' });
  const [ratio, setRatio] = useState(16 / 9);
  const [drawing, setDrawing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const anchor = useRef<{ x: number; y: number } | null>(null);
  const frame = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  function point(e: PointerEvent<HTMLDivElement>) {
    const rect = frame.current!.getBoundingClientRect();
    return { x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)) };
  }
  function start(e: PointerEvent<HTMLDivElement>) {
    if (!drawing) return;
    anchor.current = point(e); e.currentTarget.setPointerCapture(e.pointerId);
    setBox(b => ({ ...b, ...anchor.current!, w: 0.001, h: 0.001 }));
  }
  function move(e: PointerEvent<HTMLDivElement>) {
    if (!drawing || !anchor.current) return;
    const p = point(e), a = anchor.current;
    setBox(b => ({ ...b, x: Math.min(a.x, p.x), y: Math.min(a.y, p.y),
      w: Math.max(0.001, Math.abs(a.x - p.x)), h: Math.max(0.001, Math.abs(a.y - p.y)) }));
  }
  async function save() {
    setBusy(true); setError('');
    try {
      if (box.w < 0.01 || box.h < 0.01) throw new Error('Vẽ vùng đủ lớn để bao trọn dòng phụ đề.');
      await api(`/episodes/${episode.id}/roi`, { method: 'PUT', body: JSON.stringify(box) });
      const lines = await api<Segment[]>(`/episodes/${episode.id}/segments`);
      const current = await api<Episode>(`/episodes/${episode.id}`);
      const reviewCodes = ['DURATION_REWRITE_REQUIRED', 'ALIGNMENT_REVIEW_REQUIRED', 'TRANSCRIPT_REVIEW_REQUIRED', 'MISSING_TTS_CLIP'];
      if (!lines.some(s => s.needs_review || s.action === 'NEEDS_REVIEW') &&
          !current.issues.some(i => !i.resolved && reviewCodes.includes(i.code)))
        await post(`/episodes/${episode.id}/start`);
      await onSaved(); onClose();
    } catch (e) { setError(e instanceof Error ? e.message : 'Không thể lưu vùng phụ đề.'); }
    finally { setBusy(false); }
  }
  return <div className="dialogue-backdrop" role="dialog" aria-modal="true" aria-label="Chọn vùng phụ đề">
    <section className="dialogue-panel">
      <header><h2>Vùng phụ đề Trung</h2><button disabled={busy} onClick={onClose}>Đóng</button></header>
      <p>Tìm khung hình có phụ đề, dừng video rồi kéo vùng bao trọn phụ đề cần xóa.</p>
      <button onClick={() => { video.current?.pause(); anchor.current = null; setDrawing(d => !d); }}>
        {drawing ? 'Xem video' : 'Vẽ vùng phụ đề'}
      </button>
      <div ref={frame} className="roi-frame" style={{ aspectRatio: String(ratio) }}>
        <video ref={video} src={`/api/episodes/${episode.id}/source`} controls={!drawing} playsInline
          onLoadedMetadata={() => { const v = video.current!; if (v.videoWidth && v.videoHeight) setRatio(v.videoWidth / v.videoHeight); }}/>
        <div className="roi-overlay" style={{ pointerEvents: drawing ? 'auto' : 'none' }}
          onPointerDown={start} onPointerMove={move} onPointerUp={() => { anchor.current = null; }} onPointerCancel={() => { anchor.current = null; }}>
          <div className="roi-box" style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`,
            width: `${box.w * 100}%`, height: `${box.h * 100}%` }}/>
        </div>
      </div>
      <label><input type="checkbox" checked={box.scope === 'series'} onChange={e => setBox(b => ({ ...b, scope: e.target.checked ? 'series' : 'episode' }))}/> Áp dụng vùng này cho cả Series</label>
      {error && <p className="error" role="alert">{error}</p>}
      <footer><button disabled={busy} onClick={save}>Lưu vùng và tiếp tục</button></footer>
    </section>
  </div>;
}

