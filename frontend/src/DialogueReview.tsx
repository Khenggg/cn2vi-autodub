import { useEffect, useState } from 'react';
import { api, post, type Episode, type Segment } from './api';

export function DialogueReview({ episode, onClose, onSaved }: {
  episode: Episode; onClose: () => void; onSaved: () => Promise<void>;
}) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api<Segment[]>(`/episodes/${episode.id}/segments`).then(items => {
      if (!cancelled) setSegments(items);
    }).catch(e => { if (!cancelled) setError(String(e.message)); });
    return () => { cancelled = true; };
  }, [episode.id]);
  function change(index: number, value: Partial<Segment>) {
    setSegments(items => items.map((s, i) => i === index ? { ...s, ...value } : s));
  }
  async function save(resume: boolean) {
    setBusy(true); setError('');
    try {
      if (segments.some(s => s.action === 'NEEDS_REVIEW')) throw new Error('Chọn lồng tiếng hoặc giữ âm gốc cho các dòng cần kiểm tra.');
      await api<Segment[]>(`/episodes/${episode.id}/segments`, {
        method: 'PUT', body: JSON.stringify(segments.map(s => ({ ...s, needs_review: false,
          voice_id: s.voice_id?.trim() || null, speaker_id: s.speaker_id?.trim() || null }))),
      });
      if (resume) await post(`/episodes/${episode.id}/start`);
      await onSaved(); onClose();
    } catch (e) { setError(e instanceof Error ? e.message : 'Không thể lưu lời thoại.'); }
    finally { setBusy(false); }
  }
  return <div className="dialogue-backdrop" role="dialog" aria-modal="true" aria-label="Duyệt lời thoại">
    <section className="dialogue-panel">
      <header><h2>Duyệt lời thoại · {episode.filename}</h2><button onClick={onClose} disabled={busy}>Đóng</button></header>
      <p>Sửa lời Việt quá dài hoặc giữ âm gốc cho tiếng cười, thở và âm thanh không cần lồng tiếng.</p>
      <video className="dialogue-source" src={`/api/episodes/${episode.id}/source`} controls playsInline/>
      {error && <p role="alert" className="error">{error}</p>}
      {segments.map((s, index) => <article className="dialogue-row" key={s.id}>
        <strong>{(s.start_ms / 1000).toFixed(2)}–{(s.end_ms / 1000).toFixed(2)} giây</strong>
        <p>{s.zh_text}</p>
        <label>Phụ đề Việt<textarea value={s.subtitle_vi} onChange={e => change(index, { subtitle_vi: e.target.value })}/></label>
        <label>Lời lồng tiếng<textarea value={s.dub_vi} onChange={e => change(index, { dub_vi: e.target.value })}/></label>
        <label>Xử lý<select value={s.action} onChange={e => change(index, { action: e.target.value as Segment['action'], needs_review: false })}>
          <option value="NEEDS_REVIEW">Cần chọn</option><option value="DUB">Lồng tiếng</option><option value="KEEP">Giữ âm gốc</option>
        </select></label>
        <label>Giọng đọc<input value={s.voice_id ?? ''} placeholder="Giọng mặc định" onChange={e => change(index, { voice_id: e.target.value || null })}/></label>
        <label>Nhân vật<input value={s.speaker_id ?? ''} placeholder="Tùy chọn" onChange={e => change(index, { speaker_id: e.target.value || null })}/></label>
        <details><summary>Thời gian từng từ</summary>
          {s.words.map((w, wi) => <div className="word-times" key={wi}>
            <span>{w.t}</span>
            <label>Bắt đầu<input type="number" min={s.start_ms / 1000} max={s.end_ms / 1000} step="0.01" value={w.s / 1000}
              onChange={e => change(index, { words: s.words.map((v, i) => i === wi ? { ...v, s: Math.round(Number(e.target.value) * 1000) } : v) })}/></label>
            <label>Kết thúc<input type="number" min={s.start_ms / 1000} max={s.end_ms / 1000} step="0.01" value={w.e / 1000}
              onChange={e => change(index, { words: s.words.map((v, i) => i === wi ? { ...v, e: Math.round(Number(e.target.value) * 1000) } : v) })}/></label>
          </div>)}
        </details>
      </article>)}
      <footer><button disabled={busy || !segments.length} onClick={() => void save(false)}>Lưu thay đổi</button>
        <button disabled={busy || !segments.length} onClick={() => void save(true)}>Lưu và tiếp tục</button></footer>
    </section>
  </div>;
}

