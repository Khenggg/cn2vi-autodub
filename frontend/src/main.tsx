import { StrictMode, useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { createRoot } from 'react-dom/client';
import { api, ApiError, bytes, duration, money, post, type Episode, type Series, type System } from './api';
import './style.css';
import { DialogueReview } from './DialogueReview';
import { RoiReview } from './RoiReview';

const labels: Record<string, string> = {
  UPLOADING: 'Đang tải lên', QUEUED: 'Sẵn sàng', PREPARING: 'Kiểm tra media',
  CHECKPOINTED: 'Sẵn sàng lồng tiếng', FAILED: 'Cần thử lại', PREVIEW_READY: 'Chờ review',
  NEEDS_REVIEW: 'Có video và báo cáo cần kiểm tra', COMPLETED: 'Hoàn tất', V1_RUNNING: 'Đang xử lý CN2VI V1', V2_RUNNING: 'Đang thuyết minh CN2VI V2',
  ASR: 'Nhận diện thoại (ASR)', ALIGNING: 'Căn chỉnh từ (Align)', TRANSLATING: 'Dịch lời thoại',
  SEPARATING: 'Tách thoại và âm nền', TTS: 'Tạo giọng Việt', TIMING: 'Khớp thời lượng lời thoại',
  AUDIO_MIX: 'Hòa trộn âm thanh', QC: 'Kiểm tra chất lượng', TEXT_REMOVAL: 'Xóa phụ đề Trung', SUBTITLE_RENDER: 'Chèn Vietsub', ENCODING: 'Xuất video final',
};

function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [token, setToken] = useState('');
  const [series, setSeries] = useState<Series[]>([]);
  const [selected, select] = useState<string | null>(null);
  const [system, setSystem] = useState<System | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [newSeries, setNewSeries] = useState(false);
  const [title, setTitle] = useState('');
  const [priority, setPriority] = useState(0);
  const [tab, setTab] = useState<'episodes' | 'glossary'>('episodes');
  const [zh, setZh] = useState('');
  const [vi, setVi] = useState('');
  const [uploadProgress, setUploadProgress] = useState<{ name: string; percent: number; mbps: number } | null>(null);
  const [source, setSource] = useState<Episode | null>(null);
  const [review, setReview] = useState<Episode | null>(null);
  const [roiReview, setRoiReview] = useState<Episode | null>(null);
  const [subtitleMode, setSubtitleMode] = useState<'replace' | 'burn' | 'off'>('replace');
  const [playPreview, setPlayPreview] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const uploading = useRef(false);
  const active = series.find(s => s.id === selected) ?? series[0];

  const report = useCallback((e: unknown) => {
    if (e instanceof ApiError && e.status === 401) setAuthenticated(false);
    setError(e instanceof Error ? e.message : 'Không thể hoàn thành thao tác.');
  }, []);
  const refresh = useCallback(async () => {
    try {
      const [items, info] = await Promise.all([api<Series[]>('/series'), api<System>('/system/status')]);
      setSeries(items); setSystem(info); setAuthenticated(true);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) { setAuthenticated(false); setError(''); }
      else { report(e); setAuthenticated(current => current === null ? false : current); }
    }
  }, [report]);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!authenticated) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const events = new EventSource('/api/events');
    events.addEventListener('job', () => {
      clearTimeout(timer); timer = setTimeout(() => void refresh(), 200);
    });
    const interval = setInterval(() => void refresh(), 15000);
    return () => { events.close(); clearTimeout(timer); clearInterval(interval); };
  }, [authenticated, refresh]);

  async function action(task: () => Promise<unknown>) {
    setBusy(true); setError(''); setNotice('');
    try { await task(); await refresh(); } catch (e) { report(e); }
    finally { setBusy(false); }
  }
  async function login(e: FormEvent) {
    e.preventDefault();
    await action(async () => { await post('/auth/login', { token }); setToken(''); });
  }
  async function createSeries(e: FormEvent) {
    e.preventDefault();
    await action(async () => {
      const item = await post<Series>('/series', { title, priority });
      select(item.id); setTitle(''); setNewSeries(false); setTab('episodes');
    });
  }
  async function exportWorkspace() {
    await action(async () => {
      const result = await post<{ download_url: string }>('/workspace/export');
      download(result.download_url);
      setNotice('Đã xuất metadata workspace. Chức năng import sẽ được bổ sung ở giai đoạn Recovery.');
    });
  }
  function download(url: string) {
    const a = document.createElement('a'); a.href = url; a.download = ''; document.body.append(a); a.click(); a.remove();
  }
  async function uploadFiles(files: FileList | File[]) {
    if (!active || uploading.current) return;
    const target = active;
    const selectedFiles = Array.from(files);
    if (!selectedFiles.length) return;
    uploading.current = true; setBusy(true); setError(''); setNotice('');
    let ordinal = Math.max(0, ...target.episodes.map(e => e.ordinal)) + 1;
    try {
      for (const file of selectedFiles) {
        const key = `autodub:upload:${target.id}:${file.name}:${file.size}:${file.lastModified}`;
        const saved = localStorage.getItem(key);
        let episode: Episode | undefined;
        if (saved) {
          try { episode = await api<Episode>(`/uploads/${saved}`); }
          catch (e) { if (!(e instanceof ApiError) || e.status !== 404) throw e; localStorage.removeItem(key); }
        }
        if (episode && episode.status !== 'UPLOADING') {
          setNotice(`Tệp ${file.name} đã tải xong trước đó.`); continue;
        }
        if (!episode) {
          episode = await post<Episode>('/episodes', { series_id: target.id, ordinal: ordinal++, filename: file.name, total_bytes: file.size, subtitle_mode: subtitleMode });
          localStorage.setItem(key, episode.id);
        }
        let offset = episode.uploaded_bytes;
        const started = performance.now(), initial = offset;
        const chunkSize = system?.limits.chunk_bytes ?? 8 * 1024 ** 2;
        while (offset < file.size) {
          const block = file.slice(offset, Math.min(offset + chunkSize, file.size));
          let sent = false;
          for (let attempt = 0; attempt < 3 && !sent; attempt++) {
            try {
              episode = await api<Episode>(`/uploads/${episode.id}/chunks`, { method: 'PUT', body: block, headers: { 'Upload-Offset': String(offset), 'Content-Type': 'application/octet-stream' } });
              offset = episode.uploaded_bytes; sent = true;
            } catch (e) {
              if (e instanceof ApiError && ![409, 429, 500, 502, 503, 504].includes(e.status)) throw e;
              if (attempt === 2) throw e;
              await new Promise(resolve => setTimeout(resolve, 1000 * 2 ** attempt));
              episode = await api<Episode>(`/uploads/${episode.id}`);
              if (episode.uploaded_bytes !== offset) { offset = episode.uploaded_bytes; sent = true; }
            }
          }
          setUploadProgress({ name: file.name, percent: offset / file.size * 100,
            mbps: (offset - initial) * 8 / Math.max(1, performance.now() - started) / 1000 });
        }
        // Starting requires the user's separate action after upload.
        await refresh();
      }
      setNotice('Upload hoàn tất. Chọn Bắt đầu để xử lý video.');
    } catch (e) { report(e); }
    finally { uploading.current = false; setUploadProgress(null); setBusy(false); await refresh(); if (input.current) input.current.value = ''; }
  }
  async function startAll() {
    if (!active) return;
    await action(async () => {
      for (const episode of active.episodes.filter(e => e.status === 'QUEUED' && !e.queue_requested))
        await post(`/episodes/${episode.id}/start`);
    });
  }
  async function saveTerm(e: FormEvent) {
    e.preventDefault(); if (!active) return;
    await action(async () => {
      await api(`/series/${active.id}/glossary`, { method: 'PUT', body: JSON.stringify([{ zh, vi, confidence: 1, locked_by_user: true }]) });
      setZh(''); setVi('');
    });
  }

  if (authenticated === null) return <div className="connecting" role="status">Đang kết nối workspace…</div>;
  if (!authenticated) return <main className="login-page"><div className="login-brand">CN<span>2</span>VI <small>AUTODUB</small></div>
    <form className="login-form" onSubmit={login}>
      <p className="eyebrow">WORKSPACE CÁ NHÂN</p><h1>Chuyển lời thoại.<br/>Giữ trọn âm thanh.</h1>
      <p>Nhập admin token được in trong terminal khi khởi động server.</p>
      <label htmlFor="token">Admin token</label><input id="token" type="password" value={token} onChange={e => setToken(e.target.value)} autoComplete="current-password" required autoFocus/>
      {error && <p className="error" role="alert">{error}</p>}
      <button className="primary" disabled={busy}>Mở workspace →</button>
      <small>CN2VI AutoDub · Core v0.1</small>
    </form></main>;

  const allEpisodes = series.flatMap(s => s.episodes);
  return <div className="shell">
    <aside className="sidebar"><div className="brand">CN<span>2</span>VI <small>AUTODUB</small></div>
      <p className="sidebar-label">WORKSPACE</p><div className="nav-current">Thư viện Series <span>{series.length}</span></div>
      <div className="sidebar-heading"><span>SERIES · ƯU TIÊN</span><button className="icon-button" aria-label="Tạo Series" onClick={() => setNewSeries(true)}>+</button></div>
      <nav aria-label="Danh sách Series">{series.map((item, index) => <button key={item.id} className={`series-link ${active?.id === item.id ? 'selected' : ''}`} onClick={() => { select(item.id); setTab('episodes'); }}>
        <span className="series-index">{String(index + 1).padStart(2, '0')}</span><span>{item.title}<small>{item.episodes.length} tập · ưu tiên {item.priority}</small></span>
      </button>)}</nav>
      <div className="sidebar-bottom"><span className={`dot ${system?.worker_state === 'ACCEPTING' ? 'live' : ''}`}/><span>{system?.worker_state === 'ACCEPTING' ? 'Worker đang nhận việc' : system?.worker_state === 'DRAINING' ? 'Đang lưu checkpoint…' : 'Có thể tắt worker'}</span>
        <button className="text-button" onClick={() => void action(async () => { await post('/auth/logout'); setAuthenticated(false); })}>Đăng xuất</button></div>
    </aside>
    <main className="workspace">
      <header className="topbar"><span>Workspace / <strong>Thư viện</strong></span><span className="version">CORE 0.1</span></header>
      <div className="content">
        <div className="page-heading"><div><p className="eyebrow">TRUNG → VIỆT</p><h1>Xưởng lồng tiếng</h1><p>Quản lý tập phim, hàng đợi và checkpoint trong một workspace.</p></div>
          <div className="toolbar"><button disabled={busy} onClick={() => void exportWorkspace()}>Xuất workspace</button><button className="primary" onClick={() => setNewSeries(true)}>+ Tạo Series</button></div></div>
        {error && <div className="error banner" role="alert">{error}<button aria-label="Đóng lỗi" onClick={() => setError('')}>×</button></div>}
        {notice && <div className="notice banner" role="status">{notice}<button aria-label="Đóng thông báo" onClick={() => setNotice('')}>×</button></div>}
        <section className="system-strip" aria-label="Tình trạng hệ thống">
          <div><span>GPU / VRAM TRỐNG</span><strong>{system?.gpu.available ? `${system.gpu.free_mb} MB` : 'Chưa có GPU'}</strong><small>{system?.gpu.name ?? 'Core chạy được trên CPU'}</small></div>
          <div><span>RAM / DISK TRỐNG</span><strong>{system ? `${bytes(system.ram.free_bytes)} / ${bytes(system.disk.free_bytes)}` : '—'}</strong><small>FFprobe: {system?.ffprobe_available ? 'đã tìm thấy' : 'chưa tìm thấy'}</small></div>
          <div><span>CHI PHÍ ƯỚC TÍNH PHIÊN</span><strong>{money(system?.cost.projected_vnd ?? 0)}</strong><small>Theo uptime core · {money(system?.cost.cloud_rate_vnd_per_hour ?? 6000)}/giờ</small></div>
          <div><span>WORKER</span><strong>{system?.worker_state === 'ACCEPTING' ? 'Sẵn sàng' : system?.worker_state === 'DRAINING' ? 'Đang drain' : 'Đã drain'}</strong>
            <button className="text-button" disabled={busy || system?.worker_state === 'DRAINING'} onClick={() => void action(() => post(system?.worker_state === 'ACCEPTING' ? '/worker/drain' : '/worker/resume'))}>{system?.worker_state === 'ACCEPTING' ? 'Drain worker →' : 'Tiếp tục worker →'}</button></div>
        </section>
        <div className="foundation"><span className="foundation-tag">PIPELINE</span><p>Xử lý theo từng bước. Tự nhận diện vùng phụ đề. Lượt thử xuất video kèm báo cáo để bạn đánh giá.</p><span>{allEpisodes.length} tập</span></div>
        {!active ? <section className="empty-state"><div className="empty-number">01</div><h2>Bắt đầu với một Series</h2><p>Gom các tập cùng phim để dùng chung tên nhân vật,<br/>thuật ngữ và vùng phụ đề.</p><button className="primary" onClick={() => setNewSeries(true)}>Tạo Series đầu tiên</button></section> :
          <section className="series-panel"><div className="series-title"><div><p className="eyebrow">SERIES ĐANG CHỌN</p><h2>{active.title}</h2></div>
            <div className="series-actions"><label>Ưu tiên <input aria-label="Ưu tiên Series" type="number" min="0" max="10000" key={`${active.id}:${active.priority}`} defaultValue={active.priority} onBlur={e => { const value = Number(e.target.value); if (value !== active.priority) void action(() => api(`/series/${active.id}`, { method: 'PATCH', body: JSON.stringify({ priority: value }) })); }}/></label>
              <button className="danger" disabled={busy} onClick={() => { if (confirm(`Xóa Series “${active.title}” và toàn bộ tệp của Series?`)) void action(async () => { await api(`/series/${active.id}`, { method: 'DELETE' }); select(null); }); }}>Xóa Series</button></div></div>
            <div className="tabs"><button className={tab === 'episodes' ? 'current' : ''} onClick={() => setTab('episodes')}>Tập phim <span>{active.episodes.length}</span></button><button className={tab === 'glossary' ? 'current' : ''} onClick={() => setTab('glossary')}>Tên & thuật ngữ <span>{active.glossary.length}</span></button></div>
            {tab === 'episodes' ? <>
              <div className="queue-tools"><p>Số ưu tiên nhỏ chạy trước · tập phim theo thứ tự</p><button disabled={busy || system?.worker_state !== 'ACCEPTING' || !active.episodes.some(e => e.status === 'QUEUED' && !e.queue_requested)} onClick={() => void startAll()}>Bắt đầu hàng đợi</button></div>
              <label className="output-choice">Kết quả<select value={subtitleMode} onChange={e => setSubtitleMode(e.target.value as typeof subtitleMode)}><option value="replace">Lồng tiếng · xóa phụ đề Trung · Vietsub</option><option value="burn">Lồng tiếng · thêm Vietsub</option><option value="off">Chỉ lồng tiếng</option></select></label>
              <div className={`drop-zone ${busy ? 'disabled' : ''}`} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); void uploadFiles(e.dataTransfer.files); }}>
                <span className="upload-mark">↑</span><div><strong>{uploadProgress ? uploadProgress.name : 'Kéo tập phim vào đây'}</strong><p>{uploadProgress ? `${uploadProgress.percent.toFixed(0)}% · ${uploadProgress.mbps.toFixed(1)} Mbps đo được` : 'MP4, MKV, MOV, WEBM · chọn lại cùng tệp để tiếp tục upload'}</p></div>
                <button disabled={busy} onClick={() => input.current?.click()}>Chọn video</button>
                <input ref={input} type="file" accept=".mp4,.mkv,.mov,.webm,.avi,.m4v" multiple hidden onChange={e => e.target.files && void uploadFiles(e.target.files)}/>
                {uploadProgress && <progress max="100" value={uploadProgress.percent} aria-label="Tiến độ upload"/>}
              </div>
              <div className="table-scroll"><table><thead><tr><th>TẬP / TỆP NGUỒN</th><th>THỜI LƯỢNG</th><th>TRẠNG THÁI</th><th>TIẾN ĐỘ</th><th><span className="sr-only">Thao tác</span></th></tr></thead><tbody>
                {active.episodes.map(episode => <tr key={episode.id}><td><div className="episode-name"><span>{String(episode.ordinal).padStart(2, '0')}</span><div><strong>{episode.filename}</strong><small>{bytes(episode.total_bytes)}{episode.issues.filter(i => !i.resolved).map(i => <span className="issue" key={i.id}>{i.details.message ?? i.code}</span>)}</small></div></div></td><td>{duration(episode.duration_ms)}</td>
                  <td><span className={`badge ${episode.status.toLowerCase()}`}>{episode.queue_requested && episode.status === 'QUEUED' ? 'Trong hàng đợi' : labels[episode.status] ?? episode.status}</span>{episode.next_stage && <small className="next-stage">Tiếp theo: {episode.next_stage}</small>}</td>
                  <td><div className="progress-cell"><progress max="1" value={episode.status === 'UPLOADING' ? episode.uploaded_bytes / episode.total_bytes : episode.progress}/><small>{Math.round((episode.status === 'UPLOADING' ? episode.uploaded_bytes / episode.total_bytes : episode.progress) * 100)}%</small></div></td>
                  <td><div className="row-actions">{episode.status === 'NEEDS_REVIEW' && episode.issues.some(i => i.code === 'ROI_REQUIRED' && !i.resolved) && <button disabled={busy} onClick={() => setRoiReview(episode)}>Chọn vùng phụ đề</button>}{episode.status === 'NEEDS_REVIEW' && <button disabled={busy} onClick={() => setReview(episode)}>Sửa lời thoại</button>}{episode.artifacts.some(a => a.kind === 'final_video' || a.kind === 'preview_video') && <button onClick={() => { setSource(episode); setPlayPreview(true); }}>Xem bản Việt</button>}{((episode.status === 'QUEUED' && !episode.queue_requested) || episode.status === 'CHECKPOINTED') && <button disabled={busy || system?.worker_state !== 'ACCEPTING'} onClick={() => void action(() => post(`/episodes/${episode.id}/start`))}>{episode.status === 'CHECKPOINTED' ? 'Lồng tiếng' : 'Bắt đầu'}</button>}
                    {episode.status === 'FAILED' && <button disabled={busy} onClick={() => void action(() => post(`/episodes/${episode.id}/retry`))}>Thử lại</button>}
                    {episode.status !== 'UPLOADING' && <button onClick={() => { setSource(episode); setPlayPreview(false); }}>Xem nguồn</button>}
                    {episode.artifacts.map(artifact => <a className="button-link" key={artifact.id} href={`/api/download/${artifact.id}`} download>{artifact.kind === 'checkpoint' ? 'Checkpoint' : artifact.kind === 'final_video' ? 'Video Final' : artifact.kind === 'preview_audio' ? 'Audio Mix' : 'Tải xuống'}</a>)}
                    <button className="text-button muted" aria-label={`Xóa tập ${episode.ordinal}`} disabled={busy || ['V1_RUNNING', 'V2_RUNNING', 'PREPARING', 'ASR', 'ALIGNING', 'TRANSLATING', 'SEPARATING', 'TTS', 'TIMING', 'AUDIO_MIX', 'QC', 'OCR_VERIFY', 'TEXT_REMOVAL', 'SUBTITLE_RENDER', 'ENCODING'].includes(episode.status)} onClick={() => { if (confirm(`Xóa tập ${episode.ordinal} và tệp đã upload?`)) void action(() => api(`/episodes/${episode.id}`, { method: 'DELETE' })); }}>×</button></div></td></tr>)}
              </tbody></table>{!active.episodes.length && <p className="table-empty">Chưa có tập phim. Tải lên video đầu tiên để bắt đầu.</p>}</div>
            </> : <div className="glossary"><p>Thuật ngữ lưu theo Series. Sửa thủ công sẽ khóa tên; tập cũ chỉ được dịch lại khi bạn yêu cầu.</p><form onSubmit={saveTerm}><div><label htmlFor="zh">Tên / thuật ngữ tiếng Trung</label><input id="zh" value={zh} onChange={e => setZh(e.target.value)} required maxLength={200}/></div><div><label htmlFor="vi">Cách dịch tiếng Việt</label><input id="vi" value={vi} onChange={e => setVi(e.target.value)} required maxLength={300}/></div><button className="primary" disabled={busy}>Lưu thuật ngữ</button></form>
              <table><thead><tr><th>TIẾNG TRUNG</th><th>TIẾNG VIỆT</th><th>NGUỒN</th><th/></tr></thead><tbody>{active.glossary.map(term => <tr key={term.zh}><td>{term.zh}</td><td>{term.vi}</td><td>{term.locked_by_user ? 'Đã khóa bởi bạn' : 'Đề xuất model'}</td><td><button onClick={() => { setZh(term.zh); setVi(term.vi); }}>Sửa</button></td></tr>)}</tbody></table>{!active.glossary.length && <p className="table-empty">Chưa có thuật ngữ.</p>}</div>}
          </section>}
        <footer><span>Video truyền trực tiếp vào workspace · tệp được giữ đến khi bạn xóa</span><span>CN2VI AUTODUB</span></footer>
      </div>
    </main>
    {newSeries && <div className="modal-backdrop" onClick={() => setNewSeries(false)}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="new-title" onClick={e => e.stopPropagation()}><button className="close" aria-label="Đóng" onClick={() => setNewSeries(false)}>×</button><p className="eyebrow">THƯ VIỆN</p><h2 id="new-title">Tạo Series</h2><form onSubmit={createSeries}><label htmlFor="series-title">Tên phim / Series</label><input id="series-title" value={title} onChange={e => setTitle(e.target.value)} required maxLength={160} autoFocus/><label htmlFor="series-priority">Ưu tiên</label><input id="series-priority" type="number" min="0" max="10000" value={priority} onChange={e => setPriority(Number(e.target.value))}/><p>Số nhỏ hơn được xử lý trước.</p><button className="primary" disabled={busy}>Tạo Series</button></form></section></div>}
    {roiReview && <RoiReview episode={roiReview} onClose={() => setRoiReview(null)} onSaved={refresh}/> }
    {review && <DialogueReview episode={review} onClose={() => setReview(null)} onSaved={refresh}/> }
    {source && <div className="modal-backdrop" onClick={() => setSource(null)}><section className="modal video-modal" role="dialog" aria-modal="true" aria-labelledby="source-title" onClick={e => e.stopPropagation()}><button className="close" aria-label="Đóng video" onClick={() => setSource(null)}>×</button><h2 id="source-title">{source.filename}</h2><p>{playPreview ? 'Bản lồng tiếng Việt · cần nghe kiểm tra chất lượng.' : 'Video nguồn để đối chiếu.'}</p><video src={`/api/episodes/${source.id}/${playPreview ? 'preview' : 'source'}`} controls autoPlay playsInline/></section></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<StrictMode><App/></StrictMode>);
