"use strict";
const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("roi"), ctx = canvas.getContext("2d");
let regions = [], drag = null, metadata = null, ready = false, busy = false;
let uploaded = null, objectURL = null, uploadReady = false, polling = null, currentJob = null;
let events = [], eventLimit = 100;

async function api(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
function clock(seconds) {
  const s = Math.max(0, seconds || 0);
  return `${Math.floor(s / 60).toString().padStart(2,"0")}:${Math.floor(s % 60).toString().padStart(2,"0")}`;
}
function updateControls() {
  $("confirm").disabled = !regions.length || busy;
  $("clear").disabled = !regions.length || busy;
  $("run").disabled = !(ready && uploadReady && regions.length && $("confirm").checked) || busy;
  $("export").disabled = !uploaded || !regions.length || !$("confirm").checked || busy;
  $("file").disabled = busy;
  $("start").disabled = $("end").disabled = !uploaded || busy;
  $("regions").querySelectorAll("button").forEach(button=>{button.disabled=busy;});
  canvas.style.pointerEvents = busy ? "none" : "auto";
}
function draw() {
  const bounds = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * dpr);
  canvas.height = Math.round(bounds.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, bounds.width, bounds.height);
  const all = drag ? [...regions, rect(drag.a, drag.b)] : regions;
  all.forEach((r, i) => {
    const x = r.x * bounds.width, y = r.y * bounds.height;
    const w = r.width * bounds.width, h = r.height * bounds.height;
    ctx.fillStyle = "rgba(100,199,171,.25)";
    ctx.strokeStyle = "#64c7ab"; ctx.lineWidth = 2;
    ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
    ctx.fillStyle = "#e8fff8"; ctx.font = "12px Segoe UI,Arial";
    ctx.fillText(`Bỏ qua ${i + 1}`, x + 5, Math.min(bounds.height - 4, y + 15));
  });
}
function point(event) {
  const b = canvas.getBoundingClientRect();
  return {x:Math.max(0,Math.min(1,(event.clientX-b.left)/b.width)),
          y:Math.max(0,Math.min(1,(event.clientY-b.top)/b.height))};
}
function rect(a,b) {
  const x=Math.min(a.x,b.x), y=Math.min(a.y,b.y);
  return {x,y,width:Math.min(1-x,Math.abs(a.x-b.x)),height:Math.min(1-y,Math.abs(a.y-b.y))};
}
function changedRegions() {
  $("confirm").checked = false;
  $("regions").replaceChildren();
  if (!regions.length) {
    const p=document.createElement("p"); p.className="muted"; p.textContent="Chưa khoanh vùng";
    $("regions").append(p);
  }
  regions.forEach((r,i) => {
    const row=document.createElement("div"); row.className="region";
    const label=document.createElement("span");
    label.textContent=`Vùng ${i+1} · ${(r.width*100).toFixed(1)}% × ${(r.height*100).toFixed(1)}%`;
    const remove=document.createElement("button"); remove.textContent="Xóa"; remove.disabled=busy;
    remove.onclick=()=>{if(!busy){regions.splice(i,1);changedRegions();}};
    row.append(label,remove); $("regions").append(row);
  });
  draw();updateControls();
}
canvas.addEventListener("pointerdown", e => {
  if (!uploaded || busy) return;
  if (regions.length>=16) return message("Tối đa 16 vùng bỏ qua.",true);
  video.pause(); canvas.setPointerCapture(e.pointerId); drag={a:point(e),b:point(e)};draw();
});
canvas.addEventListener("pointermove",e=>{if(drag){drag.b=point(e);draw();}});
canvas.addEventListener("pointerup",e=>{
  if(!drag)return;
  const r=rect(drag.a,point(e));drag=null;
  if(r.width*video.videoWidth>=2 && r.height*video.videoHeight>=2)regions.push(r);
  changedRegions();
});
canvas.addEventListener("pointercancel",()=>{drag=null;draw();});
new ResizeObserver(draw).observe($("preview"));
$("clear").onclick=()=>{regions=[];changedRegions();};
$("confirm").onchange=updateControls;
$("play").onclick=()=>video.paused?video.play().catch(e=>message(e.message,true)):video.pause();
$("seek").oninput=()=>{video.currentTime=Number($("seek").value);};
video.addEventListener("timeupdate",()=>{
  $("seek").value=video.currentTime;$("clock").textContent=`${clock(video.currentTime)} / ${clock(video.duration)}`;
});
video.addEventListener("loadedmetadata",()=>{
  $("preview").style.aspectRatio=`${video.videoWidth} / ${video.videoHeight}`;
  $("empty").hidden=true;$("empty").style.display="none";
  $("seek").max=video.duration;$("seek").disabled=$("play").disabled=false;draw();
});
video.addEventListener("error",()=>message("Trình duyệt không xem được video này. Bản mẫu cần MP4/H.264 để khoanh đúng vùng.",true));

$("file").onchange=async()=>{
  const file=$("file").files[0];if(!file)return;
  ready=false;uploadReady=false;uploaded=null;regions=[];changedRegions();$("results").hidden=true;
  $("filename").textContent=file.name;
  if(objectURL)URL.revokeObjectURL(objectURL);
  objectURL=URL.createObjectURL(file);
  const loaded=new Promise((resolve,reject)=>{
    video.addEventListener("loadedmetadata",resolve,{once:true});
    video.addEventListener("error",()=>reject(new Error("Không xem được video MP4 này.")),{once:true});
  });
  video.src=objectURL;
  busy=true;updateControls();message("Đang mở video…");
  try {
    await loaded;
    const status=await api("/api/status");
    if(!status.analysis_enabled){
      metadata={id:null,name:file.name,media:{duration_s:video.duration,width:video.videoWidth,height:video.videoHeight}};
      uploaded=metadata;
      $("start").value=0;$("end").value=Math.min(30,video.duration);
      $("start").max=$("end").max=video.duration;
      message("Video được xem trực tiếp trên máy bạn. Khoanh sub, xác nhận rồi lưu vùng JSON. OCR chưa bật ở đây.");
      return;
    }
    message("Đang tải video lên máy thực hiện…");
    metadata=await api(`/api/videos?name=${encodeURIComponent(file.name)}`,{method:"POST",body:file});
    uploaded=metadata;uploadReady=true;
    $("start").value=0;$("end").value=Math.min(30,metadata.media.duration_s);
    $("start").max=$("end").max=metadata.media.duration_s;
    ready=status.analysis_enabled&&status.models_configured;
    message(ready?"Video đã tải lên. Khoanh vùng sub rồi xác nhận để bắt đầu.":"Video đã tải lên. OCR chưa được bật trên máy thực hiện.");
  }catch(e){message(e.message,true);}
  finally{busy=false;updateControls();}
};

$("export").onclick=()=>{
  const start=Number($("start").value),end=Number($("end").value);
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>metadata.media.duration_s){
    message("Chọn khoảng thời gian hợp lệ trước khi lưu vùng.",true);return;
  }
  const value={roi_confirmed:true,start_s:start,end_s:end,
    exclusions:regions.map(r=>({...r,start_s:0,end_s:metadata.media.duration_s}))};
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json"}));
  const link=document.createElement("a");link.href=url;link.download="vnle-request.json";link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};

function showEvents() {
  $("events").replaceChildren();
  for(const event of events.slice(0,eventLimit)){
    const li=document.createElement("li");
    const image=document.createElement("img");image.alt="Ảnh chữ đã nhận diện";image.loading="lazy";
    image.src=`/artifacts/${currentJob}/evidence/${event.id}.png`;
    const body=document.createElement("div"),text=document.createElement("p");
    text.className="text";text.textContent=event.text||"[Có vùng chữ nhưng chưa đọc được]";
    const detail=document.createElement("p");detail.className="details";
    detail.textContent=`${event.start_s.toFixed(3)}–${(event.end_s??event.last_seen_s).toFixed(3)}s · ${event.observations.length} mẫu · ${event.id}`;
    const jump=document.createElement("button");jump.textContent="Xem trên video";
    jump.onclick=()=>{video.pause();video.currentTime=event.geometry_keyframes[0].time_s;};
    body.append(text,detail,jump);
    if(event.review_reasons.length){
      const note=document.createElement("p");note.className="details review";
      note.textContent=`Cần xem lại: ${event.review_reasons.join(", ")}`;body.append(note);
    }
    li.append(image,body);$("events").append(li);
  }
  if(events.length>eventLimit){
    const li=document.createElement("li");li.className="pagination";
    const more=document.createElement("button");more.textContent=`Xem thêm (${events.length-eventLimit} mục còn lại)`;
    more.onclick=()=>{eventLimit+=100;showEvents();};li.append(more);$("events").append(li);
  }
}
async function poll() {
  try{
    const job=await api(`/api/runs/${currentJob}`),p=job.progress||{};
    const stages={INPUT_AND_MODEL_HASH:"Kiểm tra video và model",MODEL_LOAD:"Nạp model",ANALYZING:"Đang đọc chữ"};
    $("run-state").textContent=`${stages[p.stage]||p.stage||job.status} · ${p.events||0} sự kiện · ${(p.wall_s||0).toFixed(1)} giây`;
    const start=Number($("start").value),end=Number($("end").value);
    $("progress").value=Math.min(99,Math.max(0,100*((p.time_s||0)-start)/(end-start)));
    if(job.status==="RUNNING"){polling=setTimeout(poll,1500);return;}
    busy=false;$("cancel").hidden=true;updateControls();
    $("results").hidden=false;$("downloads").replaceChildren();
    for(const [name,label] of [["events.json","Sự kiện JSON"],["run-report.json","Báo cáo chạy"]]){
      const link=document.createElement("a");link.href=`/artifacts/${currentJob}/${name}`;
      link.textContent=label;link.download=name;$("downloads").append(link);
    }
    try{const result=await api(`/artifacts/${currentJob}/events.json`);events=result.events;eventLimit=100;showEvents();}catch{events=[];showEvents();}
    const done=job.status==="COMPLETED_UNVERIFIED";
    $("progress").value=done?100:$("progress").value;
    $("result-note").textContent=`${events.length} sự kiện. Khoảng thời gian được ước lượng từ các mẫu; độ chính xác cần bạn xem lại.`;
    message(done?"Đã phân tích xong. Xem chữ và ảnh chứng cứ phía dưới.":`${job.status}: ${job.error||"Kết quả hiện có được giữ lại."}`,!done);
  }catch(e){
    busy=false;$("cancel").hidden=true;updateControls();message(`Mất kết nối theo dõi: ${e.message}. Lượt chạy không được tự khởi động lại.`,true);
  }
}
$("run").onclick=async()=>{
  if(!uploaded)return;
  const start=Number($("start").value),end=Number($("end").value);
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>metadata.media.duration_s){
    message("Chọn khoảng thời gian hợp lệ trong video.",true);return;
  }
  busy=true;video.pause();updateControls();$("results").hidden=true;
  try{
    const job=await api("/api/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      video_id:uploaded.id,roi_confirmed:$("confirm").checked,start_s:start,end_s:end,
      exclusions:regions.map(r=>({...r,start_s:0,end_s:metadata.media.duration_s}))
    })});
    currentJob=job.id;$("cancel").hidden=false;$("progress").hidden=false;$("progress").value=0;
    message("Đang phân tích. Vùng sub đã được khóa cho lượt này.");await poll();
  }catch(e){busy=false;updateControls();message(e.message,true);}
};
$("cancel").onclick=async()=>{
  try{await api(`/api/runs/${currentJob}/cancel`,{method:"POST"});message("Đã yêu cầu dừng. Chờ OCR hoàn tất lượt hiện tại để lưu kết quả.");}
  catch(e){message(e.message,true);}
};
async function connect() {
  try {
    const status=await api("/api/status");
    $("server-state").textContent=status.analysis_enabled?(status.models_configured?"Máy thực hiện đã cấu hình":"Chưa cấu hình model OCR"):"Chế độ xem giao diện";
    const params=new URLSearchParams(location.search),videoId=params.get("video"),runId=params.get("run");
    if(!videoId || !runId)return;
    if(!/^[a-f0-9]{32}$/.test(videoId)||!/^[a-f0-9]{32}$/.test(runId))throw new Error("Đường dẫn kết quả không hợp lệ.");
    const [info,job,report]=await Promise.all([
      api(`/api/videos/${videoId}`),api(`/api/runs/${runId}`),api(`/artifacts/${runId}/run-report.json`)
    ]);
    if(job.video_id && job.video_id!==videoId)throw new Error("Video không thuộc lượt chạy này.");
    metadata=uploaded=info;uploadReady=true;ready=status.analysis_enabled&&status.models_configured;
    regions=report.request.exclusions.map(r=>({...r.rect}));
    changedRegions();$("confirm").checked=true; // A stored Request exists only after explicit confirmation.
    $("filename").textContent=info.name;$("start").value=report.request.start_s;$("end").value=report.request.end_s;
    $("start").max=$("end").max=info.media.duration_s;
    currentJob=runId;busy=job.status==="RUNNING";
    $("progress").hidden=false;$("cancel").hidden=!busy;updateControls();
    video.src=`/media/${videoId}`;await poll();
  }catch(e){message(e.message,true);}
}
connect();
