/* Snake AI UI — with saving indicator + Close Project (FULL, FIXED) */
(() => {
  const CONFIG = window.__CONFIG__ || {
    API_URL: "http://127.0.0.1:5000",
    WS_URL: "ws://127.0.0.1:5000/ws/stream"
  };

  const el = {
    canvas: document.getElementById("gameCanvas"),
    overlayWin: document.getElementById("overlayWin"),
    overlayLoss: document.getElementById("overlayLoss"),
    lossReason: document.getElementById("lossReason"),

    viewNone: document.getElementById("viewToggleNone"),
    viewNeural: document.getElementById("viewToggleNeural"),
    viewGraphs: document.getElementById("viewToggleGraphs"),
    nnPanel: document.getElementById("nnVisualization"),
    graphsPanel: document.getElementById("graphsPanel"),

    algoSelect: document.getElementById("algoSelect"),
    gridSize: document.getElementById("gridSizeSelect"),

    start: document.getElementById("startBtn"),
    pause: document.getElementById("pauseBtn"),
    resume: document.getElementById("resumeBtn"),
    stop: document.getElementById("stopBtn"),
    reset: document.getElementById("resetBtn"),
    save: document.getElementById("saveBtn"),
    load: document.getElementById("loadBtn"),
    close: document.getElementById("closeBtn"),
    status: document.getElementById("runStatus"),

    epsSlider: document.getElementById("epsilonSlider"),
    lrSlider: document.getElementById("lrSlider"),
    gammaSlider: document.getElementById("gammaSlider"),
    batchSlider: document.getElementById("batchSlider"),
    speedSlider: document.getElementById("speedSlider"),
    epsVal: document.getElementById("epsilonVal"),
    lrVal: document.getElementById("lrVal"),
    gammaVal: document.getElementById("gammaVal"),
    batchVal: document.getElementById("batchVal"),
    speedVal: document.getElementById("speedVal"),

    tableBody: document.getElementById("analysisTable").querySelector("tbody"),
    scoreChartEl: document.getElementById("scoreChart"),
    epsilonChartEl: document.getElementById("epsilonChart"),
    lossChartEl: document.getElementById("lossChart"),
    nnSvg: document.getElementById("nnSvg"),

    llmPanel: document.getElementById("llmPanel"),
    llmFeed: document.getElementById("llmFeed"),

    saveIndicator: document.getElementById("saveIndicator"),
  };

  const fmt = (v)=> (v===undefined||v===null ? "" : Number(v).toFixed(4));
  const debounce = (fn, ms=150)=>{ let t; return (...args)=>{ clearTimeout(t); t=setTimeout(()=>fn(...args), ms); }; };

  // --- small helper used by sliders ---
  async function postConfigUpdate(patch){
    try { return await restPost("/api/config", patch); }
    catch (e) { console.warn(e); return { ok:false, error:String(e) }; }
  }

  // Canvas/grid
  const ctx = el.canvas.getContext("2d");
  const board = { w: 20, h: 20 };
  const gridCache = document.createElement("canvas");
  function buildGridCache(){
    gridCache.width = el.canvas.width;
    gridCache.height = el.canvas.height;
    const g = gridCache.getContext("2d");
    const W = gridCache.width, H = gridCache.height;
    const cw = W / board.w, ch = H / board.h;
    g.fillStyle = "#000"; g.fillRect(0,0,W,H);
    g.strokeStyle = "rgba(255,255,255,0.18)"; g.lineWidth = 1;
    for (let x=0;x<=board.w;x++){ g.beginPath(); g.moveTo(x*cw,0); g.lineTo(x*cw,H); g.stroke(); }
    for (let y=0;y<=board.h;y++){ g.beginPath(); g.moveTo(0,y*ch); g.lineTo(0+W,y*ch); g.stroke(); }
  }
  const drawGrid = ()=> ctx.drawImage(gridCache, 0, 0);

  function renderStateFrame(frame){
    if (!frame || !frame.grid) return;
    const W=el.canvas.width, H=el.canvas.height;
    const cw = W / frame.grid.w, ch = H / frame.grid.h;
    drawGrid();

    // be robust to {x,y} or [x,y]
    let food = frame.grid.food;
    if (Array.isArray(food)) food = { x: food[0], y: food[1] };
    if (food && Number.isFinite(food.x) && Number.isFinite(food.y)) {
      ctx.fillStyle = "#f2c94c";
      ctx.fillRect(food.x*cw+1, food.y*ch+1, cw-2, ch-2);
    }

    // snake: array of {x,y} or array of [x,y]
    if (Array.isArray(frame.grid.snake)) {
      frame.grid.snake.forEach((seg, idx) => {
        const x = Array.isArray(seg) ? seg[0] : seg.x;
        const y = Array.isArray(seg) ? seg[1] : seg.y;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        ctx.fillStyle = idx===0 ? "#00ffc6" : "#1e90ff";
        ctx.fillRect(x*cw+1, y*ch+1, cw-2, ch-2);
      });
    }
  }

  const showWinOverlay = ()=>{ el.overlayWin.classList.add("show"); el.overlayWin.setAttribute("aria-hidden","false"); el.overlayLoss.classList.remove("show"); el.overlayLoss.setAttribute("aria-hidden","true"); };
  const hideOverlays  = ()=>{ el.overlayWin.classList.remove("show"); el.overlayLoss.classList.remove("show"); el.overlayWin.setAttribute("aria-hidden","true"); el.overlayLoss.setAttribute("aria-hidden","true"); };

  /* ===== Charts ===== */
  let charts = { score:null, epsilon:null, loss:null };
  let chartsInitialized = false;

  function initCharts() {
    if (!window.Chart) return;

    const common = {
      type: "line",
      options: {
        responsive: true,
        animation: false,
        maintainAspectRatio: true,
        aspectRatio: 8,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { labels: { color: "#e6f1ff" } },
          tooltip: {
            enabled: true,
            callbacks: {
              title: (items) => (items[0] ? `Episode ${items[0].label}` : ""),
              label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(3)}`
            }
          }
        },
        scales: {
          x: { ticks: { color: "#a1acc4" }, grid: { color: "rgba(255,255,255,0.08)" } },
          y: { ticks: { color: "#a1acc4" }, grid: { color: "rgba(255,255,255,0.08)" } }
        }
      }
    };

    charts.score = new Chart(el.scoreChartEl.getContext("2d"), {
      ...common,
      data: { labels: [], datasets: [{
        label: "Score", data: [],
        borderColor: "#17c964", backgroundColor: "rgba(23,201,100,.18)",
        borderWidth: 2, pointRadius: 0, tension: 0.2, fill: true
      }]}
    });

    charts.epsilon = new Chart(el.epsilonChartEl.getContext("2d"), {
      ...common,
      data: { labels: [], datasets: [{
        label: "Epsilon", data: [],
        borderColor: "#7aa2f7", backgroundColor: "rgba(122,162,247,.20)",
        borderWidth: 2, pointRadius: 0, tension: 0.2, fill: true
      }]}
    });

    charts.loss = new Chart(el.lossChartEl.getContext("2d"), {
      ...common,
      data: { labels: [], datasets: [{
        label: "Avg Loss", data: [],
        borderColor: "#ffd166", backgroundColor: "rgba(255,209,102,.25)",
        borderWidth: 2, pointRadius: 0, tension: 0.2, fill: true
      }]}
    });

    chartsInitialized = true;
  }

  function chartAppend(chart,x,y){
    if(!chart) return;
    chart.data.labels.push(x);
    chart.data.datasets[0].data.push(y);
    if(chart.data.labels.length>1000){ chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
    chart.update("none");
  }
  function ensureCharts() {
    if (!chartsInitialized) {
      initCharts();
    } else {
      charts.score && charts.score.resize();
      charts.epsilon && charts.epsilon.resize();
      charts.loss && charts.loss.resize();
    }
  }
  function fmtDuration(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n <= 0) return "";
  const totalSec = n / 1000;
  if (totalSec < 60) return `${totalSec.toFixed(1)}s`;
  const m = Math.floor(totalSec / 60);
  const s = Math.floor(totalSec - m * 60);
  return `${m}m ${s}s`;
}

  // ---- EPISODE TABLE (keep latest 1000) ----
  let episodes = [];
  function renderEpisodeTable(rows){
  if (!el.tableBody) return;
  const frag = document.createDocumentFragment();
  // newest first
  for (let i = rows.length - 1; i >= 0; i--) {
    const r = rows[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.episode ?? ""}</td>
      <td>${r.outcome ?? ""}</td>
      <td>${r.reason ?? ""}</td>
      <td>${r.score ?? ""}</td>
      <td>${r.snake_length ?? ""}</td>
      <td>${fmt(r.epsilon)}</td>
      <td>${fmt(r.avg_loss)}</td>
      <td>${r.steps ?? ""}</td>
      <td>${fmtDuration(r.duration_ms)}</td>
      <td>${r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : ""}</td>`;
    frag.appendChild(tr);
  }
  el.tableBody.innerHTML = "";
  el.tableBody.appendChild(frag);
}

  // ---- NN (static sizes, with blink) ----
  const INPUT_LABELS = [
    "dir_up","dir_down","dir_left","dir_right",
    "danger_fwd","danger_left","danger_right",
    "food_dx","food_dy",
    "head_x_norm","head_y_norm","food_x_norm","food_y_norm",
    "length_ratio","score_norm","bias"
  ];
  const NN_LAYERS = [16,12,8,4];
  const NN = { prevActs:[new Float32Array(16), new Float32Array(12), new Float32Array(8), new Float32Array(4)] };

  function drawNeuralDiagram(){
    const svg = d3.select(el.nnSvg); svg.selectAll("*").remove();
    const width = el.nnSvg.clientWidth || 800, height = el.nnSvg.clientHeight || 360;
    const layerSpacing = width / (NN_LAYERS.length + 1);

    const nodes=[], nodeMap={};
    NN_LAYERS.forEach((count, l)=>{
      const ySpacing = height / (count + 1);
      for (let i=0;i<count;i++){
        const n={id:`${l}-${i}`, layer:l, idx:i, x:(l+1)*layerSpacing, y:(i+1)*ySpacing};
        nodes.push(n); nodeMap[n.id]=n;
      }
    });

    const links=[];
    for (let l=0;l<NN_LAYERS.length-1;l++){
      for (let i=0;i<NN_LAYERS[l];i++){
        for (let j=0;j<NN_LAYERS[l+1];j++){
          links.push({source:`${l}-${i}`, target:`${l+1}-${j}`, tLayer:l+1, tIdx:j});
        }
      }
    }

    const linkSel = svg.selectAll("line.link").data(links).enter().append("line")
      .attr("class","link")
      .attr("x1",d=>nodeMap[d.source].x).attr("y1",d=>nodeMap[d.source].y)
      .attr("x2",d=>nodeMap[d.target].x).attr("y2",d=>nodeMap[d.target].y);

    const nodeSel = svg.selectAll("circle.node").data(nodes).enter().append("circle")
      .attr("class", d => d.layer===NN_LAYERS.length-1 ? "node output" : "node")
      .attr("cx",d=>d.x).attr("cy",d=>d.y).attr("r",8);

    // input labels
    svg.selectAll("text.input-label").data(nodes.filter(n=>n.layer===0))
      .enter().append("text").attr("class","node-label input-label")
      .attr("x", d=>d.x - 14).attr("y", d=>d.y + 4).attr("text-anchor","end")
      .text((d,i)=> INPUT_LABELS[i] ?? `in_${i}`);

    // output labels + Q values
    const outNames = ["Up","Down","Left","Right"];
    const lastL = NN_LAYERS.length-1;
    svg.selectAll("text.node-label.output-label").data(nodes.filter(n=>n.layer===lastL))
      .enter().append("text").attr("class","node-label output-label")
      .attr("x", d=>d.x + 14).attr("y", d=>d.y - 6)
      .text((d,i)=> outNames[i] ?? `out_${i}`);

    const outputQText = svg.selectAll("text.output-q").data(nodes.filter(n=>n.layer===lastL))
      .enter().append("text").attr("class","output-q")
      .attr("x", d=>d.x + 14).attr("y", d=>d.y + 10)
      .text("-");

    // store
    NN.nodeSel = nodeSel;
    NN.linkSel = linkSel;
    NN.outputQText = outputQText;
    NN.prevActs = NN_LAYERS.map(count => new Float32Array(count));
    NN.targetLinksByLayer = Array.from({length: NN_LAYERS.length}, (_, l)=>
      l===0 ? null : Array.from({length: NN_LAYERS[l]}, ()=>[])
    );
    links.forEach((L, idx)=> NN.targetLinksByLayer[L.tLayer][L.tIdx].push(idx));
  }

  function fitToCount(arr, n){
    const out = new Float32Array(n);
    if (!Array.isArray(arr) || arr.length === 0) return out;
    const m = arr.length;
    if (m === n) { for (let i=0;i<n;i++) out[i] = Number(arr[i]) || 0; return out; }
    for (let i=0;i<n;i++){
      const idx = Math.min(m-1, Math.round(i * (m-1) / Math.max(1,n-1)));
      out[i] = Number(arr[idx]) || 0;
    }
    return out;
  }

  function applyNN(ev){
    if (!ev || !NN.nodeSel || !NN.linkSel) return;

    const layersMsg = Array.isArray(ev.layers) ? ev.layers : [];
    const hiddenMsg = layersMsg.filter(l => Array.isArray(l.act)).map(l => l.act);
    const qEntry = layersMsg.find(l => Array.isArray(l.q));
    const outRaw = qEntry ? qEntry.q : [];

    const hiddenCounts = NN_LAYERS.slice(1, NN_LAYERS.length-1);
    const actsByLayer = [null];

    for (let li=0; li<hiddenCounts.length; li++){
      actsByLayer.push(fitToCount(hiddenMsg[li] || [], hiddenCounts[li]));
    }
    const outCount = NN_LAYERS[NN_LAYERS.length-1];
    const outActs = fitToCount(outRaw, outCount);
    actsByLayer.push(outActs);

    const th = 0.006;
    const lastIdx = NN_LAYERS.length-1;

    // node blink
    NN.nodeSel.each(function(d){
      if (d.layer === 0) return;
      const a = actsByLayer[d.layer] ? actsByLayer[d.layer][d.idx] : 0;
      const p = (NN.prevActs[d.layer] && NN.prevActs[d.layer][d.idx]) || 0;
      const delta = a - p;
      if (delta > th) { const s=d3.select(this); s.classed("blink-green",true); setTimeout(()=>s.classed("blink-green",false), 160); }
      else if (delta < -th) { const s=d3.select(this); s.classed("blink-red",true); setTimeout(()=>s.classed("blink-red",false), 160); }
    });

    // link highlight
    const linkElems = NN.linkSel.nodes();
    for (let layerIdx=1; layerIdx<=lastIdx; layerIdx++){
      const layerActs = actsByLayer[layerIdx];
      if (!layerActs) continue;
      const prevLayer = NN.prevActs[layerIdx];
      const table = NN.targetLinksByLayer[layerIdx] || [];
      for (let j=0;j<layerActs.length;j++){
        const a = layerActs[j], p = prevLayer ? prevLayer[j] : 0;
        const delta = a - p;
        if (Math.abs(delta) <= th) continue;
        const indices = table[j] || [];
        for (let k=0;k<indices.length;k++){
          const elLine = d3.select(linkElems[indices[k]]);
          elLine.classed("pos", delta > th).classed("neg", delta < -th);
          const base = 0.35, extra = Math.min(0.65, Math.abs(delta) * 3);
          elLine.style("opacity", base + extra);
        }
      }
    }

    // Q text
    if (NN.outputQText) {
      const nodes = NN.outputQText.nodes();
      for (let i=0;i<Math.min(nodes.length, outActs.length); i++){
        nodes[i].textContent = Number.isFinite(outActs[i]) ? outActs[i].toFixed(3) : "-";
      }
    }

    // commit prevActs
    for (let layerIdx=1; layerIdx<NN.prevActs.length; layerIdx++){
      const arr = actsByLayer[layerIdx];
      if (arr) {
        if (NN.prevActs[layerIdx].length !== arr.length) {
          NN.prevActs[layerIdx] = new Float32Array(arr.length);
        }
        NN.prevActs[layerIdx].set(arr);
      }
    }
  }

  // REST helpers
  async function restPost(path, body){
    const url = `${CONFIG.API_URL}${path}`;
    const res = await fetch(url, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(body||{}) });
    if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
    return await res.json();
  }
  async function restGet(path){
    const url = `${CONFIG.API_URL}${path}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  }

  // LLM feed
  function appendLLMComment(msg){
    if (!el.llmFeed) return;
    const wrapper = document.createElement("div");
    wrapper.className = "llm-msg";
    const meta = document.createElement("div");
    meta.className = "llm-meta";
    meta.textContent = `Episode ${msg.episode} • step ${msg.step}`;
    const body = document.createElement("div");
    body.textContent = msg.text || "";
    wrapper.appendChild(meta);
    wrapper.appendChild(body);
    el.llmFeed.prepend(wrapper);
    while (el.llmFeed.children.length > 50) el.llmFeed.removeChild(el.llmFeed.lastChild);
  }

  // Saving indicator
  function showSaving(){ el.saveIndicator && el.saveIndicator.classList.remove("hidden"); }
  function hideSaving(){ el.saveIndicator && el.saveIndicator.classList.add("hidden"); }

  // WebSocket
  let ws=null, lastStateFrame=null, lastNN=null, lastRenderTs=0;
  function connectWS(){
    try{
      ws = new WebSocket(CONFIG.WS_URL);
      ws.onopen  = ()=> console.log("[WS] connected");
      ws.onerror = (e)=> console.warn("[WS] error", e);
      ws.onclose = ()=> console.log("[WS] closed");
      ws.onmessage = (msg)=>{
        let data; try{ data = JSON.parse(msg.data);}catch{ return; }
        switch(data.type){
          case "state_frame": lastStateFrame = data; break;
          case "nn_activations": lastNN = data; break;
          case "metrics_tick":
            if (el.status) el.status.textContent = `Running (ep ${data.episode}, step ${data.step})`;
            break;

          case "episode_summary": {
            // keep only the most recent 1000 rows in the UI
            episodes.push(data);                       // <-- fixed (was episode.push)
            if (episodes.length > 1000) episodes = episodes.slice(-1000);

            // charts
            ensureCharts();
            chartAppend(charts.score,   data.episode, Number(data.score)||0);
            chartAppend(charts.epsilon, data.episode, Number(data.epsilon)||0);
            chartAppend(charts.loss,    data.episode, Number(data.avg_loss)||0);

            // table render (newest first)
            renderEpisodeTable(episodes);
            break;
          }

          case "game_over":
            if (data.outcome === "WIN") showWinOverlay();
            break;

          case "checkpoint_saved":
            console.log(`[ckpt] ${data.path} @ step ${data.total_steps}`);
            break;

          case "config_applied":
            if (data.params && data.params.board) {
              board.w = data.params.board.w;
              board.h = data.params.board.h;
              buildGridCache(); drawGrid();
            }
            if (typeof data.epsilon_current === "number" && el.epsVal) {
              el.epsVal.textContent = Number(data.epsilon_current).toFixed(4);
            }
            if (el.status) el.status.textContent = "Config applied";
            break;

          case "board_changed":
            if (data.board) {
              board.w = data.board.w; board.h = data.board.h;
              buildGridCache(); drawGrid();
            }
            break;

          case "llm_comment":
            appendLLMComment(data);
            break;

          case "saving_started":
            showSaving();
            break;
          case "saving_finished":
            hideSaving();
            break;
          case "saving_failed":
            hideSaving();
            alert("Save failed: " + (data.error || "unknown error"));
            break;
          case "server_closing":
            showClosingOverlay();
            break;
        }
      };
    }catch(e){ console.warn("WS connect error", e); }
  }

  function showClosingOverlay(){
    const overlay = document.createElement("div");
    overlay.style.position="fixed"; overlay.style.inset="0"; overlay.style.zIndex="9999";
    overlay.style.background="rgba(0,0,0,0.85)";
    overlay.style.color="#ffd166";
    overlay.style.display="flex"; overlay.style.alignItems="center"; overlay.style.justifyContent="center";
    overlay.style.fontSize="22px"; overlay.style.fontWeight="800";
    overlay.textContent = "Project closed — you can safely close this tab.";
    document.body.appendChild(overlay);
  }

  function rafLoop(ts){
    if (ts - lastRenderTs >= 33) {
      lastRenderTs = ts;
      if (lastStateFrame) { renderStateFrame(lastStateFrame); lastStateFrame = null; }
      if (lastNN) { applyNN(lastNN); lastNN = null; }
    }
    requestAnimationFrame(rafLoop);
  }

  // View toggle UI
  el.viewNone.addEventListener("click", ()=> selectView("none"));
  el.viewNeural.addEventListener("click", ()=> selectView("neural"));
  el.viewGraphs.addEventListener("click", ()=> { selectView("graphs"); ensureCharts(); });

  function selectView(view = "neural") {
    const isNone = view === "none", isNeural = view === "neural", isGraphs = view === "graphs";
    el.viewNone.classList.toggle("active", isNone);
    el.viewNeural.classList.toggle("active", isNeural);
    el.viewGraphs.classList.toggle("active", isGraphs);
    el.nnPanel.classList.toggle("hidden", !isNeural);
    el.graphsPanel.classList.toggle("hidden", !isGraphs);
    el.nnPanel.setAttribute("aria-hidden", String(!isNeural));
    el.graphsPanel.setAttribute("aria-hidden", String(!isGraphs));
  }

  // algorithm dropdown
  let currentAlgo = "DDQN";
  if (el.algoSelect) {
    el.algoSelect.addEventListener("change", ()=>{
      currentAlgo = el.algoSelect.value || "DDQN";
      restPost("/api/update_config", { algorithm: currentAlgo }).catch(()=>{});
    });
  }

  // grid size selector
  el.gridSize.addEventListener("change", async ()=>{
    const n = Number(el.gridSize.value) | 0;
    board.w = n; board.h = n; buildGridCache(); drawGrid();
    await restPost("/api/update_config", { board: { w: n, h: n } }).catch(()=>{});
  });

  // controls
  el.start.addEventListener("click", async ()=>{
    hideOverlays();
    if (el.status) el.status.textContent = "Starting…";
    const body = {
      algorithm: currentAlgo,
      max_episodes: 0, max_steps: 1000,
      gamma: Number(el.gammaSlider.value), lr: Number(el.lrSlider.value),
      batch_size: Number(el.batchSlider.value),
      epsilon: { start: 1.0, min: 0.05, decay: Number(el.epsSlider.value) }
    };
    try {
      const res = await restPost("/api/start", body);
      if (el.status) el.status.textContent = res?.ok ? "Running" : "Error starting";
    } catch {
      if (el.status) el.status.textContent = "Error starting";
    }
  });

  el.pause.addEventListener("click", async ()=>{
    try { const r = await restPost("/api/pause"); if (el.status) el.status.textContent = r?.ok ? "Paused" : "Error"; }
    catch { if (el.status) el.status.textContent = "Error"; }
  });
  el.resume.addEventListener("click", async ()=>{
    try { const r = await restPost("/api/resume"); if (el.status) el.status.textContent = r?.ok ? "Running" : "Error"; }
    catch { if (el.status) el.status.textContent = "Error"; }
  });
  el.stop.addEventListener("click", async ()=>{
    try { const r = await restPost("/api/stop"); if (el.status) el.status.textContent = r?.ok ? "Stopped" : "Error"; }
    catch { if (el.status) el.status.textContent = "Error"; }
  });
  el.reset.addEventListener("click", async ()=>{
    hideOverlays();
    try { const r = await restPost("/api/reset"); if (el.status) el.status.textContent = r?.ok ? "Reset" : "Error"; }
    catch { if (el.status) el.status.textContent = "Error"; }
  });
  el.save.addEventListener("click", async ()=>{
    showSaving();
    try { const r = await restPost("/api/save"); if (!r?.ok) hideSaving(); }
    catch { hideSaving(); }
  });
  el.load.addEventListener("click", async ()=>{
    try { const r = await restPost("/api/load"); if (el.status) el.status.textContent = r?.ok ? "Loaded" : "Error"; }
    catch { if (el.status) el.status.textContent = "Error"; }
  });
  el.close.addEventListener("click", async ()=>{
    showSaving();
    try {
      const r = await restPost("/api/close");
      if (r?.ok) { window.close(); showClosingOverlay(); }
      else { hideSaving(); alert("Close failed"); }
    } catch { hideSaving(); alert("Close failed"); }
  });

  // Speed slider -> live label + /api/config
  const postSpeedConfig = debounce(async () => {
    const step_delay_ms = Number(el.speedSlider.value) | 0;
    try { await restPost("/api/config", { step_delay_ms }); } catch (e) { console.warn(e); }
  }, 120);

  if (el.speedSlider) {
    el.speedSlider.addEventListener("input", () => {
      const v = Number(el.speedSlider.value) | 0;
      if (el.speedVal) el.speedVal.textContent = String(v);
      postSpeedConfig();
    });
  }

  function liveBind(sliderEl, labelEl, formatFn, sendFn) {
    const ff = (v) => (formatFn ? formatFn(v) : v);
    const onInput = () => {
      const v = Number(sliderEl.value);
      if (labelEl) labelEl.textContent = ff(v);
      sendFn(v);
    };
    sliderEl.addEventListener("input", onInput);
    if (labelEl) labelEl.textContent = ff(Number(sliderEl.value));
  }

  liveBind(el.lrSlider,    el.lrVal,    v => v.toFixed(4), (v) => postConfigUpdate({ lr: v }));
  liveBind(el.gammaSlider, el.gammaVal, v => v.toFixed(2), (v) => postConfigUpdate({ gamma: v }));
  liveBind(el.epsSlider,   el.epsVal,   v => v.toFixed(4), (v) => postConfigUpdate({ epsilon: { decay: v } }));
  liveBind(el.batchSlider, el.batchVal, v => String(Math.round(v)), (v) => postConfigUpdate({ batch_size: Math.round(v) }));

  // Bootstrap
  async function bootstrap(){
    try{
      const cfg = await restGet("/api/config");
      if (cfg?.board?.w && cfg?.board?.h){
        board.w=cfg.board.w; board.h=cfg.board.h;
        const opt = [5,8,10,12,20].includes(board.w) ? String(board.w) : "20";
        el.gridSize.value = opt;
      }
      if (cfg?.step_delay_ms !== undefined) {
        el.speedSlider.value = String(cfg.step_delay_ms|0);
        if (el.speedVal) el.speedVal.textContent = String(cfg.step_delay_ms|0);
      }
      if (cfg?.algorithm) {
        el.algoSelect.value = String(cfg.algorithm);
      }
    }catch{}
    buildGridCache(); drawGrid(); drawNeuralDiagram(); connectWS(); requestAnimationFrame(rafLoop);
    selectView("neural");
  }
  bootstrap();

})();
