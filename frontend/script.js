/* Snake AI UI — view 3-way toggle; algorithm selector; grid size selector; live sliders; LLM feed */
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

    // Algorithm segment control
    algoSeg: document.getElementById("algoSeg"),

    // Grid size
    gridSize: document.getElementById("gridSizeSelect"),

    start: document.getElementById("startBtn"),
    pause: document.getElementById("pauseBtn"),
    resume: document.getElementById("resumeBtn"),
    stop: document.getElementById("stopBtn"),
    reset: document.getElementById("resetBtn"),
    save: document.getElementById("saveBtn"),
    load: document.getElementById("loadBtn"),
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
  };

  const fmt = (v)=> (v===undefined||v===null ? "" : Number(v).toFixed(4));
  const debounce = (fn, ms=150)=>{ let t; return (...args)=>{ clearTimeout(t); t=setTimeout(()=>fn(...args), ms); }; };

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
    for (let y=0;y<=board.h;y++){ g.beginPath(); g.moveTo(0,y*ch); g.lineTo(W,y*ch); g.stroke(); }
  }
  const drawGrid = ()=> ctx.drawImage(gridCache, 0, 0);
  function renderStateFrame(frame){
    if (!frame || !frame.grid) return;
    const W=el.canvas.width, H=el.canvas.height;
    const cw = W / frame.grid.w, ch = H / frame.grid.h;
    drawGrid();
    if (frame.grid.food) {
      ctx.fillStyle = "#f2c94c";
      ctx.fillRect(frame.grid.food[0]*cw+1, frame.grid.food[1]*ch+1, cw-2, ch-2);
    }
    if (Array.isArray(frame.grid.snake)) {
      frame.grid.snake.forEach((seg, idx) => {
        const [x,y]=seg;
        ctx.fillStyle = idx===0 ? "#00ffc6" : "#1e90ff";
        ctx.fillRect(x*cw+1, y*ch+1, cw-2, ch-2);
      });
    }
  }

  const showWinOverlay = ()=>{
    el.overlayWin.classList.add("show");
    el.overlayWin.setAttribute("aria-hidden","false");
    el.overlayLoss.classList.remove("show");
    el.overlayLoss.setAttribute("aria-hidden","true");
  };
  const hideOverlays = ()=>{
    el.overlayWin.classList.remove("show");
    el.overlayLoss.classList.remove("show");
    el.overlayWin.setAttribute("aria-hidden","true");
    el.overlayLoss.setAttribute("aria-hidden","true");
  };

  // Charts
  let charts = { score:null, epsilon:null, loss:null };
  function initCharts(){
    if (!window.Chart) return;
    charts.score = new Chart(el.scoreChartEl.getContext("2d"), { type:"line", data:{labels:[], datasets:[{label:"Score", data:[]}]}, options:{responsive:true, animation:false}});
    charts.epsilon = new Chart(el.epsilonChartEl.getContext("2d"), { type:"line", data:{labels:[], datasets:[{label:"Epsilon", data:[]}]}, options:{responsive:true, animation:false}});
    charts.loss = new Chart(el.lossChartEl.getContext("2d"), { type:"line", data:{labels:[], datasets:[{label:"Avg Loss", data:[]}]}, options:{responsive:true, animation:false}});
  }
  function chartAppend(chart,x,y){
    if(!chart) return;
    chart.data.labels.push(x);
    chart.data.datasets[0].data.push(y);
    if(chart.data.labels.length>200){ chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
    chart.update();
  }

  // Table
  function addEpisodeRow(row){
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.episode??""}</td><td>${row.outcome??""}</td><td>${row.reason??""}</td>
      <td>${row.score??""}</td><td>${row.snake_length??""}</td><td>${fmt(row.epsilon)}</td>
      <td>${fmt(row.avg_loss)}</td><td>${row.steps??""}</td><td>${row.duration_ms??""}</td><td>${row.timestamp??""}</td>`;
    el.tableBody.prepend(tr);
  }

  // Neural diagram (16→12→8→4)
  const INPUT_LABELS = [
    "dir_up","dir_down","dir_left","dir_right",
    "danger_fwd","danger_left","danger_right",
    "food_dx","food_dy",
    "head_x_norm","head_y_norm","food_x_norm","food_y_norm",
    "length_ratio","score_norm","bias"
  ];
  const NN_LAYERS = [16,12,8,4];
  const NN = { nodes:[], links:[], nodeSel:null, linkSel:null, outputQText:null,
    prevActs:[new Float32Array(16), new Float32Array(12), new Float32Array(8), new Float32Array(4)], targetLinksByLayer:[] };

  function drawNeuralDiagram(){
    const svg = d3.select(el.nnSvg); svg.selectAll("*").remove();
    const width = el.nnSvg.clientWidth || 800, height = el.nnSvg.clientHeight || 360;
    const layerSpacing = width / (NN_LAYERS.length + 1);

    const nodes = [], nodeMap = {};
    NN_LAYERS.forEach((count, l)=>{
      const ySpacing = height / (count + 1);
      for (let i=0;i<count;i++){
        const n = { id:`${l}-${i}`, layer:l, idx:i, x:(l+1)*layerSpacing, y:(i+1)*ySpacing };
        nodes.push(n); nodeMap[n.id] = n;
      }
    });

    const links = [];
    for (let l=0;l<NN_LAYERS.length-1;l++){
      for (let i=0;i<NN_LAYERS[l];i++){
        for (let j=0;j<NN_LAYERS[l+1];j++){
          links.push({ source:`${l}-${i}`, target:`${l+1}-${j}`, tLayer:l+1, tIdx:j });
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

    svg.selectAll("text.input-label").data(nodes.filter(n=>n.layer===0))
      .enter().append("text").attr("class","node-label input-label")
      .attr("x", d=>d.x - 14).attr("y", d=>d.y + 4).attr("text-anchor","end")
      .text((d,i)=> INPUT_LABELS[i] ?? `in_${i}`);

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

    const targetLinksByLayer = Array.from({length: NN_LAYERS.length}, (_, l)=>
      l===0 ? null : Array.from({length: NN_LAYERS[l]}, ()=>[])
    );
    links.forEach((L, idx)=> targetLinksByLayer[L.tLayer][L.tIdx].push(idx));

    NN.nodes = nodes; NN.links = links;
    NN.nodeSel = nodeSel; NN.linkSel = linkSel; NN.outputQText = outputQText;
    NN.targetLinksByLayer = targetLinksByLayer;
  }

  function fitArray(arr, outLen){
    if (!Array.isArray(arr) || arr.length === 0) return new Float32Array(outLen);
    const out = new Float32Array(outLen);
    const n = arr.length;
    for (let i=0;i<outLen;i++){
      const src = Math.min(n-1, Math.round(i * (n-1) / Math.max(1,outLen-1)));
      out[i] = Math.max(0, Math.min(1, Number(arr[src]) || 0));
    }
    return out;
  }

  function applyNN(ev){
    const L = ev && Array.isArray(ev.layers) ? ev.layers : [];
    const h1 = fitArray((L[0] && (L[0].act || L[0].a)) || [], NN_LAYERS[1]);
    const h2 = fitArray((L[1] && (L[1].act || L[1].a)) || h1, NN_LAYERS[2]);
    const out = fitArray((L[L.length-1] && (L[L.length-1].q || L[L.length-1].act || L[L.length-1].a)) || [], NN_LAYERS[3]);

    const th = 0.002;
    const prev = NN.prevActs;

    NN.nodeSel.each(function(d){
      if (d.layer === 0) return;
      const a = (d.layer===1? h1[d.idx] : d.layer===2? h2[d.idx] : out[d.idx]);
      const p = prev[d.layer][d.idx];
      const delta = a - p;
      if (delta > th) { const s=d3.select(this); s.classed("blink-green", true); setTimeout(()=>s.classed("blink-green", false), 160); }
      else if (delta < -th) { const s=d3.select(this); s.classed("blink-red", true); setTimeout(()=>s.classed("blink-red", false), 160); }
    });

    const linkElems = NN.linkSel.nodes();
    function bumpLinks(layerIdx, activations){
      const table = NN.targetLinksByLayer[layerIdx];
      for (let j=0;j<activations.length;j++){
        const a = activations[j], p = prev[layerIdx][j], delta = a - p;
        if (Math.abs(delta) <= th) continue;
        const indices = table[j];
        for (let k=0;k<indices.length;k++){
          const el = d3.select(linkElems[indices[k]]);
          el.classed("pos", delta > th).classed("neg", delta < -th);
          const base = 0.35, extra = Math.min(0.65, Math.abs(delta) * 3);
          el.style("opacity", base + extra);
        }
      }
    }
    bumpLinks(1, h1); bumpLinks(2, h2); bumpLinks(3, out);

    if (NN.outputQText) {
      const nodes = NN.outputQText.nodes();
      for (let i=0;i<out.length && i<nodes.length;i++){
        nodes[i].textContent = Number.isFinite(out[i]) ? out[i].toFixed(3) : "-";
      }
    }

    prev[1].set(h1); prev[2].set(h2); prev[3].set(out);
  }

  // View toggle (3-way)
  function selectView(view="neural"){
    const isNone = view==="none", isNeural = view==="neural", isGraphs = view==="graphs";
    el.viewNone.classList.toggle("active", isNone);
    el.viewNeural.classList.toggle("active", isNeural);
    el.viewGraphs.classList.toggle("active", isGraphs);

    el.nnPanel.classList.toggle("hidden", !isNeural);
    el.graphsPanel.classList.toggle("hidden", !isGraphs);
    el.graphsPanel.setAttribute("aria-hidden", String(!isGraphs));
  }

  async function restPost(path, body){
    try {
      const res = await fetch(`${CONFIG.API_URL}${path}`, {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify(body||{})
      });
      return await res.json();
    } catch (e) { return { ok:false, error:String(e) }; }
  }
  async function restGet(path){
    try { const res = await fetch(`${CONFIG.API_URL}${path}`); return await res.json(); }
    catch { return null; }
  }

  function appendLLMComment(msg){
    if (!el.llmFeed) return;
    const wrapper = document.createElement("div");
    wrapper.className = "llm-msg";
    const meta = document.createElement("div");
    meta.className = "llm-meta";
    meta.textContent = `Episode ${msg.episode} • step ${msg.step}`;
    const body = document.createElement("div");
    body.textContent = msg.text;
    wrapper.appendChild(meta);
    wrapper.appendChild(body);
    el.llmFeed.prepend(wrapper);
    while (el.llmFeed.children.length > 50) el.llmFeed.removeChild(el.llmFeed.lastChild);
  }

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
            el.status.textContent = `Running (ep ${data.episode}, step ${data.step})`; break;
          case "episode_summary":
            addEpisodeRow(data);
            chartAppend(charts.score, data.episode, data.score);
            chartAppend(charts.epsilon, data.episode, data.epsilon);
            chartAppend(charts.loss, data.episode, data.avg_loss);
            if (data.outcome === "WIN") showWinOverlay();
            break;
          case "game_over":
            if (data.outcome === "WIN") showWinOverlay();
            break;
          case "checkpoint_saved":
            console.log(`[ckpt] ${data.path} @ step ${data.total_steps}`);
            break;
          case "config_applied":
            console.log("[WS] config_applied", data.params);
            if (data.params && data.params.board) {
              // update board and rebuild grid immediately for visuals
              board.w = data.params.board.w;
              board.h = data.params.board.h;
              buildGridCache(); drawGrid();
            }
            el.status.textContent = "Config applied";
            break;
          case "board_changed":
            // definitive confirmation after env rebuild
            if (data.board) {
              board.w = data.board.w; board.h = data.board.h;
              buildGridCache(); drawGrid();
            }
            break;
          case "llm_comment":
            appendLLMComment(data);
            break;
        }
      };
    }catch(e){ console.warn("WS connect error", e); }
  }

  function rafLoop(ts){
    if (ts - lastRenderTs >= 33) {
      lastRenderTs = ts;
      if (lastStateFrame) { renderStateFrame(lastStateFrame); lastStateFrame = null; }
      if (lastNN) { applyNN(lastNN); lastNN = null; }
    }
    requestAnimationFrame(rafLoop);
  }

  // ----- Controls -----
  el.viewNone.addEventListener("click", ()=> selectView("none"));
  el.viewNeural.addEventListener("click", ()=> selectView("neural"));
  el.viewGraphs.addEventListener("click", ()=> selectView("graphs"));

  // algorithm segment
  let currentAlgo = "DDQN";
  el.algoSeg.addEventListener("click", (e)=>{
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    el.algoSeg.querySelectorAll(".seg-btn").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    currentAlgo = btn.dataset.algo || "DDQN";
    // notify backend (non-breaking if unused)
    restPost("/api/update_config", { algorithm: currentAlgo });
  });

  // grid size selector
  el.gridSize.addEventListener("change", async ()=>{
    const n = Number(el.gridSize.value) | 0;
    // pre-apply visuals
    board.w = n; board.h = n; buildGridCache(); drawGrid();
    // ask backend to resize (applies next episode)
    await restPost("/api/update_config", { board: { w: n, h: n } });
  });

  el.start.addEventListener("click", async ()=>{
    hideOverlays();
    el.status.textContent = "Starting…";
    const body = {
      algorithm: currentAlgo,
      max_episodes: 0, max_steps: 1000,
      gamma: Number(el.gammaSlider.value), lr: Number(el.lrSlider.value),
      batch_size: Number(el.batchSlider.value),
      epsilon: { start: 1.0, min: 0.05, decay: Number(el.epsSlider.value) }
    };
    const res = await restPost("/api/start", body);
    el.status.textContent = res?.ok ? "Running" : "Error starting";
  });
  el.pause.addEventListener("click", async ()=>{ const r=await restPost("/api/pause"); el.status.textContent = r?.ok ? "Paused" : "Error"; });
  el.resume.addEventListener("click", async ()=>{ const r=await restPost("/api/resume"); el.status.textContent = r?.ok ? "Running" : "Error"; });
  el.stop.addEventListener("click", async ()=>{ const r=await restPost("/api/stop"); el.status.textContent = r?.ok ? "Stopped" : "Error"; });
  el.reset.addEventListener("click", async ()=>{ hideOverlays(); drawGrid(); const r=await restPost("/api/reset"); el.status.textContent = r?.ok ? "Reset" : "Error"; });
  el.save.addEventListener("click", async ()=>{ const r=await restPost("/api/save"); el.status.textContent = r?.ok ? "Saved" : "Error"; });
  el.load.addEventListener("click", async ()=>{ const r=await restPost("/api/load"); el.status.textContent = r?.ok ? "Loaded" : "Error"; });

  // reflect slider labels
  const mirror = (slider, label, fmtFn=(v)=>v)=>{ const f=()=>{ label.textContent = fmtFn(slider.value); }; slider.addEventListener("input", f); f(); };
  mirror(el.epsSlider,   el.epsVal,   v=>Number(v).toFixed(4));
  mirror(el.lrSlider,    el.lrVal,    v=>Number(v).toFixed(4));
  mirror(el.gammaSlider, el.gammaVal, v=>Number(v).toFixed(2));
  mirror(el.batchSlider, el.batchVal, v=>String(v));
  mirror(el.speedSlider, el.speedVal, v=>String(v));

  // Speed slider -> backend (/api/config)
  const postSpeedConfig = debounce(async ()=>{
    const step_delay_ms = Number(el.speedSlider.value) | 0;
    await restPost("/api/config", { step_delay_ms });
  }, 120);
  el.speedSlider.addEventListener("input", postSpeedConfig);

  // Live hyperparameters -> /api/update_config
  const postConfigUpdate = debounce(async (payload) => {
    await restPost("/api/update_config", payload);
  }, 120);

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
        // reflect select
        const opt = [5,8,10,12,20].includes(board.w) ? String(board.w) : "20";
        el.gridSize.value = opt;
      }
      if (cfg?.step_delay_ms !== undefined) {
        el.speedSlider.value = String(cfg.step_delay_ms|0);
        el.speedVal.textContent = String(cfg.step_delay_ms|0);
      }
    }catch{}
    buildGridCache(); drawGrid(); initCharts(); drawNeuralDiagram(); connectWS(); requestAnimationFrame(rafLoop);

    // default view = neural
    selectView("neural");
  }
  bootstrap();

  // optional helper
  window.SnakeUI = window.SnakeUI || {};
  window.SnakeUI.setLLM = (enabled, freq=1000) => fetch(`${CONFIG.API_URL}/api/update_config`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ llm: { enabled, freq } })
  });
})();
