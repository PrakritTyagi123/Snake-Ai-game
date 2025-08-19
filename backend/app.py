from __future__ import annotations
import asyncio
import os
import signal
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

from backend.train.loop import Trainer

app = FastAPI(title="Snake AI Backend", version="1.2")

# CORS for the static frontend on 8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BroadcastManager:
    def __init__(self) -> None:
        self.conns: List[WebSocket] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for ws in list(self.conns):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                self.conns.remove(ws)
            except ValueError:
                pass

    async def close_all(self) -> None:
        # best-effort close of all websockets
        conns = list(self.conns)
        self.conns.clear()
        for ws in conns:
            try:
                await ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass

broadcaster = BroadcastManager()
trainer = Trainer(broadcaster.send_json)
try:
    trainer.load_checkpoint
except Exception:
    pass

# -------- REST --------
@app.get("/api/config")
def api_get_config() -> Dict[str, Any]:
    return trainer.get_config()

@app.post("/api/config")
def api_set_config(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    trainer.update_config(payload)
    return {"ok": True}

@app.post("/api/update_config")
def api_update_config(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    trainer.update_config(payload)
    return {"ok": True, "params": trainer.get_config()}

@app.post("/api/start")
def api_start(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    # 1) Load whatever exists for the currently selected algo (if any)
    loaded = trainer.load_checkpoint()
    # 2) Re-apply UI config AFTER loading, so UI overrides the checkpoint values
    trainer.update_config(payload or {})
    # 3) Start training
    trainer.start()
    return {"ok": True, "loaded": loaded}

@app.post("/api/pause")
async def api_pause() -> Dict[str, Any]:
    await trainer.pause_and_save()
    return {"ok": True}

@app.post("/api/resume")
def api_resume() -> Dict[str, Any]:
    trainer.resume()
    return {"ok": True}

@app.post("/api/stop")
def api_stop() -> Dict[str, Any]:
    trainer.stop()
    return {"ok": True}

@app.post("/api/reset")
def api_reset() -> Dict[str, Any]:
    trainer.reset()
    return {"ok": True}

@app.post("/api/save")
async def api_save() -> Dict[str, Any]:
    await trainer.save_checkpoint(reason="api_save")
    return {"ok": True}

@app.post("/api/load")
def api_load() -> Dict[str, Any]:
    ok = trainer.load_checkpoint()
    return {"ok": ok}

@app.post("/api/close")
async def api_close() -> Dict[str, Any]:
    # 1) pause + save (allow WS events to show "Saving…")
    await trainer.pause_and_save()
    try:
        trainer.flush_episode_log()
    except Exception:
        pass
    # 2) tell clients we're closing
    await broadcaster.send_json({"type": "server_closing"})
    # 3) stop sending late events, then close sockets
    trainer.closing = True
    await broadcaster.close_all()
    # 4) shutdown trainer threads/loops
    trainer.shutdown()
    # 5) schedule process exit after response is sent
    loop = asyncio.get_running_loop()
    loop.call_later(0.2, lambda: os.kill(os.getpid(), signal.SIGINT))
    return {"ok": True}

# -------- lifecycle hooks --------
@app.on_event("shutdown")
async def on_shutdown():
    try:
        trainer.closing = True
        await broadcaster.close_all()
        try:
            trainer.flush_episode_log()
        except Exception:
            pass
        trainer.shutdown()
    except Exception:
        pass

# -------- WebSocket --------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    broadcaster.conns.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        try:
            broadcaster.conns.remove(ws)
        except ValueError:
            pass
