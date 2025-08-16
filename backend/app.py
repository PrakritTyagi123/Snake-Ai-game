from __future__ import annotations
import asyncio
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

from .train.loop import Trainer, BroadcastManager

app = FastAPI(title="Snake AI Backend", version="1.0")

# CORS for the static frontend on 8080
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create trainer/broadcaster, bind loop on startup
broadcaster = BroadcastManager(loop=None)
trainer = Trainer(broadcaster=broadcaster)

@app.on_event("startup")
async def _bind_loop():
    broadcaster.loop = asyncio.get_running_loop()

# -------- REST --------
@app.get("/api/config")
def api_get_config():
    # support w/h attributes named either w/h or width/height
    w = getattr(trainer.env, "w", None) or getattr(trainer.env, "width", None)
    h = getattr(trainer.env, "h", None) or getattr(trainer.env, "height", None)
    return {
        "ok": True,
        "step_delay_ms": trainer.params.get("step_delay_ms", 0),
        "board": {"w": w, "h": h},
        "params": trainer.params,
        "total_steps": trainer.total_steps,
    }

@app.post("/api/config")
def api_post_config(payload: dict = Body(...)):
    return trainer.update_config(payload)

@app.post("/api/update_config")
def api_update_config(payload: dict = Body(...)):
    return trainer.update_config(payload)

@app.post("/api/start")
def api_start(payload: dict = Body(default={})):
    trainer.start(payload)
    return {"ok": True}

@app.post("/api/pause")
def api_pause():
    trainer.pause(); return {"ok": True}

@app.post("/api/resume")
def api_resume():
    trainer.resume(); return {"ok": True}

@app.post("/api/stop")
def api_stop():
    trainer.stop(); return {"ok": True}

@app.post("/api/reset")
def api_reset():
    trainer.reset_env(); return {"ok": True}

@app.post("/api/save")
def api_save():
    return trainer.save()

@app.post("/api/load")
def api_load(payload: dict = Body(default={})):
    return trainer.load(payload.get("path"))

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
