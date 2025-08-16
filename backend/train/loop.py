from __future__ import annotations
import time, threading, csv, os, json, asyncio, tempfile, queue
from typing import Optional, Dict, Any, List

import torch

from ..config import (
    DEFAULT_MAX_EPISODES, DEFAULT_MAX_STEPS, DEFAULT_GAMMA, DEFAULT_LR, DEFAULT_BATCH,
    DEFAULT_EPS_START, DEFAULT_EPS_MIN, DEFAULT_EPS_DECAY, TARGET_UPDATE_FREQ,
    CHECKPOINT_DIR, LOG_DIR, BOARD_WIDTH, BOARD_HEIGHT, WIN_LENGTH, REQUIRE_CUDA, STEP_DELAY_MS,
)
from .. import config as cfg
from ..env.snake_env import SnakeEnv
from ..agent.ddqn import DDQNAgent
from ..agent.replay_buffer import ReplayBuffer
try:
    from ..llm import LLMReporter  # if llm.py is inside backend/
except ImportError:
    from llm import LLMReporter    # if llm.py is at project root


class BroadcastManager:
    def __init__(self, loop: asyncio.AbstractEventLoop | None):
        self.loop = loop
        self.conns: List[Any] = []
        self._lock = threading.Lock()

    async def _broadcast(self, payload: Dict[str, Any]):
        dead = []
        for ws in list(self.conns):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for d in dead:
                    if d in self.conns:
                        self.conns.remove(d)

    def broadcast(self, payload: Dict[str, Any]):
        if not self.loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)


# ------------------ async/light checkpointer ------------------
class AsyncCheckpointer:
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer
        self.q: "queue.Queue[tuple[str,int|None]]" = queue.Queue(maxsize=1)
        self._last_auto_ts = 0.0
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def request_auto(self, total_steps: int):
        min_gap = float(self.trainer.params.get("autosave_interval_s", 3.0))
        now = time.time()
        if now - self._last_auto_ts < min_gap:
            return
        self._last_auto_ts = now
        self._offer(("auto", total_steps))

    def _offer(self, item: tuple[str, int | None]):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(item)
        except queue.Full:
            pass

    def _worker(self):
        while True:
            kind, total_steps = self.q.get()
            try:
                if kind == "auto":
                    ckpt = self.trainer._compose_light_checkpoint()
                    path = os.path.join(CHECKPOINT_DIR, "latest_auto.ckpt")
                    self.trainer._write_ckpt_to_path(ckpt, path, make_latest=False)
                    self.trainer.b.broadcast({
                        "type": "checkpoint_saved", "path": os.path.basename(path),
                        "total_steps": int(total_steps or self.trainer.total_steps),
                        "mode": "light"
                    })
            except Exception as e:
                self.trainer.b.broadcast({"type": "checkpoint_error", "error": str(e)})


class Trainer:
    def __init__(self, broadcaster: BroadcastManager):
        self.b = broadcaster
        self.training_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._running = False
        self.history: List[Dict[str, Any]] = []

        # ----- env / agent -----
        self.env = SnakeEnv(BOARD_WIDTH, BOARD_HEIGHT, WIN_LENGTH, seed=0)
        state_dim = len(self.env.get_state())
        if REQUIRE_CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but not available.")

        self.agent = DDQNAgent(state_dim, 4, lr=DEFAULT_LR, gamma=DEFAULT_GAMMA, device="cuda")
        self.buffer = ReplayBuffer(100_000, seed=0)

        # ----- params (runtime editable) -----
        base_local_dir = os.getenv("LLM_LOCAL_DIR", getattr(cfg, "LLM_LOCAL_DIR", None))
        self.params = {
            "algorithm": "DDQN",
            "max_episodes": DEFAULT_MAX_EPISODES,   # <=0 → run forever
            "max_steps": DEFAULT_MAX_STEPS,         # 0 → unlimited
            "gamma": DEFAULT_GAMMA,
            "lr": DEFAULT_LR,
            "batch_size": DEFAULT_BATCH,
            "epsilon": {"start": DEFAULT_EPS_START, "min": DEFAULT_EPS_MIN, "decay": DEFAULT_EPS_DECAY},
            "step_delay_ms": STEP_DELAY_MS,
            "autosave_steps": getattr(cfg, "DEFAULT_AUTOSAVE_STEPS", 1000),
            "autosave_interval_s": 3.0,
            "autosave_light": True,
            "llm": {
                "enabled": bool(getattr(cfg, "LLM_ENABLED", True)),
                "freq": int(getattr(cfg, "LLM_COMMENT_FREQ", 1000)),
                "backend": str(getattr(cfg, "LLM_BACKEND", "hf")).lower(),
                "model_id": str(getattr(cfg, "LLM_MODEL_HF_ID", "Qwen/Qwen2-0.5B-Instruct")),
                "local_dir": base_local_dir,
                "max_tokens": int(getattr(cfg, "LLM_MAX_TOKENS", 250)),
            },
        }

        self.total_steps = 0
        self._eps_override: Optional[float] = None

        # pending env resize
        self._pending_board: Optional[tuple[int,int]] = None
        self._force_ep_end = False

        # ----- LLM reporter -----
        self.llm = LLMReporter(
            backend=self.params["llm"]["backend"],
            model_id=self.params["llm"]["model_id"],
            local_dir=self.params["llm"]["local_dir"],
            max_tokens=self.params["llm"]["max_tokens"],
        )
        self._llm_busy = False

        # ----- async saver -----
        self._saver = AsyncCheckpointer(self)

    # -------- control --------
    def start(self, params: Optional[Dict[str, Any]] = None):
        if params:
            self.params.update(params)
            if params.get("reset_epsilon", False):
                self._eps_override = self.params["epsilon"]["start"]
            if "algorithm" in params:
                self.params["algorithm"] = str(params["algorithm"]).upper()
        if self._running:
            return
        self._stop.clear()
        self._pause.clear()
        self.training_thread = threading.Thread(target=self._train_loop, daemon=True)
        self.training_thread.start()
        self._running = True

    def pause(self): self._pause.set()
    def resume(self): self._pause.clear()
    def stop(self): self._stop.set(); self._running = False
    def reset_env(self): self.env.reset()

    # -------- checkpoint composition/writing --------
    def _compose_light_checkpoint(self) -> Dict[str, Any]:
        try:
            scaler_state = getattr(self.agent, "scaler", None)
            scaler_state = scaler_state.state_dict() if scaler_state is not None else None
        except Exception:
            scaler_state = None
        return {
            "agent": {
                "policy": {k: v.detach().cpu() for k, v in self.agent.policy.state_dict().items()},
                "target": {k: v.detach().cpu() for k, v in self.agent.target.state_dict().items()},
                "opt": self.agent.optimizer.state_dict(),
                "scaler": scaler_state,
            },
            "trainer": {
                "total_steps": self.total_steps,
                "epsilon": self._eps_override,
                "params": self.params,
            },
            "buffer": None,
            "meta": {"ts": int(time.time()), "algo": self.params.get("algorithm","DDQN"), "version": 2, "kind": "light"},
        }

    def _compose_checkpoint(self) -> Dict[str, Any]:
        try:
            scaler_state = getattr(self.agent, "scaler", None)
            scaler_state = scaler_state.state_dict() if scaler_state is not None else None
        except Exception:
            scaler_state = None
        return {
            "agent": {
                "policy": self.agent.policy.state_dict(),
                "target": self.agent.target.state_dict(),
                "opt": self.agent.optimizer.state_dict(),
                "scaler": scaler_state,
            },
            "trainer": {
                "total_steps": self.total_steps,
                "epsilon": self._eps_override,
                "params": self.params,
            },
            "buffer": self.buffer.state_dict(),
            "meta": {"ts": int(time.time()), "algo": self.params.get("algorithm","DDQN"), "version": 2, "kind": "full"},
        }

    def _write_ckpt_to_path(self, ckpt: Dict[str, Any], path: str, make_latest: bool = True) -> str:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=CHECKPOINT_DIR, prefix="._tmp_", suffix=".ckpt")
        os.close(fd)
        torch.save(ckpt, tmp_path, pickle_protocol=4, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
        if make_latest:
            latest = os.path.join(CHECKPOINT_DIR, "latest.ckpt")
            try:
                if os.path.exists(latest): os.remove(latest)
                os.link(path, latest)
            except Exception:
                try:
                    import shutil; shutil.copy2(path, latest)
                except Exception:
                    pass
        return path

    def save(self):
        path = os.path.join(CHECKPOINT_DIR, f"trainer_{int(time.time())}.ckpt")
        final = self._write_ckpt_to_path(self._compose_checkpoint(), path, make_latest=True)
        return {"ok": True, "path": final}

    def load(self, path: Optional[str] = None):
        def _pick_latest():
            files = []
            for f in os.listdir(CHECKPOINT_DIR):
                if f.endswith(('.ckpt', '.pth')):
                    files.append(os.path.join(CHECKPOINT_DIR, f))
            if not files: return None
            files.sort(key=lambda p: os.path.getmtime(p))
            return files[-1]

        if not path:
            path = _pick_latest()
            if not path:
                return {"ok": False, "error": "no checkpoint found"}

        try:
            loaded = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            return {"ok": False, "error": f"load failed: {e}"}

        if isinstance(loaded, dict) and ("agent" in loaded or "policy" in loaded):
            if "agent" in loaded:
                agent = loaded["agent"]
                self.agent.policy.load_state_dict(agent["policy"])
                self.agent.target.load_state_dict(agent["target"])
                self.agent.optimizer.load_state_dict(agent["opt"])
                if agent.get("scaler") and getattr(self.agent, "scaler", None) is not None:
                    try: self.agent.scaler.load_state_dict(agent["scaler"])
                    except Exception: pass

                tr = loaded.get("trainer", {})
                self.total_steps = int(tr.get("total_steps", 0))
                self._eps_override = tr.get("epsilon", None)
                if isinstance(tr.get("params"), dict):
                    self.params.update(tr["params"])

                buf = loaded.get("buffer", None)
                if isinstance(buf, dict):
                    self.buffer.load_state_dict(buf)
                return {"ok": True, "path": path, "resumed": True}
            else:
                try:
                    self.agent.policy.load_state_dict(loaded["policy"])
                    self.agent.target.load_state_dict(loaded["target"])
                    self.agent.optimizer.load_state_dict(loaded["opt"])
                except Exception:
                    return {"ok": False, "error": "invalid legacy checkpoint"}
                return {"ok": True, "path": path, "resumed": False}

        try:
            self.agent.load(path)
            return {"ok": True, "path": path, "resumed": False}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # -------- live config (sliders + algorithm + board) --------
    def update_config(self, cfg: Dict[str, Any]):
        if "algorithm" in cfg:
            try: self.params["algorithm"] = str(cfg["algorithm"]).upper()
            except Exception: pass

        if "board" in cfg and isinstance(cfg["board"], dict):
            w = int(cfg["board"].get("w", 0)); h = int(cfg["board"].get("h", 0))
            if w == h and w in (5, 8, 10, 12, 20):
                self._pending_board = (w, h)
                self._force_ep_end = True  # end the current episode quickly to apply
                # pre-announce (frontend will update grid visuals, then final confirmation after apply)
                try:
                    self.b.broadcast({"type": "config_applied", "params": {"board": {"w": w, "h": h}}})
                except Exception:
                    pass

        if "step_delay_ms" in cfg:
            try: self.params["step_delay_ms"] = int(cfg["step_delay_ms"])
            except Exception: pass
        if "max_steps" in cfg:
            try: self.params["max_steps"] = int(cfg["max_steps"])
            except Exception: pass
        if "batch_size" in cfg:
            try: self.params["batch_size"] = int(cfg["batch_size"])
            except Exception: pass
        if "autosave_steps" in cfg:
            try: self.params["autosave_steps"] = int(cfg["autosave_steps"])
            except Exception: pass
        if "autosave_interval_s" in cfg:
            try: self.params["autosave_interval_s"] = float(cfg["autosave_interval_s"])
            except Exception: pass
        if "autosave_light" in cfg:
            try: self.params["autosave_light"] = bool(cfg["autosave_light"])
            except Exception: pass

        if "lr" in cfg:
            try:
                new_lr = float(cfg["lr"]); self.params["lr"] = new_lr
                for g in self.agent.optimizer.param_groups: g["lr"] = new_lr
            except Exception: pass
        if "gamma" in cfg:
            try:
                g = float(cfg["gamma"]); self.params["gamma"] = g
                if hasattr(self.agent, "gamma"): self.agent.gamma = g
            except Exception: pass

        if "epsilon" in cfg and isinstance(cfg["epsilon"], dict):
            e = cfg["epsilon"]
            if "start" in e:
                try: self.params["epsilon"]["start"] = float(e["start"])
                except Exception: pass
            if "min" in e:
                try: self.params["epsilon"]["min"] = float(e["min"])
                except Exception: pass
            if "decay" in e:
                try: self.params["epsilon"]["decay"] = float(e["decay"])
                except Exception: pass
            if "value" in e:
                try: self._eps_override = float(e["value"])
                except Exception: pass
        if "epsilon_value" in cfg:
            try: self._eps_override = float(cfg["epsilon_value"])
            except Exception: pass

        if "llm" in cfg and isinstance(cfg["llm"], dict):
            L = cfg["llm"]
            if "enabled" in L: self.params["llm"]["enabled"] = bool(L["enabled"])
            if "freq" in L:
                try: self.params["llm"]["freq"] = max(1, int(L["freq"]))
                except Exception: pass
            if "backend" in L and L["backend"]:
                self.params["llm"]["backend"] = str(L["backend"]).lower()
            if "model_id" in L and L["model_id"]:
                self.params["llm"]["model_id"] = str(L["model_id"])
            if "local_dir" in L and L["local_dir"]:
                self.params["llm"]["local_dir"] = str(L["local_dir"])
            if "max_tokens" in L:
                try: self.params["llm"]["max_tokens"] = int(L["max_tokens"])
                except Exception: pass
            self.llm = LLMReporter(
                backend=self.params["llm"]["backend"],
                model_id=self.params["llm"]["model_id"],
                local_dir=self.params["llm"]["local_dir"],
                max_tokens=self.params["llm"]["max_tokens"],
            )

        # general echo (includes algorithm + autosave fields)
        try:
            w = getattr(self.env, "w", None) or getattr(self.env, "width", None)
            h = getattr(self.env, "h", None) or getattr(self.env, "height", None)
            self.b.broadcast({"type": "config_applied", "params": {
                "algorithm": self.params.get("algorithm", "DDQN"),
                "lr": self.params["lr"], "gamma": self.params["gamma"],
                "batch_size": self.params["batch_size"], "max_steps": self.params["max_steps"],
                "step_delay_ms": self.params["step_delay_ms"], "autosave_steps": self.params["autosave_steps"],
                "autosave_interval_s": self.params["autosave_interval_s"], "autosave_light": self.params["autosave_light"],
                "epsilon_value": self._eps_override if self._eps_override is not None else "auto",
                "epsilon_min": self.params["epsilon"]["min"], "epsilon_decay": self.params["epsilon"]["decay"],
                "llm": self.params.get("llm", {}),
                "board": {"w": w, "h": h},
            }})
        except Exception:
            pass

        return {"ok": True, **self.params, "epsilon_value": self._eps_override if self._eps_override is not None else "auto"}

    # -------- LLM helper --------
    def _kickoff_llm_comment(self, episode: int, step: int, epsilon: float, avg_loss: float, score_so_far: int, q_values):
        if self._llm_busy or not self.params["llm"]["enabled"]:
            return
        self._llm_busy = True

        recent = list(self.history[-10:])
        snap = {
            "step": int(step),
            "episode": int(episode),
            "epsilon": float(epsilon),
            "avg_loss": float(avg_loss),
            "score_so_far": int(score_so_far),
            "recent": recent[::-1],
            "q_values": [float(x) for x in (q_values or [])][:4] if q_values is not None else None,
        }

        def worker():
            try:
                text = self.llm.generate_comment(snap)
            except Exception:
                text = "LLM commentary unavailable."
            try:
                self.b.broadcast({"type": "llm_comment", "episode": episode, "step": step, "text": text})
            finally:
                self._llm_busy = False

        threading.Thread(target=worker, daemon=True).start()

    # -------- training --------
    def _train_loop(self):
        eps = self._eps_override if self._eps_override is not None else self.params["epsilon"]["start"]

        self.env.reset()
        csv_path = os.path.join(LOG_DIR, "episodes.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(["episode","outcome","reason","score","snake_length","epsilon","avg_loss","steps","duration_ms","timestamp"])

        ep = 1
        max_ep = self.params["max_episodes"]
        infinite = (max_ep is None) or (max_ep <= 0)

        while infinite or ep <= max_ep:
            # apply pending board resize at episode boundary
            if self._pending_board:
                w, h = self._pending_board
                self.env = SnakeEnv(w, h, WIN_LENGTH, seed=0)
                self._pending_board = None
                try:
                    self.b.broadcast({"type": "board_changed", "board": {"w": w, "h": h}})
                except Exception:
                    pass

            if self._stop.is_set(): break
            while self._pause.is_set():
                time.sleep(0.1)
                if self._stop.is_set(): break
            if self._stop.is_set(): break

            state = self.env.reset()
            ep_loss_sum = 0.0; ep_loss_count = 0; score = 0; t0 = time.time(); step = 1

            while True:
                if self._stop.is_set() or self._pause.is_set(): break
                if self._force_ep_end:
                    # end this episode quickly to apply changes
                    self._force_ep_end = False
                    info = {"outcome": "LOSS", "reason": "board_resize"}
                    break

                cur_max_steps = int(self.params.get("max_steps", 0))
                if cur_max_steps > 0 and step > cur_max_steps: break

                delay = int(self.params.get("step_delay_ms", 0))
                if delay > 0: time.sleep(delay/1000.0)

                action = self.agent.act(state, eps)
                next_state, reward, done, info = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                score += (1 if reward > 0.5 else 0)

                batch = int(self.params.get("batch_size", DEFAULT_BATCH))
                if len(self.buffer) >= batch:
                    batch_data = self.buffer.sample(batch)
                    target_update = (self.total_steps % TARGET_UPDATE_FREQ == 0)
                    loss = self.agent.learn(batch_data, target_net_update=target_update)
                    ep_loss_sum += loss; ep_loss_count += 1

                if self.total_steps % 3 == 0:
                    self.b.broadcast({"type":"state_frame", "grid": self.env.render_spec(), "step": self.total_steps, "episode": ep})
                    acts = self.agent.activation_summary(); q = self.agent.q_values(state)
                    self.b.broadcast({"type":"nn_activations", "layers":[
                        {"name":"dense1","act":acts.get("dense1", [])},
                        {"name":"dense2","act":acts.get("dense2", [])},
                        {"name":"output","q":q}
                    ], "chosen_action": ["Up","Down","Left","Right"][action]})
                    self.b.broadcast({"type":"metrics_tick", "episode": ep, "step": step, "epsilon": eps,
                                       "rolling_loss": (ep_loss_sum/max(1,ep_loss_count)) if ep_loss_count else 0.0,
                                       "score_so_far": score})

                self.total_steps += 1

                # async LIGHT autosave
                autosave_steps = int(self.params.get("autosave_steps", 0))
                if autosave_steps > 0 and (self.total_steps % autosave_steps == 0) and self.params.get("autosave_light", True):
                    self._saver.request_auto(self.total_steps)

                # LLM commentary trigger
                llm_cfg = self.params.get("llm", {})
                freq = int(llm_cfg.get("freq", 0))
                if llm_cfg.get("enabled", False) and freq > 0 and (self.total_steps % freq == 0):
                    rolling_loss = (ep_loss_sum/max(1,ep_loss_count)) if ep_loss_count else 0.0
                    q_vals = None
                    try: q_vals = self.agent.q_values(state)
                    except Exception: pass
                    self._kickoff_llm_comment(ep, self.total_steps, eps, rolling_loss, score, q_vals)

                # epsilon
                if self._eps_override is not None:
                    eps = self._eps_override
                eps_min = float(self.params["epsilon"]["min"])
                eps_decay = float(self.params["epsilon"]["decay"])
                eps = max(eps_min, eps * eps_decay)
                self._eps_override = eps

                if done: break
                step += 1

            duration_ms = int((time.time() - t0) * 1000)
            avg_loss = (ep_loss_sum/max(1,ep_loss_count)) if ep_loss_count else 0.0
            row = {
                "episode": ep, "outcome": info.get("outcome","LOSS") if 'info' in locals() else "LOSS",
                "reason": info.get("reason","") if 'info' in locals() else "",
                "score": self.env.score, "snake_length": len(self.env.snake),
                "epsilon": eps, "avg_loss": avg_loss, "steps": step, "duration_ms": duration_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            self.history.append(row)
            csv_writer.writerow([row[k] for k in ["episode","outcome","reason","score","snake_length","epsilon","avg_loss","steps","duration_ms","timestamp"]])
            csv_file.flush()

            self.b.broadcast({"type":"episode_summary", **row})
            if row["outcome"] == "WIN":
                self.b.broadcast({"type":"game_over","episode":ep,"outcome":"WIN","win_length":len(self.env.snake),"message":"YOU WIN!"})

            ep += 1

        csv_file.close()
        self._running = False
