# start.py — web UI + local LLM (no pygame)
"""
Unified launcher for Snake AI that can also download & wire up the local LLM.

Examples
--------
# Open the web UI and use the default small instruct model (downloads if missing)
python start.py --open

# Disable the LLM entirely
python start.py --open --nollm

# Use a different model/dir
python start.py --llm-model Qwen/Qwen2-0.5B-Instruct --llm-dir ai_chatmodel/Qwen2-0.5B-Instruct

# Only download the model then exit
python start.py --download-only
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import os
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

# On Windows, prefer Selector loop to avoid Proactor shutdown noise
if sys.platform.startswith("win"):
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# ---------- constants ----------
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FRONT_HOST = "127.0.0.1"
DEFAULT_FRONT_PORT = 8080
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 5000

# Defaults mirror backend/config.py
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL_HF_ID", "Qwen/Qwen2-0.5B-Instruct")
DEFAULT_LLM_DIR   = os.getenv("LLM_LOCAL_DIR", "ai_chatmodel/Qwen2-0.5B-Instruct")
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "250"))

# ---------- small utils ----------
def port_is_free(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) != 0

def wait_for_http(host: str, port: int, timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1.0)
            conn.request("GET", "/")
            conn.getresponse()
            return True
        except Exception:
            time.sleep(0.25)
    return False

# ---------- LLM download ----------
@dataclass
class LLMPlan:
    model_id: str
    local_dir: Path
    enabled: bool
    max_tokens: int
    require_cuda: Optional[bool] = None
    token: Optional[str] = None

def ensure_llm(plan: LLMPlan) -> bool:
    """
    Ensure the HF model repo is present at plan.local_dir.
    Returns True if the directory exists (downloaded or already present).
    """
    if not plan.enabled:
        print("[LLM] Disabled via flags; skipping download.")
        return True

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[LLM] huggingface_hub is not installed. Please `pip install -r requirements.txt`.")
        print("      Error:", e)
        return False

    plan.local_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic: if we already have a config.json or safetensors in the folder, assume it's downloaded
    has_config = (plan.local_dir / "config.json").exists()
    has_weights = any(plan.local_dir.glob("*.safetensors")) or (plan.local_dir / "model.safetensors").exists()
    if has_config and has_weights:
        print(f"[LLM] Found existing model files in {plan.local_dir} — skipping download.")
        return True

    print(f"[LLM] Downloading {plan.model_id} → {plan.local_dir} (first run may take a while)")
    try:
        snapshot_download(
            repo_id=plan.model_id,
            local_dir=str(plan.local_dir),
            local_dir_use_symlinks=False,
            token=plan.token or os.getenv("HUGGINGFACE_TOKEN"),
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "generation_config.json",
                "*.bin",
                "*.safetensors",
                "*.json",
                "*.txt",
                "*.model",
            ],
            ignore_patterns=[
                "flax_model.*",
                "tf_model.*",
                "*.onnx",
                "*-onnx/*",
            ],
        )
        print("[LLM] Download complete.")
        return True
    except Exception as e:
        print("[LLM] Failed to download model:", e)
        return False

# ---------- static web server ----------
class _CwdHandler(SimpleHTTPRequestHandler):
    # silence noisy logs
    def log_message(self, fmt, *args):  # noqa: N802
        pass

def serve_frontend(root: Path, host: str, port: int) -> ThreadingHTTPServer:
    os.chdir(str(root))
    httpd = ThreadingHTTPServer((host, port), _CwdHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[FE] Serving static frontend from {root} → http://{host}:{port}")
    return httpd

# ---------- backend (uvicorn) ----------
def run_backend(host: str, port: int):
    from uvicorn import Config, Server

    # Prefer package path "backend.app:app", fall back to plain "app:app"
    module = "backend.app:app"
    if not (REPO_ROOT / "backend" / "app.py").exists():
        module = "app:app"

    config = Config(module, host=host, port=port, reload=False, log_level="info")
    server = Server(config)
    print(f"[API] Starting FastAPI at http://{host}:{port} (module {module})")
    server.run()

def start_backend_thread(host: str, port: int) -> threading.Thread:
    t = threading.Thread(target=run_backend, args=(host, port), daemon=True)
    t.start()
    return t

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Snake AI launcher with optional LLM download.")
    p.add_argument("--open", action="store_true", help="Open the web UI in your browser.")
    p.add_argument("--front-host", default=DEFAULT_FRONT_HOST)
    p.add_argument("--front-port", type=int, default=DEFAULT_FRONT_PORT)
    p.add_argument("--api-host", default=DEFAULT_API_HOST)
    p.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)

    # LLM flags
    p.add_argument("--nollm", action="store_true", help="Disable LLM commentary completely.")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Hugging Face model id to download/use.")
    p.add_argument("--llm-dir", default=DEFAULT_LLM_DIR, help="Local directory to store/use the model files.")
    p.add_argument("--llm-max-tokens", dest="llm_max_tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    p.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_TOKEN"), help="HF token (env HUGGINGFACE_TOKEN also honored).")
    p.add_argument("--download-only", action="store_true", help="Download the LLM and exit.")

    # device pref
    dev = p.add_argument_group("device")
    dev.add_argument("--cpu", action="store_true", help="Force CPU (sets REQUIRE_CUDA=0).")
    dev.add_argument("--require-cuda", action="store_true", help="Require CUDA (sets REQUIRE_CUDA=1).")

    args = p.parse_args()

    # Decide frontend root automatically: prefer ./frontend/index.html, else ./index.html
    front_root = (REPO_ROOT / "frontend")
    if not (front_root / "index.html").exists():
        if (REPO_ROOT / "index.html").exists():
            front_root = REPO_ROOT
        else:
            front_root.mkdir(parents=True, exist_ok=True)
            (front_root / "index.html").write_text("<!doctype html><title>Snake AI</title><h1>Backend running.</h1>")

    # Make sure storage dirs exist
    (REPO_ROOT / "storage" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "storage" / "episodes").mkdir(parents=True, exist_ok=True)

    # LLM plan
    llm_enabled = not args.nollm
    plan = LLMPlan(
        model_id=args.llm_model,
        local_dir=(REPO_ROOT / args.llm_dir),
        enabled=llm_enabled,
        max_tokens=int(args.llm_max_tokens),
        require_cuda=(True if args.require_cuda else (False if args.cpu else None)),
        token=args.hf_token,
    )

    # Set env for backend/config.py consumption
    os.environ["LLM_ENABLED"] = "1" if plan.enabled else "0"
    os.environ["LLM_HARD_DISABLE"] = "0" if plan.enabled else "1"
    os.environ["LLM_BACKEND"] = "hf"
    os.environ["LLM_MODEL_HF_ID"] = plan.model_id
    os.environ["LLM_LOCAL_DIR"] = str(plan.local_dir)
    os.environ["LLM_MAX_TOKENS"] = str(plan.max_tokens)

    if plan.require_cuda is not None:
        os.environ["REQUIRE_CUDA"] = "1" if plan.require_cuda else "0"

    # If enabled, ensure model is present
    if not ensure_llm(plan):
        if args.download_only:
            sys.exit(1)
        else:
            print("[LLM] Proceeding without the LLM (fallback text will be used).")
            os.environ["LLM_ENABLED"] = "0"
            os.environ["LLM_HARD_DISABLE"] = "1"

    if args.download_only:
        print("[start.py] Download-only requested; exiting now.")
        return

    # Start frontend first (if port free)
    if not port_is_free(args.front_host, args.front_port):
        print(f"[FE] Port {args.front_port} is busy; pick another with --front-port.")
        sys.exit(2)
    fe_server = serve_frontend(front_root, args.front_host, args.front_port)

    # Then start backend (if port free)
    if not port_is_free(args.api_host, args.api_port):
        print(f"[API] Port {args.api_port} is busy; pick another with --api-port.")
        fe_server.shutdown()
        fe_server.server_close()
        sys.exit(2)
    be_thread = start_backend_thread(args.api_host, args.api_port)

    # Optionally open the browser to the frontend
    if args.open:
        url = f"http://{args.front_host}:{args.front_port}"
        time.sleep(0.25)
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    # Nudge the backend to warm up by hitting the root once
    wait_for_http(args.api_host, args.api_port, timeout=8.0)

    # Block the main thread, but keep handling Ctrl+C gracefully
    print("[start.py] Press Ctrl+C to stop.")
    try:
        while be_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[start.py] Shutting down…")
    finally:
        try:
            fe_server.shutdown()
        except Exception:
            pass
        try:
            fe_server.server_close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
