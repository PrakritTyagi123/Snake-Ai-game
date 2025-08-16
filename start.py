import os, sys, pathlib, threading, time, socket, functools

# --- Paths/ports ---
ROOT = pathlib.Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_PORT = 8080
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5000

# --- Local model setup (Qwen2-0.5B-Instruct) ---
MODEL_ROOT = ROOT / "ai_chatmodel"
MODEL_ID = os.getenv("LLM_MODEL_HF_ID", "Qwen/Qwen2-0.5B-Instruct")
MODEL_DIR = MODEL_ROOT / MODEL_ID.split("/")[-1]

# Use ai_chatmodel as the cache/home to keep everything local to the project
os.environ.setdefault("HF_HOME", str(MODEL_ROOT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_ROOT))

# App LLM config (Hugging Face local backend)
os.environ.setdefault("LLM_LOCAL_DIR", str(MODEL_DIR))
os.environ.setdefault("LLM_BACKEND", "hf")
os.environ.setdefault("LLM_ENABLED", "1")

def ensure_local_model():
    """Download HF repo snapshot into ai_chatmodel/<repo-name> if missing."""
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        return
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[start.py] Missing dependency 'huggingface_hub'. Run: pip install huggingface_hub")
        raise
    print(f"[start.py] Downloading {MODEL_ID} to {MODEL_DIR} ... (first run only)")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        revision=None,
    )
    print("[start.py] Download complete.")

def is_port_open(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

def serve_frontend_in_thread():
    """Serve ./frontend on FRONTEND_PORT if possible."""
    if not FRONTEND_DIR.exists():
        return False
    if is_port_open("127.0.0.1", FRONTEND_PORT):
        return True
    try:
        import http.server, socketserver
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FRONTEND_DIR))
        httpd = socketserver.TCPServer(("127.0.0.1", FRONTEND_PORT), handler)
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        print(f"[start.py] Frontend served from {FRONTEND_DIR} at http://127.0.0.1:{FRONTEND_PORT}")
        return True
    except Exception as e:
        print("[start.py] Could not start static server for frontend:", e)
        return False

def open_browser_later():
    import webbrowser
    # prefer frontend if available
    target = f"http://127.0.0.1:{FRONTEND_PORT}" if is_port_open("127.0.0.1", FRONTEND_PORT) \
             else f"http://{BACKEND_HOST}:{BACKEND_PORT}/docs"
    threading.Timer(0.8, lambda: webbrowser.open(target)).start()

# Optional: autorun training via REST once backend is up
def wait_for_backend(host="127.0.0.1", port=5000, timeout=20):
    try:
        import requests
    except Exception:
        return False
    url = f"http://{host}:{port}/api/config"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=1.5)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def autorun_training(host="127.0.0.1", port=5000):
    try:
        import requests
    except Exception as e:
        print("[start.py] autorun requires 'requests' (pip install requests)"); return
    if not wait_for_backend(host, port):
        print("[start.py] backend not ready for autorun")
        return
    try:
        payload = {
            "run_title": "Auto Run",
            "max_episodes": 0,
            "max_steps": 1000,
            "gamma": 0.99,
            "lr": 0.0005,
            "batch_size": 64,
            "epsilon": {"start": 1.0, "min": 0.05, "decay": 0.999},
        }
        r = requests.post(f"http://{host}:{port}/api/start", json=payload, timeout=5)
        print("[start.py] autorun /api/start →", r.status_code)
    except Exception as e:
        print("[start.py] autorun failed:", e)

# Ensure model presence before launching backend
ensure_local_model()

# CLI flags
args = set(sys.argv[1:])
if "--serve-frontend" in args or "--open" in args:
    serve_frontend_in_thread()
if "--open" in args:
    open_browser_later()
if "--autorun" in args:
    threading.Thread(target=lambda: autorun_training(BACKEND_HOST, BACKEND_PORT), daemon=True).start()

# --- Now import and run the backend app ---
from backend.app import app  # adjust path if your app lives elsewhere

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, log_level="info")
