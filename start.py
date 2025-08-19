# start.py
"""
Launcher for Snake AI project.

Usage:
  python start.py --open            # open browser to the frontend
  python start.py --open --nollm    # launch with LLM fully disabled

It serves the frontend on http://127.0.0.1:8080 and runs the FastAPI backend on http://127.0.0.1:5000.
"""
from __future__ import annotations
import sys, os, asyncio, argparse, threading, webbrowser, socket
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# --- Windows: use Selector event loop (avoids Proactor assertion at shutdown) ---
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
# -------------------------------------------------------------------------------

# ----------------------------- CLI ARGUMENTS -----------------------------------
parser = argparse.ArgumentParser(description="Run Snake AI frontend + backend")
parser.add_argument("--open", action="store_true", help="Open browser to the frontend")
parser.add_argument("--nollm", action="store_true", help="Disable LLM (no model load or commentary)")
parser.add_argument("--front-host", default="127.0.0.1")
parser.add_argument("--front-port", type=int, default=8080)
parser.add_argument("--api-host", default="127.0.0.1")
parser.add_argument("--api-port", type=int, default=5000)
args = parser.parse_args()

# --- Hard-disable LLM if requested (must happen BEFORE importing backend.app) ---
if args.nollm:
    os.environ["LLM_ENABLED"] = "0"
    os.environ["LLM_HARD_DISABLE"] = "1"
# -------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FRONT_DIR = ROOT / "frontend"

def _find_free_port(host: str, desired: int) -> int:
    # return desired if free, otherwise a free ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, desired))
            return desired
        except OSError:
            s.bind((host, 0))
            return s.getsockname()[1]

class _StaticHandler(SimpleHTTPRequestHandler):
    # Serve files from the given directory without changing CWD (py>=3.7 supports 'directory')
    def __init__(self, *handler_args, directory=None, **handler_kwargs):
        super().__init__(*handler_args, directory=directory, **handler_kwargs)

def start_frontend(host: str, port: int, webroot: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    port = _find_free_port(host, port)
    handler = lambda *a, **k: _StaticHandler(*a, directory=str(webroot), **k)  # noqa: E731
    httpd = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[start.py] Frontend served from {webroot} at http://{host}:{port}")
    return httpd, t

def main() -> None:
    if not FRONT_DIR.exists():
        print(f"[start.py] WARNING: frontend dir not found: {FRONT_DIR}")

    # 1) Start the static server for the frontend
    fe_server, fe_thread = start_frontend(args.front_host, args.front_port, FRONT_DIR)

    # 2) Optionally open the browser
    if args.open:
        try:
            webbrowser.open(f"http://{args.front_host}:{args.front_port}", new=1, autoraise=True)
        except Exception:
            pass

    # 3) Start the FastAPI backend (uvicorn) in the main thread (blocking)
    try:
        import uvicorn
        # Import app AFTER env is set so backend reads LLM_* correctly on init
        from backend.app import app  # backend must be a package (backend/__init__.py present)

        print(f"[start.py] Backend at http://{args.api_host}:{args.api_port}")
        uvicorn.run(app, host=args.api_host, port=args.api_port, log_level="info")
    except KeyboardInterrupt:
        print("[start.py] KeyboardInterrupt received - shutting down gracefully...")
    finally:
        # 4) Shutdown frontend server when backend exits
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
