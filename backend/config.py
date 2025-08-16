import os

# ===== Runtime / Device =====
REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "1") == "1"

HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
PORT = int(os.getenv("BACKEND_PORT", "5000"))

# ===== Board / Game =====
BOARD_WIDTH = int(os.getenv("BOARD_WIDTH", "20"))
BOARD_HEIGHT = int(os.getenv("BOARD_HEIGHT", "20"))
# Default WIN is "fill the board" (minus head start)
WIN_LENGTH = int(os.getenv("WIN_LENGTH", str(BOARD_WIDTH * BOARD_HEIGHT - 1)))

# ===== Training Defaults =====
DEFAULT_MAX_EPISODES = int(os.getenv("MAX_EPISODES", "0"))    # 0 or negative → run indefinitely
DEFAULT_MAX_STEPS = int(os.getenv("MAX_STEPS", "1000"))
DEFAULT_GAMMA = float(os.getenv("GAMMA", "0.95"))
DEFAULT_LR = float(os.getenv("LR", "0.001"))
DEFAULT_BATCH = int(os.getenv("BATCH_SIZE", "64"))
DEFAULT_EPS_START = float(os.getenv("EPS_START", "1.0"))
DEFAULT_EPS_MIN = float(os.getenv("EPS_MIN", "0.05"))
DEFAULT_EPS_DECAY = float(os.getenv("EPS_DECAY", "0.995"))
DEFAULT_AUTOSAVE_STEPS = int(os.getenv("AUTOSAVE_STEPS", "1000"))
TARGET_UPDATE_FREQ = int(os.getenv("TARGET_UPDATE_FREQ", "1000"))  # steps

# ===== Live Controls =====
STEP_DELAY_MS = int(os.getenv("STEP_DELAY_MS", "0"))  # 0 = fastest

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(BASE_DIR, "storage", "checkpoints"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "storage", "logs"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===== LLM (optional) =====
ENABLE_LLM = os.getenv("ENABLE_LLM", "1") == "1"
LLM_MODEL_ID = os.getenv("LLM_MODEL", "distilgpt2")
LLM_THOUGHT_EVERY_N_STEPS = int(os.getenv("LLM_THOUGHT_N", "5"))

# ---- LLM local defaults (Qwen2-0.5B-Instruct) ----
def _as_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "on")

LLM_ENABLED = _as_bool(os.getenv("LLM_ENABLED", "1"))
LLM_COMMENT_FREQ = int(os.getenv("LLM_COMMENT_FREQ", "1000"))

# We default to the local Hugging Face backend
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf").lower()
LLM_MODEL_HF_ID = os.getenv("LLM_MODEL_HF_ID", "Qwen/Qwen2-0.5B-Instruct")

# Root path where start.py downloads the model (can be overridden by env)
try:
    import pathlib
    _ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
except Exception:
    _ROOT = None
LLM_LOCAL_DIR = os.getenv("LLM_LOCAL_DIR", str((_ROOT / "ai_chatmodel" / "Qwen2-0.5B-Instruct") if _ROOT else "ai_chatmodel/Qwen2-0.5B-Instruct"))

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "250"))

