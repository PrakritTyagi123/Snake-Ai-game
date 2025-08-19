# config.py
"""
Central configuration for Snake AI.

This file reads environment variables so you can override defaults without
changing code. Relevant env vars:

- LLM_ENABLED (0/1)
- LLM_HARD_DISABLE (0/1)
- REQUIRE_CUDA (0/1)
- CHECKPOINT_DIR
- DEFAULT_* (see below)
- BOARD_WIDTH, BOARD_HEIGHT, WIN_LENGTH
"""

import os

def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _s(name: str, default: str) -> str:
    return os.getenv(name, default)

# --------- Hardware / runtime ---------
REQUIRE_CUDA = _i("REQUIRE_CUDA", 0)  # set to 1 to require GPU, else falls back to CPU

# --------- Checkpointing ---------
CHECKPOINT_DIR = _s("CHECKPOINT_DIR", "storage/checkpoints")

# --------- Board / game ---------
BOARD_WIDTH  = _i("BOARD_WIDTH", 20)
BOARD_HEIGHT = _i("BOARD_HEIGHT", 20)
# If not provided, default win length is fill-the-board minus 1
WIN_LENGTH   = _i("WIN_LENGTH", BOARD_WIDTH * BOARD_HEIGHT - 1)

# --------- Training defaults ---------
DEFAULT_GAMMA          = _f("DEFAULT_GAMMA", 0.95)
DEFAULT_LR             = _f("DEFAULT_LR", 1e-3)
DEFAULT_BATCH          = _i("DEFAULT_BATCH", 64)
DEFAULT_MAX_EPISODES   = _i("DEFAULT_MAX_EPISODES", 0)      # 0 => endless
DEFAULT_MAX_STEPS      = _i("DEFAULT_MAX_STEPS", 1000)
STEP_DELAY_MS          = _i("STEP_DELAY_MS", 0)
TARGET_UPDATE_FREQ     = _i("TARGET_UPDATE_FREQ", 250)

# --------- Epsilon schedule ---------
DEFAULT_EPS_START      = _f("DEFAULT_EPS_START", 1.0)
DEFAULT_EPS_MIN        = _f("DEFAULT_EPS_MIN", 0.05)
DEFAULT_EPS_DECAY      = _f("DEFAULT_EPS_DECAY", 0.995)

# --------- LLM / commentary ---------
LLM_ENABLED            = _i("LLM_ENABLED", 1)               # set to 0 via --nollm
LLM_HARD_DISABLE       = _i("LLM_HARD_DISABLE", 0)          # set to 1 via --nollm to block re-enable mid-run
LLM_BACKEND            = _s("LLM_BACKEND", "hf")            # "hf" or your custom backend in llm.py
LLM_MODEL_HF_ID        = _s("LLM_MODEL_HF_ID", "Qwen/Qwen2-0.5B-Instruct")
LLM_LOCAL_DIR          = _s("LLM_LOCAL_DIR", "ai_chatmodel/Qwen2-0.5B-Instruct")
LLM_MAX_TOKENS         = _i("LLM_MAX_TOKENS", 250)

# config.py — ADD these defaults near other config constants
EPISODES_DIR = os.getenv("EPISODES_DIR", "storage/episodes")
EPISODES_WRITE_BATCH = int(os.getenv("EPISODES_WRITE_BATCH", "100"))

