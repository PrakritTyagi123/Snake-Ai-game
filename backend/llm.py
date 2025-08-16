from __future__ import annotations
import os, textwrap, threading
from typing import Dict, Any, Optional

# Lazy import to keep startup fast
_torch = None
_tf = None
def _lazy_import_tf():
    global _torch, _tf
    if _tf is None:
        import torch as _torch  # noqa
        import transformers as _tf  # noqa
    return _torch, _tf

class LLMReporter:
    """
    Local commentary generator backed by Hugging Face Transformers (offline).
    Default model: Qwen/Qwen2-0.5B-Instruct downloaded by start.py to ai_chatmodel.
    Falls back to a deterministic text if model isn't available.
    """
    def __init__(
        self,
        backend: str = "hf",
        model_id: str = "Qwen/Qwen2-0.5B-Instruct",
        local_dir: Optional[str] = None,
        max_tokens: int = 250,
    ):
        self.backend = (backend or "hf").lower()
        self.model_id = model_id
        self.local_dir = local_dir
        self.max_tokens = int(max_tokens)
        self._pipe = None
        self._pipe_lock = threading.Lock()

    # ----- prompt + fallback -----
    def _build_prompt(self, snap: Dict[str, Any]) -> str:
        return textwrap.dedent(f"""
        You are a terse RL training commentator. Summarize what the Snake DDQN agent is
        currently doing and what it has learned, using 3–6 short bullet lines.

        Context:
        - step: {snap.get('step')}
        - episode: {snap.get('episode')}
        - epsilon: {snap.get('epsilon'):.4f}
        - avg_loss_this_ep: {snap.get('avg_loss'):.6f}
        - score_so_far: {snap.get('score_so_far')}
        - q_values: {snap.get('q_values')}
        - recent_episodes (most recent first, up to 10):
          {snap.get('recent')}
        Guidance:
        - Be concrete (mention ε, loss trends, behaviors like wall avoidance, food seeking).
        - Avoid fluff; keep it practical.
        - 3–6 lines, each starting with "•".
        """).strip()

    def _fallback(self, snap: Dict[str, Any]) -> str:
        last_k = snap.get("recent", []) or []
        wins = sum(1 for r in last_k if r.get("outcome") == "WIN")
        wr = (100.0 * wins / max(1, len(last_k))) if last_k else 0.0
        avg_len = sum(r.get("snake_length", 0) for r in last_k) / max(1, len(last_k)) if last_k else 0.0
        q = snap.get("q_values")
        qtxt = "n/a" if q is None else ", ".join(f"{float(x):.2f}" for x in q)
        return textwrap.dedent(f"""
        • step {snap.get('step')} (ep {snap.get('episode')}), ε={snap.get('epsilon'):.3f}, loss≈{snap.get('avg_loss'):.4f}, score={snap.get('score_so_far')}
        • recent {len(last_k)} eps → win rate ≈ {wr:.1f}%, avg length ≈ {avg_len:.1f}
        • Q snapshot: {qtxt}
        • Learning trend: fewer wall collisions; prefers safe paths toward food when available.
        • Next: tune ε/decay or LR to stabilize turns near edges and reduce tail hits.
        """).strip()

    # ----- HF pipeline (lazy) -----
    def _ensure_pipe(self) -> bool:
        if self._pipe is not None:
            return True
        if self.backend != "hf":
            return False
        if not self.local_dir or not os.path.exists(os.path.join(self.local_dir, "config.json")):
            return False
        with self._pipe_lock:
            if self._pipe is not None:
                return True
            try:
                torch, tf = _lazy_import_tf()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                tok = tf.AutoTokenizer.from_pretrained(self.local_dir, trust_remote_code=True)
                model = tf.AutoModelForCausalLM.from_pretrained(
                    self.local_dir,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                self._pipe = tf.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tok,
                    device=0 if device == "cuda" else -1,
                )
                return True
            except Exception:
                self._pipe = None
                return False

    # ----- public -----
    def generate_comment(self, snap: Dict[str, Any]) -> str:
        if self.backend != "hf":
            return self._fallback(snap)
        if not self._ensure_pipe():
            return self._fallback(snap)

        prompt = self._build_prompt(snap)
        try:
            out = self._pipe(
                prompt,
                max_new_tokens=int(self.max_tokens),
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                eos_token_id=self._pipe.tokenizer.eos_token_id,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
            )
            text = (out[0]["generated_text"] or "").strip()
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            bullets = []
            for ln in lines:
                if not ln.startswith("•"):
                    ln = "• " + ln.lstrip("-• ")
                bullets.append(ln)
            return "\n".join(bullets[:8]) or self._fallback(snap)
        except Exception:
            return self._fallback(snap)
