from __future__ import annotations
import random
from collections import deque
from typing import Deque, Tuple, List, Dict, Any

Transition = Tuple[list, int, float, list, bool]

class ReplayBuffer:
    def __init__(self, capacity:int=100_000, seed:int=0):
        self.buf: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self._capacity = capacity
        self._seed = seed

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size:int) -> List[Transition]:
        return self.rng.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)

    # --------- NEW: persistence ---------
    def state_dict(self) -> Dict[str, Any]:
        """Lightweight serialization of the buffer."""
        return {
            "capacity": self.buf.maxlen or self._capacity,
            "seed": self._seed,
            "data": list(self.buf),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        cap = int(state.get("capacity", self._capacity))
        data = state.get("data", [])
        self._seed = int(state.get("seed", self._seed))
        self.buf = deque(data, maxlen=cap)
        self.rng = random.Random(self._seed)
