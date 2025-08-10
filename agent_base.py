# agent_base.py
from abc import ABC, abstractmethod
import torch
from constants import GRID_W, GRID_H, device

class AgentBase(ABC):
    """Base class: defines state encoding and public interface."""

    def state(self, snake, food):
        hx, hy = snake.head()
        dx, dy = snake.dir
        ldx, ldy =  dy, -dx
        rdx, rdy = -dy,  dx

        def danger(offset):
            px, py = hx + offset[0], hy + offset[1]
            if px < 0 or px >= GRID_W or py < 0 or py >= GRID_H:
                return True
            return (px, py) in snake.body

        return torch.tensor([
            danger((dx,  dy)),      # danger straight
            danger((ldx, ldy)),     # danger left
            danger((rdx, rdy)),     # danger right
            food.pos[0] < hx,       # food left
            food.pos[0] > hx,       # food right
            food.pos[1] < hy,       # food up
            food.pos[1] > hy,       # food down
            1.0                     # bias
        ], dtype=torch.float32, device=device)

    @abstractmethod
    def act(self, state): ...
    @abstractmethod
    def step(self, state, action, reward, next_state, done): ...

    # Optional for ε-greedy agents
    def decay_eps(self): pass

    # Simple checkpoint helpers
    def save(self, path): torch.save(self.net.state_dict(), path)
    def load(self, path): self.net.load_state_dict(torch.load(path, map_location=device))
