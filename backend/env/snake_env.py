
from __future__ import annotations
import random
from typing import List, Tuple, Dict, Optional

class SnakeEnv:
    """Simple Snake environment with grid and classic rules.

    State for learning: you can choose a compact feature vector outside this class.
    This class focuses on game mechanics and exposes grid for rendering.
    """

    ACTIONS = [(0,-1),(0,1),(-1,0),(1,0)]  # Up, Down, Left, Right

    def __init__(self, width:int=20, height:int=20, win_length:int=50, seed:int=0):
        self.w = width
        self.h = height
        self.rng = random.Random(seed)
        self.win_length = win_length
        self.reset()

    def reset(self):
        cx, cy = self.w//2, self.h//2
        self.snake: List[Tuple[int,int]] = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self.direction = (1,0)  # moving right
        self.spawn_food()
        self.steps = 0
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            fx = self.rng.randrange(self.w)
            fy = self.rng.randrange(self.h)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break

    def step(self, action:int):
        self.steps += 1
        if action is not None and 0 <= action < 4:
            proposed = SnakeEnv.ACTIONS[action]
            # prevent 180 reversal
            if (proposed[0] * -1, proposed[1] * -1) != self.direction:
                self.direction = proposed

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        nx, ny = head_x + dx, head_y + dy

        # check collisions
        if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h:
            return self.get_state(), -1.0, True, {"outcome":"LOSS", "reason":"hit_wall"}
        if (nx, ny) in self.snake:
            return self.get_state(), -1.0, True, {"outcome":"LOSS", "reason":"self_bite"}

        # move
        self.snake.insert(0, (nx, ny))

        reward = -0.01  # small living penalty
        done = False
        info: Dict = {"outcome":"RUNNING"}

        if (nx, ny) == self.food:
            self.score += 1
            reward = 1.0
            self.spawn_food()
        else:
            self.snake.pop()

        # win condition
        if len(self.snake) >= self.win_length:
            done = True
            info = {"outcome":"WIN","reason":"WIN"}

        # Optional timeout (prevent infinite loops)
        if self.steps > self.w * self.h * 10 and not done:
            done = True
            reward = -1.0
            info = {"outcome":"LOSS", "reason":"timeout"}

        return self.get_state(), reward, done, info

    # --------------- helpers ---------------
    def get_state(self):
        """Return a compact state vector for learning (16 dims)."""
        head = self.snake[0]
        food = self.food
        dir_vec = self.direction
        left = (-dir_vec[1], dir_vec[0])
        right = (dir_vec[1], -dir_vec[0])

        def danger(a):
            nx, ny = head[0] + a[0], head[1] + a[1]
            if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h: return 1.0
            if (nx, ny) in self.snake: return 1.0
            return 0.0

        sx, sy = head
        fx, fy = food
        # normalized deltas
        dx = (fx - sx) / max(1, self.w)
        dy = (fy - sy) / max(1, self.h)

        state = [
            float(dir_vec[0]==0 and dir_vec[1]==-1),  # up
            float(dir_vec[0]==0 and dir_vec[1]== 1),  # down
            float(dir_vec[0]==-1 and dir_vec[1]==0),  # left
            float(dir_vec[0]== 1 and dir_vec[1]==0),  # right
            danger(dir_vec), danger(left), danger(right),
            dx, dy,
            sx/self.w, sy/self.h, fx/self.w, fy/self.h,
            len(self.snake)/float(self.w*self.h),
            self.score/100.0,
            1.0  # bias
        ]
        return state

    def render_spec(self):
        """Grid spec for frontend canvas."""
        return {
            "w": self.w,
            "h": self.h,
            "snake": [[x,y] for (x,y) in self.snake],
            "food": [self.food[0], self.food[1]]
        }
