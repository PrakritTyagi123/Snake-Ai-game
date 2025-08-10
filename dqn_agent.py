# dqn_agent.py
# Vanilla Deep Q-Network with replay buffer & target network
import random
import collections
from typing import Deque, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from constants import device
from models import Net
from agent_base import AgentBase


class DQNAgent(AgentBase):
    """Standard DQN implementation."""

    def __init__(
        self,
        lr: float = 1e-3,
        eps: float = 1.0,
        decay: float = 0.995,
        eps_min: float = 0.05,
        batch_size: int = 64,
        mem_size: int = 10_000,
        gamma: float = 0.9,
        target_update_freq: int = 1_000
    ) -> None:
        # Networks
        self.online_net = Net().to(device)
        self.target_net = Net().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.net = self.online_net          # for save / load

        # Optimiser & loss
        self.optim   = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()    # Huber

        # Replay memory
        self.memory: Deque[
            Tuple[torch.Tensor, int, float, torch.Tensor, bool]
        ] = collections.deque(maxlen=mem_size)

        # Hyper-parameters
        self.eps, self.decay, self.eps_min = eps, decay, eps_min
        self.batch_size, self.gamma = batch_size, gamma
        self.target_update_freq = target_update_freq

        # Counters
        self.total_steps = 0

    # ---------------  Interaction API  ---------------
    def act(self, state: torch.Tensor) -> int:
        if random.random() < self.eps:
            return random.randint(0, 3)
        with torch.no_grad():
            q = self.online_net(state.unsqueeze(0))[0]
            return int(torch.argmax(q))

    def step(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        # 1) Store transition
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return

        # 2) Sample mini-batch
        batch   = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        actions     = torch.tensor(actions,  dtype=torch.long,  device=device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones       = torch.tensor(dones,   dtype=torch.bool,  device=device)

        # 3) Q-learning target
        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(1)[0]
            q_target = rewards + (~dones) * self.gamma * q_next

        # 4) Optimise
        loss = self.loss_fn(q_pred, q_target)
        self.optim.zero_grad(); loss.backward(); self.optim.step()

        # 5) House-keeping
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_eps(self):           # ε-annealing
        self.eps = max(self.eps_min, self.eps * self.decay)
