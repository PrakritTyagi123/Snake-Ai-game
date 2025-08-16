
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # modern API

class PolicyNet(nn.Module):
    def __init__(self, in_dim:int, hidden:Tuple[int,int]=(128,64), out_dim:int=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], out_dim)
        self.activations: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        # x: [B, in_dim], keep model in FP32; AMP handles mixed precision
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        q = self.out(h2)
        # store activations for visualization (detach for safety)
        self.activations = {
            "dense1": h1.detach(),
            "dense2": h2.detach(),
            "output": q.detach()
        }
        return q

class DDQNAgent:
    def __init__(self, in_dim:int, out_dim:int=4, lr=1e-3, gamma=0.95, device="cuda"):
        self.device = torch.device(device)
        self.gamma = gamma
        # Keep model parameters in FP32; autocast() + GradScaler handles precision
        self.policy = PolicyNet(in_dim, (128,64), out_dim).to(self.device)
        self.target = PolicyNet(in_dim, (128,64), out_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scaler = GradScaler('cuda')

    def act(self, state, epsilon: float):
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(0, 4, (1,), device=self.device).item())
        with torch.no_grad(), autocast('cuda'):
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s)
            return int(torch.argmax(q, dim=1).item())

    def learn(self, batch, target_net_update=None):
        # batch: list of (s, a, r, s2, done)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        with autocast('cuda'):
            q_pred = self.policy(states).gather(1, actions)  # [B,1]

            # Double DQN: use policy net to choose action, target net to evaluate
            next_q_policy = self.policy(next_states)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)  # [B,1]

            next_q_target = self.target(next_states).gather(1, next_actions)  # [B,1]
            y = rewards + (1.0 - dones) * self.gamma * next_q_target

            loss = F.smooth_l1_loss(q_pred, y)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if target_net_update is not None and target_net_update:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.detach().float().mean().item())

    def q_values(self, state):
        with torch.no_grad(), autocast('cuda'):
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s)
            return q.squeeze(0).float().tolist()

    def activation_summary(self) -> Dict[str, List[float]]:
        out = {}
        for name, tensor in self.policy.activations.items():
            if tensor is None: continue
            t = tensor.float().mean(dim=0)  # [features]
            t = t[:64]  # cap payload
            out[name] = t.cpu().tolist()
        return out

    def save(self, path:str):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.optimizer.state_dict(),
        }, path)

    def load(self, path:str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["opt"])
