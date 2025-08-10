# models.py
import torch
import torch.nn as nn

class Net(nn.Module):
    """Simple feed-forward network for DQN."""
    def __init__(self, in_dim: int = 8, hid: int = 64, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
