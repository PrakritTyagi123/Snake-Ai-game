import torch

GRID_W, GRID_H = 20, 20

FPS_INIT, FPS_MIN, FPS_MAX = 600, 5, 600

BLINK_DUR = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
