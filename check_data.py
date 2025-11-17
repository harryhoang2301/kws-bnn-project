import numpy as np
from pathlib import Path

path = Path("data/processed/logmel_train.npz")
data = np.load(path)
x, y = data["x"], data["y"]

print("x shape:", x.shape)  # (N, 40, T)
print("y shape:", y.shape)  # (N,)
print("First 10 labels:", y[:10])
