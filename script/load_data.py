import numpy as np

data = np.load("train.npz")
print(data["traj_lengths"].shape)
print(data["traj_lengths"])
print(np.sum(data["traj_lengths"]))
print(data["states"].shape)
print(data["actions"].shape)