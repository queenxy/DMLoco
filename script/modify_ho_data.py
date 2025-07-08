import numpy as np

data = np.load("data/aliengo/multi_vel_onehot_300.npz")
print(data["states"].shape)
print(data["actions"].shape)

obs = data["states"]
action = data["actions"]
traj_lengths = data["traj_lengths"]

obs = obs[:, 1372:]
print(obs.shape)

np.savez("data/aliengo/multi_vel_onehot_2.npz", states=obs, actions=action, traj_lengths=traj_lengths)
