import numpy as np

data = np.load("multi_vel.npz")
print(data["states"].shape)
print(data["actions"].shape)

obs = data["states"]
print(obs[0,0,-73:])
print(obs[0,1,-73:])
print(obs[0,2,-73:])
print(obs[0,3,-73:])
print(obs[0,4,-73:])
print(obs[0,5,-73:])
print(obs[0,6,-73:])
print(obs[0,7,-73:])
print(obs[0,8,-73:])
print(obs[0,9,-73:])
print(obs[0,10,-73:])

# print(obs[0,5,-73:])
# print(obs[1,5,-73:])
# print(obs[2,5,-73:])
# print(obs[3,5,-73:])
# print(obs[4,5,-73:])