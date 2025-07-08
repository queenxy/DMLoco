"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys
import pretty_errors
import logging
import isaacgym

import math
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from matplotlib import pyplot as plt
import time

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)

    # Reset env before iteration starts
    agent.model.eval()
    firsts_trajs = np.zeros((agent.n_steps + 1, agent.n_envs))
    prev_obs_venv = agent.reset_env_all()
    firsts_trajs[0] = 1
    reward_trajs = np.zeros((agent.n_steps, agent.n_envs))
    fail_cout = 0
    tracking_error = np.zeros((agent.n_steps, agent.n_envs))
    # Collect a set of trajectories from env
    t = []
    for step in range(agent.n_steps):
        if step % 10 == 0:
            print(f"Processed step {step} of {agent.n_steps}")
        with torch.no_grad():
            cond = {
                "state": torch.from_numpy(prev_obs_venv["state"])
                .float()
                .to(agent.device)
            }
            tm = time.time()
            samples = agent.model(cond=cond, deterministic=True)
            t.append(time.time() - tm)
            output_venv = (
                samples.trajectories.cpu().numpy()
            )  # n_env x horizon x act
        action_venv = output_venv[:, : agent.act_steps]

        # Apply multi-step action
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            agent.venv.step(action_venv)
        )
        reward_trajs[step] = reward_venv
        firsts_trajs[step + 1] = terminated_venv | truncated_venv

        fail_cout += np.sum(terminated_venv | truncated_venv)

        measured_vels = agent.venv.env.base_lin_vel[:, :3].cpu().numpy()
        expect_vels = obs_venv["state"][:, -7:-4] * np.array([0.5,0.5,0.5])
        tracking_error[step] = np.linalg.norm(measured_vels - expect_vels, axis=1)**2
        # update for next step
        prev_obs_venv = obs_venv

    lower_percentile = np.percentile(tracking_error, 5)
    upper_percentile = np.percentile(tracking_error, 95)
    filtered_data = tracking_error[(tracking_error >= lower_percentile) & (tracking_error <= upper_percentile)]
    mean_90_percent = np.mean(filtered_data)

    print("Reward:", np.average(np.sum(reward_trajs,axis=0)))
    print("Success rate:", np.mean((np.sum(firsts_trajs[1:-1,:], axis=0) == 0).astype(float)))
    print("Tracking error:", mean_90_percent)
    # print(sum(t)/len(t))

if __name__ == "__main__":
    main()
