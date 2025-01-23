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

    agent.venv.env.start_recording()

    # print(agent.env_name)
    # if agent.env_name == "aliengo":
    #     for steps in range(100):
    #         actions = np.zeros((1, 2, 12),dtype=np.float32)
    #         agent.venv.env.p_gains = 80.0
    #         agent.venv.env.d_gains = 4.0
    #         obs, rew, _, done, info = agent.venv.step(actions)
    #     agent.venv.env.p_gains = 40.0
    #     agent.venv.env.d_gains = 2.0
    measured_x_vels = np.zeros(agent.n_steps)
    measured_y_vels = np.zeros(agent.n_steps)
    measured_ang_vels = np.zeros(agent.n_steps)

    # Collect a set of trajectories from env
    for step in range(agent.n_steps):
        if step % 10 == 0:
            print(f"Processed step {step} of {agent.n_steps}")
        with torch.no_grad():
            cond = {
                "state": torch.from_numpy(prev_obs_venv["state"])
                .float()
                .to(agent.device)
            }
            samples = agent.model(cond=cond, deterministic=True)
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

        measured_x_vels[step] = agent.venv.env.base_lin_vel[0, 0]
        measured_y_vels[step] = agent.venv.env.base_lin_vel[0, 1]
        measured_ang_vels[step] = agent.venv.env.base_ang_vel[0, 2]

        # update for next step
        prev_obs_venv = obs_venv

    plt.figure()
    plt.plot(np.arange(agent.n_steps),measured_x_vels,label='measured_x_vel')
    plt.plot(np.arange(agent.n_steps),measured_y_vels,label='measured_y_vel')
    plt.plot(np.arange(agent.n_steps),measured_ang_vels,label='measured_ang_vel')
    plt.legend()
    plt.savefig('vel.png')

    import cv2
    video_writer = cv2.VideoWriter('temp_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (360, 240))
    for frame in agent.venv.env.video_frames:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(image)
    video_writer.release()

    import subprocess
    subprocess.run([
        "ffmpeg",
        "-i", "temp_video.mp4",  # 输入文件
        "-vcodec", "libx264",  # 视频编码格式
        "-acodec", "aac",  # 音频编码格式
        "-pix_fmt", "yuv420p",  # 像素格式
        "-movflags", "+faststart",  # 优化 MP4 文件
        "video.mp4"  # 输出文件
    ])

    import os
    os.remove("temp_video.mp4")

    print(reward_trajs)
    print("Reward:", np.sum(reward_trajs,axis=0))

if __name__ == "__main__":
    main()
