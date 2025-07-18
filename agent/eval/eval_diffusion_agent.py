"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.

"""

import os
import numpy as np
import torch
import logging
import time as tm

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent


class EvalDiffusionAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):

        # Start training loop
        timer = Timer()

        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))

        # print(self.env_name)
        # if self.env_name == "aliengo":
        #     for steps in range(100):
        #         actions = np.zeros((1, 2, 12),dtype=np.float32)
        #         self.venv.env.p_gains = 80.0
        #         self.venv.env.d_gains = 4.0
        #         obs, rew, _, done, info = self.venv.step(actions)
        #     self.venv.env.p_gains = 40.0
        #     self.venv.env.d_gains = 2.0
        # data = np.load("/home/qxy/dppo/data/aliengo/trotting_straight/trotting_clean_50_scaled.npz")
        # states = data["states"]
        # action = data["actions"]
        # prev_obs_venv["state"] = states[0:1,:]

        # Collect a set of trajectories from env
        for step in range(self.n_steps):
            if step % 10 == 0:
                print(f"Processed step {step} of {self.n_steps}")
            # print(prev_obs_venv["state"][0,-49:])
            # Select action
            # t1 = tm.time()
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples = self.model(cond=cond, deterministic=True)
                output_venv = (
                    samples.trajectories.cpu().numpy()
                )  # n_env x horizon x act
            action_venv = output_venv[:, : self.act_steps]
            # print(tm.time()-t1)
            # print(action_venv)

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            # update for next step
            prev_obs_venv = obs_venv

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if (
                self.furniture_sparse_reward
            ):  # only for furniture tasks, where reward only occurs in one env step
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            log.info("[WARNING] No episode completed within the iteration!")

        # Log loss and save metrics
        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )
