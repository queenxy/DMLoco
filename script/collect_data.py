import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from aliengo_gym.envs import *
from aliengo_gym.envs.base.legged_robot_config import Cfg
from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
import random

def load_policy():
    body = torch.jit.load('./pretrained/body_latest.jit')
    adaptation_module = torch.jit.load('./pretrained/adaptation_module_latest.jit')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(headless=False):
    from aliengo_gym.envs.aliengo.aliengo_config import config_aliengo
    config_aliengo(Cfg)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.randomize_Kd_factor = True
    Cfg.domain_rand.randomize_Kp_factor = True
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = True

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 16
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 100
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    Cfg.terrain.teleport_thresh = 50.0

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "P"
    Cfg.env.episode_length_s = 10        # max episode length is episode_length_s / control_dt, 500 steps when episode_length_s = 10

    from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy()

    return env, policy


def play_go1(headless=True):
    env, policy = load_env(headless=headless)

    num_eval_steps = 20000
    num_envs = 16
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["bounding"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    obs = env.reset()

    obs_buf = np.zeros((num_eval_steps, num_envs, env.num_obs_history))
    action_buf = np.zeros((num_eval_steps,num_envs, 12))
    rew_buf = np.zeros((num_eval_steps, num_envs, 1))
    done_buf = np.zeros((num_eval_steps, num_envs, 1))
    cmd_buf = np.zeros((num_eval_steps, num_envs, 15))

    for i in tqdm(range(num_eval_steps)):
        obs_buf[i,:,:] = obs["obs_history"].cpu().numpy()
        cmd_buf[i,:,:] = env.commands.cpu().numpy()
        with torch.no_grad():
            actions = policy(obs)
        action_buf[i,:,:] = actions.cpu().numpy()
        # env.commands[:, 0] = x_vel_cmd
        # env.commands[:, 1] = y_vel_cmd
        # env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        # env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)
        rew_buf[i,:,:] = rew.reshape(-1,1).cpu().numpy()
        done_buf[i,:,:] = done.long().reshape(-1,1).cpu().numpy()

    np.savez("multi_vel.npz",states=obs_buf,actions=action_buf,rews=rew_buf,dones=done_buf,cmd=cmd_buf)




if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
