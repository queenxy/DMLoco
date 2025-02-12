import gym
import torch
import numpy as np

class AliengoILWrapper(gym.Wrapper):
    def __init__(self, env, n_action_steps, num_obs, obs_history_length):
        super().__init__(env)
        self.env = env

        self.obs_history_length = obs_history_length
        self.num_obs = num_obs
        self.n_action_steps = n_action_steps

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)

    def step(self, action):
        # privileged information and observation history are stored in info
        action = torch.from_numpy(action).to(self.env.device)
        total_rew = torch.zeros(action.shape[0],device=self.env.device)
        for i in range(self.n_action_steps):
            # self.delay_action = self.delay_action[1:] + [action[:,i,:]]
            # act = self.delay_action[0]
            act = action[:,i,:]
            # print(act)
            obs, rew, done, info = self.env.step(act)
            obs = self.convert_obs(obs)
            # print(obs)
            total_rew += rew
            env_ids = done.nonzero(as_tuple=False).flatten()
            self.reset_one_arg(env_ids)
            self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        return {'state': self.obs_history.cpu().numpy()}, total_rew.cpu().numpy(), torch.zeros_like(done).cpu().numpy(), done.cpu().numpy(), info

    def get_observations(self):
        obs = self.env.get_observations()
        obs = self.convert_obs(obs)
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        return {'state': self.obs_history.clone().cpu().numpy()}

    def reset_one_arg(self, env_ind):  # it might be a problem that this isn't getting called!!
        self.obs_history[env_ind, :] = 0


    def reset_arg(self, options_list=None):
        ret = super().reset()
        obs = self.convert_obs(ret)
        # print(obs)
        self.obs_history[:, :] = 0
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        print(self.obs_history.shape)
        return {'state': self.obs_history.clone().cpu().numpy()}
    
    def convert_obs(self, obs):
        o = torch.zeros((self.env.num_envs, 49),device=self.env.device)
        o[:,0:6] = obs[:,0:6]
        o[:,6:42] = obs[:,21:57]
        o[:,30:42] *= 10
        o[:,42:45] = torch.tensor([0.0, 1.0, 0.0], device=self.env.device)
        o[:,-4:] = torch.tensor([1., 0., 0., 0.],device=self.env.device)
        return o

        # obs[:,-7:-4] = torch.tensor([2.0, 0.0, 0.0], device=self.env.device)
        # obs[:,-4:] = torch.tensor([0., 0., 0., 1.],device=self.env.device)
        # return obs