import torch
import numpy as np
from aliengo_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class VelTrackingRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    # def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.env.base_lin_vel[:, 2])

    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
    #                      dim=1)

    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
    #     out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_pos(self):
    #     # Penalize dof positions
    #     return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.env.dof_vel), dim=1)

    # def _reward_action_smoothness_1(self):
    #     # Penalize changes in actions
    #     diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
    #     diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
    #     return torch.sum(diff, dim=1)

    # def _reward_action_smoothness_2(self):
    #     # Penalize changes in actions
    #     diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
    #     diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
    #     diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
    #     return torch.sum(diff, dim=1)

    # def _reward_feet_slip(self):
    #     contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.env.last_contacts)
    #     self.env.last_contacts = contact
    #     foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
    #     rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
    #     return rew_slip

    # def _reward_feet_clearance_cmd_linear(self):
    #     phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    #     foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
    #     target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
    #     rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
    #     return torch.sum(rew_foot_clearance, dim=1)

    # def _reward_feet_impact_vel(self):
    #     prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
    #     contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

    #     rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

    #     return torch.sum(rew_foot_impact_vel, dim=1)