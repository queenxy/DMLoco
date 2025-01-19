from typing import Union
from params_proto import Meta
from aliengo_gym.envs.base.legged_robot_config import Cfg


def config_aliengo(Cnfg: Union[Cfg, Meta]):
    # Set initial state
    Cnfg.init_state.pos = [0.0, 0.0, 0.43]  # x,y,z [m]
    Cnfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.0,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.0,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

    # Control settings
    Cnfg.control.control_type = "P"
    Cnfg.control.stiffness = {'joint': 40.}  # [N*m/rad]
    Cnfg.control.damping = {'joint': 2.0}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    Cnfg.control.action_scale = 0.25
    Cnfg.control.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    Cnfg.control.decimation = 4

    Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf'
    Cnfg.asset.foot_name = "foot"
    Cnfg.asset.penalize_contacts_on = ["thigh", "calf"]
    Cnfg.asset.terminate_after_contacts_on = ["base"]
    Cnfg.asset.self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
    Cnfg.asset.flip_visual_attachments = True
    Cnfg.asset.fix_base_link = False

    # Environmental settings
    Cnfg.env.num_envs = 4000
    Cnfg.env.num_observations = 73
    Cnfg.env.num_scalar_observations = 73
    Cnfg.env.num_privileged_obs = 6
    Cnfg.env.num_observation_history = 30
    Cnfg.env.observe_two_prev_actions = True
    Cnfg.env.observe_yaw = False
    Cnfg.env.observe_only_ang_vel = True
    Cnfg.env.observe_vel = False
    Cnfg.env.observe_gait_commands = True
    Cnfg.env.observe_timing_parameter = False
    Cnfg.env.observe_clock_inputs = True
    Cnfg.env.priv_observe_motion = False
    Cnfg.env.priv_observe_gravity_transformed_motion = False
    Cnfg.env.priv_observe_friction_indep = False
    Cnfg.env.priv_observe_friction = True
    Cnfg.env.priv_observe_restitution = True
    Cnfg.env.priv_observe_base_mass = False
    Cnfg.env.priv_observe_gravity = False
    Cnfg.env.priv_observe_com_displacement = False
    Cnfg.env.priv_observe_ground_friction = False
    Cnfg.env.priv_observe_ground_friction_per_foot = False
    Cnfg.env.priv_observe_motor_strength = False
    Cnfg.env.priv_observe_motor_offset = False
    Cnfg.env.priv_observe_Kp_factor = False
    Cnfg.env.priv_observe_Kd_factor = False
    Cnfg.env.priv_observe_body_velocity = True
    Cnfg.env.priv_observe_body_height = True
    Cnfg.env.priv_observe_desired_contact_states = False
    Cnfg.env.priv_observe_contact_forces = False
    Cnfg.env.priv_observe_foot_displacement = False
    Cnfg.env.priv_observe_gravity_transformed_foot_displacement = False

    # Domain randomization settings
    Cnfg.domain_rand.lag_timesteps = 6
    Cnfg.domain_rand.randomize_lag_timesteps = True
    Cnfg.domain_rand.randomize_rigids_after_start = False
    Cnfg.domain_rand.randomize_friction_indep = False
    Cnfg.domain_rand.randomize_friction = True
    Cnfg.domain_rand.friction_range = [0.1, 4.5]
    Cnfg.domain_rand.randomize_restitution = True
    Cnfg.domain_rand.restitution_range = [0.0, 0.4]
    Cnfg.domain_rand.randomize_base_mass = True
    Cnfg.domain_rand.added_mass_range = [-1.0, 3.5]
    Cnfg.domain_rand.randomize_gravity = True
    Cnfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cnfg.domain_rand.gravity_rand_interval_s = 8.0
    Cnfg.domain_rand.gravity_impulse_duration = 0.99
    Cnfg.domain_rand.randomize_com_displacement = True
    Cnfg.domain_rand.com_displacement_range = [-0.1, 0.1]
    Cnfg.domain_rand.randomize_ground_friction = True
    Cnfg.domain_rand.ground_friction_range = [0.1, 4.5]
    Cnfg.domain_rand.randomize_motor_strength = True
    Cnfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cnfg.domain_rand.randomize_motor_offset = True
    Cnfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cnfg.domain_rand.push_robots = False
    Cnfg.domain_rand.max_push_vel_xy = 0.5
    Cnfg.domain_rand.randomize_Kp_factor = True
    Cnfg.domain_rand.Kp_factor_range = [0.9, 1.1]
    Cnfg.domain_rand.randomize_Kd_factor = True
    Cnfg.domain_rand.Kd_factor_range = [0.8, 1.3]

    # Rewards settings
    Cnfg.rewards.use_terminal_foot_height = False
    Cnfg.rewards.use_terminal_body_height = True
    Cnfg.rewards.terminal_body_height = 0.20
    Cnfg.rewards.use_terminal_roll_pitch = True
    Cnfg.rewards.terminal_body_ori = 1.6
    Cnfg.rewards.base_height_target = 0.38
    Cnfg.rewards.kappa_gait_probs = 0.07
    Cnfg.rewards.gait_force_sigma = 100.
    Cnfg.rewards.gait_vel_sigma = 10.
    Cnfg.rewards.reward_container_name = "CoRLRewards"
    Cnfg.rewards.only_positive_rewards = False
    Cnfg.rewards.only_positive_rewards_ji22_style = True
    Cnfg.rewards.sigma_rew_neg = 0.02
    Cnfg.rewards.soft_dof_pos_limit = 0.9
    
    # Set reward scales
    Cnfg.reward_scales.tracking_lin_vel = 1.0
    Cnfg.reward_scales.tracking_ang_vel = 0.5
    Cnfg.reward_scales.feet_slip = -0.04
    Cnfg.reward_scales.action_smoothness_1 = -0.1
    Cnfg.reward_scales.action_smoothness_2 = -0.1
    Cnfg.reward_scales.dof_vel = -1e-4
    Cnfg.reward_scales.dof_acc = -2.5e-7
    Cnfg.reward_scales.jump = 10.0
    Cnfg.reward_scales.raibert_heuristic = -10.0
    Cnfg.reward_scales.feet_clearance_cmd_linear = -30.0
    Cnfg.reward_scales.orientation_control = -5.0
    Cnfg.reward_scales.lin_vel_z = -0.02
    Cnfg.reward_scales.ang_vel_xy = -0.001
    Cnfg.reward_scales.tracking_contacts_shaped_force = 5.0
    Cnfg.reward_scales.tracking_contacts_shaped_vel = 5.0
    Cnfg.reward_scales.collision = -5.0
    Cnfg.reward_scales.torques = -0.0001
    Cnfg.reward_scales.action_rate = -0.01
    Cnfg.reward_scales.dof_pos_limits = -10.0

    Cnfg.obs_scales.ang_vel = 2.0

    # Terrain settings
    Cnfg.terrain.border_size = 0.0
    Cnfg.terrain.mesh_type = "trimesh"
    Cnfg.terrain.num_cols = 30
    Cnfg.terrain.num_rows = 30
    Cnfg.terrain.terrain_width = 5.0
    Cnfg.terrain.terrain_length = 5.0
    Cnfg.terrain.x_init_range = 0.2
    Cnfg.terrain.y_init_range = 0.2
    Cnfg.terrain.teleport_thresh = 0.3
    Cnfg.terrain.teleport_robots = False
    Cnfg.terrain.center_robots = True
    Cnfg.terrain.center_span = 4
    Cnfg.terrain.horizontal_scale = 0.10
    Cnfg.terrain.yaw_init_range = 3.14
    Cnfg.terrain.measure_heights = False
    Cnfg.terrain.terrain_noise_magnitude = 0.0
    Cnfg.terrain.teleport_robots = True
    Cnfg.terrain.border_size = 50
    Cnfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    Cnfg.terrain.curriculum = False
    
    # Domain Randomization Tile Height Settings
    Cnfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cnfg.domain_rand.tile_height_curriculum = False
    Cnfg.domain_rand.tile_height_update_interval = 1000000
    Cnfg.domain_rand.tile_height_curriculum_step = 0.01

    Cnfg.normalization.friction_range = [0, 1]
    Cnfg.normalization.ground_friction_range = [0, 1]
    Cnfg.normalization.clip_actions = 10.0


    # Curriculum settings
    Cnfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cnfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cnfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cnfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    # Command settings
    Cnfg.commands.heading_command = False
    Cnfg.commands.command_curriculum = True
    Cnfg.commands.num_lin_vel_bins = 30
    Cnfg.commands.num_ang_vel_bins = 30
    Cnfg.commands.distributional_commands = True
    Cnfg.commands.num_commands = 15
    Cnfg.commands.resampling_time = 10
    Cnfg.commands.lin_vel_x = [-1.0, 1.0]
    Cnfg.commands.lin_vel_y = [-0.6, 0.6]
    Cnfg.commands.ang_vel_yaw = [-0.5, 0.5]
    Cnfg.commands.body_height_cmd = [-0.10, 0.05]
    Cnfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    Cnfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cnfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cnfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cnfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cnfg.commands.footswing_height_range = [0.03, 0.35]
    Cnfg.commands.body_pitch_range = [-0.4, 0.4]
    Cnfg.commands.body_roll_range = [-0.0, 0.0]
    Cnfg.commands.stance_width_range = [0.10, 0.45]
    Cnfg.commands.stance_length_range = [0.35, 0.45]
    Cnfg.commands.limit_vel_x = [-5.0, 5.0]
    Cnfg.commands.limit_vel_y = [-0.6, 0.6]
    Cnfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cnfg.commands.limit_body_height = [-0.15, 0.05]
    Cnfg.commands.limit_gait_frequency = [2.0, 4.0]
    Cnfg.commands.limit_gait_phase = [0.0, 1.0]
    Cnfg.commands.limit_gait_offset = [0.0, 1.0]
    Cnfg.commands.limit_gait_bound = [0.0, 1.0]
    Cnfg.commands.limit_gait_duration = [0.5, 0.5]
    Cnfg.commands.limit_footswing_height = [0.03, 0.35]
    Cnfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cnfg.commands.limit_body_roll = [-0.0, 0.0]
    Cnfg.commands.limit_stance_width = [0.10, 0.45]
    Cnfg.commands.limit_stance_length = [0.35, 0.45]

    Cnfg.commands.num_bins_vel_x = 21
    Cnfg.commands.num_bins_vel_y = 1
    Cnfg.commands.num_bins_vel_yaw = 21
    Cnfg.commands.num_bins_body_height = 1
    Cnfg.commands.num_bins_gait_frequency = 1
    Cnfg.commands.num_bins_gait_phase = 1
    Cnfg.commands.num_bins_gait_offset = 1
    Cnfg.commands.num_bins_gait_bound = 1
    Cnfg.commands.num_bins_gait_duration = 1
    Cnfg.commands.num_bins_footswing_height = 1
    Cnfg.commands.num_bins_body_roll = 1
    Cnfg.commands.num_bins_body_pitch = 1
    Cnfg.commands.num_bins_stance_width = 1

    Cnfg.commands.exclusive_phase_offset = False
    Cnfg.commands.pacing_offset = False
    Cnfg.commands.binary_phases = True
    Cnfg.commands.gaitwise_curricula = True