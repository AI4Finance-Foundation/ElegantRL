# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from elegantrl.envs.utils.torch_jit_utils import *
from elegantrl.envs.isaac_tasks.base.vec_task import VecTask


class Ant(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 8

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, sensors_per_env * 6
        )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(
            self.dof_pos, device=self.device, dtype=torch.float
        )
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(
            self.dof_limits_lower > zero_tensor,
            self.dof_limits_lower,
            torch.where(
                self.dof_limits_upper < zero_tensor,
                self.dof_limits_upper,
                self.initial_dof_pos,
            ),
        )
        self.initial_dof_vel = torch.zeros_like(
            self.dof_vel, device=self.device, dtype=torch.float
        )

        # initialize some data used later on
        self.up_vec = to_torch(
            get_axis_params(1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat(
            (self.num_envs, 1)
        )

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000.0 / self.dt], device=self.device).repeat(
            self.num_envs
        )
        self.prev_potentials = self.potentials.clone()

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../isaac_assets"
        )
        asset_file = "mjcf/nv_ant.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        self.start_rotation = torch.tensor(
            [start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
            device=self.device,
        )

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [
            self.gym.get_asset_rigid_body_name(ant_asset, i)
            for i in range(self.num_bodies)
        ]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(
            len(extremity_names), dtype=torch.long, device=self.device
        )

        # create force sensors attached to the "feet"
        extremity_indices = [
            self.gym.find_asset_rigid_body_index(ant_asset, name)
            for name in extremity_names
        ]
        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)

        self.ant_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            ant_handle = self.gym.create_actor(
                env_ptr, ant_asset, start_pose, "ant", i, 1, 0
            )

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    ant_handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.97, 0.38, 0.06),
                )

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.ant_handles[0], extremity_names[i]
            )

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # print("Feet forces and torques: ", self.vec_sensor_tensor[0, :])
        # print(self.vec_sensor_tensor.shape)

        (
            self.obs_buf[:],
            self.potentials[:],
            self.prev_potentials[:],
            self.up_vec[:],
            self.heading_vec[:],
        ) = compute_ant_observations(
            self.obs_buf,
            self.root_states,
            self.targets,
            self.potentials,
            self.inv_start_rot,
            self.dof_pos,
            self.dof_vel,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.dof_vel_scale,
            self.vec_sensor_tensor,
            self.actions,
            self.dt,
            self.contact_force_scale,
            self.basis_vec0,
            self.basis_vec1,
            self.up_axis_idx,
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(
            -0.2, 0.2, (len(env_ids), self.num_dof), device=self.device
        )
        velocities = torch_rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device
        )

        self.dof_pos[env_ids] = tensor_clamp(
            self.initial_dof_pos[env_ids] + positions,
            self.dof_limits_lower,
            self.dof_limits_upper,
        )
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        forces = self.actions * self.joint_gears * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(
                    origin.x + pose[0], origin.y + pose[1], origin.z + pose[2]
                )
                points.append(
                    [
                        glob_pos.x,
                        glob_pos.y,
                        glob_pos.z,
                        glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                        glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                        glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy(),
                    ]
                )
                colors.append([0.97, 0.1, 0.06])
                points.append(
                    [
                        glob_pos.x,
                        glob_pos.y,
                        glob_pos.z,
                        glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(),
                        glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                        glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy(),
                    ]
                )
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    death_cost,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(
        obs_buf[:, 11] > 0.8,
        heading_weight_tensor,
        heading_weight * obs_buf[:, 11] / 0.8,
    )

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost * joints_at_limit_cost_scale
    )

    # adjust reward for fallen agents
    total_reward = torch.where(
        obs_buf[:, 0] < termination_height,
        torch.ones_like(total_reward) * death_cost,
        total_reward,
    )

    # reset agents
    reset = torch.where(
        obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf
    )
    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
    )

    return total_reward, reset


@torch.jit.script
def compute_ant_observations(
    obs_buf,
    root_states,
    targets,
    potentials,
    inv_start_rot,
    dof_pos,
    dof_vel,
    dof_limits_lower,
    dof_limits_upper,
    dof_vel_scale,
    sensor_force_torques,
    actions,
    dt,
    contact_force_scale,
    basis_vec0,
    basis_vec1,
    up_axis_idx,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    obs = torch.cat(
        (
            torso_position[:, up_axis_idx].view(-1, 1),
            vel_loc,
            angvel_loc,
            yaw.unsqueeze(-1),
            roll.unsqueeze(-1),
            angle_to_target.unsqueeze(-1),
            up_proj.unsqueeze(-1),
            heading_proj.unsqueeze(-1),
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            sensor_force_torques.view(-1, 24) * contact_force_scale,
            actions,
        ),
        dim=-1,
    )

    return obs, potentials, prev_potentials_new, up_vec, heading_vec
