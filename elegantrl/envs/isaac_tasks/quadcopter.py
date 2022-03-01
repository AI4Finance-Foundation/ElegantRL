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

import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from isaacgym import gymutil, gymtorch, gymapi
from elegantrl.envs.utils.torch_jit_utils import *
from elegantrl.envs.isaac_tasks.base.vec_task import VecTask


class Quadcopter(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        dofs_per_env = 8
        bodies_per_env = 9

        # Observations:
        # 0:13 - root state
        # 13:29 - DOF states
        num_obs = 21

        # Actions:
        # 0:8 - rotor DOF position targets
        # 8:12 - rotor thrust magnitudes
        num_acts = 12

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(
            self.num_envs, dofs_per_env, 2
        )

        self.root_states = vec_root_tensor
        self.root_positions = vec_root_tensor[..., 0:3]
        self.root_quats = vec_root_tensor[..., 3:7]
        self.root_linvels = vec_root_tensor[..., 7:10]
        self.root_angvels = vec_root_tensor[..., 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = vec_root_tensor.clone()
        self.initial_dof_states = vec_dof_tensor.clone()

        max_thrust = 2
        self.thrust_lower_limits = torch.zeros(
            4, device=self.device, dtype=torch.float32
        )
        self.thrust_upper_limits = max_thrust * torch.ones(
            4, device=self.device, dtype=torch.float32
        )

        # control tensors
        self.dof_position_targets = torch.zeros(
            (self.num_envs, dofs_per_env),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.thrusts = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.forces = torch.zeros(
            (self.num_envs, bodies_per_env, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.all_actor_indices = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(
                self.num_envs, bodies_per_env, 13
            )
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self.dt = self.sim_params.dt
        self._create_quadcopter_asset()
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_quadcopter_asset(self):

        chassis_radius = 0.1
        chassis_thickness = 0.03
        rotor_radius = 0.04
        rotor_thickness = 0.01
        rotor_arm_radius = 0.01

        root = ET.Element("mujoco")
        root.attrib["model"] = "Quadcopter"
        compiler = ET.SubElement(root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        worldbody = ET.SubElement(root, "worldbody")

        chassis = ET.SubElement(worldbody, "body")
        chassis.attrib["name"] = "chassis"
        chassis.attrib["pos"] = "%g %g %g" % (0, 0, 0)
        chassis_geom = ET.SubElement(chassis, "geom")
        chassis_geom.attrib["type"] = "cylinder"
        chassis_geom.attrib["size"] = "%g %g" % (
            chassis_radius,
            0.5 * chassis_thickness,
        )
        chassis_geom.attrib["pos"] = "0 0 0"
        chassis_geom.attrib["density"] = "50"
        chassis_joint = ET.SubElement(chassis, "joint")
        chassis_joint.attrib["name"] = "root_joint"
        chassis_joint.attrib["type"] = "free"

        zaxis = gymapi.Vec3(0, 0, 1)
        rotor_arm_offset = gymapi.Vec3(chassis_radius + 0.25 * rotor_arm_radius, 0, 0)
        pitch_joint_offset = gymapi.Vec3(0, 0, 0)

        rotor_offset = gymapi.Vec3(rotor_radius + 0.25 * rotor_arm_radius, 0, 0)

        rotor_angles = [0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi]
        for i in range(len(rotor_angles)):
            angle = rotor_angles[i]

            rotor_arm_quat = gymapi.Quat.from_axis_angle(zaxis, angle)
            rotor_arm_pos = rotor_arm_quat.rotate(rotor_arm_offset)
            pitch_joint_pos = pitch_joint_offset
            rotor_pos = rotor_offset
            rotor_quat = gymapi.Quat()

            rotor_arm = ET.SubElement(chassis, "body")
            rotor_arm.attrib["name"] = "rotor_arm" + str(i)
            rotor_arm.attrib["pos"] = "%g %g %g" % (
                rotor_arm_pos.x,
                rotor_arm_pos.y,
                rotor_arm_pos.z,
            )
            rotor_arm.attrib["quat"] = "%g %g %g %g" % (
                rotor_arm_quat.w,
                rotor_arm_quat.x,
                rotor_arm_quat.y,
                rotor_arm_quat.z,
            )
            rotor_arm_geom = ET.SubElement(rotor_arm, "geom")
            rotor_arm_geom.attrib["type"] = "sphere"
            rotor_arm_geom.attrib["size"] = "%g" % rotor_arm_radius
            rotor_arm_geom.attrib["density"] = "200"

            pitch_joint = ET.SubElement(rotor_arm, "joint")
            pitch_joint.attrib["name"] = "rotor_pitch" + str(i)
            pitch_joint.attrib["type"] = "hinge"
            pitch_joint.attrib["pos"] = "%g %g %g" % (0, 0, 0)
            pitch_joint.attrib["axis"] = "0 1 0"
            pitch_joint.attrib["limited"] = "true"
            pitch_joint.attrib["range"] = "-30 30"

            rotor = ET.SubElement(rotor_arm, "body")
            rotor.attrib["name"] = "rotor" + str(i)
            rotor.attrib["pos"] = "%g %g %g" % (rotor_pos.x, rotor_pos.y, rotor_pos.z)
            rotor.attrib["quat"] = "%g %g %g %g" % (
                rotor_quat.w,
                rotor_quat.x,
                rotor_quat.y,
                rotor_quat.z,
            )
            rotor_geom = ET.SubElement(rotor, "geom")
            rotor_geom.attrib["type"] = "cylinder"
            rotor_geom.attrib["size"] = "%g %g" % (rotor_radius, 0.5 * rotor_thickness)
            # rotor_geom.attrib["type"] = "box"
            # rotor_geom.attrib["size"] = "%g %g %g" % (rotor_radius, rotor_radius, 0.5 * rotor_thickness)
            rotor_geom.attrib["density"] = "1000"

            roll_joint = ET.SubElement(rotor, "joint")
            roll_joint.attrib["name"] = "rotor_roll" + str(i)
            roll_joint.attrib["type"] = "hinge"
            roll_joint.attrib["pos"] = "%g %g %g" % (0, 0, 0)
            roll_joint.attrib["axis"] = "1 0 0"
            roll_joint.attrib["limited"] = "true"
            roll_joint.attrib["range"] = "-30 30"

        gymutil._indent_xml(root)
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../isaac_assets",
            "quadcopter.xml",
        )
        ET.ElementTree(root).write(xml_path)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "elegantrl/envs/isaac_assets"
        asset_file = "quadcopter.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dofs = self.gym.get_asset_dof_count(asset)

        dof_props = self.gym.get_asset_dof_properties(asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for i in range(self.num_dofs):
            self.dof_lower_limits.append(dof_props["lower"][i])
            self.dof_upper_limits.append(dof_props["upper"][i])

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.dof_ranges = self.dof_upper_limits - self.dof_lower_limits

        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(
                env, asset, default_pose, "quadcopter", i, 1, 0
            )

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(1000.0)
            dof_props["damping"].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            # pretty colors
            chassis_color = gymapi.Vec3(0.8, 0.6, 0.2)
            rotor_color = gymapi.Vec3(0.1, 0.2, 0.6)
            arm_color = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.set_rigid_body_color(
                env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 1, gymapi.MESH_VISUAL_AND_COLLISION, arm_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 3, gymapi.MESH_VISUAL_AND_COLLISION, arm_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 5, gymapi.MESH_VISUAL_AND_COLLISION, arm_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 7, gymapi.MESH_VISUAL_AND_COLLISION, arm_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 2, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 6, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color
            )
            self.gym.set_rigid_body_color(
                env, actor_handle, 8, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color
            )
            # self.gym.set_rigid_body_color(env, actor_handle, 2, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
            # self.gym.set_rigid_body_color(env, actor_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))
            # self.gym.set_rigid_body_color(env, actor_handle, 6, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
            # self.gym.set_rigid_body_color(env, actor_handle, 8, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 0))

            self.envs.append(env)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros(
                (self.num_envs, 4, 3), device=self.device
            )
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    def reset_idx(self, env_ids):

        num_resets = len(env_ids)

        self.dof_states[env_ids] = self.initial_dof_states[env_ids]

        actor_indices = self.all_actor_indices[env_ids].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(
            -1.5, 1.5, (num_resets, 1), self.device
        ).flatten()
        self.root_states[env_ids, 1] += torch_rand_float(
            -1.5, 1.5, (num_resets, 1), self.device
        ).flatten()
        self.root_states[env_ids, 2] += torch_rand_float(
            -0.2, 1.5, (num_resets, 1), self.device
        ).flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self.root_tensor,
            gymtorch.unwrap_tensor(actor_indices),
            num_resets,
        )

        self.dof_positions[env_ids] = torch_rand_float(
            -0.2, 0.2, (num_resets, 8), self.device
        )
        self.dof_velocities[env_ids] = 0.0
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self.dof_state_tensor,
            gymtorch.unwrap_tensor(actor_indices),
            num_resets,
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, _actions):

        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = _actions.to(self.device)

        dof_action_speed_scale = 8 * math.pi
        self.dof_position_targets += self.dt * dof_action_speed_scale * actions[:, 0:8]
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.dof_lower_limits, self.dof_upper_limits
        )

        thrust_action_speed_scale = 200
        self.thrusts += self.dt * thrust_action_speed_scale * actions[:, 8:12]
        self.thrusts[:] = tensor_clamp(
            self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits
        )

        self.forces[:, 2, 2] = self.thrusts[:, 0]
        self.forces[:, 4, 2] = self.thrusts[:, 1]
        self.forces[:, 6, 2] = self.thrusts[:, 2]
        self.forces[:, 8, 2] = self.thrusts[:, 3]

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0
        self.dof_position_targets[reset_env_ids] = self.dof_positions[reset_env_ids]

        # apply actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_position_targets)
        )
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE
        )

    def post_physics_step(self):

        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(
                self.num_envs, 4, 3
            )
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

    def compute_observations(self):
        target_x = 0.0
        target_y = 0.0
        target_z = 1.0
        self.obs_buf[..., 0] = (target_x - self.root_positions[..., 0]) / 3
        self.obs_buf[..., 1] = (target_y - self.root_positions[..., 1]) / 3
        self.obs_buf[..., 2] = (target_z - self.root_positions[..., 2]) / 3
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        self.obs_buf[..., 13:21] = self.dof_positions
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_quadcopter_reward(
    root_positions,
    root_quats,
    root_linvels,
    root_angvels,
    reset_buf,
    progress_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(
        root_positions[..., 0] * root_positions[..., 0]
        + root_positions[..., 1] * root_positions[..., 1]
        + (1 - root_positions[..., 2]) * (1 - root_positions[..., 2])
    )
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 3.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.3, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
