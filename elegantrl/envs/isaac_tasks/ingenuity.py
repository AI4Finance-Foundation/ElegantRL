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

from elegantrl.envs.utils.torch_jit_utils import *
from elegantrl.envs.isaac_tasks.base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi


class Ingenuity(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Observations:
        # 0:13 - root state
        self.cfg["env"]["numObservations"] = 13

        # Actions:
        # 0:3 - xyz force vector for lower rotor
        # 4:6 - xyz force vector for upper rotor
        self.cfg["env"]["numActions"] = 6

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )

        dofs_per_env = 4
        bodies_per_env = 6

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(
            self.num_envs, 2, 13
        )
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(
            self.num_envs, dofs_per_env, 2
        )

        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.target_root_positions = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.target_root_positions[:, 2] = 1
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        self.thrust_lower_limit = 0
        self.thrust_upper_limit = 2000
        self.thrust_lateral_component = 0.2

        # control tensors
        self.thrusts = torch.zeros(
            (self.num_envs, 2, 3),
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
            self.num_envs * 2, dtype=torch.int32, device=self.device
        ).reshape((self.num_envs, 2))

        if self.viewer:
            cam_pos = gymapi.Vec3(2.25, 2.25, 3.0)
            cam_target = gymapi.Vec3(3.5, 4.0, 1.9)
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

        # Mars gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -3.721

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self.dt = self.sim_params.dt
        self._create_ingenuity_asset()
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ingenuity_asset(self):
        chassis_size = 0.06
        rotor_axis_length = 0.2
        rotor_radius = 0.15
        rotor_thickness = 0.01
        rotor_arm_radius = 0.01

        root = ET.Element("mujoco")
        root.attrib["model"] = "Ingenuity"

        compiler = ET.SubElement(root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"

        mesh_asset = ET.SubElement(root, "asset")

        model_path = "../isaac_assets/glb/ingenuity/"
        mesh = ET.SubElement(mesh_asset, "mesh")
        mesh.attrib["file"] = model_path + "chassis.glb"
        mesh.attrib["name"] = "ingenuity_mesh"

        lower_prop_mesh = ET.SubElement(mesh_asset, "mesh")
        lower_prop_mesh.attrib["file"] = model_path + "lower_prop.glb"
        lower_prop_mesh.attrib["name"] = "lower_prop_mesh"

        upper_prop_mesh = ET.SubElement(mesh_asset, "mesh")
        upper_prop_mesh.attrib["file"] = model_path + "upper_prop.glb"
        upper_prop_mesh.attrib["name"] = "upper_prop_mesh"

        worldbody = ET.SubElement(root, "worldbody")

        chassis = ET.SubElement(worldbody, "body")
        chassis.attrib["name"] = "chassis"
        chassis.attrib["pos"] = "%g %g %g" % (0, 0, 0)

        chassis_geom = ET.SubElement(chassis, "geom")
        chassis_geom.attrib["type"] = "box"
        chassis_geom.attrib["size"] = "%g %g %g" % (
            chassis_size,
            chassis_size,
            chassis_size,
        )
        chassis_geom.attrib["pos"] = "0 0 0"
        chassis_geom.attrib["density"] = "50"

        mesh_quat = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)

        mesh_geom = ET.SubElement(chassis, "geom")
        mesh_geom.attrib["type"] = "mesh"
        mesh_geom.attrib["quat"] = "%g %g %g %g" % (
            mesh_quat.w,
            mesh_quat.x,
            mesh_quat.y,
            mesh_quat.z,
        )
        mesh_geom.attrib["mesh"] = "ingenuity_mesh"
        mesh_geom.attrib["pos"] = "%g %g %g" % (0, 0, 0)
        mesh_geom.attrib["contype"] = "0"
        mesh_geom.attrib["conaffinity"] = "0"

        chassis_joint = ET.SubElement(chassis, "joint")
        chassis_joint.attrib["name"] = "root_joint"
        chassis_joint.attrib["type"] = "hinge"
        chassis_joint.attrib["limited"] = "true"
        chassis_joint.attrib["range"] = "0 0"

        zaxis = gymapi.Vec3(0, 0, 1)

        low_rotor_pos = gymapi.Vec3(0, 0, 0)
        rotor_separation = gymapi.Vec3(0, 0, 0.025)

        for i, mesh_name in enumerate(["lower_prop_mesh", "upper_prop_mesh"]):
            angle = 0

            rotor_quat = gymapi.Quat.from_axis_angle(zaxis, angle)
            rotor_pos = low_rotor_pos + (rotor_separation * i)

            rotor = ET.SubElement(chassis, "body")
            rotor.attrib["name"] = "rotor_physics_" + str(i)
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
            rotor_geom.attrib["density"] = "1000"

            roll_joint = ET.SubElement(rotor, "joint")
            roll_joint.attrib["name"] = "rotor_roll" + str(i)
            roll_joint.attrib["type"] = "hinge"
            roll_joint.attrib["limited"] = "true"
            roll_joint.attrib["range"] = "0 0"
            roll_joint.attrib["pos"] = "%g %g %g" % (0, 0, 0)

            rotor_dummy = ET.SubElement(chassis, "body")
            rotor_dummy.attrib["name"] = "rotor_visual_" + str(i)
            rotor_dummy.attrib["pos"] = "%g %g %g" % (
                rotor_pos.x,
                rotor_pos.y,
                rotor_pos.z,
            )
            rotor_dummy.attrib["quat"] = "%g %g %g %g" % (
                rotor_quat.w,
                rotor_quat.x,
                rotor_quat.y,
                rotor_quat.z,
            )

            rotor_mesh_geom = ET.SubElement(rotor_dummy, "geom")
            rotor_mesh_geom.attrib["type"] = "mesh"
            rotor_mesh_geom.attrib["mesh"] = mesh_name
            rotor_mesh_quat = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            rotor_mesh_geom.attrib["quat"] = "%g %g %g %g" % (
                rotor_mesh_quat.w,
                rotor_mesh_quat.x,
                rotor_mesh_quat.y,
                rotor_mesh_quat.z,
            )
            rotor_mesh_geom.attrib["contype"] = "0"
            rotor_mesh_geom.attrib["conaffinity"] = "0"

            dummy_roll_joint = ET.SubElement(rotor_dummy, "joint")
            dummy_roll_joint.attrib["name"] = "rotor_roll" + str(i)
            dummy_roll_joint.attrib["type"] = "hinge"
            dummy_roll_joint.attrib["axis"] = "0 0 1"
            dummy_roll_joint.attrib["pos"] = "%g %g %g" % (0, 0, 0)

        gymutil._indent_xml(root)
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../isaac_assets",
            "ingenuity.xml",
        )
        ET.ElementTree(root).write(xml_path)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../isaac_assets"
        )
        asset_file = "ingenuity.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)

        default_pose = gymapi.Transform()
        default_pose.p.z = 1.0

        self.envs = []
        self.actor_handles = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(
                env, asset, default_pose, "ingenuity", i, 1, 1
            )

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props["stiffness"].fill(0)
            dof_props["damping"].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            marker_handle = self.gym.create_actor(
                env, marker_asset, default_pose, "marker", i, 1, 1
            )
            self.gym.set_rigid_body_color(
                env,
                marker_handle,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(1, 0, 0),
            )

            self.actor_handles.append(actor_handle)
            self.envs.append(env)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros(
                (self.num_envs, 2, 3), device=self.device
            )
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        # set target position randomly with x, y in (-5, 5) and z in (1, 2)
        self.target_root_positions[env_ids, 0:2] = (
            torch.rand(num_sets, 2, device=self.device) * 10
        ) - 5
        self.target_root_positions[env_ids, 2] = (
            torch.rand(num_sets, device=self.device) + 1
        )
        self.marker_positions[env_ids] = self.target_root_positions[env_ids]
        # copter "position" is at the bottom of the legs, so shift the target up so it visually aligns better
        self.marker_positions[env_ids, 2] += 0.4
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()

        return actor_indices

    def reset_idx(self, env_ids):

        # set rotor speeds
        self.dof_velocities[:, 1] = -50
        self.dof_velocities[:, 3] = 50

        num_resets = len(env_ids)

        target_actor_indices = self.set_targets(env_ids)

        actor_indices = self.all_actor_indices[env_ids, 0].flatten()

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

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self.dof_state_tensor,
            gymtorch.unwrap_tensor(actor_indices),
            num_resets,
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        return torch.unique(torch.cat([target_actor_indices, actor_indices]))

    def pre_physics_step(self, _actions):

        # resets
        set_target_ids = (
            (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        )
        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(set_target_ids) > 0:
            target_actor_indices = self.set_targets(set_target_ids)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)

        reset_indices = torch.unique(torch.cat([target_actor_indices, actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                self.root_tensor,
                gymtorch.unwrap_tensor(reset_indices),
                len(reset_indices),
            )

        actions = _actions.to(self.device)

        thrust_action_speed_scale = 2000
        vertical_thrust_prop_0 = torch.clamp(
            actions[:, 2] * thrust_action_speed_scale,
            -self.thrust_upper_limit,
            self.thrust_upper_limit,
        )
        vertical_thrust_prop_1 = torch.clamp(
            actions[:, 5] * thrust_action_speed_scale,
            -self.thrust_upper_limit,
            self.thrust_upper_limit,
        )
        lateral_fraction_prop_0 = torch.clamp(
            actions[:, 0:2],
            -self.thrust_lateral_component,
            self.thrust_lateral_component,
        )
        lateral_fraction_prop_1 = torch.clamp(
            actions[:, 3:5],
            -self.thrust_lateral_component,
            self.thrust_lateral_component,
        )

        self.thrusts[:, 0, 2] = self.dt * vertical_thrust_prop_0
        self.thrusts[:, 0, 0:2] = self.thrusts[:, 0, 2, None] * lateral_fraction_prop_0
        self.thrusts[:, 1, 2] = self.dt * vertical_thrust_prop_1
        self.thrusts[:, 1, 0:2] = self.thrusts[:, 1, 2, None] * lateral_fraction_prop_1

        self.forces[:, 1] = self.thrusts[:, 0]
        self.forces[:, 3] = self.thrusts[:, 1]

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0

        # apply actions
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
        self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions) / 3
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_ingenuity_reward(
            self.root_positions,
            self.target_root_positions,
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
def compute_ingenuity_reward(
    root_positions,
    target_root_positions,
    root_quats,
    root_linvels,
    root_angvels,
    reset_buf,
    progress_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(
        torch.square(target_root_positions - root_positions).sum(-1)
    )
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 5.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 8.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.5, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
