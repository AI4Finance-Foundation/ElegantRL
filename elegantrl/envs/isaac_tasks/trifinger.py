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
from isaacgym.torch_utils import *

from collections import OrderedDict

project_dir = os.path.abspath(os.path.dirname(__file__))

from elegantrl.envs.utils.torch_jit_utils import *
from elegantrl.envs.isaac_tasks.base.vec_task import VecTask
from types import SimpleNamespace
from collections import deque
from typing import Deque, Dict, Tuple, Union


# python
import enum
import numpy as np

# ################### #
# Dimensions of robot #
# ################### #


class TrifingerDimensions(enum.Enum):
    """
    Dimensions of the tri-finger robot.

    Note: While it may not seem necessary for tri-finger robot since it is fixed base, for floating
    base systems having this dimensions class is useful.
    """

    # general state
    # cartesian position + quaternion orientation
    PoseDim = (7,)
    # linear velocity + angular velcoity
    VelocityDim = 6
    # state: pose + velocity
    StateDim = 13
    # force + torque
    WrenchDim = 6
    # for robot
    # number of fingers
    NumFingers = 3
    # for three fingers
    JointPositionDim = 9
    JointVelocityDim = 9
    JointTorqueDim = 9
    # generalized coordinates
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim
    # for objects
    ObjectPoseDim = 7
    ObjectVelocityDim = 6


# ################# #
# Different objects #
# ################# #


# readius of the area
ARENA_RADIUS = 0.195


class CuboidalObject:
    """
    Fields for a cuboidal object.

    @note Motivation for this class is that if domain randomization is performed over the
          size of the cuboid, then its attributes are automatically updated as well.
    """

    # 3D radius of the cuboid
    radius_3d: float
    # distance from wall to the center
    max_com_distance_to_center: float
    # minimum and mximum height for spawning the object
    min_height: float
    max_height = 0.1

    NumKeypoints = 8
    ObjectPositionDim = 3
    KeypointsCoordsDim = NumKeypoints * ObjectPositionDim

    def __init__(self, size: Union[float, Tuple[float, float, float]]):
        """Initialize the cuboidal object.

        Args:
            size: The size of the object along x, y, z in meters. If a single float is provided, then it is assumed that
                  object is a cube.
        """
        # decide the size depedning on input type
        if isinstance(size, float):
            self._size = (size, size, size)
        else:
            self._size = size
        # compute remaning attributes
        self.__compute()

    """
    Properties
    """

    @property
    def size(self) -> Tuple[float, float, float]:
        """
        Returns the dimensions of the cuboid object (x, y, z) in meters.
        """
        return self._size

    """
    Configurations
    """

    @size.setter
    def size(self, size: Union[float, Tuple[float, float, float]]):
        """Set size of the object.

        Args:
            size: The size of the object along x, y, z in meters. If a single float is provided, then it is assumed
                  that object is a cube.
        """
        # decide the size depedning on input type
        if isinstance(size, float):
            self._size = (size, size, size)
        else:
            self._size = size
        # compute attributes
        self.__compute()

    """
    Private members
    """

    def __compute(self):
        """Compute the attributes for the object."""
        # compute 3D radius of the cuboid
        max_len = max(self._size)
        self.radius_3d = max_len * np.sqrt(3) / 2
        # compute distance from wall to the center
        self.max_com_distance_to_center = ARENA_RADIUS - self.radius_3d
        # minimum height for spawning the object
        self.min_height = self._size[2] / 2


class Trifinger(VecTask):

    # constants
    # directory where assets for the simulator are present
    _trifinger_assets_dir = os.path.join(
        project_dir, "../", "isaac_assets", "trifinger"
    )
    # robot urdf (path relative to `_trifinger_assets_dir`)
    _robot_urdf_file = "robot_properties_fingers/urdf/pro/trifingerpro.urdf"
    # stage urdf (path relative to `_trifinger_assets_dir`)
    # _stage_urdf_file = "robot_properties_fingers/urdf/trifinger_stage.urdf"
    _table_urdf_file = "robot_properties_fingers/urdf/table_without_border.urdf"
    _boundary_urdf_file = "robot_properties_fingers/urdf/high_table_boundary.urdf"
    # object urdf (path relative to `_trifinger_assets_dir`)
    # TODO: Make object URDF configurable.
    _object_urdf_file = "objects/urdf/cube_multicolor_rrc.urdf"

    # physical dimensions of the object
    # TODO: Make object dimensions configurable.
    _object_dims = CuboidalObject(0.065)
    # dimensions of the system
    _dims = TrifingerDimensions
    # Constants for limits
    # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/trifinger_platform.py#L68
    # maximum joint torque (in N-m) applicable on each actuator
    _max_torque_Nm = 0.36
    # maximum joint velocity (in rad/s) on each actuator
    _max_velocity_radps = 10

    # History of state: Number of timesteps to save history for
    # Note: Currently used only to manage history of object and frame states.
    #       This can be extended to other observations (as done in ANYmal).
    _state_history_len = 2

    # buffers to store the simulation data
    # goal poses for the object [num. of instances, 7] where 7: (x, y, z, quat)
    _object_goal_poses_buf: torch.Tensor
    # DOF state of the system [num. of instances, num. of dof, 2] where last index: pos, vel
    _dof_state: torch.Tensor
    # Rigid body state of the system [num. of instances, num. of bodies, 13] where 13: (x, y, z, quat, v, omega)
    _rigid_body_state: torch.Tensor
    # Root prim states [num. of actors, 13] where 13: (x, y, z, quat, v, omega)
    _actors_root_state: torch.Tensor
    # Force-torque sensor array [num. of instances, num. of bodies * wrench]
    _ft_sensors_values: torch.Tensor
    # DOF position of the system [num. of instances, num. of dof]
    _dof_position: torch.Tensor
    # DOF velocity of the system [num. of instances, num. of dof]
    _dof_velocity: torch.Tensor
    # DOF torque of the system [num. of instances, num. of dof]
    _dof_torque: torch.Tensor
    # Fingertip links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _fingertips_frames_state_history: Deque[torch.Tensor] = deque(
        maxlen=_state_history_len
    )
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # stores the last action output
    _last_action: torch.Tensor
    # keeps track of the number of goal resets
    _successes: torch.Tensor
    # keeps track of number of consecutive successes
    _consecutive_successes: float

    _robot_limits: dict = {
        "joint_position": SimpleNamespace(
            # matches those on the real robot
            low=np.array([-0.33, 0.0, -2.7] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([1.0, 1.57, 0.0] * _dims.NumFingers.value, dtype=np.float32),
            default=np.array(
                [0.0, 0.9, -2.0] * _dims.NumFingers.value, dtype=np.float32
            ),
        ),
        "joint_velocity": SimpleNamespace(
            low=np.full(
                _dims.JointVelocityDim.value, -_max_velocity_radps, dtype=np.float32
            ),
            high=np.full(
                _dims.JointVelocityDim.value, _max_velocity_radps, dtype=np.float32
            ),
            default=np.zeros(_dims.JointVelocityDim.value, dtype=np.float32),
        ),
        "joint_torque": SimpleNamespace(
            low=np.full(_dims.JointTorqueDim.value, -_max_torque_Nm, dtype=np.float32),
            high=np.full(_dims.JointTorqueDim.value, _max_torque_Nm, dtype=np.float32),
            default=np.zeros(_dims.JointTorqueDim.value, dtype=np.float32),
        ),
        "fingertip_position": SimpleNamespace(
            low=np.array([-0.4, -0.4, 0], dtype=np.float32),
            high=np.array([0.4, 0.4, 0.5], dtype=np.float32),
        ),
        "fingertip_orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        ),
        "fingertip_velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -0.2, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 0.2, dtype=np.float32),
        ),
        "fingertip_wrench": SimpleNamespace(
            low=np.full(_dims.WrenchDim.value, -1.0, dtype=np.float32),
            high=np.full(_dims.WrenchDim.value, 1.0, dtype=np.float32),
        ),
        # used if we want to have joint stiffness/damping as parameters`
        "joint_stiffness": SimpleNamespace(
            low=np.array([1.0, 1.0, 1.0] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array(
                [50.0, 50.0, 50.0] * _dims.NumFingers.value, dtype=np.float32
            ),
        ),
        "joint_damping": SimpleNamespace(
            low=np.array(
                [0.01, 0.03, 0.0001] * _dims.NumFingers.value, dtype=np.float32
            ),
            high=np.array([1.0, 3.0, 0.01] * _dims.NumFingers.value, dtype=np.float32),
        ),
    }
    # limits of the object (mapped later: str -> torch.tensor)
    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-0.3, -0.3, 0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            default=np.array([0, 0, _object_dims.min_height], dtype=np.float32),
        ),
        # difference between two positions
        "position_delta": SimpleNamespace(
            low=np.array([-0.6, -0.6, 0], dtype=np.float32),
            high=np.array([0.6, 0.6, 0.3], dtype=np.float32),
            default=np.array([0, 0, 0], dtype=np.float32),
        ),
        "orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -0.5, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 0.5, dtype=np.float32),
            default=np.zeros(_dims.VelocityDim.value, dtype=np.float32),
        ),
        "scale": SimpleNamespace(
            low=np.full(1, 0.0, dtype=np.float32),
            high=np.full(1, 1.0, dtype=np.float32),
        ),
    }
    # PD gains for the robot (mapped later: str -> torch.tensor)
    # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/sim_finger.py#L49-L65
    _robot_dof_gains = {
        # The kp and kd gains of the PD control of the fingers.
        # Note: This depends on simulation step size and is set for a rate of 250 Hz.
        "stiffness": [10.0, 10.0, 10.0] * _dims.NumFingers.value,
        "damping": [0.1, 0.3, 0.001] * _dims.NumFingers.value,
        # The kd gains used for damping the joint motor velocities during the
        # safety torque check on the joint motors.
        "safety_damping": [0.08, 0.08, 0.04] * _dims.NumFingers.value,
    }
    action_dim = _dims.JointTorqueDim.value

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.obs_spec = {
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            "object_q": self._dims.ObjectPoseDim.value,
            "object_q_des": self._dims.ObjectPoseDim.value,
            "command": self.action_dim,
        }
        if self.cfg["env"]["asymmetric_obs"]:
            self.state_spec = {
                # observations spec
                **self.obs_spec,
                # extra observations (added separately to make computations simpler)
                "object_u": self._dims.ObjectVelocityDim.value,
                "fingertip_state": self._dims.NumFingers.value
                * self._dims.StateDim.value,
                "robot_a": self._dims.GeneralizedVelocityDim.value,
                "fingertip_wrench": self._dims.NumFingers.value
                * self._dims.WrenchDim.value,
            }
        else:
            self.state_spec = self.obs_spec

        self.action_spec = {"command": self.action_dim}

        self.cfg["env"]["numObservations"] = sum(self.obs_spec.values())
        self.cfg["env"]["numStates"] = sum(self.state_spec.values())
        self.cfg["env"]["numActions"] = sum(self.action_spec.values())
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        # define prims present in the scene
        prim_names = ["robot", "table", "boundary", "object", "goal_object"]
        # mapping from name to asset instance
        self.gym_assets = dict.fromkeys(prim_names)
        # mapping from name to gym indices
        self.gym_indices = dict.fromkeys(prim_names)
        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        fingertips_frames = [
            "finger_tip_link_0",
            "finger_tip_link_120",
            "finger_tip_link_240",
        ]
        self._fingertips_handles = OrderedDict.fromkeys(fingertips_frames, None)
        # mapping from name to gym dof index
        robot_dof_names = list()
        for finger_pos in ["0", "120", "240"]:
            robot_dof_names += [
                f"finger_base_to_upper_joint_{finger_pos}",
                f"finger_upper_to_middle_joint_{finger_pos}",
                f"finger_middle_to_lower_joint_{finger_pos}",
            ]
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # change constant buffers from numpy/lists into torch tensors
        # limits for robot
        for limit_name in self._robot_limits:
            # extract limit simple-namespace
            limit_dict = self._robot_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(
                    value, dtype=torch.float, device=self.device
                )
        # limits for the object
        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(
                    value, dtype=torch.float, device=self.device
                )
        # PD gains for actuation
        for gain_name, value in self._robot_dof_gains.items():
            self._robot_dof_gains[gain_name] = torch.tensor(
                value, dtype=torch.float, device=self.device
            )

        # store the sampled goal poses for the object: [num. of instances, 7]
        self._object_goal_poses_buf = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float
        )
        # get force torque sensor if enabled
        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            # # joint torques
            # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            # self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
            #                                                                self._dims.JointTorqueDim.value)
            # # force-torque sensor
            num_ft_dims = self._dims.NumFingers.value * self._dims.WrenchDim.value
            # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            # self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, num_ft_dims)

            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(
                self.num_envs, num_ft_dims
            )

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self._dims.JointTorqueDim.value
            )

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # refresh the buffer (to copy memory?)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # create wrapper tensors for reference (consider everything as pointer to actual memory)
        # DOF
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._dof_position = self._dof_state[..., 0]
        self._dof_velocity = self._dof_state[..., 1]
        # rigid body
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        # root actors
        self._actors_root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )
        # frames history
        action_dim = sum(self.action_spec.values())
        self._last_action = torch.zeros(
            self.num_envs, action_dim, dtype=torch.float, device=self.device
        )
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self.gym_indices["object"]
        # timestep 0 is current tensor
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            # add tensors to history list
            print(self._rigid_body_state.shape)
            self._fingertips_frames_state_history.append(
                self._rigid_body_state[:, fingertip_handles_indices]
            )
            self._object_state_history.append(self._actors_root_state[object_indices])
            # update current history length
            curr_history_length += 1

        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._states_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._successes_pos = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._successes_quat = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self.__configure_mdp_spaces()

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._create_ground_plane()
        self._create_scene_assets()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.013
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_scene_assets(self):
        """Define Gym assets for stage, robot and object."""
        # define assets
        self.gym_assets["robot"] = self.__define_robot_asset()
        self.gym_assets["table"] = self.__define_table_asset()
        self.gym_assets["boundary"] = self.__define_boundary_asset()
        self.gym_assets["object"] = self.__define_object_asset()
        self.gym_assets["goal_object"] = self.__define_goal_object_asset()
        # display the properties (only for debugging)
        # robot
        print("Trifinger Robot Asset: ")
        print(
            f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.gym_assets["robot"])}'
        )
        print(
            f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.gym_assets["robot"])}'
        )
        print(
            f'\t Number of dofs: {self.gym.get_asset_dof_count(self.gym_assets["robot"])}'
        )
        print(f"\t Number of actuated dofs: {self._dims.JointTorqueDim.value}")
        # stage
        print("Trifinger Table Asset: ")
        print(
            f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.gym_assets["table"])}'
        )
        print(
            f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.gym_assets["table"])}'
        )
        print("Trifinger Boundary Asset: ")
        print(
            f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.gym_assets["boundary"])}'
        )
        print(
            f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.gym_assets["boundary"])}'
        )

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define the dof properties for the robot
        robot_dof_props = self.gym.get_asset_dof_properties(self.gym_assets["robot"])
        # set dof properites based on the control mode
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            # note: since safety checks are employed, the simulator PD controller is not
            #       used. Instead the torque is computed manually and applied, even if the
            #       command mode is 'position'.
            robot_dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_EFFORT
            robot_dof_props["stiffness"][dof_index] = 0.0
            robot_dof_props["damping"][dof_index] = 0.0
            # set dof limits
            robot_dof_props["effort"][dof_index] = self._max_torque_Nm
            robot_dof_props["velocity"][dof_index] = self._max_velocity_radps
            robot_dof_props["lower"][dof_index] = float(
                self._robot_limits["joint_position"].low[k]
            )
            robot_dof_props["upper"][dof_index] = float(
                self._robot_limits["joint_position"].high[k]
            )

        self.envs = []

        # define lower and upper region bound for each environment
        env_lower_bound = gymapi.Vec3(
            -self.cfg["env"]["envSpacing"], -self.cfg["env"]["envSpacing"], 0.0
        )
        env_upper_bound = gymapi.Vec3(
            self.cfg["env"]["envSpacing"],
            self.cfg["env"]["envSpacing"],
            self.cfg["env"]["envSpacing"],
        )
        num_envs_per_row = int(np.sqrt(self.num_envs))
        # initialize gym indices buffer as a list
        # note: later the list is converted to torch tensor for ease in interfacing with IsaacGym.
        for asset_name in self.gym_indices.keys():
            self.gym_indices[asset_name] = list()
        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset in self.gym_assets.values():
            max_agg_bodies += self.gym.get_asset_rigid_body_count(asset)
            max_agg_shapes += self.gym.get_asset_rigid_shape_count(asset)
        # iterate and create environment instances
        for env_index in range(self.num_envs):
            # create environment
            env_ptr = self.gym.create_env(
                self.sim, env_lower_bound, env_upper_bound, num_envs_per_row
            )
            # begin aggregration mode if enabled - this can improve simulation performance
            if self.cfg["env"]["aggregate_mode"]:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # add trifinger robot to environment
            trifinger_actor = self.gym.create_actor(
                env_ptr,
                self.gym_assets["robot"],
                gymapi.Transform(),
                "robot",
                env_index,
                0,
                0,
            )
            trifinger_idx = self.gym.get_actor_index(
                env_ptr, trifinger_actor, gymapi.DOMAIN_SIM
            )

            # add table to environment
            table_handle = self.gym.create_actor(
                env_ptr,
                self.gym_assets["table"],
                gymapi.Transform(),
                "table",
                env_index,
                1,
                0,
            )
            table_idx = self.gym.get_actor_index(
                env_ptr, table_handle, gymapi.DOMAIN_SIM
            )

            # add stage to environment
            boundary_handle = self.gym.create_actor(
                env_ptr,
                self.gym_assets["boundary"],
                gymapi.Transform(),
                "boundary",
                env_index,
                1,
                0,
            )
            boundary_idx = self.gym.get_actor_index(
                env_ptr, boundary_handle, gymapi.DOMAIN_SIM
            )

            # add object to environment
            object_handle = self.gym.create_actor(
                env_ptr,
                self.gym_assets["object"],
                gymapi.Transform(),
                "object",
                env_index,
                0,
                0,
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            # add goal object to environment
            goal_handle = self.gym.create_actor(
                env_ptr,
                self.gym_assets["goal_object"],
                gymapi.Transform(),
                "goal_object",
                env_index + self.num_envs,
                0,
                0,
            )
            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_handle, gymapi.DOMAIN_SIM
            )
            # change settings of DOF
            self.gym.set_actor_dof_properties(env_ptr, trifinger_actor, robot_dof_props)
            # add color to instances
            stage_color = gymapi.Vec3(0.73, 0.68, 0.72)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, stage_color
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                boundary_handle,
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                stage_color,
            )
            # end aggregation mode if enabled
            if self.cfg["env"]["aggregate_mode"]:
                self.gym.end_aggregate(env_ptr)
            # add instances to list
            self.envs.append(env_ptr)
            self.gym_indices["robot"].append(trifinger_idx)
            self.gym_indices["table"].append(table_idx)
            self.gym_indices["boundary"].append(boundary_idx)
            self.gym_indices["object"].append(object_idx)
            self.gym_indices["goal_object"].append(goal_object_idx)
        # convert gym indices from list to tensor
        for asset_name, asset_indices in self.gym_indices.items():
            self.gym_indices[asset_name] = torch.tensor(
                asset_indices, dtype=torch.long, device=self.device
            )

    def __configure_mdp_spaces(self):
        """
        Configures the observations, state and action spaces.
        """
        # Action scale for the MDP
        # Note: This is order sensitive.
        if self.cfg["env"]["command_mode"] == "position":
            # action space is joint positions
            self._action_scale.low = self._robot_limits["joint_position"].low
            self._action_scale.high = self._robot_limits["joint_position"].high
        elif self.cfg["env"]["command_mode"] == "torque":
            # action space is joint torques
            self._action_scale.low = self._robot_limits["joint_torque"].low
            self._action_scale.high = self._robot_limits["joint_torque"].high
        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)

        # Observations scale for the MDP
        # check if policy outputs normalized action [-1, 1] or not.
        if self.cfg["env"]["normalize_action"]:
            obs_action_scale = SimpleNamespace(
                low=torch.full(
                    (self.action_dim,), -1, dtype=torch.float, device=self.device
                ),
                high=torch.full(
                    (self.action_dim,), 1, dtype=torch.float, device=self.device
                ),
            )
        else:
            obs_action_scale = self._action_scale

        object_obs_low = torch.cat(
            [
                self._object_limits["position"].low,
                self._object_limits["orientation"].low,
            ]
            * 2
        )
        object_obs_high = torch.cat(
            [
                self._object_limits["position"].high,
                self._object_limits["orientation"].high,
            ]
            * 2
        )

        # Note: This is order sensitive.
        self._observations_scale.low = torch.cat(
            [
                self._robot_limits["joint_position"].low,
                self._robot_limits["joint_velocity"].low,
                object_obs_low,
                obs_action_scale.low,
            ]
        )
        self._observations_scale.high = torch.cat(
            [
                self._robot_limits["joint_position"].high,
                self._robot_limits["joint_velocity"].high,
                object_obs_high,
                obs_action_scale.high,
            ]
        )
        # State scale for the MDP
        if self.cfg["env"]["asymmetric_obs"]:
            # finger tip scaling
            fingertip_state_scale = SimpleNamespace(
                low=torch.cat(
                    [
                        self._robot_limits["fingertip_position"].low,
                        self._robot_limits["fingertip_orientation"].low,
                        self._robot_limits["fingertip_velocity"].low,
                    ]
                ),
                high=torch.cat(
                    [
                        self._robot_limits["fingertip_position"].high,
                        self._robot_limits["fingertip_orientation"].high,
                        self._robot_limits["fingertip_velocity"].high,
                    ]
                ),
            )
            states_low = [
                self._observations_scale.low,
                self._object_limits["velocity"].low,
                fingertip_state_scale.low.repeat(self._dims.NumFingers.value),
                self._robot_limits["joint_torque"].low,
                self._robot_limits["fingertip_wrench"].low.repeat(
                    self._dims.NumFingers.value
                ),
            ]
            states_high = [
                self._observations_scale.high,
                self._object_limits["velocity"].high,
                fingertip_state_scale.high.repeat(self._dims.NumFingers.value),
                self._robot_limits["joint_torque"].high,
                self._robot_limits["fingertip_wrench"].high.repeat(
                    self._dims.NumFingers.value
                ),
            ]
            # Note: This is order sensitive.
            self._states_scale.low = torch.cat(states_low)
            self._states_scale.high = torch.cat(states_high)
        # check that dimensions of scalings are correct
        # count number of dimensions
        state_dim = sum(self.state_spec.values())
        obs_dim = sum(self.obs_spec.values())
        action_dim = sum(self.action_spec.values())
        # check that dimensions match
        # observations
        if (
            self._observations_scale.low.shape[0] != obs_dim
            or self._observations_scale.high.shape[0] != obs_dim
        ):
            msg = (
                f"Observation scaling dimensions mismatch. "
                f"\tLow: {self._observations_scale.low.shape[0]}, "
                f"\tHigh: {self._observations_scale.high.shape[0]}, "
                f"\tExpected: {obs_dim}."
            )
            raise AssertionError(msg)
        # state
        if self.cfg["env"]["asymmetric_obs"] and (
            self._states_scale.low.shape[0] != state_dim
            or self._states_scale.high.shape[0] != state_dim
        ):
            msg = (
                f"States scaling dimensions mismatch. "
                f"\tLow: {self._states_scale.low.shape[0]}, "
                f"\tHigh: {self._states_scale.high.shape[0]}, "
                f"\tExpected: {state_dim}."
            )
            raise AssertionError(msg)
        # actions
        if (
            self._action_scale.low.shape[0] != action_dim
            or self._action_scale.high.shape[0] != action_dim
        ):
            msg = (
                f"Actions scaling dimensions mismatch. "
                f"\tLow: {self._action_scale.low.shape[0]}, "
                f"\tHigh: {self._action_scale.high.shape[0]}, "
                f"\tExpected: {action_dim}."
            )
            raise AssertionError(msg)
        # print the scaling
        print(
            f"MDP Raw observation bounds\n"
            f"\tLow: {self._observations_scale.low}\n"
            f"\tHigh: {self._observations_scale.high}"
        )
        print(
            f"MDP Raw state bounds\n"
            f"\tLow: {self._states_scale.low}\n"
            f"\tHigh: {self._states_scale.high}"
        )
        print(
            f"MDP Raw action bounds\n"
            f"\tLow: {self._action_scale.low}\n"
            f"\tHigh: {self._action_scale.high}"
        )

    def compute_reward(self, actions):
        self.rew_buf[:] = 0.0
        self.reset_buf[:] = 0.0

        self.rew_buf[:], self.reset_buf[:], log_dict = compute_trifinger_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.cfg["sim"]["dt"],
            self.cfg["env"]["reward_terms"]["finger_move_penalty"]["weight"],
            self.cfg["env"]["reward_terms"]["finger_reach_object_rate"]["weight"],
            self.cfg["env"]["reward_terms"]["object_dist"]["weight"],
            self.cfg["env"]["reward_terms"]["object_rot"]["weight"],
            self.env_steps_count,
            self._object_goal_poses_buf,
            self._object_state_history[0],
            self._object_state_history[1],
            self._fingertips_frames_state_history[0],
            self._fingertips_frames_state_history[1],
            self.cfg["env"]["reward_terms"]["keypoints_dist"]["activate"],
        )

        self.extras.update({"env/rewards/" + k: v.mean() for k, v in log_dict.items()})

    def compute_observations(self):
        # refresh memory buffers
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            joint_torques = self._dof_torque
            tip_wrenches = self._ft_sensors_values

        else:
            joint_torques = torch.zeros(
                self.num_envs,
                self._dims.JointTorqueDim.value,
                dtype=torch.float32,
                device=self.device,
            )
            tip_wrenches = torch.zeros(
                self.num_envs,
                self._dims.NumFingers.value * self._dims.WrenchDim.value,
                dtype=torch.float32,
                device=self.device,
            )

        # extract frame handles
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self.gym_indices["object"]
        # update state histories
        self._fingertips_frames_state_history.appendleft(
            self._rigid_body_state[:, fingertip_handles_indices]
        )
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        # fill the observations and states buffer

        self.obs_buf[:], self.states_buf[:] = compute_trifinger_observations_states(
            self.cfg["env"]["asymmetric_obs"],
            self._dof_position,
            self._dof_velocity,
            self._object_state_history[0],
            self._object_goal_poses_buf,
            self.actions,
            self._fingertips_frames_state_history[0],
            joint_torques,
            tip_wrenches,
        )

        # normalize observations if flag is enabled
        if self.cfg["env"]["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high,
            )

    def reset_idx(self, env_ids):

        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # A) Reset episode stats buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self._successes[env_ids] = 0
        self._successes_pos[env_ids] = 0
        self._successes_quat[env_ids] = 0
        # B) Various randomizations at the start of the episode:
        # -- Robot base position.
        # -- Stage position.
        # -- Coefficient of restituion and friction for robot, object, stage.
        # -- Mass and size of the object
        # -- Mass of robot links
        # -- Robot joint state
        robot_initial_state_config = self.cfg["env"]["reset_distribution"][
            "robot_initial_state"
        ]
        self._sample_robot_state(
            env_ids,
            distribution=robot_initial_state_config["type"],
            dof_pos_stddev=robot_initial_state_config["dof_pos_stddev"],
            dof_vel_stddev=robot_initial_state_config["dof_vel_stddev"],
        )
        # -- Sampling of initial pose of the object
        object_initial_state_config = self.cfg["env"]["reset_distribution"][
            "object_initial_state"
        ]
        self._sample_object_poses(
            env_ids,
            distribution=object_initial_state_config["type"],
        )
        # -- Sampling of goal pose of the object
        self._sample_object_goal_poses(
            env_ids, difficulty=self.cfg["env"]["task_difficulty"]
        )
        # C) Extract trifinger indices to reset
        robot_indices = self.gym_indices["robot"][env_ids].to(torch.int32)
        object_indices = self.gym_indices["object"][env_ids].to(torch.int32)
        goal_object_indices = self.gym_indices["goal_object"][env_ids].to(torch.int32)
        all_indices = torch.unique(
            torch.cat([robot_indices, object_indices, goal_object_indices])
        )
        # D) Set values into simulator
        # -- DOF
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(robot_indices),
            len(robot_indices),
        )
        # -- actor root states
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._actors_root_state),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )

    def _sample_robot_state(
        self,
        instances: torch.Tensor,
        distribution: str = "default",
        dof_pos_stddev: float = 0.0,
        dof_vel_stddev: float = 0.0,
    ):
        """Samples the robot DOF state based on the settings.

        Type of robot initial state distribution: ["default", "random"]
             - "default" means that robot is in default configuration.
             - "random" means that noise is added to default configuration
             - "none" means that robot is configuration is not reset between episodes.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
            dof_pos_stddev: Noise scale to DOF position (used if 'type' is 'random')
            dof_vel_stddev: Noise scale to DOF velocity (used if 'type' is 'random')
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample dof state based on distribution type
        if distribution == "none":
            return
        elif distribution == "default":
            # set to default configuration
            self._dof_position[instances] = self._robot_limits["joint_position"].default
            self._dof_velocity[instances] = self._robot_limits["joint_velocity"].default
        elif distribution == "random":
            # sample uniform random from (-1, 1)
            dof_state_dim = (
                self._dims.JointPositionDim.value + self._dims.JointVelocityDim.value
            )
            dof_state_noise = (
                2
                * torch.rand(
                    (
                        num_samples,
                        dof_state_dim,
                    ),
                    dtype=torch.float,
                    device=self.device,
                )
                - 1
            )
            # set to default configuration
            self._dof_position[instances] = self._robot_limits["joint_position"].default
            self._dof_velocity[instances] = self._robot_limits["joint_velocity"].default
            # add noise
            # DOF position
            start_offset = 0
            end_offset = self._dims.JointPositionDim.value
            self._dof_position[instances] += (
                dof_pos_stddev * dof_state_noise[:, start_offset:end_offset]
            )
            # DOF velocity
            start_offset = end_offset
            end_offset += self._dims.JointVelocityDim.value
            self._dof_velocity[instances] += (
                dof_vel_stddev * dof_state_noise[:, start_offset:end_offset]
            )
        else:
            msg = f"Invalid robot initial state distribution. Input: {distribution} not in [`default`, `random`]."
            raise ValueError(msg)
        # reset robot fingertips state history
        for idx in range(1, self._state_history_len):
            self._fingertips_frames_state_history[idx][instances] = 0.0

    def _sample_object_poses(self, instances: torch.Tensor, distribution: str):
        """Sample poses for the cube.

        Type of distribution: ["default", "random", "none"]
             - "default" means that pose is default configuration.
             - "random" means that pose is randomly sampled on the table.
             - "none" means no resetting of object pose between episodes.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample poses based on distribution type
        if distribution == "none":
            return
        elif distribution == "default":
            pos_x, pos_y, pos_z = self._object_limits["position"].default
            orientation = self._object_limits["orientation"].default
        elif distribution == "random":
            # For initialization
            pos_x, pos_y = random_xy(
                num_samples, self._object_dims.max_com_distance_to_center, self.device
            )
            # add a small offset to the height to account for scale randomisation (prevent ground intersection)
            pos_z = self._object_dims.size[2] / 2 + 0.0015
            orientation = random_yaw_orientation(num_samples, self.device)
        else:
            msg = (
                f"Invalid object initial state distribution. Input: {distribution} "
                "not in [`default`, `random`, `none`]."
            )
            raise ValueError(msg)
        # set buffers into simulator
        # extract indices for goal object
        object_indices = self.gym_indices["object"][instances]
        # set values into buffer
        # object buffer
        self._object_state_history[0][instances, 0] = pos_x
        self._object_state_history[0][instances, 1] = pos_y
        self._object_state_history[0][instances, 2] = pos_z
        self._object_state_history[0][instances, 3:7] = orientation
        self._object_state_history[0][instances, 7:13] = 0
        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][instances] = 0.0
        # root actor buffer
        self._actors_root_state[object_indices] = self._object_state_history[0][
            instances
        ]

    def _sample_object_goal_poses(self, instances: torch.Tensor, difficulty: int):
        """Sample goal poses for the cube and sets them into the desired goal pose buffer.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            difficulty: Difficulty level. The higher, the more difficult is the goal.

        Possible levels are:
            - -1:  Random goal position on the table, including yaw orientation.
            - 1: Random goal position on the table, no orientation.
            - 2: Fixed goal position in the air with x,y = 0.  No orientation.
            - 3: Random goal position in the air, no orientation.
            - 4: Random goal pose in the air, including orientation.
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample poses based on task difficulty
        if difficulty == -1:
            # For initialization
            pos_x, pos_y = random_xy(
                num_samples, self._object_dims.max_com_distance_to_center, self.device
            )
            pos_z = self._object_dims.size[2] / 2
            orientation = random_yaw_orientation(num_samples, self.device)
        elif difficulty == 1:
            # Random goal position on the table, no orientation.
            pos_x, pos_y = random_xy(
                num_samples, self._object_dims.max_com_distance_to_center, self.device
            )
            pos_z = self._object_dims.size[2] / 2
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 2:
            # Fixed goal position in the air with x,y = 0.  No orientation.
            pos_x, pos_y = 0.0, 0.0
            pos_z = self._object_dims.min_height + 0.05
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 3:
            # Random goal position in the air, no orientation.
            pos_x, pos_y = random_xy(
                num_samples, self._object_dims.max_com_distance_to_center, self.device
            )
            pos_z = random_z(
                num_samples,
                self._object_dims.min_height,
                self._object_dims.max_height,
                self.device,
            )
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 4:
            # Random goal pose in the air, including orientation.
            # Note: Set minimum height such that the cube does not intersect with the
            #       ground in any orientation
            max_goal_radius = self._object_dims.max_com_distance_to_center
            max_height = self._object_dims.max_height
            orientation = random_orientation(num_samples, self.device)

            # pick x, y, z according to the maximum height / radius at the current point
            # in the cirriculum
            pos_x, pos_y = random_xy(num_samples, max_goal_radius, self.device)
            pos_z = random_z(
                num_samples, self._object_dims.radius_3d, max_height, self.device
            )
        else:
            msg = f"Invalid difficulty index for task: {difficulty}."
            raise ValueError(msg)

        # extract indices for goal object
        goal_object_indices = self.gym_indices["goal_object"][instances]
        # set values into buffer
        # object goal buffer
        self._object_goal_poses_buf[instances, 0] = pos_x
        self._object_goal_poses_buf[instances, 1] = pos_y
        self._object_goal_poses_buf[instances, 2] = pos_z
        self._object_goal_poses_buf[instances, 3:7] = orientation
        # root actor buffer
        self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[
            instances
        ]
        # self._actors_root_state[goal_object_indices, 2] = -10

    def pre_physics_step(self, actions):

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.gym.simulate(self.sim)

        self.actions = actions.clone().to(self.device)

        # if normalized_action is true, then denormalize them.
        if self.cfg["env"]["normalize_action"]:
            # TODO: Default action should correspond to normalized value of 0.
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high,
            )
        else:
            action_transformed = self.actions

        # compute command on the basis of mode selected
        if self.cfg["env"]["command_mode"] == "torque":
            # command is the desired joint torque
            computed_torque = action_transformed
        elif self.cfg["env"]["command_mode"] == "position":
            # command is the desired joint positions
            desired_dof_position = action_transformed
            # compute torque to apply
            computed_torque = self._robot_dof_gains["stiffness"] * (
                desired_dof_position - self._dof_position
            )
            computed_torque -= self._robot_dof_gains["damping"] * self._dof_velocity
        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)
        # apply clamping of computed torque to actuator limits
        applied_torque = saturate(
            computed_torque,
            lower=self._robot_limits["joint_torque"].low,
            upper=self._robot_limits["joint_torque"].high,
        )
        # apply safety damping and clamping of the action torque if enabled
        if self.cfg["env"]["apply_safety_damping"]:
            # apply damping by joint velocity
            applied_torque -= (
                self._robot_dof_gains["safety_damping"] * self._dof_velocity
            )
            # clamp input
            applied_torque = saturate(
                applied_torque,
                lower=self._robot_limits["joint_torque"].low,
                upper=self._robot_limits["joint_torque"].high,
            )
        # set computed torques to simulator buffer.
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(applied_torque)
        )

    def post_physics_step(self):

        self._step_info = {}

        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        # check termination conditions (success only)
        self._check_termination()

        if torch.sum(self.reset_buf) > 0:
            self._step_info["consecutive_successes"] = np.mean(
                self._successes.float().cpu().numpy()
            )
            self._step_info["consecutive_successes_pos"] = np.mean(
                self._successes_pos.float().cpu().numpy()
            )
            self._step_info["consecutive_successes_quat"] = np.mean(
                self._successes_quat.float().cpu().numpy()
            )

    def _check_termination(self):
        """Check whether the episode is done per environment."""
        # Extract configuration for termination conditions
        termination_config = self.cfg["env"]["termination_conditions"]
        # Termination condition - successful completition
        # Calculate distance between current object and goal
        object_goal_position_dist = torch.norm(
            self._object_goal_poses_buf[:, 0:3] - self._object_state_history[0][:, 0:3],
            p=2,
            dim=-1,
        )
        # log theoretical number of r eseats
        goal_position_reset = torch.le(
            object_goal_position_dist,
            termination_config["success"]["position_tolerance"],
        )
        self._step_info["env/current_position_goal/per_env"] = np.mean(
            goal_position_reset.float().cpu().numpy()
        )
        # For task with difficulty 4, we need to check if orientation matches as well.
        # Compute the difference in orientation between object and goal pose
        object_goal_orientation_dist = quat_diff_rad(
            self._object_state_history[0][:, 3:7], self._object_goal_poses_buf[:, 3:7]
        )
        # Check for distance within tolerance
        goal_orientation_reset = torch.le(
            object_goal_orientation_dist,
            termination_config["success"]["orientation_tolerance"],
        )
        self._step_info["env/current_orientation_goal/per_env"] = np.mean(
            goal_orientation_reset.float().cpu().numpy()
        )

        if self.cfg["env"]["task_difficulty"] < 4:
            # Check for task completion if position goal is within a threshold
            task_completion_reset = goal_position_reset
        elif self.cfg["env"]["task_difficulty"] == 4:
            # Check for task completion if both position + orientation goal is within a threshold
            task_completion_reset = torch.logical_and(
                goal_position_reset, goal_orientation_reset
            )
        else:
            # Check for task completion if both orientation goal is within a threshold
            task_completion_reset = goal_orientation_reset

        self._successes = task_completion_reset
        self._successes_pos = goal_position_reset
        self._successes_quat = goal_orientation_reset

    """
    Helper functions - define assets
    """

    def __define_robot_asset(self):
        """Define Gym asset for robot."""
        # define tri-finger asset
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.flip_visual_attachments = False
        robot_asset_options.fix_base_link = True
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = False
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        robot_asset_options.thickness = 0.001
        robot_asset_options.angular_damping = 0.01

        robot_asset_options.vhacd_enabled = True
        robot_asset_options.vhacd_params = gymapi.VhacdParams()
        robot_asset_options.vhacd_params.resolution = 100000
        robot_asset_options.vhacd_params.concavity = 0.0025
        robot_asset_options.vhacd_params.alpha = 0.04
        robot_asset_options.vhacd_params.beta = 1.0
        robot_asset_options.vhacd_params.convex_hull_downsampling = 4
        robot_asset_options.vhacd_params.max_num_vertices_per_ch = 256

        if self.physics_engine == gymapi.SIM_PHYSX:
            robot_asset_options.use_physx_armature = True
        # load tri-finger asset
        trifinger_asset = self.gym.load_asset(
            self.sim,
            self._trifinger_assets_dir,
            self._robot_urdf_file,
            robot_asset_options,
        )
        # set the link properties for the robot
        # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/sim_finger.py#L563
        trifinger_props = self.gym.get_asset_rigid_shape_properties(trifinger_asset)
        for p in trifinger_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(trifinger_asset, trifinger_props)
        # extract the frame handles
        for frame_name in self._fingertips_handles.keys():
            self._fingertips_handles[frame_name] = self.gym.find_asset_rigid_body_index(
                trifinger_asset, frame_name
            )
            # check valid handle
            if self._fingertips_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print(msg)

        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            sensor_pose = gymapi.Transform()
            for fingertip_handle in self._fingertips_handles.values():
                self.gym.create_asset_force_sensor(
                    trifinger_asset, fingertip_handle, sensor_pose
                )
        # extract the dof indices
        # Note: need to write actuated dofs manually since the system contains fixed joints as well which show up.
        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self.gym.find_asset_dof_index(
                trifinger_asset, dof_name
            )
            # check valid handle
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print(msg)
        # return the asset
        return trifinger_asset

    def __define_table_asset(self):
        """Define Gym asset for stage."""
        # define stage asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.thickness = 0.001

        # load stage asset
        table_asset = self.gym.load_asset(
            self.sim,
            self._trifinger_assets_dir,
            self._table_urdf_file,
            table_asset_options,
        )
        # set stage properties
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        # iterate over each mesh
        for p in table_props:
            p.friction = 0.1
            p.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        # return the asset
        return table_asset

    def __define_boundary_asset(self):
        """Define Gym asset for stage."""
        # define stage asset
        boundary_asset_options = gymapi.AssetOptions()
        boundary_asset_options.disable_gravity = True
        boundary_asset_options.fix_base_link = True
        boundary_asset_options.thickness = 0.001

        boundary_asset_options.vhacd_enabled = True
        boundary_asset_options.vhacd_params = gymapi.VhacdParams()
        boundary_asset_options.vhacd_params.resolution = 100000
        boundary_asset_options.vhacd_params.concavity = 0.0
        boundary_asset_options.vhacd_params.alpha = 0.04
        boundary_asset_options.vhacd_params.beta = 1.0
        boundary_asset_options.vhacd_params.max_num_vertices_per_ch = 1024

        # load stage asset
        boundary_asset = self.gym.load_asset(
            self.sim,
            self._trifinger_assets_dir,
            self._boundary_urdf_file,
            boundary_asset_options,
        )
        # set stage properties
        boundary_props = self.gym.get_asset_rigid_shape_properties(boundary_asset)

        self.gym.set_asset_rigid_shape_properties(boundary_asset, boundary_props)
        # return the asset
        return boundary_asset

    def __define_object_asset(self):
        """Define Gym asset for object."""
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True
        # load object asset
        object_asset = self.gym.load_asset(
            self.sim,
            self._trifinger_assets_dir,
            self._object_urdf_file,
            object_asset_options,
        )
        # set object properties
        # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/collision_objects.py#L96
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        for p in object_props:
            p.friction = 1.0
            p.torsion_friction = 0.001
            p.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(object_asset, object_props)
        # return the asset
        return object_asset

    def __define_goal_object_asset(self):
        """Define Gym asset for goal object."""
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True
        # load object asset
        goal_object_asset = self.gym.load_asset(
            self.sim,
            self._trifinger_assets_dir,
            self._object_urdf_file,
            object_asset_options,
        )
        # return the asset
        return goal_object_asset

    @property
    def env_steps_count(self) -> int:
        """Returns the total number of environment steps aggregated across parallel environments."""
        return self.gym.get_frame_count(self.sim) * self.num_envs


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0, eps: float = 2) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function (controls how wide the 'bell' shape is')
        eps: Controls how 'tall' the 'bell' shape is.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())


@torch.jit.script
def gen_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: Tuple[float, float, float] = (0.065, 0.065, 0.065),
):

    num_envs = pose.shape[0]

    keypoints_buf = torch.ones(
        num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device
    )

    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = (
            torch.tensor(corner_loc, dtype=torch.float32, device=pose.device)
            * keypoints_buf[:, i, :]
        )
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf


@torch.jit.script
def compute_trifinger_reward(
    obs_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    episode_length: int,
    dt: float,
    finger_move_penalty_weight: float,
    finger_reach_object_weight: float,
    object_dist_weight: float,
    object_rot_weight: float,
    env_steps_count: int,
    object_goal_poses_buf: torch.Tensor,
    object_state: torch.Tensor,
    last_object_state: torch.Tensor,
    fingertip_state: torch.Tensor,
    last_fingertip_state: torch.Tensor,
    use_keypoints: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    ft_sched_start = 0
    ft_sched_end = 5e7

    # Reward penalising finger movement

    fingertip_vel = (fingertip_state[:, :, 0:3] - last_fingertip_state[:, :, 0:3]) / dt

    finger_movement_penalty = finger_move_penalty_weight * fingertip_vel.pow(2).view(
        -1, 9
    ).sum(dim=-1)

    # Reward for finger reaching the object

    # distance from each finger to the centroid of the object, shape (N, 3).
    curr_norms = torch.stack(
        [
            torch.norm(fingertip_state[:, i, 0:3] - object_state[:, 0:3], p=2, dim=-1)
            for i in range(3)
        ],
        dim=-1,
    )
    # distance from each finger to the centroid of the object in the last timestep, shape (N, 3).
    prev_norms = torch.stack(
        [
            torch.norm(
                last_fingertip_state[:, i, 0:3] - last_object_state[:, 0:3], p=2, dim=-1
            )
            for i in range(3)
        ],
        dim=-1,
    )

    ft_sched_val = 1.0 if ft_sched_start <= env_steps_count <= ft_sched_end else 0.0
    finger_reach_object_reward = (
        finger_reach_object_weight
        * ft_sched_val
        * (curr_norms - prev_norms).sum(dim=-1)
    )

    if use_keypoints:
        object_keypoints = gen_keypoints(object_state[:, 0:7])
        goal_keypoints = gen_keypoints(object_goal_poses_buf[:, 0:7])

        delta = object_keypoints - goal_keypoints

        dist_l2 = torch.norm(delta, p=2, dim=-1)

        keypoints_kernel_sum = lgsk_kernel(dist_l2, scale=30.0, eps=2.0).mean(dim=-1)

        pose_reward = object_dist_weight * dt * keypoints_kernel_sum

    else:

        # Reward for object distance
        object_dist = torch.norm(
            object_state[:, 0:3] - object_goal_poses_buf[:, 0:3], p=2, dim=-1
        )
        object_dist_reward = (
            object_dist_weight * dt * lgsk_kernel(object_dist, scale=50.0, eps=2.0)
        )

        # Reward for object rotation

        # extract quaternion orientation
        quat_a = object_state[:, 3:7]
        quat_b = object_goal_poses_buf[:, 3:7]

        angles = quat_diff_rad(quat_a, quat_b)
        object_rot_reward = object_rot_weight * dt / (3.0 * torch.abs(angles) + 0.01)

        pose_reward = object_dist_reward + object_rot_reward

    total_reward = finger_movement_penalty + finger_reach_object_reward + pose_reward

    # reset agents
    reset = torch.zeros_like(reset_buf)
    reset = torch.where(
        progress_buf >= episode_length - 1, torch.ones_like(reset_buf), reset
    )

    info: Dict[str, torch.Tensor] = {
        "finger_movement_penalty": finger_movement_penalty,
        "finger_reach_object_reward": finger_reach_object_reward,
        "pose_reward": finger_reach_object_reward,
        "reward": total_reward,
    }

    return total_reward, reset, info


@torch.jit.script
def compute_trifinger_observations_states(
    asymmetric_obs: bool,
    dof_position: torch.Tensor,
    dof_velocity: torch.Tensor,
    object_state: torch.Tensor,
    object_goal_poses: torch.Tensor,
    actions: torch.Tensor,
    fingertip_state: torch.Tensor,
    joint_torques: torch.Tensor,
    tip_wrenches: torch.Tensor,
):

    num_envs = dof_position.shape[0]

    obs_buf = torch.cat(
        [
            dof_position,
            dof_velocity,
            object_state[:, 0:7],  # pose
            object_goal_poses,
            actions,
        ],
        dim=-1,
    )

    if asymmetric_obs:
        states_buf = torch.cat(
            [
                obs_buf,
                object_state[:, 7:13],  # linear / angular velocity
                fingertip_state.reshape(num_envs, -1),
                joint_torques,
                tip_wrenches,
            ],
            dim=-1,
        )
    else:
        states_buf = obs_buf

    return obs_buf, states_buf


"""
Sampling of cuboidal object
"""


@torch.jit.script
def random_xy(
    num: int, max_com_distance_to_center: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def random_z(
    num: int, min_height: float, max_height: float, device: str
) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros(
        (
            num,
            4,
        ),
        dtype=torch.float,
        device=device,
    )
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn(
        (
            num,
            4,
        ),
        dtype=torch.float,
        device=device,
    )
    # normalize the quaternion
    quat = torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)

    return quat


@torch.jit.script
def random_orientation_within_angle(
    num: int, device: str, base: torch.Tensor, max_angle: float
):
    """Generates random quaternions within max_angle of base
    Ref: https://math.stackexchange.com/a/3448434
    """
    quat = torch.zeros(
        (
            num,
            4,
        ),
        dtype=torch.float,
        device=device,
    )

    rand = torch.rand((num, 3), dtype=torch.float, device=device)

    c = torch.cos(rand[:, 0] * max_angle)
    n = torch.sqrt((1.0 - c) / 2.0)

    quat[:, 3] = torch.sqrt((1 + c) / 2.0)
    quat[:, 2] = (rand[:, 1] * 2.0 - 1.0) * n
    quat[:, 0] = (
        torch.sqrt(1 - quat[:, 2] ** 2.0) * torch.cos(2 * np.pi * rand[:, 2])
    ) * n
    quat[:, 1] = (
        torch.sqrt(1 - quat[:, 2] ** 2.0) * torch.sin(2 * np.pi * rand[:, 2])
    ) * n

    # floating point errors can cause it to  be slightly off, re-normalise
    quat = torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)

    return quat_mul(quat, base)


@torch.jit.script
def random_angular_vel(num: int, device: str, magnitude_stdev: float) -> torch.Tensor:
    """Samples a random angular velocity with standard deviation `magnitude_stdev`"""

    axis = torch.randn(
        (
            num,
            3,
        ),
        dtype=torch.float,
        device=device,
    )
    axis /= torch.norm(axis, p=2, dim=-1).view(-1, 1)
    magnitude = torch.randn(
        (
            num,
            1,
        ),
        dtype=torch.float,
        device=device,
    )
    magnitude *= magnitude_stdev
    return magnitude * axis


@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)
