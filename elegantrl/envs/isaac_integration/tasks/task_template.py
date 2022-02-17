import gym
import torch
from isaacgym import gymtorch
from isaacgym import gymapi
from elegantrl.envs.isaac_integration.tasks.base.vec_task import VecTask
from typing import Dict


class IsaacVecEnv(VecTask):
    def __init__(
        self, env_name: str, config_dict: Dict, sim_device: int, headless: bool
    ):
        super().__init__(cfg=config_dict)
        self.initialize_state_tensors()

    def initialize_state_tensors(self):
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        pass

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        pass

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        pass
