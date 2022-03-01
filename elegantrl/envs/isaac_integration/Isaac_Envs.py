import isaacgym
import numpy as np
import torch
from elegantrl.envs.isaac_integration.tasks import isaacgym_task_map
from elegantrl.envs.isaac_integration.tasks.base.vec_task import VecTask
from elegantrl.envs.isaac_integration.utils.utils import set_seed
from elegantrl.envs.isaac_integration.utils.config_utils import load_task_config
from pprint import pprint
from typing import Dict, Tuple


class IsaacVecEnv:
    def __init__(
        self,
        env_name: str,
        env_num=-1,
        sim_device_id=0,
        rl_device_id=0,
        headless=False,
        should_print=False,
    ):
        """
        Preprocesses an Isaac Gym environment for RL training.
        [Isaac Gym](https://developer.nvidia.com/isaac-gym)
        NVIDIA Isaac Gym - Preview 3 Release
        """
        task_config = load_task_config(env_name)
        sim_device = f"cuda:{sim_device_id}" if sim_device_id >= 0 else "cpu"
        isaac_task = isaacgym_task_map[env_name]
        self._override_default_env_num(env_num, task_config)
        set_seed(-1, False)

        env: VecTask = isaac_task(
            cfg=task_config,
            sim_device=sim_device,
            graphics_device_id=rl_device_id,
            headless=headless,
        )

        is_discrete = "float" not in str(env.action_space.dtype)

        state_dimension = env.num_obs
        assert isinstance(state_dimension, int)

        action_dimension = env.action_space.n if is_discrete else env.num_acts
        if not is_discrete:
            assert all(env.action_space.high == np.ones(action_dimension))
            assert all(-env.action_space.low == np.ones(action_dimension))

        # FIXME: figure out a better way to determine these
        target_return = 10**10
        max_step = 1000

        self.device = torch.device(rl_device_id)
        self.env = env
        self.env_num = env.num_envs
        self.env_name = env_name
        self.state_dim = state_dimension
        self.action_dim = action_dimension
        self.if_discrete = is_discrete
        self.target_return = target_return

        if should_print:
            pprint(
                {
                    "num_envs": env.num_envs,
                    "env_name": env_name,
                    "max_step": max_step,
                    "state_dim": state_dimension,
                    "action_dim": action_dimension,
                    "if_discrete": is_discrete,
                    "target_return": target_return,
                }
            )

    def _override_default_env_num(self, num_envs: int, config_args: Dict):
        """Overrides the default number of environments if it's passed in.

        Args:
            num_envs (int): new number of environments
            config_args (Dict): configuration retrieved
        """
        if num_envs > 0:
            config_args["env"]["numEnvs"] = num_envs

    def reset(self) -> torch.Tensor:
        states = self.env.reset()
        if isinstance(states, Dict):
            states = states["obs"]
        assert isinstance(states, torch.Tensor)
        return states

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        (observations_dict, rewards, dones, info_dict) = self.env.step(actions)
        observations = observations_dict["obs"]
        return (observations, rewards, dones, info_dict)


class IsaacOneEnv(IsaacVecEnv):
    def __init__(self, env_name: str, device_id=0, headless=False, should_print=False):
        """
        Preprocess an Isaac Gym single environment for RL evaluating.
        [Isaac Gym](https://developer.nvidia.com/isaac-gym)
        NVIDIA Isaac Gym - Preview 3 Release
        """
        super().__init__(
            env_name=env_name,
            env_num=1,
            sim_device_id=device_id,
            rl_device_id=device_id,
            headless=headless,
            should_print=should_print,
        )

    def reset(self) -> np.ndarray:
        tensor_state_dict = self.env.reset()
        tensor_states = tensor_state_dict["obs"]
        first_state = tensor_states[0]
        return first_state.cpu().detach().numpy()  # state

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        tensor_action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        tensor_state_dict, tensor_reward, tensor_done, info_dict = self.env.step(
            tensor_action
        )
        tensor_state = tensor_state_dict["obs"]
        state = tensor_state[0].cpu().detach().numpy()
        reward = tensor_reward[0].item()
        done = tensor_done[0].item()
        return state, reward, done, info_dict
