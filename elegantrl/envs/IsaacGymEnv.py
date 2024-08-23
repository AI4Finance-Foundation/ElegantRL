import gym.spaces
import isaacgym
import numpy as np
import torch
from elegantrl.envs.isaac_tasks import isaacgym_task_map
from elegantrl.envs.isaac_tasks.base.vec_task import VecTask
from elegantrl.envs.utils.utils import set_seed
from elegantrl.envs.utils.config_utils import load_task_config, get_max_step_from_config
from pprint import pprint
from typing import Dict, Tuple

'''[ElegantRL.2022.06.06](github.com/AI4Fiance-Foundation/ElegantRL)'''

"""
Source: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs (I hate `import hydra` in IsaacGym Preview 3)
Modify: https://github.com/hmomin (hmomin's code is quite good!)
Modify: https://github.com/Yonv1943 (I make a little change based on hmomin's code)

There are still cuda:0 BUG in Isaac Gym Preview 3:
    Isaac Gym Preview 3 will force the cuda:0 to be used even you set the `sim_device_id=1, rl_device_id=1`
    You can only use `export CUDA_VISIBLE_DEVICES=1,2,3` to let Isaac Gym use a specified GPU.


isaacgym/gymdeps.py", line 21, in _import_deps
raise ImportError("PyTorch was imported before isaacgym modules.  
                   Please import th after isaacgym modules.")             

run the following code in bash before running.
export LD_LIBRARY_PATH=/xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib
can't use os.environ['LD_LIBRARY_PATH'] = /xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib

cd isaacgym/python/ElegantRL-1212
conda activate rlgpu
export LD_LIBRARY_PATH=~/anaconda3/envs/rlgpu/lib
"""

Tensor = torch.Tensor
Array = np.ndarray


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
        """Preprocesses a vectorized Isaac Gym environment for RL training.
        [Isaac Gym - Preview 3 Release](https://developer.nvidia.com/isaac-gym)

        Args:
            env_name (str): the name of the environment to be processed.
            env_num (int, optional): the number of environments to simulate on the
                device. Defaults to whatever is specified in the corresponding config
                file.
            sim_device_id (int, optional): the GPU device id to render physics on.
                Defaults to 0.
            rl_device_id (int, optional): the GPU device id to perform RL training on.
                Defaults to 0.
            headless (bool, optional): whether or not the Isaac Gym environment should
                render on-screen. Defaults to False.
            should_print (bool, optional): whether or not the arguments should be
                printed. Defaults to False.
        """
        task_config = load_task_config(env_name)
        sim_device = f"cuda:{sim_device_id}" if sim_device_id >= 0 else "cpu"
        self.device = sim_device
        isaac_task = isaacgym_task_map[env_name]
        self._override_default_env_num(env_num, task_config)
        set_seed(-1, False)

        env: VecTask = isaac_task(
            cfg=task_config,
            sim_device=sim_device,
            graphics_device_id=rl_device_id,
            headless=headless,
        )

        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # is_discrete = not isinstance(env.action_space, gym.spaces.Box)  # Continuous action space

        state_dimension = env.num_obs
        assert isinstance(state_dimension, int)

        action_dim = getattr(env.action_space, 'n') if is_discrete else env.num_acts
        if not is_discrete:
            try:
                assert all(getattr(env.action_space, 'high') == np.ones(action_dim))
                assert all(-getattr(env.action_space, 'low') == np.ones(action_dim))
            except AssertionError:
                print(f"\n| IsaacGymEnv env.action_space.high {getattr(env.action_space, 'high')}"
                      f"\n| IsaacGymEnv env.action_space.low  {getattr(env.action_space, 'low')}")
                raise AssertionError("| IsaacGymEnv env.action_space should be (-1.0, +1.0)")

        target_return = 10 ** 10  # TODO:  plan to make `target_returns` optional

        env_config = task_config["env"]
        max_step = get_max_step_from_config(env_config)

        self.device = torch.device(rl_device_id)
        self.env = env
        self.env_num = env.num_envs
        self.env_name = env_name
        self.max_step = max_step
        self.state_dim = state_dimension
        self.action_dim = action_dim
        self.if_discrete = is_discrete
        self.target_return = target_return

        if should_print:
            pprint(
                {
                    "num_envs": env.num_envs,
                    "env_name": env_name,
                    "max_step": max_step,
                    "state_dim": state_dimension,
                    "action_dim": action_dim,
                    "if_discrete": is_discrete,
                    "target_return": target_return,
                }
            )

    def convert_obs_to_state_device(self, obs_dict) -> Tensor:
        return obs_dict['obs'].to(self.device)

    @staticmethod
    def _override_default_env_num(num_envs: int, config_args: Dict):
        """Overrides the default number of environments if it's passed in.

        Args:
            num_envs (int): new number of environments.
            config_args (Dict): configuration retrieved.
        """
        if num_envs > 0:
            config_args["env"]["numEnvs"] = num_envs

    def reset(self) -> Tensor:
        """Resets the environments in the VecTask that need to be reset.

        Returns:
            torch.Tensor: the next states in the simulation.
        """
        tensor_state_dict = self.env.reset()
        return self.convert_obs_to_state_device(tensor_state_dict)

    def step(self, actions: Tensor) -> (Tensor, Tensor, Tensor, Dict):
        """Steps through the vectorized environment.

        Args:
            actions (torch.Tensor): a multidimensional tensor of actions to perform on
                *each* environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]: a tuple containing
                observations, rewards, dones, and extra info.
        """
        observations_dict, rewards, dones, info_dict = self.env.step(actions)
        states = self.convert_obs_to_state_device(self.env.reset())
        return states, rewards.to(self.device), dones.to(self.device), info_dict


class IsaacOneEnv(IsaacVecEnv):
    def __init__(self, env_name: str, device_id=0, headless=False, should_print=False):
        """Preprocesses a single Isaac Gym environment for RL evaluating.
        [Isaac Gym - Preview 3 Release](https://developer.nvidia.com/isaac-gym)

        Args:
            env_name (str): the name of the environment to be processed.
            device_id (int, optional): the GPU device id to render physics and perform
                RL training. Defaults to 0.
            headless (bool, optional): whether or not the Isaac Gym environment should
                render on-screen. Defaults to False.
            should_print (bool, optional): whether or not the arguments should be
                printed. Defaults to False.
        """
        super().__init__(
            env_name=env_name,
            env_num=1,
            sim_device_id=device_id,
            rl_device_id=device_id,
            headless=headless,
            should_print=should_print,
        )

    @staticmethod
    def convert_obs_to_state_numpy(obs_dict) -> Array:
        return obs_dict['obs'].detach().cpu().numpy()[0]

    def reset(self) -> Array:
        """Resets the environments in the VecTask that need to be reset.

        Returns:
            np.ndarray: a numpy array containing the new state of the single
                environment.
        """
        tensor_state_dict = self.env.reset()
        return self.convert_obs_to_state_numpy(tensor_state_dict)  # state

    def step(self, action: Array) -> (Array, Array, bool, dict):
        """Steps through the single environment.

        Args:
            action (np.ndarray): a (possibly multidimensional) numpy array of actions
                to perform on the single environment.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]: a tuple containing
                observations, rewards, dones, and extra info.
        """
        tensor_action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        tensor_state_dict, tensor_reward, tensor_done, info_dict = self.env.step(tensor_action)
        state = self.convert_obs_to_state_numpy(tensor_state_dict)
        reward = tensor_reward[0].item()
        done = tensor_done[0].item()
        return state, reward, done, info_dict


def check_isaac_gym(env_name='Ant', gpu_id=0):
    assert env_name in {
        'AllegroHand',
        'Ant',
        'Anymal',
        'AnymalTerrain',
        'BallBalance',
        'Cartpole',
        'FrankaCabinet',
        'Humanoid',
        'Ingenuity',
        'Quadcopter',
        'ShadowHand',
        'Trifinger',
    }  # raise NameError by input an incorrect environment name to see the avaliable env_name
    env = IsaacVecEnv(env_name=env_name, env_num=1024, sim_device_id=gpu_id, rl_device_id=gpu_id, should_print=True)
    states = env.reset()
    print('\n\nstates.shape', states.shape)

    import torch

    action = torch.rand((env.env_num, env.action_dim), dtype=torch.float32)
    print('\n\naction.shape', action.shape)

    states, rewards, dones, info_dict = env.step(action)
    print(f'\nstates.shape  {states.shape}'
          f'\nrewards.shape {rewards.shape}'
          f'\ndones.shape   {dones.shape}'
          f'\nrepr(info.dict) {repr(info_dict)}')

    from tqdm import trange

    device = torch.device(f"cuda:{gpu_id}")
    rewards_ary = []
    dones_ary = []
    env.reset()
    print()
    for _ in trange(env.max_step * 2):
        action = torch.rand((env.env_num, env.action_dim), dtype=torch.float32, device=device)
        states, rewards, dones, info_dict = env.step(action)

        rewards_ary.append(rewards)
        dones_ary.append(dones)

    rewards_ary = torch.stack(rewards_ary)  # rewards_ary.shape == (env.max_step, env.num_envs)
    dones_ary = torch.stack(dones_ary)
    print(f'\nrewards_ary.shape {rewards_ary.shape}'
          f'\ndones_ary.shape   {dones_ary.shape}')

    reward_list = []
    steps_list = []
    print()
    for i in trange(env.env_num):
        dones_where = torch.where(dones_ary[:, i] == 1)[0]
        episode_num = dones_where.shape[0]

        if episode_num == 0:
            continue

        j0 = 0
        rewards_env = rewards_ary[:, i]
        for j1 in dones_where + 1:
            reward_list.append(rewards_env[j0:j1].sum())
            steps_list.append(j1 - j0 + 1)
            j0 = j1

    reward_list = torch.tensor(reward_list, dtype=torch.float32)
    steps_list = torch.tensor(steps_list, dtype=torch.float32)

    print(f'\n reward_list avg {reward_list.mean(0):9.2f}'
          f'\n             std {reward_list.std(0):9.2f}'
          f'\n  steps_list avg {steps_list.mean(0):9.2f}'
          f'\n             std {steps_list.std(0):9.2f}'
          f'\n     episode_num {steps_list.shape[0]}')
    return reward_list, steps_list


if __name__ == '__main__':
    check_isaac_gym()
