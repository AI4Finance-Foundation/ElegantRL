import multiprocessing as mp
import sys

import isaacgym
import numpy as np
import torch  # import torch after import IsaacGym modules

from elegantrl.envs.utils_OLD.config import (
    set_seed,
    get_args,
    parse_sim_params,
    load_cfg,
)
from elegantrl.envs.utils_OLD.parse_task import parse_task
from typing import Tuple

dir((isaacgym, torch))
"""
isaacgym/gymdeps.py", line 21, in _import_deps
raise ImportError("PyTorch was imported before isaacgym modules.  
                   Please import torch after isaacgym modules.")
;              

run the following code in bash before running.
export LD_LIBRARY_PATH=/xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib
can't use os.environ['LD_LIBRARY_PATH'] = /xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib

cd isaacgym/python/ElegantRL-1212
conda activate rlgpu
export LD_LIBRARY_PATH=~/anaconda3/envs/rlgpu/lib
"""


class IsaacVecEnv:
    def __init__(
        self, env_name, env_num=32, device_id=0, rl_device_id=None, if_print=True
    ):
        """Preprocess an Isaac Gym vector environment for RL training.
        [Isaac Gym](https://developer.nvidia.com/isaac-gym)
        NVIDIA Isaac Gym. Preview 2
        """

        """env_name"""
        sys_argv = sys.argv  # build a pure sys.argv for IsaacGym args = get_args()
        sys.argv = sys.argv[:1]  # build a pure sys.argv for IsaacGym args = get_args()
        env_target_return_dict = {
            "Ant": 14e3,  # 16e3
            "Humanoid": 9e3,  # 11e3
        }
        assert env_name in env_target_return_dict
        args = get_args(task_name=env_name, headless=True)

        # set after `args = get_args()`  # get_args()  in .../utils/config.py
        rl_device_id = device_id if rl_device_id is None else rl_device_id
        args.rl_device = f"cuda:{rl_device_id}" if rl_device_id >= 0 else "cpu"
        args.device_id = device_id  # PhyX device
        args.num_envs = env_num  # in `.../cfg/train/xxx.yaml`, `numEnvs`
        # set before load_cfg()

        cfg, cfg_train, log_dir = load_cfg(args)
        sim_params = parse_sim_params(args, cfg, cfg_train)
        set_seed(cfg_train["params"]["seed"])

        task, env = parse_task(args, cfg, cfg_train, sim_params)
        assert env_num == env.num_environments
        sys.argv = sys_argv  # build a pure sys.argv for IsaacGym args = get_args()

        """max_step"""
        max_step = getattr(task, "max_episode_length", None)
        if max_step is None:
            max_step = getattr(task, "_max_episode_steps")

        """if_discrete"""
        # import gym
        # if_discrete = isinstance(env.act_space, gym.spaces.Discrete)
        if_discrete = "float" not in str(env.action_space.dtype)

        """state_dim"""
        state_dim = task.num_obs
        assert isinstance(state_dim, int)

        """action_dim"""
        if if_discrete:
            action_dim = env.action_space.n
        else:
            action_dim = task.num_actions
            assert all(env.action_space.high == np.ones(action_dim))
            assert all((-env.action_space.low) == np.ones(action_dim))

        """target_return"""
        target_return = env_target_return_dict[env_name]

        self.device = torch.device(env.rl_device)
        self.env = env
        self.env_num = env_num
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = if_discrete
        self.target_return = target_return

        if if_print:
            env_args = {
                "env_num": env_num,
                "env_name": env_name,
                "max_step": max_step,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "if_discrete": if_discrete,
                "target_return": target_return,
            }
            env_args_repr = repr(env_args)
            env_args_repr = env_args_repr.replace(",", ",\n   ")
            env_args_repr = env_args_repr.replace("{", "{\n    ")
            env_args_repr = env_args_repr.replace("}", ",\n}")
            print(f"env_args = {env_args_repr}")

    def reset(self) -> torch.Tensor:
        return self.env.reset()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        return self.env.step(actions)


class IsaacOneEnv(IsaacVecEnv):
    def __init__(self, env_name, device_id=0, if_print=True):
        """Preprocess an Isaac Gym single environment for RL evaluating.
        [Isaac Gym](https://developer.nvidia.com/isaac-gym)
        NVIDIA Isaac Gym. Preview 2
        """
        super().__init__(
            env_name=env_name,
            env_num=1,
            device_id=device_id,
            rl_device_id=-1,
            if_print=if_print,
        )

    def reset(self) -> np.ndarray:
        ten_states = self.env.reset()
        return ten_states[0].detach().numpy()  # state

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        ten_action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        ten_state, ten_reward, ten_done, info_dict = self.env.step(ten_action)

        state = ten_state[0].detach().numpy()
        reward = ten_reward[0].item()
        done = ten_done[0].item()
        return state, reward, done, info_dict


"""check"""


def run_isaac_env(env_name="Ant", if_vec_env=True):
    # from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv
    # env = IsaacVecEnv(env_name='Ant', env_num=32, device_id=0, if_print=True)
    env_func = IsaacVecEnv if if_vec_env else IsaacOneEnv
    if env_name == "Ant":
        env_args = {
            "env_num": 32,
            "env_name": "Ant",
            "max_step": 1000,
            "state_dim": 60,
            "action_dim": 8,
            "if_discrete": False,
            "target_return": 14000.0,
            "device_id": None,  # set by worker
            "if_print": False,  # if_print=False in default
        }
    elif env_name == "Humanoid":
        env_args = {
            "env_num": 32,
            "env_name": "Humanoid",
            "max_step": 1000,
            "state_dim": 108,
            "action_dim": 21,
            "if_discrete": False,
            "target_return": 9000.0,
            "device_id": None,  # set by worker
            "if_print": False,  # if_print=False in default
        }
    else:
        raise KeyError(f"| run_isaac_env: env_name={env_name}")

    if not if_vec_env:
        env_args["env_num"] = 1

    # from elegantrl.train.config import build_env
    # env = build_env(env=None, env_func=env_func, env_args=env_args)
    from elegantrl.train.config import check_env

    env = check_env(env=None, env_func=env_func, env_args=env_args, gpu_id=0)
    dir(env)


def run_isaac_gym_multiple_process():
    process_list = [mp.Process(target=run_isaac_env, args=("Ant", True))]  # VecEnv
    process_list.append(
        mp.Process(
            target=run_isaac_env,
            args=("Ant", False),
        )
    )  # OneEnv

    mp.set_start_method(method="spawn")  # should be
    [p.start() for p in process_list]
    [p.join() for p in process_list]


if __name__ == "__main__":
    run_isaac_env(env_name="Ant", if_vec_env=True)
    # run_isaac_env(env_name='Ant', if_vec_env=False)
    # run_isaac_gym_multiple_process()
