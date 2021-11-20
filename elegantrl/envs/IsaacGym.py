from elegantrl.envs.utils.config import set_seed, get_args, parse_sim_params, load_cfg
from elegantrl.envs.utils.parse_task import parse_task
import numpy as np
import torch  # import torch after isaacgym modules
import multiprocessing as mp

"""[ElegantRL.2021.11.04](https://github.com/AI4Finance-Foundation/ElegantRL)"""

"""
run the following code in bash before running.
export LD_LIBRARY_PATH=/xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib
can't use os.environ['LD_LIBRARY_PATH'] = /xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib
"""


class PreprocessIsaacVecEnv:  # environment wrapper
    def __init__(self, env_name, target_return=None, if_print=False, headless=True, data_type=torch.float32,
                 env_num=32, device_id=0, rl_device_id=-1):
        """Preprocess an Isaac Gym vec environment for RL training.
        [Isaac Gym](https://developer.nvidia.com/isaac-gym)"""
        # Override env_name if passed on the command line
        args = get_args(task_name=env_name, headless=headless)

        # set after `args = get_args()`  # get_args()  in .../utils/config.py
        args.device_id = device_id  # PhyX device
        args.rl_device = f"cuda:{rl_device_id}" if rl_device_id >= 0 else 'cpu'
        args.num_envs = env_num  # in `.../cfg/train/xxx.yaml`, `numEnvs`
        # set before load_cfg()

        cfg, cfg_train, logdir = load_cfg(args)
        sim_params = parse_sim_params(args, cfg, cfg_train)
        set_seed(cfg_train["seed"])

        task, env = parse_task(args, cfg, cfg_train, sim_params)

        self.env_name = env_name
        self.env = env
        self.data_type = data_type
        self.device = torch.device(env.rl_device)
        self.env_num = env.num_environments

        state = self.env.reset()
        self.env_num = state.shape[0]

        self.target_return = target_return

        max_step = getattr(task, 'max_episode_length', None)
        max_step_default = getattr(task, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        import gym
        if_discrete = isinstance(env.act_space, gym.spaces.Discrete)

        self.state_dim = task.num_obs
        if if_discrete:
            self.action_dim = env.action_space.n
            raise RuntimeError("| Not support for discrete environment now. :(")
        elif isinstance(env.act_space, gym.spaces.Box):
            self.action_dim = task.num_actions
            action_max = float(env.action_space.high[0])
            # check: whether the action_max is correct, delete before uploading to github, vincent
            assert not any(env.action_space.high + env.action_space.low)
        else:
            raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

        self.action_max, self.max_step = action_max, max_step
        self.if_discrete = if_discrete

        if if_print:
            print(f"\n| env_name:  {self.env_name}, action space if_discrete: {self.if_discrete}"
                  f"\n| state_dim: {self.state_dim:4}, action_dim: {self.action_dim}, action_max: {self.action_max}"
                  f"\n| max_step:  {self.max_step:4}, target_return: {self.target_return}")

    def reset(self) -> torch.Tensor:
        return self.env.reset()

    def step(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, None):
        return self.env.step(actions)[:3]


class PreprocessIsaacOneEnv(PreprocessIsaacVecEnv):  # environment wrapper
    def __init__(self, env_name, target_return=None, if_print=False, headless=True, data_type=torch.float32,
                 env_num=1, device_id=0):
        assert env_num == 1
        super().__init__(env_name=env_name,
                         target_return=target_return,
                         if_print=if_print,
                         headless=headless,
                         data_type=data_type,
                         env_num=1,
                         device_id=device_id)

    def reset(self) -> torch.Tensor:
        state = self.env.reset()
        return state[0].detach().numpy()

    def step(self, action: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, None):
        ten_action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        ten_state, ten_reward, ten_done, info_dict = self.env.step(ten_action)

        state = ten_state[0].detach().numpy()
        reward = ten_reward[0].detach().numpy()
        done = ten_done[0].detach().numpy()
        return state, reward, done, info_dict


def build_isaac_gym_env(env, if_print=False, device_id=0):
    env_name = getattr(env, 'env_name', env)
    assert isinstance(env_name, str)

    env_last_name = env_name[11:]
    assert env_last_name in {'Ant', 'Humanoid'}
    target_return = {'Ant': 4000, 'Humanoid': 7000}[env_last_name]

    if env_name.find('IsaacOneEnv') != -1:
        env = PreprocessIsaacOneEnv(env_last_name, target_return=target_return, if_print=if_print,
                                    env_num=1, device_id=device_id)
    elif env_name.find('IsaacVecEnv') != -1:
        env = PreprocessIsaacVecEnv(env_last_name, target_return=target_return, if_print=if_print,
                                    env_num=32, device_id=device_id)
    else:
        raise ValueError(f'| build_env_from_env_name: need register: {env_name}')

    return env


'''check'''


def run_isaac_env(env_name, device_id):
    env = build_isaac_gym_env(env_name, if_print=True, device_id=device_id)

    if env.env_num == 1:
        def get_random_action():
            return torch.rand(env.action_dim, dtype=torch.float32) * 2 - 1
    else:
        def get_random_action():
            return torch.rand((env.env_num, env.action_dim), dtype=torch.float32) * 2 - 1

    total_step = 2 ** 4
    print("| total_step", total_step)
    for step_i in range(total_step):
        action = get_random_action()
        state, reward, done, info_dict = env.step(action)
        print('|', device_id, step_i, state.dtype)
    print('| env_num', env.env_num)


def run_multiple_process():
    env_last_name = ['Ant', 'Humanoid'][0]
    one_env_name = f"IsaacOneEnv{env_last_name}"
    vec_env_name = f"IsaacVecEnv{env_last_name}"

    process_list = list()
    process_list.append(mp.Process(target=run_isaac_env, args=(one_env_name, 4,)))
    process_list.append(mp.Process(target=run_isaac_env, args=(vec_env_name, 5,)))

    mp.set_start_method(method='spawn')  # should be
    [p.start() for p in process_list]
    [p.join() for p in process_list]


if __name__ == '__main__':
    run_isaac_env(env_name='IsaacVecEnvAnt', device_id=3)
    # run_multiple_process()
