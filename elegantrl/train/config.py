import os
import torch as th
import numpy as np
from typing import Tuple
from multiprocessing import Pipe, Process

TEN = th.Tensor


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.num_envs = None  # `num_envs==1` in a single environment. `num_envs > 1` in a vectorized environment.
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = self.get_if_off_policy()  # whether off-policy or on-policy of DRL algorithm

        '''Argument of environment'''
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None,
                        'num_envs': 1,
                        'max_step': 12345,
                        'state_dim': None,
                        'action_dim': None,
                        'if_discrete': None, }
        env_args.setdefault('num_envs', 1)  # `num_envs=1` in default in single env.
        env_args.setdefault('max_step', 12345)  # `max_step=12345` in default, which is a large enough value.
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.num_envs = env_args['num_envs']  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.max_step = env_args['max_step']  # the max step number of an episode. set as 12345 in default.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = [128, 128]  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # the learning rate for network updating
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
        self.continue_train = False  # continue train use last train saved models
        if self.if_off_policy:  # off-policy
            self.batch_size = int(64)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(512)  # collect horizon_len step while exploring, then update networks
            self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
            self.lambda_fit_cum_r = 0.0  # critic fits the mean of a batch cumulative rewards
            self.buffer_init_size = int(self.batch_size * 8)  # train after samples over buffer_init_size for off-policy
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(2048)  # collect horizon_len step while exploring, then update network
            self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_vtrace = True  # use V-trace + GAE (Generalized Advantage Estimation) for sparse reward
            self.buffer_init_size = None  # train after samples over buffer_init_size for off-policy

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.num_workers = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for pytorch, `th.set_num_threads(self.num_threads)`
        self.random_seed = None  # initialize random seed in self.init_before_training(), None means set GPU_ID as seed
        self.learner_gpu_ids = ()  # multiple gpu id Tuple[int, ...] for learners. () means single GPU or CPU.

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = np.inf  # break training if 'total_step > break_step'
        self.break_score = np.inf  # break training if `cumulative_rewards > break_score`
        self.if_keep_save = True  # keeping save the checkpoint. False means save until stop training.
        self.if_over_write = False  # overwrite the best policy network. `self.cwd/actor.pth`
        self.if_save_buffer = False  # if save the replay buffer for continuous training after stop training

        self.save_gap = int(8)  # save actor f"{cwd}/actor_*.pth" for learning curve.
        self.eval_times = int(3)  # number of times that get the average episodic cumulative return
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps
        self.eval_env_class = None  # eval_env = eval_env_class(*eval_env_args)
        self.eval_env_args = None  # eval_env = eval_env_class(*eval_env_args)
        self.eval_record_step = 0  # evaluator start recording after the exploration reaches this step.

    def init_before_training(self):
        if self.random_seed is None:
            self.random_seed = max(0, self.gpu_id)
        np.random.seed(self.random_seed)
        th.manual_seed(self.random_seed)
        th.set_num_threads(self.num_threads)
        th.set_default_dtype(th.float32)

        '''set cwd (current working directory) for saving model'''
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{self.random_seed}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}", flush=True)
        else:
            print(f"| Arguments Keep cwd: {self.cwd}", flush=True)
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])

    def print_config(self):
        from pprint import pprint
        print(pprint(vars(self)), flush=True)  # prints out args in a neat, readable format


def build_env(env_class=None, env_args: dict = None, gpu_id: int = -1):
    import warnings
    warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated.*")
    env_args['gpu_id'] = gpu_id  # set gpu_id for vectorized env before build it

    if env_args.get('if_build_vec_env'):
        num_envs = env_args['num_envs']
        env = VecEnv(env_class=env_class, env_args=env_args, num_envs=num_envs, gpu_id=gpu_id)
    elif env_class.__module__ == 'gymnasium.envs.registration':
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))

    env_args.setdefault('num_envs', 1)
    env_args.setdefault('max_step', 12345)

    for attr_str in ('env_name', 'num_envs', 'max_step', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def get_gym_env_args(env, if_print: bool) -> dict:
    """get a dict about a standard OpenAI gym env information.

    param env: a standard OpenAI gym env
    param if_print: [bool] print the dict about env information.
    return: env_args [dict]

    env_args = {
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'num_envs': num_envs.       # [int] the number of sub envs in vectorized env. `num_envs=1` in single env.
        'max_step': max_step,       # [int] the max step number of an episode.
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated.*")

    if_gym_standard_env = {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env))

    if if_gym_standard_env and (not hasattr(env, 'num_envs')):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        num_envs = getattr(env, 'num_envs', 1)
        max_step = getattr(env, '_max_episode_steps', 12345)

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = str(env.action_space).find('Discrete') >= 0
        if if_discrete:  # make sure it is discrete action space
            action_dim = getattr(env.action_space, 'n')
        elif str(env.action_space).find('Box') >= 0:  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if any(env.action_space.high - 1):
                print(f'| WARNING: env.action_space.high {env.action_space.high}', flush=True)
            if any(env.action_space.low + 1):
                print(f'| WARNING: env.action_space.low {env.action_space.low}', flush=True)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info(). Please set these value manually:'
                               '\n  `state_dim=int; action_dim=int; if_discrete=bool;`'
                               '\n  And keep action_space in range (-1, 1).')
    else:
        env_name = getattr(env, 'env_name', 'env')
        num_envs = getattr(env, 'num_envs', 1)
        max_step = getattr(env, 'max_step', 12345)
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_name': env_name,
                'num_envs': num_envs,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_str = repr(env_args).replace(',', f",\n{'':11}")
        print(f"env_args = {env_args_str}", flush=True)
    return env_args


"""vectorized env"""


class SubEnv(Process):
    def __init__(self, sub_pipe0: Pipe, vec_pipe1: Pipe,
                 env_class, env_args: dict, env_id: int = 0):
        super().__init__()
        self.sub_pipe0 = sub_pipe0
        self.vec_pipe1 = vec_pipe1

        self.env_class = env_class
        self.env_args = env_args
        self.env_id = env_id

    def run(self):
        th.set_grad_enabled(False)

        '''build env'''
        if self.env_class.__module__ == 'gymnasium.envs.registration':  # is standard OpenAI Gym env
            env = self.env_class(id=self.env_args['env_name'])
        else:
            env = self.env_class(**kwargs_filter(self.env_class.__init__, self.env_args.copy()))

        '''set env random seed'''
        random_seed = self.env_id
        np.random.seed(random_seed)
        th.manual_seed(random_seed)

        while True:
            action = self.sub_pipe0.recv()
            if action is None:
                state, info_dict = env.reset()
                self.vec_pipe1.send((self.env_id, state))
            else:
                state, reward, terminal, truncate, info_dict = env.step(action)

                done = terminal or truncate
                state = env.reset()[0] if done else state
                self.vec_pipe1.send((self.env_id, state, reward, terminal, truncate))


class VecEnv:
    def __init__(self, env_class: object, env_args: dict, num_envs: int, gpu_id: int = -1):
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.num_envs = num_envs  # the number of sub env in vectorized env.

        '''the necessary env information when you design a custom env'''
        self.env_name = env_args['env_name']  # the name of this env.
        self.max_step = env_args['max_step']  # the max step number in an episode for evaluation
        self.state_dim = env_args['state_dim']  # feature number of state
        self.action_dim = env_args['action_dim']  # feature number of action
        self.if_discrete = env_args['if_discrete']  # discrete action or continuous action

        '''speed up with multiprocessing: Process, Pipe'''
        assert self.num_envs <= 64
        self.res_list = [[] for _ in range(self.num_envs)]

        sub_pipe0s, sub_pipe1s = list(zip(*[Pipe(duplex=False) for _ in range(self.num_envs)]))
        self.sub_pipe1s = sub_pipe1s

        vec_pipe0, vec_pipe1 = Pipe(duplex=False)  # recv, send
        self.vec_pipe0 = vec_pipe0

        self.sub_envs = [
            SubEnv(sub_pipe0=sub_pipe0, vec_pipe1=vec_pipe1,
                   env_class=env_class, env_args=env_args, env_id=env_id)
            for env_id, sub_pipe0 in enumerate(sub_pipe0s)
        ]

        [setattr(p, 'daemon', True) for p in self.sub_envs]  # set before process start to exit safely
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tuple[TEN, dict]:  # reset the agent in env
        th.set_grad_enabled(False)

        for pipe in self.sub_pipe1s:
            pipe.send(None)
        states, = self.get_orderly_zip_list_return()
        states = th.tensor(np.stack(states), dtype=th.float32, device=self.device)
        info_dicts = dict()
        return states, info_dicts

    def step(self, action: TEN) -> Tuple[TEN, TEN, TEN, TEN, dict]:  # agent interacts in env
        action = action.detach().cpu().numpy()
        for pipe, a in zip(self.sub_pipe1s, action):
            pipe.send(a)

        states, rewards, terminal, truncate = self.get_orderly_zip_list_return()
        states = th.tensor(np.stack(states), dtype=th.float32, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        terminal = th.tensor(terminal, dtype=th.bool, device=self.device)
        truncate = th.tensor(truncate, dtype=th.bool, device=self.device)
        info_dicts = dict()
        return states, rewards, terminal, truncate, info_dicts

    def close(self):
        [process.terminate() for process in self.sub_envs]

    def get_orderly_zip_list_return(self):
        for _ in range(self.num_envs):
            res = self.vec_pipe0.recv()
            self.res_list[res[0]] = res[1:]
        return list(zip(*self.res_list))


def check_vec_env():
    import gymnasium as gym
    num_envs = 3
    gpu_id = 0

    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'CartPole-v1',
                'max_step': 500,
                'state_dim': 4,
                'action_dim': 2,
                'if_discrete': True, }

    env = VecEnv(env_class=env_class, env_args=env_args, num_envs=num_envs, gpu_id=gpu_id)

    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state, info_dict = env.reset()
    print(f"| num_envs {num_envs}  state.shape {state.shape}", flush=True)

    for i in range(4):
        if env.if_discrete:  # state -> action
            action = th.randint(0, env.action_dim, size=(num_envs,), device=device)
        else:
            action = th.zeros(size=(num_envs,), dtype=th.float32, device=device)
        state, reward, terminal, truncate, info_dict = env.step(action)

        print(f"| num_envs {num_envs}  {[t.shape for t in (state, reward, terminal, truncate)]}", flush=True)
    env.close() if hasattr(env, 'close') else None


if __name__ == '__main__':
    check_vec_env()
