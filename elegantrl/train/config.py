import inspect
import os
from copy import deepcopy

import numpy as np
import numpy.random as rd
import torch


class Arguments:
    """
    Configuration map.

    :param env[object]: the environment object in ElegantRL.
    :param agent[object]: the agent object in ElegantRL.

    **Attributes for environment setup**

    Attributes
    ----------------
        env : object
            environment object in ElegantRL.
        env_num : int
            number of sub-environments. For VecEnv, env_num > 1.
        max_step : int
            max step of an episode.
        state_dim : int
            state dimension of the environment.
        action_dim : int
            action dimension of the environment.
        if_discrete : boolean
            discrete or continuous action space.
        target_return : float
            target average episodic return.

    **Attributes for model training**

    Attributes
    ----------------
        agent : object
            agent object in ElegantRL.
        if_off_policy : boolean
            off-policy or on-policy for the DRL algorithm.
        net_dim : int
            neural network width.
        max_memo : int
            capacity of replay buffer.
        batch_size : int
            number of transitions sampled in one iteration.
        target_step : int
            repeatedly update network to keep critic's loss small.
        repeat_times : int
            collect target_step, then update network.
        break_step : int
            break training after total_step > break_step.
        if_allow_break : boolean
            allow break training when reach goal (early termination).
        if_per_or_gae : boolean
            use Prioritized Experience Replay (PER) or not for off-policy algorithms.

            use Generalized Advantage Estimation or not for on-policy algorithms.
        gamma : float
            discount factor of future rewards.
        reward_scale : int
            an approximate target reward.
        learning_rate : float
            the learning rate.
        soft_update_tau : float
            soft update parameter for target networks.

    **Attributes for model evaluation**

    Attributes
    ----------------
        eval_env : object
            environment object for model evaluation.
        eval_gap : int
            time gap for periodical evaluation (in seconds).
        eval_times1 : int
            number of times that get episode return in first.
        eval_times2 : int
            number of times that get episode return in second.
        eval_gpu_id : int or None
            the GPU id for the evaluation environment.

            -1 means use cpu, >=0 means use GPU, None means set as learner_gpus[0].
        if_overwrite : boolean
            save policy networks with different episodic return separately or overwrite.

    **Attributes for resource allocation**

    Attributes
    ----------------
        worker_num : int
            rollout workers number per GPU (adjust it to get high GPU usage).
        thread_num : int
            cpu_num for evaluate model.
        random_seed : int
            initialize random seed in ``init_before_training``.
        learner_gpus : list
            GPU ids for learner.
        workers_gpus : list
            GPU ids for worker.
        ensemble_gpus : list
            GPU ids for population-based training (PBT).
        ensemble_gap : list
            time gap for leaderboard update in tournament-based ensemble training.
        cwd : string
            directory path to save the model.
        if_remove : boolean
            remove the cwd folder? (True, False, None:ask me).
    """

    def __init__(self, agent, env=None, env_func=None, env_args=None):
        self.env = env  # the environment for training
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.update_attr(
            "env_num"
        )  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.update_attr(
            "max_step"
        )  # the env name. Be used to set 'cwd'.
        self.env_name = self.update_attr("env_name")  # the max step of an episode
        self.state_dim = self.update_attr(
            "state_dim"
        )  # vector dimension (feature number) of state
        self.action_dim = self.update_attr(
            "action_dim"
        )  # vector dimension (feature number) of action
        self.if_discrete = self.update_attr(
            "if_discrete"
        )  # discrete or continuous action space
        self.target_return = self.update_attr(
            "target_return"
        )  # target average episode return

        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.if_off_policy = agent.if_off_policy  # agent is on-policy or off-policy
        if self.if_off_policy:  # off-policy
            self.net_dim = 2**8  # the network width
            self.max_memo = 2**21  # capacity of replay buffer
            self.batch_size = (
                self.net_dim
            )  # num of transitions sampled from replay buffer.
            self.target_step = (
                2**10
            )  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2**0  # collect target_step, then update network
            self.if_per_or_gae = (
                False  # use PER (Prioritized Experience Replay) for sparse reward
            )
        else:  # on-policy
            self.net_dim = 2**9  # the network width
            self.max_memo = 2**12  # capacity of replay buffer
            self.batch_size = (
                self.net_dim * 2
            )  # num of transitions sampled from replay buffer.
            self.target_step = (
                self.max_memo
            )  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2**3  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        """Arguments for training"""
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = (
            2**0
        )  # an approximate target reward usually be closed to 256
        self.learning_rate = 2**-15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2**-8  # 2 ** -8 ~= 5e-3

        """Arguments for device"""
        self.worker_num = (
            2  # rollout workers number per GPU (adjust it to get high GPU usage)
        )
        self.thread_num = (
            8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        )
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = (
            -1
        )  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.workers_gpus = self.learner_gpus  # for GPU_VectorEnv (such as isaac gym)

        """Arguments for evaluate and save"""
        self.cwd = None  # the directory path to save the model
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.if_allow_break = (
            True  # allow break training when reach goal (early termination)
        )

        self.eval_env = (
            None  # the environment for evaluating. None means set automatically.
        )
        self.eval_env_func = None  # env = env_func(*env_args)
        self.eval_env_args = None  # env = env_func(*env_args)

        self.eval_gap = 2**8  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2**2  # number of times that get episode return in first
        self.eval_times2 = 2**4  # number of times that get episode return in second
        self.eval_gpu_id = None  # -1 means use cpu, >=0 means use GPU, None means set as learner_gpus[0]
        self.if_overwrite = (
            True  # Save policy networks with different episode return or overwrite
        )

        self.save_gap = 2**9  # save the agent per save_gap seconds (for ensemble DRL)
        self.save_dir = "./LeaderBoard"  # a directory to save the `pod_save_{episode_returns}` for ensemble DRL

    def init_before_training(self, agent_id=0):
        """
        Check parameters before training.
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        """env"""
        assert isinstance(self.env_num, int)
        assert isinstance(self.env_name, str)
        assert isinstance(self.max_step, int)
        assert isinstance(self.state_dim, (int, tuple))
        assert isinstance(self.action_dim, (int, tuple))
        assert isinstance(self.if_discrete, (int, bool))
        assert isinstance(self.target_return, (int, float))

        """agent"""
        assert hasattr(self.agent, "init")
        assert hasattr(self.agent, "update_net")
        assert hasattr(self.agent, "explore_env")
        assert hasattr(self.agent, "select_actions")

        """auto set"""
        if isinstance(self.learner_gpus, int):
            self.learner_gpus = (self.learner_gpus,)

        from collections.abc import Iterable

        if isinstance(self.learner_gpus[0], Iterable):  # ensemble DRL
            self.learner_gpus = self.learner_gpus[agent_id]
            self.cwd = f"{self.save_dir}/pod_{agent_id:04}"
        else:
            agent_name = self.agent.__class__.__name__[5:]
            self.cwd = self.update_value(
                self.cwd,
                f"./{self.env_name}_{agent_id:02}_{agent_name}_{self.learner_gpus}",
            )

        self.eval_env_func = self.update_value(self.eval_env_func, self.env_func)
        self.eval_env_args = self.update_value(self.eval_env_args, self.env_args)
        self.eval_gpu_id = self.update_value(self.eval_gpu_id, self.learner_gpus[0])
        self.eval_env = self.update_value(self.eval_env, self.env)

        """remove history"""
        if self.if_remove is None:
            self.if_remove = bool(
                input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == "y"
            )
        elif self.if_remove:
            import shutil

            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    @staticmethod
    def update_value(src, dst):
        if src is None:
            src = dst
        return src

    def update_attr(self, attr: str):
        return getattr(self.env, attr) if self.env_args is None else self.env_args[attr]


def build_env(
    env=None, env_func=None, env_args=None, gpu_id=-1
):  # [ElegantRL.2021.12.12]
    if env is not None:
        env = deepcopy(env)
    else:
        try:
            env_args0 = deepcopy(env_args)
            env_args0["device_id"] = gpu_id  # -1 means CPU, int >=1 means GPU id
            env_args1 = kwargs_filter(env_func.__init__, env_args0)

            env = env_func(**env_args1)
        except TypeError as error:
            if (
                repr(error)
                != """TypeError("make() missing 1 required positional argument: 'id'")"""
            ):
                raise TypeError(f"Meet ERROR: {error}\nCheck env_args: {env_args}")
            import gym

            gym.logger.set_level(40)
            env = env_func(id=env_args["id"])
    env.max_step = env.max_step if hasattr(env, "max_step") else env_args["max_step"]
    env.if_discrete = (
        env.if_discrete if hasattr(env, "if_discrete") else env_args["if_discrete"]
    )
    return env


def check_env(env=None, env_func=None, env_args=None, gpu_id=-1):
    if env is None:
        env = build_env(env=env, env_func=env_func, env_args=env_args, gpu_id=gpu_id)

    env_num = env_args["env_num"]
    max_step = env_args["max_step"]
    env_name = env_args["env_name"]
    state_dim = env_args["state_dim"]
    action_dim = env_args["action_dim"]
    if_discrete = env_args["if_discrete"]
    target_return = env_args["target_return"]

    assert isinstance(env_num, int)
    assert isinstance(env_name, str)
    assert isinstance(max_step, int)
    assert isinstance(state_dim, int) or isinstance(state_dim, tuple)
    assert isinstance(action_dim, int) or isinstance(action_dim, tuple)
    assert isinstance(if_discrete, int) or isinstance(if_discrete, bool)
    assert isinstance(target_return, int) or isinstance(target_return, float)

    get_action = None
    if gpu_id == -1 or env_num == 1:  # CPU or OneEnv
        if if_discrete:

            def get_action():
                return rd.randint(action_dim)

        else:

            def get_action():
                return rd.rand(action_dim) * 2 - 1

    if gpu_id >= 0 and env_num > 1:  # GPU
        device = torch.device(f"cuda:{gpu_id}")
        if if_discrete:

            def get_action():
                return torch.randint(high=action_dim, size=(env_num,), device=device)

        else:

            def get_action():
                _action = torch.rand(
                    size=(env_num, action_dim), dtype=torch.float32, device=device
                )
                return _action * 2 - 1

    dones = []
    state = env.reset()
    if len(state.shape) == 1:
        assert state.shape == (state_dim,)

        for i in range(max_step):
            action = get_action()

            state, reward, done, _ = env.step(action)
            assert state.shape == (state_dim,)
            dones.append(done)

        assert any(dones)

    else:
        assert state.shape == (env_num, state_dim)

        for i in range(max_step):
            action = (
                torch.randint(action_dim, size=(env_num,)).to(env.device)
                if if_discrete
                else torch.rand(env_num, action_dim).to(env.device) * 2 - 1
            )
            state, reward, done, _ = env.step(action)
            assert state.shape == (
                env_num,
                state_dim,
            )
            assert reward.shape == (env_num,)
            assert done.shape == (env_num,)

            dones.append(any(done))

        assert any(dones)

    print("| config.py check_env(): Finish. ")


"""private utils"""


def kwargs_filter(func, kwargs: dict):  # [ElegantRL.2021.12.12]
    """How does one ignore `unexpected keyword arguments passed to a function`?
    https://stackoverflow.com/a/67713935/9293137

    class ClassTest:
        def __init__(self, a, b=1):
            print(f'| ClassTest: a + b == {a + b}')

    old_kwargs = {'a': 1, 'b': 2, 'c': 3}
    new_kwargs = kwargs_filter(ClassTest.__init__, old_kwargs)
    assert new_kwargs == {'a': 1, 'b': 2}
    test = ClassTest(**new_kwargs)

    :param func: func(**kwargs)
    :param kwargs: the KeyWordArguments wait for
    :return: kwargs: [dict] filtered kwargs
    """

    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs
