import time

from elegantrl_helloworld.agent import *
from elegantrl_helloworld.env import *
from typing import Tuple


class Arguments:
    """
    Configuration map. Detailed explanation please refer to https://elegantrl.readthedocs.io/en/latest/api/config.html.

    :param agent[object]: the agent object in ElegantRL.
    :param env: an existed environment. (please pass None for now)
    :param env_func: the function for creating an env.
    :param env_args: the args for the env. Please take look at the demo.
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

        self.agent = agent  # DRL algorithm
        self.if_off_policy = (
            self.get_if_off_policy()
        )  # agent is on-policy or off-policy
        self.if_act_target = False  # use actor target network for stable training
        self.if_cri_target = True  # use critic target network for stable training
        self.if_use_old_traj = (
            False  # splice old and new data to get a complete trajectory in vector env
        )
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
            self.repeat_times = 2**4  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = (
            2**0
        )  # an approximate target reward usually be closed to 256
        self.learning_rate = 2**-15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2**-8  # 2 ** -8 ~= 5e-3

        self.worker_num = (
            2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        )
        self.thread_num = (
            8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        )
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        self.cwd = (
            None  # current work directory to save model. None means set automatically
        )
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.if_allow_break = (
            True  # allow break training when reach goal (early termination)
        )

        self.eval_gap = 2**7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2**2  # number of times that get episode return in first
        self.eval_times2 = 2**4  # number of times that get episode return in second

    def init_before_training(self):
        """
        Check parameters before training.
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        """auto set"""
        if self.cwd is None:
            self.cwd = (
                f"./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}"
            )

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

    def update_attr(self, attr: str):
        return getattr(self.env, attr) if self.env_args is None else self.env_args[attr]

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find("PPO") == -1, name.find("A2C") == -1))  # if_off_policy


def train_and_evaluate(args):
    """
    The training and evaluating loop.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    """
    args.init_before_training()
    gpu_id = args.learner_gpus

    """init"""
    env = build_env(args.env, args.env_func, args.env_args)

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)

    agent.state = env.reset()
    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.target_step)
        buffer.update_buffer((trajectory,))

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    target_step = args.target_step
    if_allow_break = args.if_allow_break
    del args

    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory = agent.explore_env(env, target_step)
            steps, r_exp = buffer.update_buffer((trajectory,))

        logging_tuple = agent.update_net(buffer)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(
                agent.act, steps, r_exp, logging_tuple
            )
            if_train = not (
                (if_allow_break and if_reach_goal)
                or evaluator.total_step > break_step
                or os.path.exists(f"{cwd}/stop")
            )
    print(f"| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}")
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None


def init_agent(args, gpu_id, env=None):
    """
    Initialize an ``Agent``.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param env: an object of environment.
    :return: an Agent.
    """
    agent = args.agent(
        args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args
    )
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        """init states"""
        if args.env_num == 1:
            states = [
                env.reset(),
            ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_evaluator(args, gpu_id):
    """
    Initialize an ``Evaluator``.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :return: an Evaluator.
    """
    eval_env = build_env(args.env, args.env_func, args.env_args)
    return Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)


def init_buffer(args, gpu_id):
    """
    Initialize an ``ReplayBuffer``.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :return: a ReplayBuffer.
    """
    if args.if_off_policy:
        buffer = ReplayBuffer(
            gpu_id=gpu_id,
            max_len=args.max_memo,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
        )
        buffer.save_or_load_history(args.cwd, if_save=False)

    else:
        buffer = ReplayBufferList()
    return buffer


"""evaluator"""


class Evaluator:
    """
    An ``evaluator`` evaluates agent's performance and saves models.

    :param cwd: directory path to save the model.
    :param agent_id: agent id.
    :param eval_env: environment object for model evaluation.
    :param args: an object of ``Arguments`` class.
    """

    def __init__(self, cwd, agent_id, eval_env, args):
        self.recorder = []  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f"{cwd}/recorder.npy"

        self.cwd = cwd
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = args.eval_gap
        self.eval_times = args.eval_times
        self.target_return = args.target_return

        self.r_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(
            f"{'#' * 80}\n"
            f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
            f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
            f"{'expR':>8}{'objC':>7}{'etc.':>7}"
        )

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> Tuple[bool, bool]:
        """
        Evaluate and save the model.

        :param act: Actor (policy) network.
        :param steps: training steps for last update.
        :param r_exp: mean reward.
        :param log_tuple: log information.
        :return: a boolean for whether terminates the training process and a boolean for whether save the model.
        """
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
        else:
            self.eval_time = time.time()

            """evaluate first time"""
            rewards_steps_list = [
                get_episode_return_and_step(self.eval_env, act)
                for _ in range(self.eval_times)
            ]
            rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
            r_avg, s_avg = rewards_steps_ary.mean(
                axis=0
            )  # average of episode return and episode step
            r_std, s_std = rewards_steps_ary.std(
                axis=0
            )  # standard dev. of episode return and episode step

            """save the policy network"""
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_path = f"{self.cwd}/actor.pth"
                torch.save(act.state_dict(), act_path)  # save policy network in *.pth

                print(
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                )  # save policy and print

            self.recorder.append(
                (self.total_step, r_avg, r_std, r_exp, *log_tuple)
            )  # update recorder

            """print some information to Terminal"""
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(
                    f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                    f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                    f"{'UsedTime':>8}  ########\n"
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                    f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                    f"{self.used_time:>8}  ########"
                )

            print(
                f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}"
            )

            if hasattr(self.eval_env, "curriculum_learning_for_evaluator"):
                self.eval_env.curriculum_learning_for_evaluator(r_avg)
        return if_reach_goal


def get_episode_return_and_step(env, act) -> Tuple[float, int]:
    """
    Evaluate the actor (policy) network on testing environment.

    :param env: environment object in ElegantRL.
    :param act: Actor (policy) network.
    :return: episodic reward and number of steps needed.
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    episode_step = None
    episode_return = 0.0  # sum of rewards in an episode
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = (
            a_tensor.detach().cpu().numpy()[0]
        )  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, "episode_return", episode_return)
    episode_step += 1
    return episode_return, episode_step
