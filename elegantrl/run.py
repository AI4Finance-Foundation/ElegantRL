import gym
import time
import multiprocessing as mp

from elegantrl.agent import *

'''[ElegantRL.2022.01.01](github.com/AI4Fiance-Foundation/ElegantRL)'''


class Arguments:
    def __init__(self, agent, env=None, env_func=None, env_args=None):
        self.env = env  # the environment for training
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.update_attr('env_num')  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.update_attr('max_step')  # the env name. Be used to set 'cwd'.
        self.env_name = self.update_attr('env_name')  # the max step of an episode
        self.state_dim = self.update_attr('state_dim')  # vector dimension (feature number) of state
        self.action_dim = self.update_attr('action_dim')  # vector dimension (feature number) of action
        self.if_discrete = self.update_attr('if_discrete')  # discrete or continuous action space
        self.target_return = self.update_attr('target_return')  # target average episode return

        self.agent = agent  # DRL algorithm
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        self.if_act_target = False  # use actor target network for stable training
        self.if_cri_target = True  # use critic target network for stable training
        self.if_use_old_traj = False  # splice old and new data to get a complete trajectory in vector env
        if self.if_off_policy:  # off-policy
            self.net_dim = 2 ** 8  # the network width
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 0  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER (Prioritized Experience Replay) for sparse reward
        else:  # on-policy
            self.net_dim = 2 ** 9  # the network width
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current work directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set'''
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
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
        return all((name.find('PPO') == -1, name.find('A2C') == -1))  # if_off_policy


'''train single process'''


def train_and_evaluate(args):
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env, args.env_func, args.env_args)

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)

    agent.state = env.reset()
    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.target_step)
        buffer.update_buffer((trajectory, ))

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    target_step = args.target_step
    if_allow_break = args.if_allow_break
    del args

    if_train = True
    while if_train:
        with torch.no_grad():  # todo
            trajectory = agent.explore_env(env, target_step)
            steps, r_exp = buffer.update_buffer((trajectory, ))

        logging_tuple = agent.update_net(buffer)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None


def init_agent(args, gpu_id, env=None):
    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''init states'''
        if args.env_num == 1:
            states = [env.reset(), ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_evaluator(args, gpu_id):
    eval_env = build_env(args.env, args.env_func, args.env_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)
    return evaluator


def init_buffer(args, gpu_id):
    if args.if_off_policy:
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_len=args.max_memo,
                              state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim, )
        buffer.save_or_load_history(args.cwd, if_save=False)

    else:
        buffer = ReplayBufferList()
    return buffer


'''train multiple process'''


def train_and_evaluate_mp(args):
    args.init_before_training()

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorker(args.worker_num)
    process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.worker_num)])

    learner_pipe = PipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    [p.start() for p in process]
    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, worker_num):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args, worker_id):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        '''loop'''
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args, comm_eva, comm_exp):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        '''loop'''
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent)
            steps, r_exp = buffer.update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if_train = comm_eva.evaluate_and_save_mp(agent, steps, r_exp, logging_tuple)
        agent.save_or_load_agent(args.cwd, if_save=True)
        print(f'| Learner: Save in {args.cwd}')

        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {args.cwd}")
            buffer.save_or_load_history(args.cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, agent, steps, r_exp, logging_tuple):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train = self.pipe1.recv()
            act_cpu_dict = deepcopy(agent.act.state_dict())
        else:
            if_train = True
            act_cpu_dict = None

        self.pipe1.send((act_cpu_dict, steps, r_exp, logging_tuple))
        return if_train

    def run(self, args):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        '''loop'''
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_train = True
        if_reach_goal = False
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal = evaluator.evaluate_and_save(act, steps, r_exp, logging_tuple)
            else:
                evaluator.total_step += steps

            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            self.pipe0.send(if_train)

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)


'''evaluator'''


class Evaluator:  # [ElegantRL.2022.01.01]
    def __init__(self, cwd, agent_id, eval_env, args):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'

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
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
        else:
            self.eval_time = time.time()

            '''evaluate first time'''
            rewards_steps_list = [get_episode_return_and_step(self.eval_env, act)
                                  for _ in range(self.eval_times)]
            rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
            r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
            r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step

            '''save the policy network'''
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_path = f"{self.cwd}/actor.pth"
                torch.save(act.state_dict(), act_path)  # save policy network in *.pth

                print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                      f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                      f"{'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                      f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8}  ########")

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")

            if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
                self.eval_env.curriculum_learning_for_evaluator(r_avg)
        return if_reach_goal


def get_episode_return_and_step(env, act) -> (float, int):  # [ElegantRL.2022.01.01]
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
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    episode_step += 1
    return episode_return, episode_step


'''env'''


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """get a dict `env_args` about a standard OpenAI gym env information.

    env_args = {
        'env_num': 1,
        'env_name': env_name,            # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,            # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,          # [int] the dimension of state
        'action_dim': action_dim,        # [int] the dimension of action
        'if_discrete': if_discrete,      # [bool] action space is discrete or continuous
        'target_return': target_return,  # [float] We train agent to reach this target episode return.
    }

    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env inforamtion.
    :return: env_args [dict]
    """

    env_num = getattr(env, 'env_num') if hasattr(env, 'env_num') else 1

    if isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        target_return = getattr(env, 'target_return', None)
        target_return_default = getattr(env.spec, 'reward_threshold', None)
        if target_return is None:
            target_return = target_return_default
        if target_return is None:
            target_return = 2 ** 16

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            assert not any(env.action_space.high - 1)
            assert not any(env.action_space.low + 1)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        target_return = env.target_return

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete,
                'target_return': target_return, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n   ")
        env_args_repr = env_args_repr.replace('{', "{\n    ")
        env_args_repr = env_args_repr.replace('}', ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args


def kwargs_filter(func, kwargs: dict):  # [ElegantRL.2021.12.12]
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env=None, env_func=None, env_args=None):  # [ElegantRL.2021.12.12]
    if env is not None:
        env = deepcopy(env)
    elif env_func.__module__ == 'gym.envs.registration':
        import gym
        gym.logger.set_level(40)  # Block warning
        env = env_func(id=env_args['id'])
    else:
        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    env.max_step = env.max_step if hasattr(env, 'max_step') else env_args['max_step']
    env.if_discrete = env.if_discrete if hasattr(env, 'if_discrete') else env_args['if_discrete']
    return env
