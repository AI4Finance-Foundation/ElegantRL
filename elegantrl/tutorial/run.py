import gym
import time
from elegantrl.tutorial.agent import *

gym.logger.set_level(40)  # Block warning


class Arguments:
    def __init__(self, agent=None, env=None, if_on_policy=False):
        self.agent = agent  # DRL algorithm
        self.env = env  # env for training

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None)
        self.break_step = 2 ** 20  # terminate training after 'total_step > break_step'
        self.if_allow_break = True  # terminate training when reaching a target reward

        self.visible_gpu = '0'  # e.g., os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # #rollout workers per GPU
        self.num_threads = 8  # cpu_num to evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 20  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times = 2  # number of times that get episode return in first
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}_{self.env.env_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=True)

    '''init: Agent'''
    env = args.env
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate, args.if_per_or_gae)
    agent.save_or_load_agent(args.cwd, if_save=False)

    '''init Evaluator'''
    eval_env = deepcopy(env) if args.eval_env is None else args.eval_env
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_times, args.eval_gap)

    '''init ReplayBuffer'''
    if agent.if_on_policy:
        buffer = list()

        def update_buffer(_trajectory):
            _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
            ten_state = torch.as_tensor(_trajectory[0])
            ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
            ten_action = torch.as_tensor(_trajectory[3])
            ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

            buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

            _steps = ten_reward.shape[0]
            _r_exp = ten_reward.mean()
            return _steps, _r_exp
    else:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim)
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(_trajectory):
            ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
            ary_other = torch.as_tensor([item[1] for item in _trajectory])
            ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ten_reward
            ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

            buffer.extend_buffer(ten_state, ary_other)

            _steps = ten_state.shape[0]
            _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
            return _steps, _r_exp

    '''start training'''
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    agent.state = env.reset()
    if not agent.if_on_policy:
        trajectory = agent.explore_env(env, target_step)
        update_buffer(trajectory)

    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(trajectory)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not agent.if_on_policy else None


class Evaluator:
    def __init__(self, cwd, agent_id, device, env, eval_times, eval_gap, ):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        self.r_max = -np.inf
        self.total_step = 0

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times = eval_times
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = 0
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            return False  # if_reach_goal

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                              range(self.eval_times)]
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

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
        return if_reach_goal

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        state, reward, done, info_dict = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info_dict


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else env_name

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env.spec, 'reward_threshold', 2 ** 16)

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # for discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # for continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return
