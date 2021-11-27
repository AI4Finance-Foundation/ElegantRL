import os
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from copy import deepcopy

gym.logger.set_level(40)  # Block warning

"""net.py"""


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        a_prob = self.soft_max(self.net(state))
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state.size(0))
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state):
        return self.net(state)  # Advantage value


"""agent.py"""


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.state = None
        self.device = None
        self.action_dim = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

        '''init modify'''
        self.ClassCri = CriticAdv
        self.ClassAct = ActorPPO

        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.trajectory_list = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step):
        trajectory_temp = list()

        state = self.state
        last_done = 0
        for i in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))
            trajectory_temp.append((state, reward, done, action, noise))
            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        self.trajectory_list = trajectory_temp[last_done:]
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
        return obj_critic.item(), obj_actor.item(), a_std_log.mean().item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1.0 - tau))


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorDiscretePPO

    def explore_env(self, env, target_step):
        trajectory_temp = list()

        state = self.state
        last_done = 0
        for i in range(target_step):
            action, a_prob = self.select_action(state)  # different
            a_int = int(action)  # different
            next_state, reward, done, _ = env.step(a_int)  # different
            trajectory_temp.append((state, reward, done, a_int, a_prob))  # different

            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        self.trajectory_list \
            = trajectory_temp[last_done:]
        return trajectory_list


'''run.py'''


class Arguments:
    def __init__(self, agent=None, env=None):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.net_dim = 2 ** 9  # the network width
        self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 3  # collect target_step, then update network
        self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
        self.max_memo = self.target_step  # capacity of replay buffer
        self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.

        '''Arguments for evaluate'''
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
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae)

    '''init Evaluator'''
    eval_env = deepcopy(env)
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env, args.eval_times, args.eval_gap)

    '''init ReplayBuffer'''
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

    if_train = True
    while if_train:
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(trajectory_list)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


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
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
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
    env_name = env.unwrapped.spec.id if env_name is None else None

    if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        raise RuntimeError("| <class 'gym.spaces.discrete.Discrete'> does not support environment with discrete observation (state) space.")
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
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return


'''demo.py'''


def demo_continuous_action():
    args = Arguments()  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = '1'

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))  # env='Pendulum-v0' is OK.
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.target_step = args.env.max_step * 4
        args.if_per_or_gae = True
        args.gamma = 0.98

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.gamma = 0.98
        args.if_per_or_gae = True

    train_and_evaluate(args)


def demo_discrete_action():
    args = Arguments()  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    args.visible_gpu = '2'

    if_train_cart_pole = 1
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, PPO"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    train_and_evaluate(args)


if __name__ == '__main__':
    # demo_continuous_action()
    demo_discrete_action()
