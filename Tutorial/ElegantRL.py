import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd

"""AgentNet"""


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )

    def forward(self, s):  # state
        return self.net(s).tanh()

    def get_noisy_action(self, s, a_std):  # state, action_std
        action = self.net(s).tanh()
        return (action + torch.randn_like(action) * a_std).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), )  # network of state
        self.net__a = nn.Linear(mid_dim, action_dim)  # network of action_average
        self.net__d = nn.Linear(mid_dim, action_dim)  # network of action_log_std

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # constant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s):
        x = self.net__s(s)
        return self.net__a(x).tanh()

    def get__noise_action(self, s):
        x = self.net__s(s)
        a_avg = self.net__a(x)  # action_average
        a_std_log = self.net__d(x).clamp(-16, 2)  # action_log_std
        return torch.normal(a_avg, a_std_log.exp()).tanh()

    def get__a__log_prob(self, s):
        x = self.net__s(s)
        a_avg = self.net__a(x)
        a_std_log = self.net__d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        a = a_avg + a_std * torch.randn_like(a_avg, requires_grad=True, device=self.device)  # todo ?
        a_tanh = a.tanh()

        log_prob = (((a_avg - a) / a_std).pow(2) * 0.5 + a_std_log + self.sqrt_2pi_log
                    + (-a_tanh.pow(2) + 1.000001).log().sum(1, keepdim=True))
        return a_tanh, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), )
        self.net_q1 = nn.Linear(mid_dim, 1)
        self.net_q2 = nn.Linear(mid_dim, 1)

    def forward(self, state, action):
        x = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(x), self.net_q2(x)


"""AgentZoo"""


class AgentBaseAC:
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        self.actor_obj, self.critic_obj = 0.0, 0.5
        self.explore_noise, self.policy__noise = 0.1, 0.2

        self.state = self.action = None
        self.reward_sum = 0.0
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.SmoothL1Loss()

        self.act = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.act_target = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = Critic(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.cri_target = Critic(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

    def select_actions(self, states):  # states = (state, )
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get_noisy_action(states, self.explore_noise)
        return actions.detach().cpu().numpy()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_state, reward, done, _ = env.step(action)
            buffer.append_memo((reward * reward_scale, 0.0 if done else gamma, *self.state, *action, *next_state))
            self.state = env.reset() if done else next_state

    def update_buffer__pipe(self, pipes, buffer, max_step):
        env_num = len(pipes)
        env_num2 = env_num // 2
        for _ in range(max_step // env_num):
            for i in range(env_num2):
                reward_mask, next_state = pipes[i].recv()
                buffer.append_memo((*reward_mask, *self.state[i], *self.action[i], *next_state))
                self.state[i] = next_state
            self.action[:env_num2] = self.select_actions(self.state[:env_num2])
            for i in range(env_num2):
                pipes[i].send(self.action[i])

            for i in range(env_num2, env_num):
                reward_mask, next_state = pipes[i].recv()
                buffer.append_memo((*reward_mask, *self.state[i], *self.action[i], *next_state))
                self.state[i] = next_state
            self.action[env_num2:] = self.select_actions(self.state[env_num2:])
            for i in range(env_num2, env_num):
                pipes[i].send(self.action[i])

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k * repeat_times)

        for i in range(update_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_, )

                next_action = self.act_target.get_noisy_action(next_s, self.policy__noise)
                q_label = reward + mask * self.cri_target(next_s, next_action)

            q1, q2 = self.cri(state, action)
            critic_obj = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.critic_obj = 0.995 * self.critic_obj + 0.005 * critic_obj.item() / 2
            lamb = np.exp(-self.critic_obj ** 2)
            self.cri_optimizer.zero_grad()  # todo try mega
            critic_obj.backward()
            self.cri_optimizer.step()

            actor_obj = -self.cri(state, self.act(state)).mean() * lamb
            self.actor_obj = 0.995 * self.actor_obj + 0.005 * q_label.mean().item()
            self.act_optimizer.zero_grad()
            actor_obj.backward()
            self.act_optimizer.step()

            _soft_target_update(self.cri_target, self.cri)
            _soft_target_update(self.act_target, self.act)

    def save_or_load_model(self, cwd, if_save):
        for net, name in ((self.act, 'act'), (self.cri, 'cri')):
            if name not in dir(self):
                continue
            save_path = f'{cwd}/{name}.pth'
            if if_save:
                torch.save(net.state_dict(), save_path)
                # print("Saved act and cri:", cwd)
            elif os.path.exists(save_path):
                net = torch.load(save_path, map_location=lambda storage, loc: storage)
                net.load_state_dict(net)
                print("Loaded act and cri:", cwd)
            else:
                print(f"FileNotFound when load_model: {cwd}")


class AgentModSAC(AgentBaseAC):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        super(AgentBaseAC, self).__init__()
        self.obj_a, self.critic_obj = 0.0, (-np.log(0.5)) ** 0.5
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,),
                                      dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=learning_rate)

        self.state = self.action = None
        self.reward_sum = 0.0
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.SmoothL1Loss()

        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.act_target = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = Critic(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.cri_target = Critic(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        alpha = self.alpha_log.exp().detach()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(max_step * k * repeat_times)

        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                q_label = reward + mask * (torch.min(*self.cri_target(next_s, next_a_noise)) + next_log_prob * alpha)

            """critic_obj"""
            q1, q2 = self.cri(state, action)
            critic_obj = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.critic_obj = 0.995 * self.critic_obj + 0.0025 * critic_obj.item()

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()
            _soft_target_update(self.cri_target, self.cri)

            a_noise_pg, log_prob = self.act.get__a__log_prob(state)  # policy gradient

            alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            lamb = np.exp(-self.critic_obj ** 2)
            if update_a / update_c < 1 / (2 - lamb):  # auto TTUR
                update_a += 1

                actor_obj = -(torch.min(*self.cri(state, a_noise_pg)) + log_prob * alpha).mean()
                self.actor_obj = 0.995 * self.actor_obj + 0.005 * q_label.mean().item()

                self.act_optimizer.zero_grad()
                actor_obj.backward()
                self.act_optimizer.step()
                _soft_target_update(self.act_target, self.act)


def _soft_target_update(target, current, tau=5e-3):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


"""AgentRun"""


class Arguments:
    def __init__(self, rl_agent=None, env=None, gpu_id=None):
        self.rl_agent = rl_agent
        self.gpu_id = gpu_id
        self.cwd = None  # init cwd in def init_for_training()
        self.env = env

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)

        '''Arguments for evaluate'''
        self.break_step = 2 ** 17  # break training after 'total_step > break_step'
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.if_remove_history = True  # remove the cwd folder? (True, False, None:ask me)
        self.show_gap = 2 ** 8  # show the Reward and Loss of actor and critic per show_gap seconds
        self.eval_times1 = 2 ** 3  # evaluation times if 'eval_reward > old_max_reward'
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > target_reward'
        self.random_seed = 1943  # Github: YonV 1943

    def init_for_training(self, cpu_threads=6):
        if not hasattr(self.env, 'env_name'):
            raise RuntimeError('| What is env.env_name? use env = build_env(env) to decorate env')

        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.rl_agent.__name__}/{self.env.env_name}_{self.gpu_id}'
        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
        _whether_remove_history(self.cwd, self.if_remove_history)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(cpu_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def _whether_remove_history(cwd, is_remove=None):
    import shutil
    if is_remove is None:
        is_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(cwd)) == 'y')
    if is_remove:
        shutil.rmtree(cwd, ignore_errors=True)
        print("| Remove")
    os.makedirs(cwd, exist_ok=True)
    del shutil


class FinanceMultiStockEnv:  # 2021-01-01
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, initial_account=1e6, transaction_fee_percent=1e-3, max_stock=100):
        self.stock_dim = 30
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        self.ary = self.load_training_data_for_multi_stock()
        assert self.ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21

        '''env information'''
        self.env_name = 'FinanceStock-v1'
        self.state_dim = 1 + (5 + 1) * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_reward = 15
        self.max_step = self.ary.shape[0]

        self.gamma_r = 0.0

    def reset(self):
        self.account = self.initial_account * rd.uniform(0.99, 1.00)  # notice reset()
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        return state

    def step(self, actions):
        actions = actions * self.max_stock

        """bug or sell stock"""
        for index in range(self.stock_dim):
            action = actions[index]
            adj = self.day_npy[index]
            if action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_r = self.gamma_r * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_r
            self.gamma_r = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_account

        return state, reward, done, dict()

    @staticmethod
    def load_training_data_for_multi_stock(if_load=True):  # need more independent
        npy_path = './Result/FinanceMultiStock.npy'
        if if_load and os.path.exists(npy_path):
            data_ary = np.load(npy_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'\n| FinanceMultiStockEnv(): Can you download and put it into: {npy_path}'
                f'\n| https://github.com/Yonv1943/ElegantRL/blob/master/Result/FinanceMultiStock.npy'
                f'\n| Or you can use the following code to generate it from a csv file.'
            )


def train_agent_mp(args):  # 2021-01-01
    act_workers = args.rollout_num

    import multiprocessing as mp
    eva_pipe1, eva_pipe2 = mp.Pipe(duplex=True)
    process = list()

    exp_pipe2s = list()
    for i in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        exp_pipe2s.append(exp_pipe1)
        process.append(mp.Process(target=mp_explore_in_env, args=(args, exp_pipe2, i)))
    process.extend([
        mp.Process(target=mp_evaluate_agent, args=(args, eva_pipe1)),
        mp.Process(target=mp__update_params, args=(args, eva_pipe2, exp_pipe2s)),
    ])

    [p.start() for p in process]
    process[-1].join()
    process[-2].join()
    [p.terminate() for p in process]
    print('\n')


def mp__update_params(args, eva_pipe, pipes):  # 2020-12-22
    rl_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env = args.env
    reward_scale = args.reward_scale
    if_stop = args.if_break_early
    gamma = args.gamma
    del args

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''build agent and act_cpu'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = [pipe.recv() for pipe in pipes]
    agent.action = agent.select_actions(agent.state)
    for i in range(len(pipes)):  # todo Error: PPO return a_noise, noise
        pipes[i].send(agent.action[i])  # pipes 1 send

    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''build replay buffer, init: total_step, reward_avg'''
    reward_avg = None
    total_step = 0
    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentInterPPO'}):
        buffer = BufferArrayGPU(max_memo + max_step, state_dim, action_dim, if_ppo=True)
    else:
        buffer = BufferArrayGPU(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)

        '''initial exploration'''
        with torch.no_grad():  # update replay buffer
            rewards, steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        reward_avg = np.average(rewards)
        step_sum = sum(steps)

        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        if 'act_target' in dir(agent):
            agent.act_target.load_state_dict(agent.act.state_dict())

        total_step += step_sum
    if reward_avg is None:
        reward_avg = get_episode_return(env, agent.act, max_step, agent.device, if_discrete)

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer__pipe(pipes, buffer, max_step)  # pipes n send

        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update__now_len__before_sample()
        loss_a_avg, loss_c_avg = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        act_cpu.load_state_dict(agent.act.state_dict())
        eva_pipe.send((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # eva_pipe act

        if eva_pipe.poll():
            if_solve = eva_pipe.recv()  # eva_pipe if_solve

        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    eva_pipe.send('stop')  # eva_pipe stop  # send to mp_evaluate_agent
    time.sleep(4)
    # print('; quit: params')


def mp_explore_in_env(args, pipe, _act_id):
    env = args.env
    reward_scale = args.reward_scale
    gamma = args.gamma
    del args

    next_state = env.reset()
    pipe.send(next_state)
    while True:
        action = pipe.recv()
        next_state, reward, done, _ = env.step(action)  # pipes 1 recv, pipes n recv

        reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
        if done:
            next_state = env.reset()

        pipe.send((reward_mask, next_state))


def mp_evaluate_agent(args, eva_pipe):  # 2020-12-12
    env = args.env
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_memo
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    '''init: env'''
    if_discrete = env.if_discrete
    target_reward = env.target_reward

    '''build evaluated only actor'''
    act = eva_pipe.recv()  # eva_pipe 1, act == act.to(device_cpu), requires_grad=False
    obj_a, obj_c = 0., 0.

    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update_recorder(env, act, max_step, device, if_discrete, obj_a, obj_c)

    if_train = True
    with torch.no_grad():  # for saving the GPU buffer
        while if_train:
            is_saved = recorder.update_recorder(env, act, max_step, device, if_discrete, obj_a, obj_c)
            recorder.save_act(cwd, act, gpu_id) if is_saved else None

            is_solved = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)
            eva_pipe.send(is_solved)  # eva_pipe is_solved

            '''update actor'''
            while not eva_pipe.poll():  # wait until eva_pipe not empty
                time.sleep(1)
            while eva_pipe.poll():  # receive the latest object from pipe
                q_i_eva_get = eva_pipe.recv()  # eva_pipe act
                if q_i_eva_get == 'stop':
                    if_train = False
                    break  # it should break 'while q_i_eva.qsize():' and 'while if_train:'
                act, obj_a, obj_c = q_i_eva_get

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    else:
        print(f'| SavedDir: {new_cwd}    WARNING: file exit')
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - recorder.start_time:.0f}')

    while eva_pipe.poll():  # empty the pipe
        eva_pipe.recv()
    # print('; quit: evaluate')


def explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim):
    state = env.reset()

    rewards = list()
    episode_return = 0.0  # 2020-12-12 episode_return
    steps = list()
    step = 0

    if if_discrete:
        def random_action__discrete():
            return rd.randint(action_dim)

        get_random_action = random_action__discrete
    else:
        def random_action__continuous():
            return rd.uniform(-1, 1, size=action_dim)

        get_random_action = random_action__continuous

    global_step = 0
    while global_step < max_step or len(rewards) == 0:  # warning 2020-10-10?
        # action = np.tanh(rd.normal(0, 0.25, size=action_dim))  # zero-mean gauss exploration
        action = get_random_action()
        next_state, reward, done, _ = env.step(action * if_discrete)
        episode_return += reward
        step += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        buffer.append_memo((adjust_reward, mask, state, action, next_state))

        state = next_state
        if done:
            episode_return = env.episode_return if hasattr(env, 'episode_return') else episode_return

            rewards.append(episode_return)
            steps.append(step)
            global_step += step

            state = env.reset()  # reset the environment
            episode_return = 0.0
            step = 1

    buffer.update__now_len__before_sample()
    return rewards, steps


class Recorder:
    def __init__(self, eval_size1=3, eval_size2=9):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.is_solved = False
        self.total_step = 0

        '''constant'''
        self.eva_size1 = eval_size1
        self.eva_size2 = eval_size2

        '''print_reward'''
        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()

        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def update_recorder(self, env, act, max_step, device, if_discrete, obj_a, obj_c):
        if self.total_step == self.recorder[-1][0]:
            return None  # plan to be more elegant

        is_saved = False
        reward_list = [get_episode_return(env, act, max_step, device, if_discrete)
                       for _ in range(self.eva_size1)]

        r_avg = np.average(reward_list)
        if r_avg > self.r_max:  # check 1
            reward_list.extend([get_episode_return(env, act, max_step, device, if_discrete)
                                for _ in range(self.eva_size2 - self.eva_size1)])
            r_avg = np.average(reward_list)
            if r_avg > self.r_max:  # check final
                self.r_max = r_avg
                is_saved = True

        r_std = float(np.std(reward_list))
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        return is_saved

    def save_act(self, cwd, act, agent_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")

    def check__if_solved(self, target_reward, agent_id, show_gap, cwd):
        total_step, r_avg, r_std, obj_a, obj_c = self.recorder[-1]
        if self.r_max > target_reward:
            self.is_solved = True
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                      f"{agent_id:<2}  {total_step:8.2e}  {target_reward:8.2f} |"
                      f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")
        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()
            print(f"{agent_id:<2}  {total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
            self.save_npy__draw_plot(cwd)
        return self.is_solved

    def save_npy__draw_plot(self, cwd):  # 2020-12-12
        if len(self.recorder) == 0:
            print(f"| save_npy__draw_plot() WARNING: len(self.recorder) == {len(self.recorder)}")
            return None
        np.save(f'{cwd}/recorder.npy', self.recorder)

        train_time = time.time() - self.start_time
        max_reward = self.r_max

        recorder = np.array(self.recorder[1:], dtype=np.float32)  # 2020-12-12 Compatibility
        if len(recorder) == 0:  # not elegant
            return None

        train_time = int(train_time)
        total_step = int(recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{max_reward:.3f}"
        save_path = f"{cwd}/{save_title}.jpg"

        import matplotlib as mpl  # draw figure in Terminal
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2)
        plt.title(save_title, y=2.3)

        steps = recorder[:, 0]
        r_avg, r_std = recorder[:, 1], recorder[:, 2]
        obj_a, obj_c = recorder[:, 3], recorder[:, 4]

        r_color = 'lightcoral'  # episode return (learning curve)
        axs[0].plot(steps, r_avg, label='EvaR', color=r_color)
        axs[0].fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=r_color, alpha=0.3, )

        axs[1].plot(steps, obj_a, label='objA', color=r_color)
        axs[1].tick_params(axis='y', labelcolor=r_color)
        axs[1].set_ylabel('objA', color=r_color)

        l_color = 'royalblue'  # loss function of critic (objective of critic)
        axs_1_twin = axs[1].twinx()
        axs_1_twin.fill_between(steps, obj_c, facecolor=l_color, alpha=0.2, )
        axs_1_twin.tick_params(axis='y', labelcolor=l_color)
        axs_1_twin.set_ylabel('objC', color=l_color)

        prev_save_names = [name for name in os.listdir(cwd) if name[:9] == save_title[:9]]  # remove previous plot
        os.remove(f'{cwd}/{prev_save_names[0]}') if len(prev_save_names) > 0 else None

        plt.savefig(save_path)
        plt.close()


def get_episode_return(env, act, max_step, device, if_discrete) -> float:  # 2020-12-21
    if hasattr(env, 'episode_return'):  # Compatibility for ElegantRL 2020-12-21
        return env.episode_return
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step if hasattr(env, 'max_step') else max_step
    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.tensor((state,), dtype=torch.float32, device=device)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.cpu().data.numpy()[0]
        state, reward, done, _ = env.step(action)[1]
        episode_return += reward
        if done:
            break
    return episode_return


class BufferArrayGPU:  # 2021-01-01
    def __init__(self, memo_max_len, state_dim, action_dim, if_ppo=False):
        state_dim = state_dim if isinstance(state_dim, int) else np.prod(state_dim)  # pixel-level state

        if if_ppo:  # for Offline PPO
            memo_dim = 1 + 1 + state_dim + action_dim + action_dim
        else:
            memo_dim = 1 + 1 + state_dim + action_dim + state_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memories = torch.empty((memo_max_len, memo_dim), dtype=torch.float32, device=self.device)

        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim=1, done_dim=1
        self.action_idx = self.state_idx + action_dim

    def append_memo(self, memo_tuple):  # memo_tuple = (reward, mask, state, action, next_state)
        self.memories[self.next_idx] = torch.as_tensor(np.hstack(memo_tuple), device=self.device)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.is_full = False
        self.now_len = 0

    def random_sample(self, batch_size):  # _device should remove
        indices = rd.randint(self.now_len, size=batch_size)
        memory = self.memories[indices]

        return (memory[:, 0:1],  # rewards
                memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
                memory[:, 2:self.state_idx],  # state
                memory[:, self.state_idx:self.action_idx],  # actions
                memory[:, self.action_idx:])  # next_states or actions_noise

    def all_sample(self):  # 2020-11-11 fix bug for ModPPO
        return (self.memories[:self.now_len, 0:1],  # rewards
                self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
                self.memories[:self.now_len, 2:self.state_idx],  # state
                self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
                self.memories[:self.now_len, self.action_idx:])  # next_states or log_prob_sum


def train__demo():
    env = FinanceMultiStockEnv()  # 2020-12-24

    args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
    args.break_step = 2 ** 12  # todo just for test
    args.show_gap = 2 ** 1  # todo just for test
    args.eval_times1 = 1
    args.eval_times2 = 2

    # args.break_step = 2 ** 22  # UsedTime:
    args.net_dim = 2 ** 7
    args.max_memo = 2 ** 18
    args.batch_size = 2 ** 8
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    from AgentZoo import AgentPPO
    args = Arguments(rl_agent=AgentPPO, env=env)
    args.eval_times1 = 1
    args.eval_times2 = 1
    args.rollout_num = 4
    args.if_break_early = True

    args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = 1699 * 16
    args.batch_size = 2 ** 10
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


if __name__ == '__main__':
    train__demo()
