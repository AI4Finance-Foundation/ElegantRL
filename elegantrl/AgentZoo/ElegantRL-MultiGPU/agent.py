import os
from copy import deepcopy  # deepcopy target_network

import torch
import numpy as np
import numpy.random as rd
from elegantrl.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from elegantrl.net import Actor, ActorSAC, ActorPPO
from elegantrl.net import Critic, CriticAdv, CriticTwin
from elegantrl.net import InterDPG, InterSPG, InterPPO

"""ElegantRL (Pytorch 3 files model-free DRL Library)
GitHub.com: YonV1943, Zhihu.com: 曾伊言

reference:
TD3 https://github.com/sfujim/TD3 good++
TD3 https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2 good
PPO https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py good+
PPO https://github.com/xtma/pytorch_car_caring good
PPO https://github.com/openai/baselines/tree/master/baselines/ppo2 normal-
SAC https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac normal -
DUEL https://github.com/gouxiangchen/dueling-DQN-pytorch good

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
In my opinion, before 2020, the policy gradient algorithm agent didn't learn s policy.
Actually, they "learn game feel" or "get a soft touch". In Chinese "shǒu gǎn 手感". 
Learn more about policy gradient algorithms in:

Policy Gradient Algorithm summary
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
How to choose DRL algorithms? MuZero/SAC/PPO/TD3/DDPG/DQN etc.
如何选择深度强化学习算法？MuZero/SAC/PPO/TD3/DDPG/DQN/等 
https://zhuanlan.zhihu.com/p/342919579
DRL fine-tuning skills(not trick): take D3QN, TD3, PPO and SAC as examples.
深度强化学习调参技巧：以D3QN、TD3、PPO、SAC算法为例
https://zhuanlan.zhihu.com/p/345353294
"""


class AgentBase:
    def __init__(self):
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None

    def init(self, net_dim, state_dim, action_dim):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :int net_dim: the dimension of networks (the width of neural networks)
        :int state_dim: the dimension of state (the number of state vector)
        :int action_dim: the dimension of action (the number of discrete action)
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def select_actions(self, states) -> np.ndarray:
        """Select actions for exploration

        :array states: (state, ) or (state, state, ...) or state.shape==(n, *state_dim)
        :return array action: action.shape==(-1, action_dim), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device).detach_()
        actions = self.act(states)
        return actions.cpu().numpy()  # -1 < action < +1

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        :buffer: Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        :int batch_size: sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


'''Value-based Methods (DQN variances)'''


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
        self.action_dim = None  # chose discrete action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri  # to keep the same from Actor-Critic framework

        self.criterion = torch.torch.nn.MSELoss()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).cpu().numpy()
        return a_int

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        next_q = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)  # next_state
                next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            q_eval = self.cri(state).gather(1, action.type(torch.long))
            obj_critic = self.criterion(q_eval, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return next_q.mean().item(), obj_critic.item()


class AgentDuelingDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim):
        """Contribution of Dueling DQN

        Advantage function --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetDuel(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.MSELoss()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.softmax = torch.nn.Softmax(dim=1)

    def init(self, net_dim, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = self.softmax(actions)[0]
            a_prob = action.detach().cpu().numpy()  # choose action according to Q value
            a_int = rd.choice(self.action_dim, p=a_prob)
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).cpu().numpy()
        return a_int

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """Contribution of DDQN (Double DQN)

        Twin Q-Network. Use min(q1, q2) to reduce over-estimation.
        """
        buffer.update_now_len_before_sample()

        next_q = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s))
                next_q = next_q.max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            act_int = action.type(torch.long)
            q1, q2 = [qs.gather(1, act_int) for qs in self.act.get_q1_q2(state)]
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return next_q.mean().item(), obj_critic.item() / 2


class AgentD3QN(AgentDoubleDQN):  # D3QN: Dueling Double DQN
    def __init__(self):
        super().__init__()

    def init(self, net_dim, state_dim, action_dim):
        """Contribution of D3QN (Dueling Double DQN)

        There are not contribution of D3QN.
        Obviously, DoubleDQN is compatible with DuelingDQN.
        Any beginner can come up with this idea (D3QN) independently.
        """
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetTwinDuel(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)


'''Actor-Critic Methods (Policy Gradient)'''


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ou_explore_noise = 0.3  # explore noise of action
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim):
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.ou_explore_noise)
        # I don't recommend to use OU-Noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.MSELoss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0].cpu().numpy()
        return (action + self.ou_noise()).ratio_clip(-1, 1)

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = self.cri_target(next_s, self.act_target(next_s))
                q_label = reward + mask * next_q
            q_value = self.cri(state, action)
            obj_critic = self.criterion(q_value, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_actor.item(), obj_critic.item()


class AgentTD3(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency, for soft target update

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.MSELoss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.cpu().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """Contribution of TD3 (Twin Delay DDPG)

        1. twin critics (DoubleDQN -> TwinCritic, good idea)
        2. policy noise ('Deterministic Policy Gradient + policy noise' looks like Stochastic PG)
        3. delay update (I think it is not very useful)
        """
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None
        for i in range(int(target_step * repeat_times)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
                q_label = reward + mask * next_q
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of actor'''
            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return obj_actor.item(), obj_critic.item() / 2


class AgentInterAC(AgentBase):  # use InterSAC instead of InterAC .Warning: sth. wrong with this code, need to check
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.2  # standard deviation of explore noise
        self.policy_noise = 0.4  # standard deviation of policy noise
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update
        self.avg_loss_c = (-np.log(0.5)) ** 0.5  # old version reliable_lambda
        self.optimizer = None

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = InterDPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """Contribution of InterAC (Integrated network for deterministic policy gradient)

        1. First try integrated network to share parameter between two **different input** network.
        1. First try Encoder-DenseNetLikeNet-Decoder network architecture.
        1. First try Reliable Lambda in bi-level optimization problems. (such as Policy Gradient and GANs)
        2. Try TTUR in RL. TTUR (Two-Time-Scale Update Rule) is useful in bi-level optimization problems.
        2. Try actor_term to stabilize training in parameter-sharing network. (different learning rate is more useful)
        3. Try Spectral Normalization and found it conflict with soft target update.
        3. Try increasing batch_size and update_times
        3. Dropout layer is useless in RL.
        -1. InterAC is a semi-finished algorithms. InterSAC is a finished algorithm.
        """
        buffer.update_now_len_before_sample()

        actor_obj = None  # just for print return

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(target_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, action, state, next_state = buffer.sample_batch(batch_size_)

                next_q_label, next_action = self.act_target.next_q_action(state, next_state, self.policy_noise)
                q_label = reward + mask * next_q_label

            """critic_obj"""
            q_eval = self.act.critic(state, action)
            critic_obj = self.criterion(q_eval, q_label)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_obj.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            '''actor correction term'''
            actor_term = self.criterion(self.act(next_state), next_action)

            if i % repeat_times == 0:
                '''actor obj'''
                action_pg = self.act(state)  # policy gradient
                actor_obj = -self.act_target.critic(state, action_pg).mean()  # policy gradient
                # NOTICE! It is very important to use act_target.critic here instead act.critic
                # Or you can use act.critic.deepcopy(). Whatever you cannot use act.critic directly.

                united_loss = critic_obj + actor_term * (1 - lamb) + actor_obj * (lamb * 0.5)
            else:
                united_loss = critic_obj + actor_term * (1 - lamb)

            """united loss"""
            self.optimizer.zero_grad()
            united_loss.backward()
            self.optimizer.step()

            if i % self.update_freq == self.update_freq and lamb > 0.1:
                self.act_target.load_state_dict(self.act.state_dict())  # Hard Target Update

        return actor_obj.item(), self.avg_loss_c


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.target_entropy = None
        self.alpha_log = None
        self.alpha_optimizer = None

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act.get_action(states)[0]
        return action.cpu().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """Contribution of SAC (Soft Actor-Critic with maximum entropy)

        1. maximum entropy (Soft Q-learning -> Soft Actor-Critic, good idea)
        2. auto alpha (automating entropy adjustment on temperature parameter alpha for maximum entropy)
        3. SAC use TD3's TwinCritics too
        """
        buffer.update_now_len_before_sample()

        alpha = self.alpha_log.exp().detach()
        obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a, next_logprob = self.act_target.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_logprob * alpha)
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            obj_alpha.backward()
            self.alpha_optimizer.step()

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + logprob * alpha).mean()

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return alpha.item(), obj_critic.item()


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self):
        super().__init__()
        self.if_use_dn = True
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.act = ActorSAC(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
        self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim, self.if_use_dn).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), self.learning_rate)

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """ModSAC (Modified SAC using Reliable lambda)

        1. Reliable Lambda is calculated based on Critic's loss function value.
        2. Increasing batch_size and update_times
        3. Auto-TTUR updates parameter in non-integer times.
        4. net_dim of critic is slightly larger than actor.
        """
        buffer.update_now_len_before_sample()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(target_step * k * repeat_times)

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        for update_c in range(1, train_steps):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size_)

                next_a, next_logprob = self.act.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_logprob * alpha)
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient

            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            obj_alpha.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = (update_a / update_c) < (1 / (2 - reliable_lambda))
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri_target.get_q1_q2(state, a_noise_pg))  # ceta3
                obj_actor = -(q_value_pg + logprob * alpha.detach()).mean()
                obj_actor = obj_actor * reliable_lambda  # max(0.01, reliable_lambda)

                self.act_optimizer.zero_grad()
                obj_actor.backward()
                self.act_optimizer.step()

        return alpha.item(), self.obj_c


class AgentInterSAC(AgentSAC):  # Integrated Soft Actor-Critic
    def __init__(self):
        super().__init__()
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.optimizer = None

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.act = InterSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            [{'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.9},  # more stable
             {'params': self.act.enc_a.parameters(), },
             {'params': self.act.net.parameters(), 'lr': self.learning_rate * 0.9},
             {'params': self.act.dec_a.parameters(), },
             {'params': self.act.dec_d.parameters(), },
             {'params': self.act.dec_q1.parameters(), },
             {'params': self.act.dec_q2.parameters(), },
             {'params': (self.alpha_log,)}], lr=self.learning_rate)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act.get_noise_action(states)[0]
        return action.cpu().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):  # 1111
        """Contribution of InterSAC (Integrated network for SAC)

        1. Encoder-DenseNetLikeNet-Decoder network architecture.
            share parameter between two **different input** network
            DenseNetLikeNet with deep and shallow network is a good approximate function suitable for RL
        2. Reliable Lambda is calculated based on Critic's loss function value.
        3. Auto-TTUR updates parameter in non-integer times.
        4. Different learning rate is better than actor_term in parameter-sharing network training.
        """
        buffer.update_now_len_before_sample()

        logprob = None  # just for print return
        alpha = self.alpha_log.exp().detach()  # auto temperature parameter

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)  # increase batch_size
        train_steps = int(target_step * k * repeat_times)  # increase training_step

        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size_)

                next_q_label, next_logprob = self.act_target.get_q_logprob(next_s)
                q_label = reward + mask * (next_q_label + next_logprob * alpha)  # auto temperature parameter

            """obj_critic"""
            q1_value, q2_value = self.act.get_q1_q2(state, action)  # CriticTwin
            obj_critic = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)
            '''auto reliable lambda'''
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item() / 2  # soft update, twin critics
            reliable_lambda = np.exp(-self.obj_c ** 2)

            action_pg, logprob = self.act.get_a_logprob(state)

            '''auto temperature parameter: alpha'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach() * reliable_lambda).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
                alpha = self.alpha_log.exp()  # .detach()

            if update_a / update_c < 1 / (2 - reliable_lambda):  # auto TTUR
                update_a += 1
                """obj_actor"""
                q_value_pg = torch.min(*self.act_target.get_q1_q2(state, action_pg)).mean()  # twin critics
                # obj_actor = -(q_value_pg + logprob * alpha).mean()  # wait to check: alpha.detach()
                obj_actor = -(q_value_pg + logprob * alpha.detach()).mean()  # policy gradient

                obj_united = obj_critic + obj_alpha + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic + obj_alpha

            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return logprob.mean().item(), self.obj_c


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.3  # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.lambda_entropy = 0.01  # could be 0.01 ~ 0.05
        self.lambda_gae_adv = 0.98  # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.if_use_gae = False  # if use Generalized Advantage Estimation
        self.if_on_policy = True  # AgentPPO is an on policy DRL algorithm

        self.noise = None
        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(state_dim, net_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])

    def select_action(self, state) -> tuple:
        """select action for PPO

        :array state: state.shape==(state_dim, )

        :return array action: state.shape==(action_dim, )
        :return array noise: noise.shape==(action_dim, ), the noise
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach()
        actions, noises = self.act.get_action_noise(states)
        return actions[0].cpu().numpy(), noises[0].cpu().numpy()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        buffer.empty_buffer_before_explore()  # NOTICE! necessary for on-policy
        # assert target_step == buffer.max_len - max_step

        actual_step = 0
        while actual_step < target_step:
            state = env.reset()
            for _ in range(env.max_step):
                action, noise = self.select_action(state)

                next_state, reward, done, _ = env.step(np.tanh(action))
                actual_step += 1

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_buffer(state, other)
                if done:
                    break
                state = next_state
        return actual_step

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return self.act.a_std_log.mean().item(), obj_critic.item()

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage


class AgentInterPPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = InterPPO(state_dim, action_dim, net_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.9},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate)

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):  # old version
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using Generalized Advantage Estimation (GAE)'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
            buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

            pre_r_sum = 0  # reward sum of previous step
            pre_advantage = 0  # advantage value of previous step
            for i in range(buf_len - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]

                buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
                pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Clipped Surrogate objective of Trust Region'''
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            advantage = buf_advantage[indices]
            old_value = buf_r_sum[indices]
            old_logprob = buf_logprob[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - old_logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip, 1 + self.clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            new_value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(new_value, old_value)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda

            obj_united = obj_actor * reliable_lambda + obj_critic / (old_value.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return self.act.a_std_log.mean().item(), self.obj_c


'''Utils'''


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy, if_gpu):
        """Experience Replay Buffer

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :bool if_on_policy: on-policy or off-policy
        :bool if_gpu: create buffer space on CPU RAM or GPU
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_all(
        self.if_on_policy = if_on_policy
        self.if_gpu = if_gpu

        if if_on_policy:
            self.if_gpu = False
            other_dim = 1 + 1 + action_dim * 2
        else:
            other_dim = 1 + 1 + action_dim

        if self.if_gpu:
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        else:
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, device=self.device)
            other = torch.as_tensor(other, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            other = torch.as_tensor(other, dtype=torch.float32, device=self.device)

        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device) if self.if_gpu \
            else rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def sample_all(self) -> tuple:
        """sample all the data in ReplayBuffer (for on-policy)

        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor noise:  noise.shape ==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        """
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],
                all_other[:, 1],
                all_other[:, 2:2 + self.action_dim],
                all_other[:, 2 + self.action_dim:],
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """we empty the buffer by set now_len=0. On-policy need to empty buffer before exploration
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        """print the state norm information: state_avg, state_std

        We don't suggest to use running stat state.
        We directly do normalization on state using the historical avg and std
        eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
        neg_avg = -states.mean()
        div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

        :array neg_avg: neg_avg.shape=(state_dim)
        :array div_std: div_std.shape=(state_dim)
        """
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| print_norm: state_avg, state_fix_std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")


class ReplayBufferMP:
    def __init__(self, max_len, state_dim, action_dim, rollout_num, if_on_policy, if_gpu):
        """Experience Replay Buffer for Multiple Processing

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :int rollout_num: the rollout workers number
        :bool if_on_policy: on-policy or off-policy
        :bool if_gpu: create buffer space on CPU RAM or GPU
        """
        self.now_len = 0
        self.max_len = max_len
        self.rollout_num = rollout_num

        self.if_gpu = if_gpu
        if if_on_policy:
            self.if_gpu = False

        _max_len = max_len // rollout_num
        self.buffers = [ReplayBuffer(_max_len, state_dim, action_dim, if_on_policy, if_gpu=True)
                        for _ in range(rollout_num)]

    def extend_buffer(self, state, other, i):
        self.buffers[i].extend_buffer(state, other)

    def sample_batch(self, batch_size) -> tuple:
        rd_batch_sizes = rd.rand(self.rollout_num)
        rd_batch_sizes = (rd_batch_sizes * (batch_size / rd_batch_sizes.sum())).astype(np.int)
        l__r_m_a_s_ns = [self.buffers[i].sample_batch(rd_batch_sizes[i])
                         for i in range(self.rollout_num) if rd_batch_sizes[i] > 2]
        return (torch.cat([item[0] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[1] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[2] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[3] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[4] for item in l__r_m_a_s_ns], dim=0))

    def sample_all(self) -> tuple:
        l__r_m_a_n_s = [self.buffers[i].sample_all()
                        for i in range(self.rollout_num)]
        return (torch.cat([item[0] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[1] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[2] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[3] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[4] for item in l__r_m_a_n_s], dim=0))

    def update_now_len_before_sample(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len_before_sample()
            self.now_len += buffer.now_len

    def empty_buffer_before_explore(self):
        for buffer in self.buffers:
            buffer.empty_buffer_before_explore()

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
