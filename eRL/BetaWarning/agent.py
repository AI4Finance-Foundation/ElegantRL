import os
from copy import deepcopy  # deepcopy target_network

import torch
import numpy as np
import numpy.random as rd
from eRL.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from eRL.net import Actor, ActorSAC, ActorPPO
from eRL.net import Critic, CriticAdv, CriticTwin
from eRL.net import InterDPG, InterSPG, InterPPO

"""ElegantRL (Pytorch 3 files model-free DRL Library)
GitHub.com: YonV1943, Zhihu.com: 曾伊言

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
In my opinion, before 2020, the policy gradient algorithm agent didn't learn s policy.
Actually, they "learn game feel" or "get a soft touch". In Chinese "shǒu gǎn 手感". 
Learn more about policy gradient algorithms in:

Policy Gradient Algorithm summary
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
如何选择深度强化学习算法？MuZero/SAC/PPO/TD3/DDPG/DQN/等 
https://zhuanlan.zhihu.com/p/342919579
深度强化学习调参技巧：以D3QN、TD3、PPO、SAC算法为例
https://zhuanlan.zhihu.com/p/345353294

reference:
TD3 https://github.com/sfujim/TD3 good++
TD3 https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2 good
PPO https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py good+
PPO https://github.com/xtma/pytorch_car_caring good
PPO https://github.com/openai/baselines/tree/master/baselines/ppo2 normal-
SAC https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac normal -
DUEL https://github.com/gouxiangchen/dueling-DQN-pytorch good
"""

'''Value-based Methods (DQN variance)'''


class AgentDQN:
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
        self.action_dim = action_dim

        self.state = None  # set for self.update_buffer(), initialize self.state before training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

    def select_actions(self, states):  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim, size=(len(states),))  # choosing action randomly
        else:
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = self.act(states)
            a_int = actions.argmax(dim=1).detach().cpu().numpy()
        return a_int

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_memo(self.state, other)
            self.state = env.reset() if done else next_s
        return max_step

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of DQN (Deep Q Network)
        1. Q-table (discrete state space) --> Q-network (continuous state space)
        2. Use experiment replay buffer to train a neural network in RL
        3. Use soft target update to stablize training in RL
        """
        buffer.update__now_len__before_sample()

        next_q = obj_critic = None
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)  # next_state
                next_q = self.act_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            q_eval = self.act(state).gather(1, action.type(torch.long))
            obj_critic = self.criterion(q_eval, q_label)

            self.optimizer.zero_grad()
            obj_critic.backward()
            self.optimizer.step()
            soft_target_update(self.act_target, self.act, tau=5e-3)
        return next_q.mean().item(), obj_critic.item()  #


class AgentDuelingDQN(AgentDQN):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

        self.act = QNetDuel(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        """Contribution of Dueling DQN
        1. Advantage function (of A2C) --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """


class AgentDoubleDQN(AgentDQN):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.explore_rate = 0.25  # epsilon-greedy, the rate of choosing random action
        self.softmax = torch.nn.Softmax(dim=1)
        self.action_dim = action_dim

        self.act = QNetTwin(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

    def select_actions(self, states):  # for discrete action space
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob_l = self.softmax(actions).detach().cpu().numpy()  # choose action according to Q value
            a_int = [rd.choice(self.action_dim, p=a_prob) for a_prob in a_prob_l]
        else:
            a_int = actions.argmax(dim=1).detach().cpu().numpy()
        return a_int

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of DDQN (Double DQN)
        1. Twin Q-Network. Use min(q1, q2) to reduce over-estimation.
        """
        buffer.update__now_len__before_sample()

        next_q = obj_critic = None
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_q = self.act_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            action = action.type(torch.long)
            q_eval1, q_eval2 = [qs.gather(1, action) for qs in self.act.get__q1_q2(state)]
            obj_critic = self.criterion(q_eval1, q_label) + self.criterion(q_eval2, q_label)

            self.optimizer.zero_grad()
            obj_critic.backward()
            self.optimizer.step()
            soft_target_update(self.act_target, self.act)
        return next_q.mean().item(), obj_critic.item() / 2


class AgentD3QN(AgentDoubleDQN):  # D3QN: Dueling Double DQN
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.explore_rate = 0.25  # epsilon-greedy, the rate of choosing random action

        self.act = QNetTwinDuel(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        """Contribution of D3QN (Dueling Double DQN)
        There are not contribution of D3QN.  
        Obviously, DoubleDQN is compatible with DuelingDQN.
        Any beginner can come up with this idea (D3QN) independently.
        """


'''Actor-Critic Methods (Policy Gradient)'''


class AgentBase:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = None  # set for self.update_buffer(), initialize before training
        self.action = None  # set for self.update_buffer__pipe(), initialize before training

        self.act = None
        self.cri = None

    def select_actions(self, states):  # states = (state, ...)
        return (None,)  # -1 < action < +1

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_memo(self.state, other)
            self.state = env.reset() if done else next_s
        return max_step

    def save_or_load_model(self, cwd, if_save):  # 2020-07-07
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_act = 'act' in dir(self)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if has_act:
                torch.save(self.act.state_dict(), act_save_path)
            if has_cri:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif has_act and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif has_cri and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


class AgentDDPG(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.explore_noise = 0.05  # explore noise of action
        # self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=0.3)  # I don't recommend OU-Noise

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_actions(self, states):  # states = (state, ...)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        # actions = actions.detach().cpu().numpy()
        # return (actions + self.ou_noise()).clip(-1, 1)
        actions = (actions + torch.randn_like(actions) * self.explore_noise).clamp(-1, 1)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """ Contribution of DDPG (Deep Deterministic Policy Gradient)
        1. Policy Gradient with Deep network: DQN + DPG -> DDPG
           Q_value = reward + gamma * next_Q_value
           Q-learning -> DQN (Deep Q-learning): (discrete state space Q-table -> continuous state space Q-net)
           DQN + DPG -> DDPG: (discrete action space Q-net -> continuous action space Policy Gradient)
        2. experiment replay buffer for stabilizing training
        3. soft target update for stabilizing training
        """
        buffer.update__now_len__before_sample()

        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(max_step * repeat_times)):
            """critic (train Critic network using Supervised Deep learning)
            the optimization objective of critic is minimizing loss function 'criterion(q_value, q_label)'
            minimize criterion(q_eval, label) to train a critic
            We input state-action to a critic (policy function), critic will output a q_value estimation.
            A better action will get higher q_value from critic.  
            """
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_q = self.cri_target(next_s, self.act_target(next_s))
                q_label = reward + mask * next_q
            q_value = self.cri(state, action)
            obj_critic = self.criterion(q_value, q_label)

            """actor (Policy Gradient)
            the optimization objective of actor is maximizing value function 'critic(state, actor(state))'
            maximize cri(state, action) is equal to minimize -cri(state, action)
            Accurately, it is more appropriate to call 'actor_obj' as 'actor_objective'.

            We train critic output q_value close to q_label
                by minimizing the error provided by loss function of critic.
            We train actor output action which gets higher q_value from critic
                by maximizing the q_value provided by policy function.
            We call it Policy Gradient (PG). The gradient for actor is provided by a policy function.
                By the way, Generative Adversarial Networks (GANs) is a kind of Policy Gradient.
                The gradient for Generator (Actor) is provided by a Discriminator (Critic).
            """
            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            """united objective
            I can write in this way:
            
            self.optimizer_of_actor.zero_grad()
            obj_actor.backward()
            self.optimizer_of_actor.step()
            
            self.optimizer_of_critic.zero_grad()
            obj_critic.backward()
            self.optimizer_of_critic.step()
            
            I use one single optimizer for both networks in order to speed up training
            """
            obj_united = obj_actor + obj_critic  # objective
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act)
        return obj_actor.item(), obj_critic.item()


class AgentTD3(AgentDDPG):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency, for soft target update

        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of TD3 (Twin Delay DDPG)
        1. twin critics (DoubleDQN -> TwinCritic, good idea)
        2. policy noise ('Deterministic Policy Gradient + policy noise' looks like Stochastic PG)
        3. delay update (I think it is not very useful)
        """
        buffer.update__now_len__before_sample()

        obj_critic = obj_actor = None
        for i in range(int(max_step * repeat_times)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
                next_q = torch.min(*self.cri_target.get__q1_q2(next_s, next_a))  # twin critics
                q_label = reward + mask * next_q
            q1, q2 = self.cri.get__q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics

            '''objective of actor'''
            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            '''united objective'''
            obj_united = obj_actor + obj_critic  # objective
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            if i % self.update_freq == 0:  # delay update
                soft_target_update(self.cri_target, self.cri)
                soft_target_update(self.act_target, self.act)
        return obj_actor.item(), obj_critic.item() / 2


class AgentInterAC(AgentBase):  # use InterSAC instead of InterAC .Warning: sth. wrong with this code, need to check
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.explore_noise = 0.2  # standard deviation of explore noise
        self.policy_noise = 0.4  # standard deviation of policy noise
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update
        self.avg_loss_c = (-np.log(0.5)) ** 0.5  # old version reliable_lambda

        self.act = InterDPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
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
        buffer.update__now_len__before_sample()

        actor_obj = None  # just for print return

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, action, state, next_state = buffer.random_sample(batch_size_)

                next_q_label, next_action = self.act_target.next__q_a(state, next_state, self.policy_noise)
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
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate},
                                           {'params': (self.alpha_log,), 'lr': learning_rate}])

    def select_actions(self, states):  # states = (state, ...)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get_action(states)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of SAC (Soft Actor-Critic with maximum entropy)
        1. maximum entropy (Soft Q-learning -> Soft Actor-Critic, good idea)
        2. auto alpha (automating entropy adjustment on temperature parameter alpha for maximum entropy)
        3. SAC use TD3's TwinCritics too
        """
        buffer.update__now_len__before_sample()

        alpha = self.alpha_log.exp().detach()
        obj_critic = None
        for _ in range(int(max_step * repeat_times)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_a, next_log_prob = self.act_target.get__action__log_prob(next_s)
                next_q = torch.min(*self.cri_target.get__q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get__q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get__action__log_prob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
            obj_actor = -(torch.min(*self.cri_target.get__q1_q2(state, action_pg)) + log_prob * alpha).mean()

            '''united objective'''
            obj_united = obj_critic + obj_alpha + obj_actor
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act)

        return alpha.item(), obj_critic.item()


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        if_use_dn = False
        self.act = ActorSAC(net_dim, state_dim, action_dim, if_use_dn).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim, if_use_dn).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=learning_rate)

        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """ModSAC (Modified SAC using Reliable lambda)
        1. Reliable Lambda is calculated based on Critic's loss function value.
        2. Increasing batch_size and update_times
        3. Auto-TTUR updates parameter in non-integer times.
        4. net_dim of critic is slightly larger than actor.
        """
        buffer.update__now_len__before_sample()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(max_step * k * repeat_times)

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        for update_c in range(1, train_steps):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size_)

                next_a, next_log_prob = self.act_target.get__action__log_prob(next_s)
                next_q = torch.min(*self.cri_target.get__q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get__q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            soft_target_update(self.cri_target, self.cri)

            a_noise_pg, log_prob = self.act.get__action__log_prob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            obj_alpha.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri.get__q1_q2(state, a_noise_pg))
                obj_actor = -(q_value_pg + log_prob * alpha).mean()

                self.act_optimizer.zero_grad()
                (obj_actor * reliable_lambda).backward()
                self.act_optimizer.step()
                soft_target_update(self.act_target, self.act)

        return alpha.item(), self.obj_c


class AgentInterSAC(AgentBase):  # Integrated Soft Actor-Critic
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.obj_c = (-np.log(0.5)) ** 0.5  # reliable_lambda
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=learning_rate)

        self.act = InterSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            [{'params': self.act.enc_s.parameters(), 'lr': learning_rate},  # more stable
             {'params': self.act.enc_a.parameters(), },
             {'params': self.act.net.parameters(), 'lr': learning_rate},
             {'params': self.act.dec_a.parameters(), },
             {'params': self.act.dec_d.parameters(), },
             {'params': self.act.dec_q1.parameters(), },
             {'params': self.act.dec_q2.parameters(), }], lr=learning_rate)

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):  # 1111
        """Contribution of InterSAC (Integrated network for SAC)
        1. Encoder-DenseNetLikeNet-Decoder network architecture.
            share parameter between two **different input** network
            DenseNetLikeNet with deep and shallow network is a good approximate function suitable for RL
        2. Reliable Lambda is calculated based on Critic's loss function value.
        3. Auto-TTUR updates parameter in non-integer times.
        4. Different learning rate is better than actor_term in parameter-sharing network training.
        """
        buffer.update__now_len__before_sample()

        log_prob = None  # just for print return
        alpha = self.alpha_log.exp().detach()  # auto temperature parameter

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)  # increase batch_size
        train_steps = int(max_step * k * repeat_times)  # increase training_step

        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size_)

                next_q_label, next_log_prob = self.act_target.get__q__log_prob(next_s)
                q_label = reward + mask * (next_q_label + next_log_prob * alpha)  # auto temperature parameter

            """obj_critic"""
            q1_value, q2_value = self.act.get__q1_q2(state, action)  # CriticTwin
            obj_critic = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)
            '''auto reliable lambda'''
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item() / 2  # soft update, twin critics
            reliable_lambda = np.exp(-self.obj_c ** 2)

            action_pg, log_prob = self.act.get__a__log_prob(state)

            '''auto temperature parameter: alpha'''
            alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach() * reliable_lambda).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
                alpha = self.alpha_log.exp()  # .detach()

            if update_a / update_c < 1 / (2 - reliable_lambda):  # auto TTUR
                update_a += 1
                """obj_actor"""
                q_value_pg = torch.min(*self.act_target.get__q1_q2(state, action_pg)).mean()  # twin critics
                obj_actor = -(q_value_pg + log_prob * alpha).mean()  # policy gradient

                obj_united = obj_critic + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic

            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            soft_target_update(self.act_target, self.act)

        return log_prob.mean().item(), self.obj_c


class AgentPPO(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # larger lambda_entropy means more exploration
        self.action = self.noise = None

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(state_dim, net_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_actions(self, states):  # states = (state, ...)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_noise, noise = self.act.get__action_noise(states)
        return a_noise.detach().cpu().numpy(), noise.detach().cpu().numpy()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        buffer.empty_memories__before_explore()  # NOTICE! necessary

        step_counter = 0
        target_step = buffer.max_len - max_step
        while step_counter < target_step:
            state = env.reset()
            for _ in range(max_step):
                action, noise = self.select_actions((state,))
                action = action[0]
                noise = noise[0]

                next_state, reward, done, _ = env.step(np.tanh(action))
                step_counter += 1

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_memo(state, other)
                if done:
                    break
                state = next_state
        return step_counter

    def update_policy(self, buffer, _max_step, batch_size, repeat_times=8):
        buffer.update__now_len__before_sample()
        max_memo = buffer.now_len

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_for_ppo()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_log_prob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(buffer, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * max_memo / batch_size)):
            indices = torch.randint(max_memo, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            log_prob = buf_log_prob[indices]
            advantage = buf_advantage[indices]

            new_log_prob = self.act.compute__log_prob(state, action)  # it is obj_actor
            ratio = (new_log_prob - log_prob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip, 1 + self.clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_log_prob.exp() * new_log_prob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return self.act.a_std_log.mean().item(), obj_critic.item()

    def compute_reward(self, buffer, buf_reward, buf_mask, buf_value):
        max_memo = buffer.now_len

        buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(max_memo - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        # buf_advantage = buf_advantage / (buf_advantage.std() + 1e-5)
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage


class AgentGaePPO(AgentPPO):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)

    def compute_reward(self, buffer, buf_reward, buf_mask, buf_value):
        max_memo = buffer.now_len

        buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(max_memo - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        # buf_advantage = buf_advantage / (buf_advantage.std() + 1e-5)
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage


class AgentInterPPO(AgentPPO):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_adv = 0.98  # could be 0.95~0.99

        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

        self.act = InterPPO(state_dim, action_dim, net_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': learning_rate * 0.9},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=learning_rate)

    def update_policy(self, buffer, _max_step, batch_size, repeat_times=8):  # old version
        buffer.update__now_len__before_sample()
        max_memo = buffer.now_len

        '''Trajectory using Generalized Advantage Estimation (GAE)'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_for_ppo()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_log_prob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
            buf_advantage = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

            pre_r_sum = 0  # reward sum of previous step
            pre_advantage = 0  # advantage value of previous step
            for i in range(max_memo - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]

                buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
                pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_adv

            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Clipped Surrogate objective of Trust Region'''
        for _ in range(int(repeat_times * max_memo / batch_size)):
            indices = torch.randint(max_memo, size=(batch_size,), device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            advantage = buf_advantage[indices]
            old_value = buf_r_sum[indices]
            old_log_prob = buf_log_prob[indices]

            new_log_prob = self.act.compute__log_prob(state, action)  # it is obj_actor
            ratio = (new_log_prob - old_log_prob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip, 1 + self.clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_log_prob.exp() * new_log_prob).mean()  # policy entropy
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
    def __init__(self, max_len, state_dim, action_dim, if_on_policy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_for_ppo(
        self.if_on_policy = if_on_policy

        if if_on_policy:
            other_dim = 1 + 1 + action_dim * 2
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
        else:
            other_dim = 1 + 1 + action_dim
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

    def append_memo(self, state, other):  # CPU array to CPU array
        if not self.if_on_policy:
            state = torch.as_tensor(state, device=self.device)
            other = torch.as_tensor(other, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_memo(self, state, other):  # CPU array to CPU array
        if not self.if_on_policy:
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

    def random_sample(self, batch_size):
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_for_ppo(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
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
    def __init__(self, max_len, state_dim, action_dim, if_on_policy, rollout_num):
        self.now_len = 0
        self.max_len = max_len
        self.rollout_num = rollout_num

        _max_len = max_len // rollout_num
        self.buffers = [ReplayBuffer(_max_len, state_dim, action_dim, if_on_policy)
                        for _ in range(rollout_num)]

    def extend_memo_mp(self, state, other, i):
        self.buffers[i].extend_memo(state, other)

    def random_sample(self, batch_size):
        _batch_size = batch_size // self.rollout_num
        rd_batch_sizes = rd.rand(self.rollout_num)
        rd_batch_sizes = (rd_batch_sizes * (_batch_size / rd_batch_sizes.sum())).astype(np.int)

        l__r_m_a_s_ns = [self.buffers[i].random_sample(rd_batch_sizes[i])
                         for i in range(self.rollout_num) if rd_batch_sizes[i] > 1]
        return (torch.cat([item[0] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[1] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[2] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[3] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[4] for item in l__r_m_a_s_ns], dim=0))

    def sample_for_ppo(self):
        l__r_m_a_n_s = [self.buffers[i].sample_for_ppo()
                        for i in range(self.rollout_num)]
        return (torch.cat([item[0] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[1] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[2] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[3] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[4] for item in l__r_m_a_n_s], dim=0))

    def update__now_len__before_sample(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update__now_len__before_sample()
            self.now_len += buffer.now_len

    def empty_memories__before_explore(self):
        for buffer in self.buffers:
            buffer.empty_memories__before_explore()

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)


def soft_target_update(target, current, tau=2 ** -8):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class OrnsteinUhlenbeckNoise:
    """ Don't abuse OU Process
    OU process has too much hyper-parameters.
    Over fine-tuning is meaningless.
    """

    def __init__(self, size, theta=0.15, sigma=0.3, x0=0.0, dt=1e-2):
        """
        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        I think that:
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        """
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.size = size

    def __call__(self):
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        x = self.x0 - self.theta * self.x0 * self.dt + noise
        self.x0 = x  # update x0
        return x
