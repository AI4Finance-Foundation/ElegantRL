import os

import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn

from AgentNet import QNet, QNetTwin, QNetDuel, QNetDuelTwin  # Q-learning based
from AgentNet import Actor, Critic, CriticTwin  # DDPG, TD3
from AgentNet import ActorSAC, CriticTwinShared  # SAC
from AgentNet import ActorPPO, CriticAdv  # PPO
from AgentNet import InterDPG, InterSPG, InterPPO  # parameter sharing

"""ZenYiYan, GitHub: YonV1943 ElegantRL (Pytorch 3 files model-free DRL Library)
reference
TD3 https://github.com/sfujim/TD3 good++
TD3 https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2 good
PPO https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py good+
PPO https://github.com/xtma/pytorch_car_caring good
PPO https://github.com/openai/baselines/tree/master/baselines/ppo2 normal-
SAC https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac normal -
SQL https://github.com/gouxiangchen/soft-Q-learning/blob/master/sql.py bad-
DUEL https://github.com/gouxiangchen/dueling-DQN-pytorch good
"""

'''Policy Gradient (Actor Critic)'''


class AgentDDPG:  # DEMO (tutorial only, simplify, low effective)
    def __init__(self, state_dim, action_dim, net_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=2e-4)

        self.act_target = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = Critic(state_dim, action_dim, net_dim).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=2e-4)

        self.cri_target = Critic(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()

        '''training record'''
        self.step = 0

        '''extension'''
        self.ou_noise = OrnsteinUhlenbeckProcess(size=action_dim, sigma=0.3)
        # I hate OU-Process in RL because of its too much hyper-parameters.

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states).cpu().data.numpy()[0]
        return (action + self.ou_noise()).clip(-1, 1)

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

        critic_obj = actor_obj = None  # just for print return
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, state, action, next_state = buffer.random_sample(batch_size)

                next_action = self.act_target(next_state)
                next_q_label = self.cri_target(next_state, next_action)
                q_label = reward + mask * next_q_label

            """critic (Supervised Deep learning)
            the optimization objective of critic is minimizing loss function 'criterion(q_value, q_label)'
            minimize criterion(q_eval, label) to train a critic
            We input state-action to a critic (policy function), critic will output a q_value estimation.
            A better action will get higher q_value from critic.  
            """
            q_value = self.cri(state, action)
            critic_obj = self.criterion(q_value, q_label)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()

            _soft_target_update(self.cri_target, self.cri)

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
            action_pg = self.act(state)
            actor_obj = -self.cri(state, action_pg).mean()

            self.act_optimizer.zero_grad()
            actor_obj.backward()
            self.act_optimizer.step()

            _soft_target_update(self.act_target, self.act)

        return actor_obj.item(), critic_obj.item()


class AgentBaseAC:  # DEMO (base class, a modify DDPG without OU-Process)
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = Critic(state_dim, action_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = Critic(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: action noise, delay update'''
        self.explore_noise = 2 ** -4  # standard deviation of explore noise
        self.policy_noise = 2 ** -3  # standard deviation of policy noise
        '''extension: delay target update'''
        self.update_freq = 2  # set as 2 or 4 for soft target update
        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states, self.explore_noise)
        return actions.cpu().data.numpy()[0]

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()
        for _ in range(max_step):
            '''inactive with environment'''
            action = self.select_action(self.state)
            next_state, reward, done, _ = env.step(action)

            self.reward_sum += reward
            self.step += 1

            '''update replay buffer'''
            # reward_ = reward * reward_scale
            # mask = 0.0 if done else gamma
            # buffer.append_memo((reward_, mask, self.state, action, next_state))
            reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
            buffer.append_memo((reward_mask, self.state, action, next_state))

            self.state = next_state
            if done:
                # Compatibility for ElegantRL 2020-12-21
                episode_return = env.episode_return if hasattr(env, 'episode_return') else self.reward_sum
                rewards.append(episode_return)
                self.reward_sum = 0.0

                steps.append(self.step)
                self.step = 0

                self.state = env.reset()
        return rewards, steps

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        q_label = None  # just for print return

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k * repeat_times)

        for i in range(update_times):
            with torch.no_grad():
                reward, mask, state, action, next_state = buffer.random_sample(batch_size_, )

                next_action = self.act_target.get__noise_action(next_state, self.policy_noise)
                next_q_label = self.cri_target(next_state, next_action)
                q_label = reward + mask * next_q_label

            """critic_obj"""
            q_value = self.cri(state, action)
            critic_obj = self.criterion(q_value, q_label)
            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_obj.item()  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            # torch.nn.utils.clip_grad_norm(self.cri.parameters(), max_norm=1.0)  # keep this comment
            self.cri_optimizer.step()

            """actor_obj"""
            action_pg = self.act(state)  # policy gradient
            actor_obj = -self.cri(state, action_pg).mean() * lamb  # q value that pass policy gradient
            # self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * lamb  # keep this comment
            self.act_optimizer.zero_grad()
            actor_obj.backward()
            self.act_optimizer.step()

            if i % self.update_freq == 0:
                _soft_target_update(self.cri_target, self.cri)
                _soft_target_update(self.act_target, self.act)

        return q_label.mean().item(), self.avg_loss_c

    def save_or_load_model(self, cwd, if_save):  # 2020-07-07
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_act = 'act' in dir(self)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path) if has_act else None
            torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", cwd)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path) if has_act else None
            load_torch_file(self.cri, cri_save_path) if has_cri else None
            print("Loaded act and cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


class AgentTD3(AgentBaseAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: action noise'''
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        '''extension: delay target update'''
        self.update_freq = 2  # delay update frequency, for soft target update

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of TD3 (TDDD, Twin Delay DDPG)
        1. twin critics (DoubleDQN -> TwinCritic, good idea)
        2. policy noise ('Deterministic Policy Gradient + policy noise' looks like Stochastic PG)
        3. delay update (I think it is not very useful)
        """
        buffer.update__now_len__before_sample()

        critic_obj = actor_obj = None
        for i in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size)

                next_a_noise = self.act_target.get__noise_action(next_s, self.policy_noise)  # policy noise
                next_q_label = torch.min(*self.cri_target.get__q1_q2(next_s, next_a_noise))  # twin critics
                q_label = reward + mask * next_q_label

            """critic_obj"""
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # twin critics
            critic_obj = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()

            """actor_obj"""
            if i % repeat_times == 0:
                action_pg = self.act(state)  # policy gradient
                actor_obj = -self.cri(state, action_pg).mean()

                self.act_optimizer.zero_grad()
                actor_obj.backward()
                self.act_optimizer.step()

            if i % self.update_freq == 0:  # delay update
                _soft_target_update(self.cri_target, self.cri)
                _soft_target_update(self.act_target, self.act)

        return actor_obj.item(), critic_obj.item()


class AgentSAC(AgentBaseAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorSAC(state_dim, action_dim, net_dim, use_dn=False).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        # SAC uses target update network for critic only. Not for actor

        self.cri = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: auto-alpha for maximum entropy'''
        self.alpha_log = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.alpha_log.exp()
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        self.target_entropy = np.log(action_dim)

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states)
        return actions.cpu().data.numpy()[0]

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of SAC (Soft Actor-Critic with maximum entropy)
        1. maximum entropy (Soft Q-learning -> Soft Actor-Critic, good idea)
        2. auto alpha (automating entropy adjustment on temperature parameter alpha for maximum entropy)
        3. SAC use TD3's TwinCritics too
        """
        buffer.update__now_len__before_sample()

        log_prob = critic_obj = None  # just for print return

        for i in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size)

                next_a_noise, next_log_prob = self.act.get__a__log_prob(next_s)
                next_q_label = self.cri_target(next_s, next_a_noise)
                q_label = reward + mask * (next_q_label + next_log_prob * self.alpha)  # fix notice by Dr. Zang

            """critic_obj"""
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_obj = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()

            _soft_target_update(self.cri_target, self.cri)

            """actor_obj"""
            action_pg, log_prob = self.act.get__a__log_prob(state)  # policy gradient
            # auto alpha
            alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # policy gradient
            self.alpha = self.alpha_log.exp()
            q_eval_pg = self.cri(state, action_pg)  # policy gradient
            actor_obj = -(q_eval_pg + log_prob * self.alpha).mean()  # policy gradient

            self.act_optimizer.zero_grad()
            actor_obj.backward()
            self.act_optimizer.step()

        return log_prob.mean().item(), critic_obj


class AgentModSAC(AgentBaseAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        use_dn = True  # and use hard target update
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorSAC(state_dim, action_dim, net_dim, use_dn).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = ActorSAC(state_dim, action_dim, net_dim, use_dn).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = CriticTwinShared(state_dim, action_dim, int(net_dim * 1.25), use_dn).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwinShared(state_dim, action_dim, int(net_dim * 1.25), use_dn).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), requires_grad=True, device=self.device)
        # todo ? or -1
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states)
        return actions.cpu().data.numpy()[0]

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Modify content of ModSAC
        1. Reliable Lambda is calculated based on Critic's loss function value.
        2. Increasing batch_size and update_times
        3. Auto-TTUR updates parameter in non-integer times.
        4. net_dim of critic is slightly larger than actor.
        """
        buffer.update__now_len__before_sample()

        log_prob = None  # just for print return
        alpha = self.alpha_log.exp().detach()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(max_step * k * repeat_times)

        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_label = self.cri_target(next_s, next_a_noise)
                q_label = reward + mask * (next_q_label + next_log_prob * alpha)  # policy entropy

            """critic_obj"""
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_obj = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_obj.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()

            _soft_target_update(self.cri_target, self.cri)

            a_noise_pg, log_prob = self.act.get__a__log_prob(state)  # policy gradient

            '''auto temperature parameter (alpha)'''
            alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)  # todo (-16, 1)
            alpha = self.alpha_log.exp().detach()

            """actor_obj"""
            if update_a / update_c < 1 / (2 - lamb):  # auto TTUR
                update_a += 1

                q_value_pg = self.cri(state, a_noise_pg)
                actor_obj = -(q_value_pg + log_prob * alpha).mean()  # policy gradient

                # self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * lamb
                self.act_optimizer.zero_grad()
                actor_obj.backward()
                self.act_optimizer.step()

                _soft_target_update(self.act_target, self.act)

        # print(f'| alpha_log: {self.alpha_log.item():.3e}')
        return log_prob.mean().item(), self.avg_loss_c


class AgentInterAC(AgentBaseAC):  # warning: sth. wrong
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterDPG(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = InterDPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

        '''extension: action noise'''
        self.explore_noise = 0.2  # standard deviation of explore noise
        self.policy_noise = 0.4  # standard deviation of policy noise
        '''extension: delay target update'''
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update

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
                reward, mask, state, action, next_state = buffer.random_sample(batch_size_)

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
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            if i % self.update_freq == self.update_freq and lamb > 0.1:
                self.act_target.load_state_dict(self.act.state_dict())  # Hard Target Update

        return actor_obj.item(), self.avg_loss_c


class AgentInterSAC(AgentBaseAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.8},  # more stable
            {'params': self.act.enc_a.parameters(), },
            {'params': self.act.net.parameters(), 'lr': self.learning_rate * 0.8},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.dec_d.parameters(), },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-self.target_entropy,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states)
        return actions.cpu().data.numpy()[0]

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
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_q_label, next_log_prob = self.act_target.get__q__log_prob(next_s)
                q_label = reward + mask * (next_q_label + next_log_prob * alpha)  # auto temperature parameter

            """critic_obj"""
            q1_value, q2_value = self.act.get__q1_q2(state, action)  # CriticTwin
            critic_obj = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)
            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_obj.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            action_pg, log_prob = self.act.get__a__log_prob(state)

            '''auto temperature parameter: alpha'''
            alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach() * lamb).mean()  # stable
            # self.alpha_optimizer.param_groups[0]['lr'] = self.learning_rate * lamb
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
                alpha = self.alpha_log.exp()  # .detach()

            if update_a / update_c < 1 / (2 - lamb):  # auto TTUR
                update_a += 1
                """actor_obj"""
                q_value_pg = torch.min(*self.act_target.get__q1_q2(state, action_pg)).mean()  # twin critics
                actor_obj = -(q_value_pg + log_prob * alpha).mean()  # policy gradient

                united_loss = critic_obj + actor_obj * lamb
            else:
                united_loss = critic_obj

            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            _soft_target_update(self.act_target, self.act, tau=2 ** -8)

        return log_prob.mean().item(), self.avg_loss_c


class AgentPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.act_optimizer = torch.optim.SGD(self.act.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.cri_optimizer = torch.optim.SGD(self.cri.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.criterion = nn.SmoothL1Loss()

    def select_action(self, states):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        a_noise, noise = self.act.get__a_noise__noise(states)
        a_noise = a_noise.cpu().data.numpy()[0]
        noise = noise.cpu().data.numpy()[0]
        return a_noise, noise  # not tanh()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        step_counter = 0
        target_step = buffer.max_len - max_step
        while step_counter < target_step:
            state = env.reset()

            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                a_noise, noise = self.select_action((state,))

                next_state, reward, done, _ = env.step(np.tanh(a_noise))
                reward_sum += reward

                reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
                buffer.append_memo((reward_mask, state, a_noise, noise))

                if done:
                    break

                state = next_state

            # Compatibility for ElegantRL 2020-12-21
            episode_return = env.episode_return if hasattr(env, 'episode_return') else reward_sum
            rewards.append(episode_return)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        """Contribution of PPO (Proximal Policy Optimization
        1. the surrogate objective of TRPO, PPO simplified calculation of TRPO
        2. use the advantage function of A3C (Asynchronous Advantage Actor-Critic)
        3. add GAE. ICLR 2016. Generalization Advantage Estimate and use trajectory to calculate Q value
        """
        buffer.update__now_len__before_sample()

        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_obj = critic_obj = None  # just for print return

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample()

        all__new_v = list()
        all_log_prob = list()
        with torch.no_grad():
            b_size = 2 ** 10
            a_std_log__sqrt_2pi_log = self.act.a_std_log + self.act.sqrt_2pi_log
            for i in range(0, all_state.size()[0], b_size):
                new_v = self.cri(all_state[i:i + b_size])
                all__new_v.append(new_v)

                log_prob = -(all_noise[i:i + b_size].pow(2) / 2 + a_std_log__sqrt_2pi_log).sum(1)
                all_log_prob.append(log_prob)

            all__new_v = torch.cat(all__new_v, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_obj) + (actor_obj)
            """

            """critic_obj"""
            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_obj
            new_value = self.cri(state)

            critic_obj = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()

            """actor_obj"""
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_obj = surrogate_obj + loss_entropy * lambda_entropy

            self.act_optimizer.zero_grad()
            actor_obj.backward()
            self.act_optimizer.step()

        return actor_obj.item(), critic_obj.item()

    def save_or_load_model(self, cwd, if_save):  # 2020-07-07
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_act = 'act' in dir(self)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path) if has_act else None
            torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", cwd)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path) if has_act else None
            load_torch_file(self.cri, cri_save_path) if has_cri else None
            print("Loaded act and cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


class AgentInterPPO(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.8},  # more stable
            # {'params': self.act.net.parameters(), 'lr': self.learning_rate},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate)

        self.criterion = nn.SmoothL1Loss()
        self.action_dim = action_dim

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample()

        b_size = 2 ** 10
        with torch.no_grad():
            all__new_v = list()
            all_log_prob = list()
            for i in range(0, all_state.size()[0], b_size):
                new_v, log_prob = self.act.get__q__log_prob(all_state[i:i + b_size], all_noise[i:i + b_size])
                all__new_v.append(new_v)
                all_log_prob.append(log_prob)

            all__new_v = torch.cat(all__new_v, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value
        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv
            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]
        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)
        # Q_value_norm is necessary. Because actor_obj = surrogate_obj + loss_entropy * lambda_entropy.

        '''mini batch sample'''
        all__old_v = all__old_v.unsqueeze(1)
        sample_times = int(repeat_times * max_memo / batch_size)

        for _ in range(sample_times):
            '''random sample'''
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices]
            action = all_action[indices]
            old_log_prob = all_log_prob[indices]

            q1_value, q2_value, new_log_prob = self.act.get__q1_q2__log_prob(state, action)

            """critic_obj"""
            critic_obj = (self.criterion(q1_value, old_value) +
                          self.criterion(q2_value, old_value)) / (old_value.std() + 1e-5)
            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_obj.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            """actor_obj"""
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()

            actor_obj = surrogate_obj + loss_entropy * lambda_entropy

            '''united_loss'''
            united_loss = critic_obj + actor_obj * lamb
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

        buffer.empty_memories__before_explore()
        return self.act.a_std_log.mean().item(), self.avg_loss_c


'''DQN Variants (Critic Only)'''


class AgentDQN(AgentBaseAC):  # 2020-06-06
    def __init__(self, state_dim, action_dim, net_dim):  # 2020-04-30
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = QNet(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: epsilon-Greedy exploration'''
        self.explore_rate = 0.1  # explore rate
        self.action_dim = action_dim

    def select_action(self, state):  # for discrete action space
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if rd.rand() < self.explore_rate:
            a_int = rd.randint(self.action_dim)
        else:
            a_int = actions.argmax(dim=1).cpu().data.numpy()[0]
        return a_int

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of DQN (Deep Q Network)
        1. Q-table (discrete state space) --> Q-network (continuous state space)
        2. Use experiment replay buffer to train a neural network in RL
        3. Use soft target update to stablize training in RL
        """
        buffer.update__now_len__before_sample()

        critic_obj = None

        update_times = int(max_step * repeat_times)
        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size)

                next_q_label = self.act(next_states).max(dim=1, keepdim=True)[0]
                q_label = rewards + masks * next_q_label

            q_eval = self.act(states).gather(1, actions.type(torch.long))
            critic_obj = self.criterion(q_eval, q_label)

            self.act_optimizer.zero_grad()
            critic_obj.backward()
            self.act_optimizer.step()

        return 0, critic_obj.item()


class AgentDoubleDQN(AgentBaseAC):  # 2020-06-06 # I'm not sure.
    def __init__(self, state_dim, action_dim, net_dim):  # 2020-04-30
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = QNetTwin(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = QNetTwin(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: epsilon-Greedy exploration'''
        self.explore_rate = 0.25  # explore rate when update_buffer()
        self.action_dim = action_dim
        '''extension: q_value prob exploration'''
        self.softmax = nn.Softmax(dim=1)

    def select_action(self, state):  # for discrete action space
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if rd.rand() < self.explore_rate:
            a_prob = self.softmax(actions).cpu().data.numpy()[0]
            a_int = rd.choice(self.action_dim, p=a_prob)
        else:
            a_int = actions.argmax(dim=1).cpu().data.numpy()[0]
        return a_int

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of DDQN (Double DQN)
        1. Twin Q-Network. Use min(q1, q2) to reduce over-estimation.
        """
        buffer.update__now_len__before_sample()

        q_label = critic_obj = None

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size_)

                next_q_label = self.act_target(next_states).max(dim=1, keepdim=True)[0]
                q_label = rewards + masks * next_q_label

            actions = actions.type(torch.long)
            q_eval1, q_eval2 = [qs.gather(1, actions) for qs in self.act.get__q1_q2(states)]
            critic_obj = self.criterion(q_eval1, q_label) + self.criterion(q_eval2, q_label)

            self.act_optimizer.zero_grad()
            critic_obj.backward()
            self.act_optimizer.step()

            _soft_target_update(self.act_target, self.act)

        return q_label.mean().item(), critic_obj.item() / 2


class AgentDuelingDQN(AgentDoubleDQN):  # 2020-11-11
    def __init__(self, state_dim, action_dim, net_dim):
        object.__init__(self)  # super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = QNetDuel(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.act.train()

        self.act_target = QNetDuel(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())
        self.act_target.eval()

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: rho and loss_c'''
        self.explore_rate = 0.5  # explore rate when update_buffer()
        self.softmax = nn.Softmax(dim=1)
        self.action_dim = action_dim

    # def select_action() same as AgentDoubleDQN

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of Dueling DQN
        1. Advantage function (of A2C) --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """
        buffer.update__now_len__before_sample()

        q_label = critic_obj = None

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size_)

                next_q_label = self.act_target(next_states).max(dim=1, keepdim=True)[0]
                q_label = rewards + masks * next_q_label

            self.act.train()
            a_ints = actions.type(torch.long)
            q_eval = self.act(states).gather(1, a_ints)
            critic_obj = self.criterion(q_eval, q_label)

            self.act_optimizer.zero_grad()
            critic_obj.backward()
            self.act_optimizer.step()

            _soft_target_update(self.act_target, self.act)
        return q_label.mean().item(), critic_obj.item()


class AgentD3QN(AgentDoubleDQN):  # 2020-11-11
    def __init__(self, state_dim, action_dim, net_dim):
        object.__init__(self)  # super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = QNetDuelTwin(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.act.train()

        self.act_target = QNetDuelTwin(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())
        self.act_target.eval()

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: rho and loss_c'''
        self.explore_rate = 0.5  # explore rate when update_buffer()
        self.softmax = nn.Softmax(dim=1)
        self.action_dim = action_dim

    # def select_action() same as AgentDoubleDQN
    # def update_policy() same as AgentDuelingDQN


"""Private Utils: other python don't need to import them"""


def _soft_target_update(target, current, tau=5e-3):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class OrnsteinUhlenbeckProcess:
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
