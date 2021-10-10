import os
import torch
import numpy as np
import numpy.random as rd

from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from elegantrl.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from elegantrl.net import Actor, ActorPPO, ActorSAC, ActorDiscretePPO
from elegantrl.net import Critic, CriticAdv, CriticTwin
from elegantrl.net import SharedDPG, SharedSPG, SharedPPO

"""[ElegantRL.2021.09.09](https://github.com/AI4Finance-LLC/ElegantRL)"""


class AgentBase:
    def __init__(self):
        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.if_off_policy = True
        self.env_num = 1
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.grad_clip_norm = 6.0
        # self.amp_scale = None  # automatic mixed precision

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4,
             if_per_or_gae=False, env_num=1, gpu_id=0):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `float learning_rate` learning rate of optimizer
        `bool if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `int env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `int agent_id` if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
        """
        self.action_dim = action_dim
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

        assert isinstance(if_per_or_gae, bool)
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Select continuous actions for exploration

        `tensor states` states.shape==(batch_size, state_dim, )
        return `tensor actions` actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma):
        """actor explores in one env, then returns the traj (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        state = self.states[0]
        traj = list()
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + self.action_dim)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2:] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s

        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state, traj_other), ]
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step, reward_scale, gamma):
        """actor explores in VectorEnv, then returns the trajectory (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat((ten_rewards.unsqueeze(0),
                                    ten_dones.unsqueeze(0),
                                    ten_actions))
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        # traj = [(env_ten, ...), ...], env_ten = (env1_ten, env2_ten, ...)
        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state[:, env_i, :], traj_other[:, env_i, :])
                     for env_i in range(len(self.states))]
        # traj_list = [traj_env_0, ...], traj_env_0 = (ten_state, ten_other)
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer.
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        `float repeat_times` the times of sample batch = int(target_step * repeat_times) in off-policy
        `float soft_update_tau` target_net = target_net * (1-tau) + current_net * tau
        `return tuple` training logging. tuple = (float, float, ...)
        """

    def optim_update(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.grad_clip_norm)
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        `nn.Module target_net` target network update via a current network, it is more stable
        `nn.Module current_net` current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.

        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    @staticmethod
    def convert_trajectory(traj_list, reward_scale, gamma):  # off-policy
        for ten_state, ten_other in traj_list:
            ten_other[:, 0] = ten_other[:, 0] * reward_scale  # ten_reward
            ten_other[:, 1] = (1.0 - ten_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma
        return traj_list


"""Value-based Methods (Q network)"""


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = QNet
        self.if_use_cri_target = True

        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:  # for discrete action space
        """Select discrete actions for exploration

        `tensor states` states.shape==(batch_size, state_dim, )
        return `tensor a_ints` a_ints.shape==(batch_size, )
        """
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = torch.randint(self.action_dim, size=state.shape[0])  # choosing action randomly
        else:
            action = self.act(state.to(self.device))
            a_int = action.argmax(dim=1)
        return a_int.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma) -> list:
        traj = list()
        state = self.states[0]
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()  # isinstance(action, int)
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + 1)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s
        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state, traj_other), ]
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def explore_vec_env(self, env, target_step, reward_scale, gamma) -> list:
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat((ten_rewards.unsqueeze(0),
                                    ten_dones.unsqueeze(0),
                                    ten_actions.unsqueeze(0)))
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state[:, env_i, :], traj_other[:, env_i, :])
                     for env_i in range(len(self.states))]
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()
        return obj_critic, q_value


class AgentDuelDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.ClassCri = QNetDuel

        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.ClassCri = QNetTwin

        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.soft_max = torch.nn.Softmax(dim=1)

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:  # for discrete action space
        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.soft_max(action)
            a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
            # a_int = rd.choice(self.action_dim, prob=a_prob)  # numpy version
        else:
            a_int = action.argmax(dim=1)
        return a_int.detach().cpu()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1


class AgentD3QN(AgentDoubleDQN):  # D3QN: Dueling Double DQN
    def __init__(self):
        super().__init__()
        self.Cri = QNetTwinDuel


'''Actor-Critic Methods (Policy Gradient)'''


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise)

        if if_per_or_gae:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = torch.as_tensor(self.ou_noise(), dtype=torch.float32, device=self.device).unsqueeze(0)
            action = (action + ou_noise).clamp(-1, 1)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()

        td_error = (q_label - q_value.detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentTD3(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = CriticTwin
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """Prioritized Experience Replay

        Contributor: Github GyChou
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q2).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = 0.995 * self.obj_critic + 0.0025 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha, self.alpha_log)

            '''objective of actor'''
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2).detach()
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + logprob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            self.soft_update(self.act_target, self.act, soft_update_tau)
        return self.obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = ((self.criterion(q1, q_label) + self.criterion(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q2).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self):
        super().__init__()
        self.if_use_act_target = True
        self.if_use_cri_target = True
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = 0.995 * self.obj_critic + 0.0025 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha, self.alpha_log)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri.get_q1_q2(state, a_noise_pg))
                obj_actor = -(q_value_pg + logprob * alpha).mean()
                self.optim_update(self.act_optim, obj_actor, self.act.parameters())
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorPPO
        self.ClassCri = CriticAdv

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        super().init(net_dim=net_dim, gpu_id=gpu_id,
                     state_dim=state_dim, action_dim=action_dim, env_num=env_num,
                     learning_rate=learning_rate, if_per_or_gae=if_per_or_gae)
        self.traj_list = [list() for _ in range(env_num)]
        self.env_num = env_num

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_actions(self, state: torch.Tensor) -> tuple:
        """
        `tensor state` state.shape = (batch_size, state_dim)
        return `tensor action` action.shape = (batch_size, action_dim)
        return `tensor noise` noise.shape = (batch_size, action_dim)
        """
        state = state.to(self.device)
        action, noise = self.act.get_action(state)
        return action.detach().cpu(), noise.detach().cpu()

    def explore_one_env(self, env, target_step, reward_scale, gamma):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_actions, ten_noises = self.select_actions(ten_states)
            action = ten_actions[0].numpy()
            next_s, reward, done, _ = env.step(np.tanh(action))

            traj.append((ten_states, reward, done, ten_actions, ten_noises))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory([traj, ], [last_done, ])
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step, reward_scale, gamma):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_actions, ten_noises = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_actions.tanh())

            for env_i in range(env_num):
                traj_list[env_i].append((ten_states[env_i], ten_rewards[env_i], ten_dones[env_i],
                                         ten_actions[env_i], ten_noises[env_i]))
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            # (ten_state, ten_reward, ten_mask, ten_action, ten_noise) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - (buf_mask * buf_value[:, 0])  # buf_advantage_value
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * (pre_adv_v - ten_value[i])  # fix a bug here
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, traj_list, last_done_list):
        for env_i in range(self.env_num):
            last_done = last_done_list[env_i]
            traj_temp = traj_list[env_i]

            traj_list[env_i] = self.traj_list[env_i] + traj_temp[:last_done + 1]
            self.traj_list[env_i] = traj_temp[last_done:]
        return traj_list

    def convert_trajectory(self, traj_list, reward_scale, gamma):
        # for traj in traj_list:  # todo one env
        #     temp = list(map(list, zip(*traj)))  # 2D-list transpose
        #
        #     ten_state = torch.cat(temp[0])
        #     ten_reward = torch.as_tensor(temp[1], dtype=torch.float32) * reward_scale
        #     ten_mask = (1.0 - torch.as_tensor(temp[2], dtype=torch.float32)) * gamma
        #     ten_action = torch.cat(temp[3])
        #     ten_noise = torch.cat(temp[4])
        #
        #     traj[:] = (ten_state, ten_reward, ten_mask, ten_action, ten_noise)
        #
        #     print(';;2')
        #     print(';;', [ten.shape for ten in traj])
        #     print(';;')
        for traj in traj_list:
            temp = list(map(list, zip(*traj)))  # 2D-list transpose

            ten_state = torch.stack(temp[0])
            ten_reward = torch.as_tensor(temp[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[2], dtype=torch.float32)) * gamma
            ten_action = torch.stack(temp[3])
            ten_noise = torch.stack(temp[4])

            traj[:] = (ten_state, ten_reward, ten_mask, ten_action, ten_noise)

            # print(';;2')
            # # print(';;', [ten.shape for ten in traj])
            # print(';;', len(temp), len(temp[0]), temp[0][0].shape)
            # print(';;', type(temp), type(temp[0]), type(temp[0][0]))
            # print(';;')
        return traj_list


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorDiscretePPO

    def explore_one_env(self, env, target_step, reward_scale, gamma):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            a_int = ten_a_ints[0].numpy()
            next_s, reward, done, _ = env.step(a_int)  # only different

            traj.append((ten_states, reward, done, ten_a_ints, ten_probs))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory([traj, ], [last_done, ])
        return self.convert_trajectory(traj_list, reward_scale, gamma)

    def explore_vec_env(self, env, target_step, reward_scale, gamma):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_a_ints.numpy())

            for env_i in range(env_num):
                traj_list[env_i].append((ten_states[env_i], ten_rewards[env_i], ten_dones[env_i],
                                         ten_a_ints[env_i], ten_probs[env_i]))
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list, reward_scale, gamma)  # [traj_env_0, ...]


'''Actor-Critic Methods (Parameter Sharing)'''


class AgentSharedAC(AgentBase):  # IAC (InterAC) waiting for check
    def __init__(self):
        super().__init__()
        self.ClassCri = SharedDPG  # self.Act = None

        self.explore_noise = 0.2  # standard deviation of explore noise
        self.policy_noise = 0.4  # standard deviation of policy noise
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update
        self.avg_loss_c = (-np.log(0.5)) ** 0.5  # old version reliable_lambda

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        reliable_lambda = None
        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        for i in range(int(buffer.now_len / batch_size * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_state = buffer.sample_batch(batch_size_)

                next_q_label, next_action = self.cri_target.next_q_action(state, next_state, self.policy_noise)
                q_label = reward + mask * next_q_label

            """obj_critic"""
            q_eval = self.cri.critic(state, action)
            obj_critic = self.criterion(q_eval, q_label)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * obj_critic.item() / 2  # soft update, twin critics
            reliable_lambda = np.exp(-self.avg_loss_c ** 2)

            '''actor correction term'''
            actor_term = self.criterion(self.cri(next_state), next_action)

            if i % repeat_times == 0:
                '''actor obj'''
                action_pg = self.cri(state)  # policy gradient
                obj_actor = -self.cri_target.critic(state, action_pg).mean()  # policy gradient
                # NOTICE! It is very important to use act_target.critic here instead act.critic
                # Or you can use act.critic.deepcopy(). Whatever you cannot use act.critic directly.

                obj_united = obj_critic + actor_term * (1 - reliable_lambda) + obj_actor * (reliable_lambda * 0.5)
            else:
                obj_united = obj_critic + actor_term * (1 - reliable_lambda)

            """united loss"""
            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            if i % self.update_freq == self.update_freq and reliable_lambda > 0.1:
                self.cri_target.load_state_dict(self.cri.state_dict())  # Hard Target Update

        return obj_critic.item(), obj_actor.item(), reliable_lambda


class AgentSharedSAC(AgentSAC):  # Integrated Soft Actor-Critic
    def __init__(self):
        super().__init__()
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.cri_optim = None

        self.target_entropy = None
        self.alpha_log = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.target_entropy = np.log(action_dim)
        self.act = self.cri = SharedSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.cri_optim = torch.optim.Adam(
            [{'params': self.act.enc_s.parameters(), 'lr': learning_rate * 0.9},  # more stable
             {'params': self.act.enc_a.parameters(), },
             {'params': self.act.net.parameters(), 'lr': learning_rate * 0.9},
             {'params': self.act.dec_a.parameters(), },
             {'params': self.act.dec_d.parameters(), },
             {'params': self.act.dec_q1.parameters(), },
             {'params': self.act.dec_q2.parameters(), },
             {'params': (self.alpha_log,)}], lr=learning_rate)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:  # 1111
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            alpha = self.alpha_log.exp()

            '''objective of critic'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = 0.995 * self.obj_critic + 0.0025 * obj_critic.item()  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda

            '''objective of alpha (temperature parameter automatic adjustment)'''
            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach() * reliable_lambda).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.act_target.get_q1_q2(state, a_noise_pg)).mean()  # twin critics
                obj_actor = -(q_value_pg + logprob * alpha.detach()).mean()  # policy gradient

                obj_united = obj_critic + obj_alpha + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic + obj_alpha

            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()


class AgentSharedPPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if if_per_or_gae:
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.act = self.cri = SharedPPO(state_dim, action_dim, net_dim).to(self.device)

        self.cri_optim = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': learning_rate * 0.9},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()

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

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / torch.std(buf_adv_v) + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]  # advantage value
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            obj_united = obj_critic + obj_actor
            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple


'''Utils'''


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
