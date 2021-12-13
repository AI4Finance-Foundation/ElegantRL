import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorSAC, CriticTwin, ShareSPG, CriticMultiple


class AgentSAC(AgentBase):  # [ElegantRL.2021.11.11]
    """
    Bases: ``AgentBase``
    
    Soft Actor-Critic algorithm. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. Tuomas Haarnoja et al.. 2018.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing. 
        """
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )

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
        """
        Select actions given an array of states.
        
        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            obj_actor = -(self.cri(state, action_pg) + logprob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentModSAC(AgentSAC):  # [ElegantRL.2021.11.11]
    """
    Bases: ``AgentSAC``
    
    Modified SAC with introducing of reliable_lambda, to realize “Delayed” Policy Updates.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self):
        AgentSAC.__init__(self)
        self.ClassCri = CriticMultiple  # REDQ ensemble (parameter sharing)
        # self.ClassCri = CriticEnsemble  # REDQ ensemble  # todo ensemble
        self.if_use_cri_target = True
        self.if_use_act_target = True

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = 0.995 * self.obj_critic + 0.005 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean()  # todo ensemble
                self.optim_update(self.act_optim, obj_actor)
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True)[0]  # multiple critics

            # todo ensemble
            q_label = (reward + mask * (next_q + next_log_prob * alpha))
            q_labels = q_label * torch.ones((1, self.cri.q_values_num), dtype=torch.float32, device=self.device)
        q_values = self.cri.get_q_values(state, action)  # todo ensemble

        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True)[0]  # multiple critics

            q_label = (reward + mask * (next_q + next_log_prob * alpha))
            q_labels = q_label * torch.ones((1, self.cri.q_values_num), dtype=torch.float32, device=self.device)
        q_values = self.cri.get_q_values(state, action)

        # obj_critic = self.criterion(q_values, q_labels)
        td_error = self.criterion(q_values, q_labels).mean(dim=1, keepdim=True)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentShareSAC(AgentSAC):  # Integrated Soft Actor-Critic
    def __init__(self):
        AgentSAC.__init__(self)
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.cri_optim = None

        self.target_entropy = None
        self.alpha_log = None

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing. 
        """
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.target_entropy = np.log(action_dim)
        self.act = self.cri = ShareSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.cri_optim = torch.optim.Adam(
            [{'params': self.act.enc_s.parameters(), 'lr': learning_rate * 1.5},
             {'params': self.act.enc_a.parameters(), },
             {'params': self.act.net.parameters(), 'lr': learning_rate * 1.5},
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

            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()


class DiscreteSAC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = DiscreteCriSAC
        self.ClassAct = DiscreteActSAC
        self.train_reward = []
        self.if_use_cri_target = True
        self.if_use_act_target = False
        self.trajectory_list = []
        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.train_iteraion = 0

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )

        #self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
        #                              requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_log = torch.zeros(1,dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)
        self.alpha = alpha = self.alpha_log.cpu().exp().item()
        self.trajectory_list = list()
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs], buf_action[i:i+bs,None]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        alpha = None
        for _ in range(int(buf_len * repeat_times / batch_size)):
            indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)
            state = buf_state[indices]
            next_s = buf_state[indices+1]
            action = buf_action[indices]
            reward = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            mask = buf_mask[indices]
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                _,next_a, next_log_prob = self.act_target.get_action(next_s)  # stochastic policy
                next_q = (next_a * (torch.min(*self.cri_target.get_q1_q2(next_s)) - alpha * next_log_prob
        )).sum(dim=1)  # twin critics
            q_label = reward + mask * next_q
            q1, q2 = self.cri.get_q1_q2(state)
            q1 = q1.gather(1, action.unsqueeze(-1)).squeeze(-1)
            q2 = q2.gather(1, action.unsqueeze(-1)).squeeze(-1)
            obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            _,action_pg, log_prob = self.act.get_action(state)  # policy gradient
            obj_alpha = (alpha * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            obj_actor = (action_pg * (-(torch.min(*self.cri.get_q1_q2(state)) - log_prob * alpha.detach()))).sum(dim=1).mean()
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

            if self.train_iteraion%main_args.update_epoch == 0:
                self.explore_env_traj(self.env, 1000, 1000)
                H = 0.
                for k in range(K):
                    if len(self.traj_s[k])>main_args.num_sample:
                        index = np.random.choice(len(self.traj_s[k]),replace=False,size=main_args.num_sample)
                    else:
                        index = np.arange(len(self.traj_s[k]))
                    s = np.array(self.traj_s[k])[index]
                    a = np.array(self.traj_a[k])[index]
                    r = np.array(self.traj_r[k])[index]
                    x = np.arange(len(index))
                    s_vec = torch.from_numpy(np.array(s).reshape(-1,self.state_dim)).to(self.device)
                    a_vec = torch.from_numpy(np.array(a).reshape(-1)).to(self.device).long()
                    r = torch.from_numpy(r).to(self.device)
                    p = self.act.soft_max(self.act.net(s_vec))[np.arange(len(a_vec)), a_vec].reshape(-1, k+1)
                    traj_p = torch.prod(p, dim=1)
                    traj_r = torch.sum(r, dim=1)
                    # print('here',r.shape,traj_r.shape)
                    H -= torch.sum(traj_p * traj_r)
                self.optim_update(self.act_optim_H, H)

            if self.train_iteraion%5==0:
                ep = 0.
                for eval_i in range(num_eval):
                    episode_reward, _, = get_episode_return_and_step(self.env, self.act, self.device)
                    ep += episode_reward 
                self.train_reward.append([ep/num_eval,self.train_iteraion])
                print(self.train_iteraion,ep/num_eval)
            self.train_iteraion += 1
            
        return obj_critic, obj_actor.item(), alpha.item()
    
    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
