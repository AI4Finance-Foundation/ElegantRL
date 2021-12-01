import os
import numpy.random as rd
from copy import deepcopy
from elegantrl_helloworld.net import *


class AgentBase:
    def __init__(self):
        """initialize
        
        replace by different DRL algorithms
        """
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        """initialize the self.object in `__init__()`
        
        replace by different DRL algorithms

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        # explict call self.init() for multiprocessing
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        """Given a state, select action for exploration
        
        :param state: states.shape==(batch_size, state_dim, )
        :return: actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()

    def explore_env(self, env, target_step) -> list:
        """actor explores in single Env, then returns the trajectory (env transitions) for ReplayBuffer
        
        :param env: RL training environment. env.reset() env.step()
        :param target_step: explored target_step number of step in env
        :return: the trajectory list with length target_step
        """
        state = self.state

        trajectory_list = list()
        for _ in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            trajectory_list.append((state, (reward, done, *action)))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    @staticmethod
    def optim_update(optimizer, objective):
        """minimize the optimization objective via update the network parameters
        
        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update target network via current network
        
        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load training files for Agent
        
        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
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


class AgentDQN(AgentBase):
    """
    Bases: ``AgentBase``
    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.
    """
    def __init__(self):
        """
        call the __init__ function from AgentBase and set specific self.object
        """
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.if_use_cri_target = True
        self.ClassCri = QNet

    def select_action(self, state) -> int:  # for discrete action space
        """
        Select discrete actions given an array of states.
        .. note::
            Using ϵ-greedy to uniformly random select actions for randomness.
        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def explore_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a environment instance.
        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        state = self.state

        trajectory_list = list()
        for _ in range(target_step):
            action = self.select_action(state)  # assert isinstance(action, int)
            next_s, reward, done, _ = env.step(action)
            trajectory_list.append((state, (reward, done, action)))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.softMax = torch.nn.Softmax(dim=1)
        self.ClassCri = QNetTwin

    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.softMax(actions)[0].detach().cpu().numpy()
            a_int = rd.choice(self.action_dim, p=a_prob)  # choose action according to Q value
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # explore noise of action
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = Critic
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state


class AgentTD3(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticTwin
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state


class AgentSAC(AgentBase):
    """
    Bases: ``AgentBase``
    
    Soft Actor-Critic algorithm. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. Tuomas Haarnoja et al.. 2018.
    """
    def __init__(self):
        """
        call the __init__ function from AgentBase and set specific self.object
        """
        super().__init__()
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_use_per=False, gpu_id=0, env_num=1):
        """
        :param net_dim[int]: the dimension of networks (the width of neural networks)
        :param state_dim[int]: the dimension of state (the number of state vector)
        :param action_dim[int]: the dimension of action (the number of discrete action)
        :param learning_rate[float]: learning rate of optimizer
        :param if_use_or_per[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
        :param gpu_id[int]: which specific gpu to use
        :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        """
        super().init(net_dim, state_dim, action_dim, learning_rate, _if_use_per, gpu_id)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

    def select_action(self, state):
        """
        Select actions given a state, or an array of states.
        
        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act.get_action(states)
        return actions.detach().cpu().numpy()[0]

    def explore_env(self, env, target_step):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.
        
        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)

            trajectory.append((state, (reward, done, *action)))
            state = env.reset() if done else next_state
        self.state = state
        return trajectory

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        .. note::
            Using advantage normalization and entropy loss.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        alpha = self.alpha_log.exp().detach()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a, next_log_prob = self.act_target.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)

            self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item(), obj_actor.item(), alpha.item()


class AgentPPO(AgentBase):
    """
    Bases: ``AgentBase``
    
    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    """
    def __init__(self):
        """
        call the __init__ function from AgentBase and set specific self.object
        """
        super().__init__()
        self.ClassCri = CriticAdv
        self.ClassAct = ActorPPO

        self.if_off_policy = False
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0, env_num=1):
        """
        :param net_dim[int]: the dimension of networks (the width of neural networks)
        :param state_dim[int]: the dimension of state (the number of state vector)
        :param action_dim[int]: the dimension of action (the number of discrete action)
        :param learning_rate[float]: learning rate of optimizer
        :param if_use_or_per[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
        :param gpu_id[int]: which specific gpu to use
        :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        """
        super().init(net_dim, state_dim, action_dim, learning_rate, if_use_gae, gpu_id)
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def select_action(self, state):
        """
        Select actions given a state, or an array of states.
        
        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.
        
        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        state = self.state

        trajectory_temp = list()
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
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        .. note::
            Using advantage normalization and entropy loss.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
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
        """
        Calculate the **reward-to-go** and **advantage estimation**.
        
        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.
        
        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
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


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorDiscretePPO

    def explore_env(self, env, target_step):
        state = self.state

        trajectory_temp = list()
        last_done = 0
        for i in range(target_step):
            # action, noise = self.select_action(state)
            # next_state, reward, done, _ = env.step(np.tanh(action))
            action, a_prob = self.select_action(state)  # different from `action, noise`
            a_int = int(action)  # different
            next_state, reward, done, _ = env.step(a_int)  # different from `np.tanh(action)`
            trajectory_temp.append((state, reward, done, a_int, a_prob))
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


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

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
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]
            buf_state = np.empty((self.max_len, state_dim), dtype=np.float16)  # sometimes np.uint8
            buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach().cpu().numpy()
            buf_other[0:temp_len] = self.buf_other[self.now_len:self.max_len].detach().cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[:self.now_len].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_other = buf_dict['buf_other']

            buf_state = torch.as_tensor(buf_state, dtype=torch.float32, device=self.device)
            buf_other = torch.as_tensor(buf_other, dtype=torch.float32, device=self.device)
            self.extend_buffer(buf_state, buf_other)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")
