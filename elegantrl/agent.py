import os
import torch
import numpy as np
import numpy.random as rd

from copy import deepcopy
from elegantrl.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from elegantrl.net import Actor, ActorPPO, ActorSAC, ActorDiscretePPO
from elegantrl.net import Critic, CriticPPO, CriticTwin
from elegantrl.net import SharedDPG, SharedSPG, SharedPPO

"""[ElegantRL.2021.09.09](https://github.com/AI4Finance-Foundation/ElegantRL)"""


class AgentBase:
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        """initialize
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.if_off_policy = True

        self.env_num = 1
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        # self.amp_scale = None  # automatic mixed precision

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """initialize the self.object in `__init__()`
        replace by different DRL algorithms
        explict call self.init() for multiprocessing.
        `net_dim` the dimension of networks (the width of neural networks)
        `state_dim` the dimension of state (the number of state vector)
        `action_dim` the dimension of action (the number of discrete action)
        `learning_rate` learning rate of optimizer
        `if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `gpu_id` the gpu_id of the training device. Use CPU when cuda is not available.
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

        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

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
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
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


class AgentDQN(AgentBase):
    """
    Bases: ``elegantrl.agent.AgentBase``
    
    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        self.ClassCri = None  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        self.if_use_dueling = True  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing. 
        """
        self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw
            
    def select_actions(self, states) -> np.ndarray:  # for discrete action space
        """
        Select discrete actions given an array of states.
        
        .. note::
            Using ϵ-greedy to uniformly random select actions for randomness.
        
        :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = torch.randint(self.action_dim, size=state.shape[0])  # choosing action randomly
        else:
            action = self.act(state.to(self.device))
            a_int = action.argmax(dim=1)
        return a_int.detach().cpu()

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.
        
        :param env[object]: the DRL environment instance.
        :param target_step[int]: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
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
    
    def explore_vec_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.
        
        :param env[object]: the DRL environment instance.
        :param target_step[int]: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
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
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()
    
    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value
    
    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value
    
    
class AgentDoubleDQN(AgentDQN):
    """
    Bases: ``elegantrl.agent.AgentDQN``
    
    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. H. V. Hasselt et al.. 2015.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentDQN.__init__(self)
        self.soft_max = torch.nn.Softmax(dim=1)

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing. 
        """
        self.ClassCri = QNetTwinDuel if self.if_use_dueling else QNetTwin
        AgentDQN.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:  # for discrete action space
        """
        Select discrete actions given an array of states.
        
        .. note::
            Using softmax to random select actions with proportional probabilities for randomness.
        
        :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.soft_max(action)
            a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
            # a_int = rd.choice(self.action_dim, prob=a_prob)  # numpy version
        else:
            a_int = action.argmax(dim=1)
        return a_int.detach().cpu()

    def get_obj_critic_raw(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1

    
class AgentDDPG(AgentBase):
    """
    Bases: ``elegantrl.agent.AgentBase``
    
    Deep Deterministic Policy Gradient algorithm. “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing. 
        """
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise)

        if if_per_or_gae:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_per_or_gae else 'mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select actions given an array of states.
        
        .. note::
            Using ϵ-greedy with Ornstein–Uhlenbeck noise to add noise to actions for randomness.
        
        :param states[np.ndarray]: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = torch.as_tensor(self.ou_noise(), dtype=torch.float32, device=self.device).unsqueeze(0)
            action = (action + ou_noise).clamp(-1, 1)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """
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
        """
        Calculate the loss of networks with **uniform sampling**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    
class AgentTD3(AgentBase):
    """
    Bases: ``elegantrl.agent.AgentBase``
    
    Twin Delayed DDPG algorithm. “Addressing Function Approximation Error in Actor-Critic Methods”. Scott Fujimoto. et al.. 2015.
    
    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """
    def __init__(self, _net_dim=256, _state_dim=8, _action_dim=2, _learning_rate=1e-4,
                 _if_per_or_gae=False, _env_num=1, _gpu_id=0):
        AgentBase.__init__(self)
        self.ClassAct = Actor
        self.ClassCri = CriticTwin
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency

    def init(self, net_dim=256, state_dim=8, action_dim=2,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing. 
        """
        AgentBase.init(self, net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, env_num, gpu_id)
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw
            
    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times[float]: the re-using times of each trajectory.
        :param soft_update_tau[float]: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
                self.optim_update(self.act_optim, obj_actor, self.act.parameters())
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state
    
    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.
        
        :param buffer[object]: the ReplayBuffer instance that stores the trajectories.
        :param batch_size[int]: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
    
    
    
