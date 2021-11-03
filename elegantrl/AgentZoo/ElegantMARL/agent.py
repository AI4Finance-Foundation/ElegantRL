import os
import torch
import numpy as np
import numpy.random as rd

from copy import deepcopy
from net import Actor
from net import Critic

"""[ElegantRL.2021.09.09](https://github.com/AI4Finance-LLC/ElegantRL)"""


class AgentBase:
    def __init__(self):
        self.states = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = False
        self.explore_rate = 1.0
        self.explore_noise = None
        self.traj_list = None  # trajectory_list
        # self.amp_scale = None  # automatic mixed precision

        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4,marl=False, n_agents = 1,
             if_per_or_gae=False,  env_num=1, agent_id=0):
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
        self.device = torch.device(f"cuda:{agent_id}" if (torch.cuda.is_available() and (agent_id >= 0)) else "cpu")
        #assert 0
        if not marl:
            self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        else:
            self.cri = self.ClassCri(int(net_dim * 1.25), state_dim * n_agents, action_dim * n_agents).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

        if env_num > 1:  # VectorEnv
            self.explore_env = self.explore_vec_env
        else:
            self.explore_env = self.explore_one_env

    def select_actions(self, states) -> np.ndarray:
        """Select continuous actions for exploration

        `array states` states.shape==(batch_size, state_dim, )
        return `array actions` actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = (actions + torch.randn_like(actions) * self.explore_noise).clamp(-1, 1)
        return actions.detach().cpu().numpy()

    def explore_one_env(self, env, target_step):
        """actor explores in one env, then returns the traj (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        traj = list()
        state = self.states[0]
        for _ in range(target_step):
            action = self.select_actions((state,))[0]
            next_s, reward, done, _ = env.step(action)
            traj.append((state, (reward, done, *action)))

            state = env.reset() if done else next_s
        self.states[0] = state

        traj_list = [traj, ]
        return traj_list  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step):
        """actor explores in VectorEnv, then returns the trajectory (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        env_num = len(self.traj_list)
        states = self.states

        traj_list = [list() for _ in range(env_num)]
        for _ in range(target_step):
            actions = self.select_actions(states)
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_state, reward, done = s_r_d_list[env_i]
                traj_list[env_i].append(
                    (states[env_i], (reward, done, *actions[env_i]))
                )
                next_states.append(next_state)
            states = next_states

        self.states = states
        return traj_list  # (traj_env_0, ..., traj_env_i)

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

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
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


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4,marl=False, n_agents=1, if_use_per=False, env_num=1,  agent_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate,marl, n_agents, if_use_per, env_num, agent_id)
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise)
        self.loss_td = torch.nn.MSELoss()
        if if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_use_per else 'mean')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none' if if_use_per else 'mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, states) -> np.ndarray:
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states).detach().cpu().numpy()
        
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = self.ou_noise()
            actions = (actions + ou_noise[np.newaxis]).clip(-1, 1)
        
        return actions[0]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

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


class AgentMADDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True
        
    def init(self,net_dim, state_dim, action_dim, learning_rate=1e-4,marl=True, n_agents = 1,   if_use_per=False, env_num=1, agent_id=0):
        self.agents = [AgentDDPG() for i in range(n_agents)]
        self.explore_env = self.explore_one_env
        self.if_on_policy = False
        self.n_agents = n_agents
        for i in range(self.n_agents):
            self.agents[i].init(net_dim, state_dim, action_dim, learning_rate=1e-4,marl=True, n_agents = self.n_agents,   if_use_per=False, env_num=1, agent_id=0)
        self.n_states = state_dim
        self.n_actions = action_dim
        
        self.batch_size = net_dim
        self.gamma = 0.95
        self.update_tau = 0
        self.device = torch.device(f"cuda:{agent_id}" if (torch.cuda.is_available() and (agent_id >= 0)) else "cpu")

        
    def update_agent(self, rewards, dones, actions, observations, next_obs, index):
        #rewards, dones, actions, observations, next_obs = buffer.sample_batch(self.batch_size)
        curr_agent = self.agents[index]
        curr_agent.cri_optim.zero_grad()
        all_target_actions = []
        for i in range(self.n_agents):
            if i == index:
                all_target_actions.append(curr_agent.act_target(next_obs[:, index]))
            if i != index:
                action = self.agents[i].act_target(next_obs[:, i])
                all_target_actions.append(action)
        action_target_all = torch.cat(all_target_actions, dim = 1).to(self.device).reshape(actions.shape[0], actions.shape[1] *actions.shape[2])
        
        target_value = rewards[:, index] + self.gamma * curr_agent.cri_target(next_obs.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]), action_target_all).detach().squeeze(dim = 1)
        #vf_in = torch.cat((observations.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]), actions.reshape(actions.shape[0], actions.shape[1],actions.shape[2])), dim = 2)
        actual_value = curr_agent.cri(observations.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]), actions.reshape(actions.shape[0], actions.shape[1]*actions.shape[2])).squeeze(dim = 1)
        vf_loss = curr_agent.loss_td(actual_value, target_value.detach())
        
        
        #vf_loss.backward()
        #curr_agent.cri_optim.step()

        curr_agent.act_optim.zero_grad()
        curr_pol_out = curr_agent.act(observations[:, index])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(0, self.n_agents):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(actions[:, i])
        #vf_in = torch.cat((observations, torch.cat(all_pol_acs, dim = 0).to(self.device).reshape(actions.size()[0], actions.size()[1], actions.size()[2])), dim = 2)

        pol_loss = -torch.mean(curr_agent.cri(observations.reshape(observations.shape[0], observations.shape[1]*observations.shape[2]), torch.cat(all_pol_acs, dim = 1).to(self.device).reshape(actions.shape[0], actions.shape[1] *actions.shape[2])))
        
        curr_agent.act_optim.zero_grad()
        pol_loss.backward()
        curr_agent.act_optim.step()     
        curr_agent.cri_optim.zero_grad()
        vf_loss.backward()
        curr_agent.cri_optim.step()


    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()
        self.batch_size = batch_size
        self.update_tau = soft_update_tau
        self.update(buffer)
        self.update_all_agents()
        return 

    def update(self, buffer):
        rewards, dones, actions, observations, next_obs = buffer.sample_batch(self.batch_size)
        for index in range(self.n_agents):
            self.update_agent(rewards, dones, actions, observations, next_obs, index)

    def update_all_agents(self):
        for agent in self.agents:
            self.soft_update(agent.cri_target, agent.cri, self.update_tau)
            self.soft_update(agent.act_target, agent.act, self.update_tau)
    
    def explore_one_env(self, env, target_step) -> list:
        traj_temp = list()
        k = 0
        for _ in range(target_step):
            k = k + 1
            actions = []
            for i in range(self.n_agents):
                action = self.agents[i].select_actions(self.states[i])
                actions.append(action)
            #print(actions)
            next_s, reward, done, _ = env.step(actions)
            traj_temp.append((self.states, reward, done, actions))
            global_done = True
            for i in range(self.n_agents):
                if global_done is not True:
                    global_done = False
                    break
            if global_done or k >100:
                state = env.reset() 
                k = 0
            else: 
                state = next_s
        self.states = state
        traj_list = traj_temp
        return traj_list
    
    def select_actions(self, states):
        actions = []
        for i in range(self.n_agents):
            action = self.agents[i].select_actions((states[i]))
            actions.append(action)
        return actions

    def save_or_load_agent(self, cwd, if_save):
        for i in range(self.n_agents):
            self.agents[i].save_or_load_agent(cwd+'/'+str(i),if_save)
    def load_actor(self, cwd):
        for i in range(self.n_agents):
            self.agents[i].act.load_state_dict(torch.load(cwd+'/actor'+str(i) + '.pth', map_location ='cpu'))



class OrnsteinUhlenbeckNoise:  # NOT suggest to use it
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
