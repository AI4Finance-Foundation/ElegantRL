import torch
import numpy as np
import numpy.random as rd
from elegantrl.agents import AgentBase,AgentDDPG
from elegantrl.agents.net import Actor, Critic, CriticTwin

class AgentMADDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = CriticTwin
        self.if_use_cri_target = True
        self.if_use_act_target = True
        
    def init(self,net_dim, state_dim, action_dim, learning_rate=1e-4,marl=True, n_agents = 1,   if_use_per=False, env_num=1, agent_id=0):
        self.agents = [AgentDDPG() for i in range(n_agents)]
        self.explore_env = self.explore_one_env
        self.if_off_policy = True
        self.n_agents = n_agents
        for i in range(self.n_agents):
            self.agents[i].cri = CriticTwin
            self.agents[i].init(net_dim, state_dim, action_dim, learning_rate=1e-4,marl=True, n_agents = self.n_agents,   if_use_per=False, env_num=1, agent_id=0)
        self.n_states = state_dim
        self.n_actions = action_dim
        
        self.batch_size = net_dim
        self.gamma = 0.95
        self.update_tau = 0
        self.device = torch.device(f"cuda:{agent_id}" if (torch.cuda.is_available() and (agent_id >= 0)) else "cpu")

        
    def update_agent(self, rewards, dones, actions, observations, next_obs, index):
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
        actual_value = curr_agent.cri(observations.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]), actions.reshape(actions.shape[0], actions.shape[1]*actions.shape[2])).squeeze(dim = 1)
        vf_loss = curr_agent.loss_td(actual_value, target_value.detach())
        curr_agent.act_optim.zero_grad()
        curr_pol_out = curr_agent.act(observations[:, index])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(0, self.n_agents):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(actions[:, i])

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
            next_s, reward, done, _ = env.step(actions)
            traj_temp.append((self.states, reward, done, actions))
            global_done = True
            for i in range(self.n_agents):
                if done[i] is not True:
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
