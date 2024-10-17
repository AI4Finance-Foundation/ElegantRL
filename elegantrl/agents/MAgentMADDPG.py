import torch

from .AgentBase import AgentBase
from .AgentTD3 import AgentDDPG
from .AgentTD3 import Actor, Critic


class AgentMADDPG(AgentBase):
    """
    Bases: ``AgentBase``

    Multi-Agent DDPG algorithm. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive”. R Lowe. et al.. 2017.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param gamma[float]: learning rate of optimizer
    :param n_agents[int]: number of agents
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param num_envs[int]: the env number of VectorEnv. num_envs == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

    def init(
            self,
            net_dim,
            state_dim,
            action_dim,
            learning_rate=1e-4,
            gamma=0.95,
            n_agents=1,
            if_use_per=False,
            env_num=1,
            agent_id=0,
    ):
        self.agents = [AgentDDPG() for i in range(n_agents)]
        self.explore_env = self._explore_one_env
        self.if_off_policy = True
        self.n_agents = n_agents

        for i in range(self.n_agents):
            self.agents[i].init(
                net_dim,
                state_dim,
                action_dim,
                learning_rate=1e-4,
                n_agents=self.n_agents,
                if_use_per=False,
                env_num=1,
                agent_id=0,
            )
        self.n_states = state_dim
        self.n_actions = action_dim

        self.batch_size = net_dim
        self.gamma = gamma
        self.update_tau = 0
        self.device = torch.device(
            f"cuda:{agent_id}"
            if (torch.cuda.is_available() and (agent_id >= 0))
            else "cpu"
        )

    def update_agent(self, rewards, dones, actions, observations, next_obs, index):
        """
        Update the single agent neural networks, called by update_net.

        :param rewards: reward list of the sampled buffer
        :param dones: done list of the sampled buffer
        :param actions: action list of the sampled buffer
        :param observations: observation list of the sampled buffer
        :param next_obs: next_observation list of the sampled buffer
        :param index: ID of the agent
        """
        curr_agent = self.agents[index]
        curr_agent.cri_optim.zero_grad()
        all_target_actions = []
        for i in range(self.n_agents):
            if i == index:
                all_target_actions.append(curr_agent.act_target(next_obs[:, index]))
            if i != index:
                action = self.agents[i].act_target(next_obs[:, i])
                all_target_actions.append(action)
        action_target_all = (
            torch.cat(all_target_actions, dim=1)
            .to(self.device)
            .reshape(actions.shape[0], actions.shape[1] * actions.shape[2])
        )

        target_value = rewards[:, index] + self.gamma * curr_agent.cri_target(
            next_obs.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]),
            action_target_all,
        ).detach().squeeze(dim=1)
        actual_value = curr_agent.cri(
            observations.reshape(
                next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]
            ),
            actions.reshape(actions.shape[0], actions.shape[1] * actions.shape[2]),
        ).squeeze(dim=1)
        vf_loss = curr_agent.loss_td(actual_value, target_value.detach())
        curr_agent.act_optim.zero_grad()
        curr_pol_out = curr_agent.act(observations[:, index])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(self.n_agents):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(actions[:, i])
        pol_loss = -torch.mean(
            curr_agent.cri(
                observations.reshape(
                    observations.shape[0], observations.shape[1] * observations.shape[2]
                ),
                torch.cat(all_pol_acs, dim=1)
                .to(self.device)
                .reshape(actions.shape[0], actions.shape[1] * actions.shape[2]),
            )
        )
        curr_agent.act_optim.zero_grad()
        pol_loss.backward()
        curr_agent.act_optim.step()
        curr_agent.cri_optim.zero_grad()
        vf_loss.backward()
        curr_agent.cri_optim.step()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        """
        buffer.update_now_len()
        self.batch_size = batch_size
        self.update_tau = soft_update_tau
        rewards, dones, actions, observations, next_obs = buffer.sample_batch(
            self.batch_size
        )
        for index in range(self.n_agents):
            self.update_agent(rewards, dones, actions, observations, next_obs, index)

        for agent in self.agents:
            self.soft_update(agent.cri_target, agent.cri, self.update_tau)
            self.soft_update(agent.act_target, agent.act, self.update_tau)

        return

    def _explore_one_env(self, env, target_step) -> list:
        """
        Exploring the environment for target_step.
        param env: the Environment instance to be explored.
        param target_step: target steps to explore.
        """
        traj_temp = []
        k = 0
        for _ in range(target_step):
            k += 1
            actions = []
            for i in range(self.n_agents):
                action = self.agents[i].select_actions(self.states[i])
                actions.append(action)
            # print(actions)
            next_s, reward, done, _ = env.step(actions)
            traj_temp.append((self.states, reward, done, actions))
            global_done = all(done[i] is True for i in range(self.n_agents))
            if global_done or k > 100:
                state = env.reset()
                k = 0
            else:
                state = next_s
        self.states = state
        return traj_temp

    def select_actions(self, states):
        """
        Select continuous actions for exploration

        :param state: states.shape==(n_agents,batch_size, state_dim, )
        :return: actions.shape==(n_agents,batch_size, action_dim, ),  -1 < action < +1
        """
        actions = []
        for i in range(self.n_agents):
            action = self.agents[i].select_actions(states[i])
            actions.append(action)
        return actions

    def save_or_load_agent(self, cwd, if_save):
        """
        save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """
        for i in range(self.n_agents):
            self.agents[i].save_or_load_agent(cwd + "/" + str(i), if_save)
