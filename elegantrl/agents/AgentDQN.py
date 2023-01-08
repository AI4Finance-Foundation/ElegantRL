import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import QNet, QNetDuel
from elegantrl.agents.net import QNetTwin, QNetTwinDuel
from elegantrl.train.config import Config
from elegantrl.train.replay_buffer import ReplayBuffer

'''[ElegantRL.2022.12.12](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDQN(AgentBase):  # [ElegantRL.2022.04.18]
    """
    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = None  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.act.explore_rate = getattr(args, "explore_rate", 0.25)
        # Using ϵ-greedy to select uniformly random actions for exploration with `explore_rate` probability.

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            num_envs == 1
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(1, 1)) if if_random \
                else get_action(state.unsqueeze(0))  # different
            states[t] = state

            ary_action = action[0, 0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            state = torch.as_tensor(env.reset() if done else ary_state, dtype=torch.float32, device=self.device)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, 1), dtype=torch.int32).to(self.device)  # different
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape = (num_envs, state_dim) for a vectorized env.

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.randint(self.action_dim, size=(self.num_envs, 1)) if if_random \
                else get_action(state).detach()  # different
            states[t] = state

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            obj_actors += q_value.mean().item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            state, action, reward, undone, next_s = buffer.sample(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long()).squeeze(1)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            state, action, reward, undone, next_s, is_weights = buffer.sample(batch_size)

            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long()).squeeze(1)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):
    """
    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. H. V. Hasselt et al.. 2015.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            state, action, reward, undone, next_s = buffer.sample(batch_size)

            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q1, q2 = [qs.gather(1, action.long()).squeeze(1) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            state, action, reward, undone, next_s, is_weights = buffer.sample(batch_size)

            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0].squeeze(1)
            q_label = reward + undone * self.gamma * next_q

        q1, q2 = [qs.gather(1, action.long()).squeeze(1) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1


'''add dueling q network'''


class AgentDuelingDQN(AgentDQN):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)


class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", QNetTwinDuel)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
