import torch
import numpy as np

from torch import Tensor
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorDiscretePPO
from elegantrl.agents.net import ActorPPO, CriticPPO
from elegantrl.train.replay_buffer import ReplayBufferList
from elegantrl.train.config import Arguments

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentPPO(AgentBase):
    """
    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    :param net_dim: the dimension of networks (the width of neural networks)
    :param state_dim: the dimension of state (the number of state vector)
    :param action_dim: the dimension of action (the number of discrete action)
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param args: the arguments for agent training. `args = Arguments()`
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        self.if_off_policy = False
        self.act_class = getattr(self, 'act_class', ActorPPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        if getattr(args, 'if_use_gae', False):
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_gae_adv', 0.95)  # could be 0.50~0.99, GAE (ICLR.2016.)
        self.act_update_gap = getattr(args, 'act_update_gap', 1)

    def explore_one_env(self, env, target_step: int) -> list:
        traj_list = list()
        last_dones = [0, ]
        state = self.states[0]

        i = 0
        done = False
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        while i < target_step or not done:
            states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            actions, noises = [item.cpu() for item in get_action(states.to(self.device))]  # different
            next_state, reward, done, _ = env.step(convert(actions)[0].numpy())

            traj_list.append((states, reward, done, actions, noises))  # different

            i += 1
            state = env.reset() if done else next_state
        self.states[0] = state
        last_dones[0] = i
        return self.convert_trajectory(traj_list, last_dones)  # traj_list

    def explore_vec_env(self, env, target_step: int) -> list:
        traj_list = list()
        last_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        states = self.states if self.if_use_old_traj else env.reset()

        i = 0
        dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        while i < target_step or not any(dones):
            actions, noises = get_action(states)  # different
            next_states, rewards, dones, _ = env.step(convert(actions))

            traj_list.append((states.clone(), rewards.clone(), dones.clone(), actions, noises))  # different

            i += 1
            last_dones[torch.where(dones)[0]] = i  # behind `i+=1`
            states = next_states

        self.states = states

        if self.if_use_old_traj:
            self.fix_noise_in_old_traj()
        return self.convert_trajectory(traj_list, last_dones)  # traj_list

    def update_net(self, buffer: ReplayBufferList):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [item.to(self.device) for item in buffer]
            buffer_size = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            batch_size = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = torch.cat([self.cri_target(buf_state[i:i + batch_size])
                                   for i in range(0, buf_state.shape[0], batch_size)], dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buffer_size, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        '''update network'''
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        assert buffer_size >= self.batch_size
        for i in range(int(1 + buffer_size * self.repeat_times / self.batch_size)):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) * self.lambda_critic
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if i % self.act_update_gap == 0:
                '''PPO: Surrogate objective of Trust Region'''
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.optimizer_update(self.act_optimizer, obj_actor)

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), action_std_log.item()  # logging_tuple

    def get_reward_sum_raw(
            self, buffer_len: int, rewards: Tensor, masks: Tensor, values: Tensor
    ) -> (Tensor, Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buffer_len: the length of the ``ReplayBuffer``.
        :param rewards: a list of rewards for the state-action pairs.
        :param masks: a list of masks computed by the product of done signal and discount factor.
        :param values: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        cur_r_sum = torch.empty(buffer_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buffer_len - 1, -1, -1):
            cur_r_sum[i] = rewards[i] + masks[i] * pre_r_sum
            pre_r_sum = cur_r_sum[i]
        buf_adv_v = cur_r_sum - values[:, 0]
        return cur_r_sum, buf_adv_v

    def get_reward_sum_gae(
            self, buffer_len: int, rewards: Tensor, masks: Tensor, values: Tensor
    ) -> (Tensor, Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buffer_len: the length of the ``ReplayBuffer``.
        :param rewards: a list of rewards for the state-action pairs.
        :param masks: a list of masks computed by the product of done signal and discount factor.
        :param values: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buffer_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buffer_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buffer_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
            buf_r_sum[i] = rewards[i] + masks[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_adv_v[i] = rewards[i] + masks[i] * pre_adv_v - values[i]
            pre_adv_v = values[i] + buf_adv_v[i] * self.lambda_gae_adv
            # ten_mask[i] * pre_adv_v == (1-done) * gamma * pre_adv_v
        return buf_r_sum, buf_adv_v

    def fix_noise_in_old_traj(self):
        # states, rewards, masks, actions, noises
        batch_size = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
        for i in range(self.env_num):
            if self.traj_list[i][0].shape[0] > 0:
                states = self.traj_list[i][0]
                actions = self.traj_list[i][3]  # action with noise

                # action0s = self.act(states)  # action without noise
                action0s = torch.cat([self.act(states[i:i + batch_size])
                                      for i in range(0, states.shape[0], batch_size)], dim=0)

                self.traj_list[i][4] = actions - action0s  # noises = actions - action0s


class AgentDiscretePPO(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        self.act_class = getattr(self, 'act_class', ActorDiscretePPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


class AgentPPOHterm(AgentPPO):  # HtermPPO2
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def update_net(self, buffer: ReplayBufferList):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            batch_size = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + batch_size]) for i in range(0, buf_len, batch_size)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term(buf_state, buf_action, buf_r_sum, buf_mask, buf_reward)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if i % self.act_update_gap == 0:
                '''PPO: Surrogate objective of Trust Region'''
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + self.get_obj_h_term()  # todo H-term
                self.optimizer_update(self.act_optimizer, obj_actor)

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), action_std_log.item()  # logging_tuple


class AgentPPOHtermV2(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def update_net(self, buffer: ReplayBufferList):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            batch_size = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + batch_size]) for i in range(0, buf_len, batch_size)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term(buf_state, buf_action, buf_r_sum, buf_mask, buf_reward)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            if_update_actor = bool(i % self.act_update_gap == 0)
            obj_critic_h_term, obj_hamilton = self.get_obj_c_obj_h_term(if_update_actor)  # todo H-term

            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_sum) + obj_critic_h_term  # todo H-term
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if if_update_actor:
                '''PPO: Surrogate objective of Trust Region'''
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + obj_hamilton
                self.optimizer_update(self.act_optimizer, obj_actor)

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), action_std_log.item()  # logging_tuple

    def get_obj_c_obj_h_term(self, if_update_act=True):
        if (self.ten_state is None) or (self.ten_state.shape[0] < 1024):
            obj_critic = torch.zeros(1, dtype=torch.float32, device=self.device)
            obj_hamilton = torch.zeros(1, dtype=torch.float32, device=self.device)
            return obj_critic, obj_hamilton

        '''rd sample'''
        total_size = 0 if self.ten_state is None else self.ten_state.shape[0]
        batch_size = int(total_size * self.h_term_sample_rate)
        indices = torch.randint(total_size, size=(batch_size,), requires_grad=False, device=self.device)

        ten_state = self.ten_state[indices]
        ten_r_sum = self.ten_reward[indices]
        ten_r_norm = self.ten_r_norm[indices]
        '''critic'''
        ten_value = self.cri(ten_state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

        obj_critic = self.criterion(ten_value, ten_r_sum)

        if if_update_act:
            ten_action = self.ten_action[indices]

            '''hamilton'''
            ten_logprob = self.act.get_logprob(ten_state, ten_action)
            ten_hamilton = ten_logprob.exp().prod(dim=1)
            obj_hamilton = -(ten_hamilton * ten_r_norm).mean() * self.h_term_lambda
        else:
            obj_hamilton = torch.zeros(1, dtype=torch.float32, device=self.device)
        return obj_critic, obj_hamilton


class AgentPPOHtermK(AgentPPO):  # HtermPPO2
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def update_net(self, buffer: ReplayBufferList):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            batch_size = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + batch_size]) for i in range(0, buf_len, batch_size)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term_k(buf_state, buf_action, buf_mask, buf_reward)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if i % self.act_update_gap == 0:
                '''PPO: Surrogate objective of Trust Region'''
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + self.get_obj_h_term_k()  # todo H-term
                self.optimizer_update(self.act_optimizer, obj_actor)

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), action_std_log.item()  # logging_tuple
