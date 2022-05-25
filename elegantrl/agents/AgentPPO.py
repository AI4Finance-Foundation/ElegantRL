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


class AgentPPOHtermK(AgentPPO):
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

            if i % self.h_term_update_gap == 0:
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


class AgentPPOHtermKV2(AgentPPO):
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
            self.get_buf_h_term_k_v2(buf_state, buf_action, buf_mask, buf_reward, buf_r_sum)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            if_update_actor = bool(i % self.act_update_gap == 0)
            obj_critic_h_term, obj_hamilton = self.get_obj_c_obj_h_term_v2(if_update_actor)  # todo H-term

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

    def get_buf_h_term_k_v2(
            self, buf_state: Tensor, buf_action: Tensor, buf_mask: Tensor, buf_reward: Tensor, buf_r_sum: Tensor,
    ):
        buf_dones = torch.where(buf_mask == 0)[0].detach().cpu() + 1
        i = 0
        for j in buf_dones:
            rewards = buf_reward[i:j]
            states = buf_state[i:j]
            actions = buf_action[i:j]
            masks = buf_mask[i:j]
            v_sum = buf_r_sum[i:j]

            r_sum = rewards.sum().item()
            r_min = rewards.min().item()
            r_max = rewards.max().item()

            self.h_term_buffer.append((states, actions, rewards, masks, r_min, r_max, r_sum, v_sum))
            i = j

        q_arg_sort = np.argsort([item[6] for item in self.h_term_buffer])
        h_term_throw = max(0, int(len(self.h_term_buffer) * self.h_term_drop_rate))
        self.h_term_buffer = [self.h_term_buffer[i] for i in q_arg_sort[h_term_throw:]]

        '''update h-term buffer (states, actions, rewards_sum_norm)'''
        self.ten_state = torch.vstack([item[0] for item in self.h_term_buffer])  # ten_state.shape == (-1, state_dim)
        self.ten_action = torch.vstack([item[1] for item in self.h_term_buffer])  # ten_action.shape == (-1, action_dim)
        self.ten_mask = torch.vstack([item[3] for item in self.h_term_buffer])  # ten_mask.shape == (-1, action_dim)
        self.ten_v_sum = torch.hstack([item[7] for item in self.h_term_buffer])  # ten_q_sum.shape == (-1, action_dim)

        r_min = np.min(np.array([item[4] for item in self.h_term_buffer]))
        r_max = np.max(np.array([item[5] for item in self.h_term_buffer]))
        ten_reward = torch.vstack([item[2] for item in self.h_term_buffer])  # ten_r_sum.shape == (-1, )
        self.ten_r_norm = (ten_reward - r_min) / (r_max - r_min)  # ten_r_norm.shape == (-1, )

    def get_obj_c_obj_h_term_v2(self, if_update_act=True):
        if self.ten_state is None or self.ten_state.shape[0] < 2 ** 12:
            return torch.zeros(1, dtype=torch.float32, device=self.device)
        total_size = self.ten_state.shape[0]

        '''rd sample'''
        k0 = self.h_term_k_step
        indices = torch.randint(k0, total_size, size=(self.batch_size,), requires_grad=False, device=self.device)

        obj_critic = 0.0
        hamilton = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        for k1 in range(k0 - 1, -1, -1):
            '''hamilton (K-spin, K=k)'''
            indices_k = indices - k1

            ten_state = self.ten_state[indices_k]
            ten_v_sum = self.ten_v_sum[indices_k]

            ten_value = self.cri(ten_state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = obj_critic + self.criterion(ten_value, ten_v_sum)

            if if_update_act:
                ten_action = self.ten_action[indices_k]
                ten_mask = self.ten_mask[indices_k]
                logprob = self.act.get_logprob(ten_state, ten_action).clamp(-20, 2)
                hamilton = logprob.sum(dim=1).exp() + hamilton * ten_mask

        # '''critic'''
        # ten_state = self.ten_state[indices]
        # ten_r_sum = self.ten_reward[indices]
        # ten_value = self.cri(ten_state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        #
        # obj_critic = self.criterion(ten_value, ten_r_sum)
        #
        # if if_update_act:
        #     ten_r_norm = self.ten_r_norm[indices]
        #     ten_action = self.ten_action[indices]
        #
        #     '''hamilton'''
        #     ten_logprob = self.act.get_logprob(ten_state, ten_action)
        #     ten_hamilton = ten_logprob.exp().prod(dim=1)
        #     obj_hamilton = -(ten_hamilton * ten_r_norm).mean() * self.h_term_lambda
        # else:
        #     obj_hamilton = torch.zeros(1, dtype=torch.float32, device=self.device)
        if if_update_act:
            ten_r_norm = self.ten_r_norm[indices]
            obj_hamilton = -(hamilton * ten_r_norm).mean() * (self.h_term_lambda / k0 * total_size / self.batch_size)
        else:
            obj_hamilton = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        return obj_critic, obj_hamilton


class AgentPPOgetObjHterm(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def get_buf_h_term_k(
            self, buf_state: Tensor, buf_action: Tensor, buf_mask: Tensor, buf_reward: Tensor
    ):
        buf_dones = torch.where(buf_mask == 0)[0].detach().cpu() + 1
        i = 0
        for j in buf_dones:
            rewards = buf_reward[i:j].to(torch.float16)
            states = buf_state[i:j].to(torch.float16)
            actions = buf_action[i:j].to(torch.float16)
            masks = buf_mask[i:j].to(torch.float16)

            r_sum = rewards.sum().item()
            r_min = rewards.min().item()
            r_max = rewards.max().item()

            self.h_term_buffer.append([states, actions, rewards, masks, r_min, r_max, r_sum])
            i = j

        # # throw low r_sum
        # q_arg_sort = np.argsort([item[6] for item in self.h_term_buffer])
        # h_term_throw = max(0, int(len(self.h_term_buffer) * self.h_term_drop_rate))
        # self.h_term_buffer = [self.h_term_buffer[i] for i in q_arg_sort[h_term_throw:]]

        '''update h-term buffer (states, actions, rewards_sum_norm)'''
        self.ten_state = torch.vstack([item[0].to(torch.float32) for item in self.h_term_buffer])
        self.ten_action = torch.vstack([item[1].to(torch.float32) for item in self.h_term_buffer])
        self.ten_mask = torch.vstack([item[3].to(torch.float32) for item in self.h_term_buffer]
                                     ).squeeze(1) * self.h_term_gamma

        # r_min = np.min(np.array([item[4] for item in self.h_term_buffer]))
        # r_max = np.max(np.array([item[5] for item in self.h_term_buffer]))
        # ten_reward = torch.vstack([item[2].to(torch.float32) for item in self.h_term_buffer])
        # self.ten_r_norm = (ten_reward - r_min) / (r_max - r_min)  # ten_r_norm.shape == (-1, )
        self.ten_reward = torch.vstack([item[2].to(torch.float32) for item in self.h_term_buffer]).squeeze(1)

    def get_obj_h_term_k(self) -> Tensor:
        # '''rd sample'''
        k0 = self.h_term_k_step
        h_term_batch_size = self.batch_size // k0
        # indices = torch.randint(k0, self.ten_state.shape[0], size=(h_term_batch_size,),
        #                         requires_grad=False, device=self.device)

        all_size = self.ten_state.shape[0]
        obj_hamilton = torch.zeros((all_size,), dtype=torch.float32, device=self.device)
        for j0 in range(0, all_size, h_term_batch_size):
            j1 = min(j0 + h_term_batch_size, all_size)
            indices = torch.arange(j0, j1, device=self.device)

            '''hamilton (K-spin, K=k)'''
            hamilton = torch.zeros((j1 - j0,), dtype=torch.float32, device=self.device)
            for k1 in range(k0 - 1, -1, -1):
                indices_k = indices - k1

                ten_state = self.ten_state[indices_k]
                ten_action = self.ten_action[indices_k]
                ten_mask = self.ten_mask[indices_k]

                logprob = self.act.get_logprob(ten_state, ten_action)
                hamilton = logprob.sum(dim=1).exp() + hamilton * ten_mask
                # hamilton = (hamilton + ten_mask) * logprob.sum(dim=1).exp()
            ten_reward = self.ten_reward[indices]

            obj_hamilton[j0:j1] = hamilton * ten_reward
        # return -(hamilton.clamp(-16, 2) * ten_r_norm).mean() * self.h_term_lambda
        return obj_hamilton

    def update_net(self, buffer: ReplayBufferList):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            # buf_len = buf_state.shape[0]
            #
            # '''get buf_r_sum, buf_logprob'''
            # batch_size = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            # buf_value = [self.cri_target(buf_state[i:i + batch_size]) for i in range(0, buf_len, batch_size)]
            # buf_value = torch.cat(buf_value, dim=0)
            # buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)
            #
            # buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            # buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term_k(buf_state, buf_action, buf_mask, buf_reward)  # todo H-term
            del buf_noise

        # '''update network'''
        # obj_critic = torch.zeros(1)
        # obj_actor = torch.zeros(1)
        # assert buf_len >= self.batch_size
        # for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
        #     indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)
        #
        #     state = buf_state[indices]
        #     r_sum = buf_r_sum[indices]
        #
        #     value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        #
        #     obj_critic = self.criterion(value, r_sum)
        #     self.optimizer_update(self.cri_optimizer, obj_critic)
        #     if self.if_cri_target:
        #         self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        #
        #     if i % self.h_term_update_gap == 0:
        #         '''PPO: Surrogate objective of Trust Region'''
        #         adv_v = buf_adv_v[indices]
        #         action = buf_action[indices]
        #         logprob = buf_logprob[indices]
        #
        #         new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
        #         ratio = (new_logprob - logprob.detach()).exp()
        #         surrogate1 = adv_v * ratio
        #         surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        #         obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
        #         obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + self.get_obj_h_term_k()  # todo H-term
        #         self.optimizer_update(self.act_optimizer, obj_actor)
        #
        # action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        # return obj_critic.item(), -obj_actor.item(), action_std_log.item()  # logging_tuple
        obj_hamilton = self.get_obj_h_term_k()
        return obj_hamilton.mean().item()
