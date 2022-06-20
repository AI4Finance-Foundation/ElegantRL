import numpy as np
import torch
from elegantrl.agents.net import ActorPPO, ActorDiscretePPO, CriticPPO, SharePPO
from elegantrl.agents.AgentBase import AgentBase
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import Tensor
from typing import List, Tuple


class AgentPPO(AgentBase):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, 'act_class', ActorPPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.get_reward_sum = self.get_reward_sum_gae
        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_gae_adv', 0.95)  # could be 0.50~0.99, GAE (ICLR.2016.)
        self.act_update_gap = getattr(args, 'act_update_gap', 1)

    def explore_one_env(self, env, horizon_len: int) -> list:
        traj_list = list()
        last_dones = [0, ]
        state = self.state[0]

        i = 0
        done = False
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        while i < horizon_len or not done:
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            actions, noises = [item.cpu() for item in get_action(state.to(self.device))]  # different
            next_state, reward, done, _ = env.step(convert(actions)[0].numpy())

            traj_list.append((state, reward, done, actions, noises))  # different

            i += 1
            state = env.reset() if done else next_state
        self.state[0] = state
        last_dones[0] = i
        return self.convert_trajectory(traj_list, last_dones)  # traj_list

    def explore_vec_env(self, env, horizon_len: int, random_exploration: bool) -> list:
        obs = torch.zeros((horizon_len, self.env_num) + (self.state_dim,)).to(self.device)
        actions = torch.zeros((horizon_len, self.env_num) + (self.action_dim,)).to(self.device)
        noises = torch.zeros((horizon_len, self.env_num) + (self.action_dim,)).to(self.device)
        rewards = torch.zeros((horizon_len, self.env_num)).to(self.device)
        dones = torch.zeros((horizon_len, self.env_num)).to(self.device)

        state = self.state if self.if_use_old_traj else env.reset()
        done = torch.zeros(self.env_num).to(self.device)

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            obs[i] = state
            dones[i] = done

            action, noise = get_action(state)
            next_state, reward, done, _ = env.step(convert(action))
            state = next_state

            actions[i] = action
            noises[i] = noise
            rewards[i] = reward

            self.current_rewards += reward
            self.current_lengths += 1
            env_done_indices = torch.where(done == 1)
            self.reward_tracker.update(self.current_rewards[env_done_indices])
            self.step_tracker.update(self.current_lengths[env_done_indices])
            not_dones = 1.0 - done.float()
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

        self.state = state

        return (obs, actions, noises, self.reward_scale * rewards, dones, state, done), horizon_len * self.env_num

    def update_net(self, buffer):
        buf_state, buf_action, buf_logprob, buf_adv, buf_r_sum = self.get_reward_sum(buffer)
        buffer_size = buf_state.size()[0]
        assert buffer_size >= self.batch_size

        '''update network'''
        obj_critic_list = list()
        obj_actor_list = list()
        indices = np.arange(buffer_size)
        for epoch in range(self.repeat_times):
            np.random.shuffle(indices)

            for i in range(0, buffer_size, self.batch_size):
                minibatch_indices = indices[i:i + self.batch_size]
                state = buf_state[minibatch_indices]
                r_sum = buf_r_sum[minibatch_indices]
                adv_v = buf_adv[minibatch_indices]
                adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)
                action = buf_action[minibatch_indices]
                logprob = buf_logprob[minibatch_indices]

                value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                obj_critic = self.criterion(value, r_sum) * self.lambda_critic
                self.optimizer_update(self.cri_optimizer, obj_critic)
                if self.if_cri_target:
                    self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

                '''PPO: Surrogate objective of Trust Region'''
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.optimizer_update(self.act_optimizer, obj_actor)

                obj_critic_list.append(obj_critic.item())
                obj_actor_list.append(-obj_actor.item())

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return np.array(obj_critic_list).mean(), np.array(obj_actor_list).mean(), action_std_log.item()  # logging_tuple

    def get_reward_sum_gae(self, buffer) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.
        """
        with torch.no_grad():
            buf_state, buf_action, buf_noise, buf_reward, buf_done, next_state, next_done = buffer
            next_state_value = self.cri(next_state)

            buf_adv = torch.zeros_like(buf_reward).to(self.device)
            values = torch.zeros_like(buf_reward).to(self.device)

            lastgaelam = 0
            horizon_len = buf_state.size()[0]
            for t in reversed(range(horizon_len)):
                values[t] = self.cri(buf_state[t]).reshape(-1, )
                if t == horizon_len - 1:
                    nextnonterminal = 1.0 - next_done
                    next_values = next_state_value
                else:
                    nextnonterminal = 1.0 - buf_done[t + 1]
                    next_values = values[t + 1]
                    delta = buf_reward[t] + self.gamma * next_values * nextnonterminal - values[t]
                    buf_adv[t] = lastgaelam = delta + self.gamma * self.lambda_gae_adv * nextnonterminal * lastgaelam
            buf_r_sum = buf_adv + values

            buf_state = buf_state.reshape((-1,) + (self.state_dim,))
            buf_action = buf_action.reshape((-1,) + (self.action_dim,))
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise.reshape((-1,) + (self.action_dim,)))
            buf_logprob = buf_logprob.reshape(-1, )
            buf_adv = buf_adv.reshape(-1, )
            buf_r_sum = buf_r_sum.reshape(-1, )

        return buf_state, buf_action, buf_logprob, buf_adv, buf_r_sum


class AgentDiscretePPO(AgentPPO):
    """
    Bases: ``AgentPPO``

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(
        self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        self.act_class = getattr(self, "act_class", ActorDiscretePPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


# FIXME: this class is incomplete
class AgentSharePPO(AgentPPO):
    def __init__(self):
        AgentPPO.__init__(self)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if if_per_or_gae:
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.act = self.cri = SharePPO(state_dim, action_dim, net_dim).to(self.device)

        self.cri_optim = torch.optim.Adam(
            [
                {"params": self.act.enc_s.parameters(), "lr": learning_rate * 0.9},
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.a_std_log,
                },
                {
                    "params": self.act.dec_q1.parameters(),
                },
                {
                    "params": self.act.dec_q2.parameters(),
                },
            ],
            lr=learning_rate,
        )
        self.criterion = torch.nn.SmoothL1Loss()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [
                ten.to(self.device) for ten in buffer
            ]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            """get buf_r_sum, buf_logprob"""
            bs = 2**10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                self.lambda_a_value / torch.std(buf_adv_v) + 1e-5
            )
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(
                buf_len, size=(batch_size,), requires_grad=False, device=self.device
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]  # advantage value
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            # it is obj_actor  # todo net.py sharePPO
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            obj_united = obj_critic + obj_actor
            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple
