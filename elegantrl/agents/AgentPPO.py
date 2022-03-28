import numpy as np
import torch
from elegantrl.agents.net import ActorPPO, ActorDiscretePPO, CriticPPO, SharePPO
from elegantrl.agents.AgentBase import AgentBase
from typing import Tuple

"""[ElegantRL.2021.12.12](github.com/AI4Fiance-Foundation/ElegantRL)"""


class AgentPPO(AgentBase):
    """
    Bases: ``AgentBase``

    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

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
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        self.if_cri_target = getattr(args, "if_cri_target", False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(
            args, "ratio_clip", 0.25
        )  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(
            args, "lambda_entropy", 0.02
        )  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(
            args, "lambda_entropy", 0.98
        )  # could be 0.95~0.99, GAE (ICLR.2016.)

        if getattr(
            args, "if_use_gae", False
        ):  # GAE (Generalized Advantage Estimation) for sparse reward
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        traj_list = list()
        last_done = [
            0,
        ]
        state = self.states[0]

        step_i = 0
        done = False
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [
                ten.cpu() for ten in get_action(ten_s.to(self.device))
            ]  # different
            next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a, ten_n))  # different

            step_i += 1
            state = env.reset() if done else next_s
        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(get_a_to_e(ten_a))

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n)
            )  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer):
        """
        Update the neural networks by sampling batch data from `ReplayBuffer`.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]
            buf_len = buf_state.shape[0]

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
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        """update network"""
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for _ in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(
                buf_len,
                size=(self.batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(
        self, buf_len, buf_reward, buf_mask, buf_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(
        self, buf_len, ten_reward, ten_mask, ten_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param ten_reward: a list of rewards for the state-action pairs.
        :param ten_mask: a list of masks computed by the product of done signal and discount factor.
        :param ten_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v - ten_value[i]
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
            # ten_mask[i] * pre_adv_v == (1-done) * gamma * pre_adv_v
        return buf_r_sum, buf_adv_v


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
