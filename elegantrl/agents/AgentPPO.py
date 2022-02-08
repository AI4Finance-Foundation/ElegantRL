import numpy as np
import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorDiscretePPO, SharePPO
from elegantrl.agents.net import ActorPPO, CriticPPO
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

    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorPPO
        self.ClassCri = CriticPPO

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = (
            None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        )
        self.if_use_old_traj = True
        self.traj_list = None

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
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )

        self.traj_list = [[[] for _ in range(5)] for _ in range(env_num)]

        self.env_num = env_num

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action via a given state.

        :param state: a state in a shape (state_dim, ).
        :return: action [array], action.shape == (action_dim, ) where each action is clipped into range(-1, 1).
        """
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return np.tanh(action)  # the only different

    def explore_one_env(self, env, target_step) -> list:  # 247 second
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        traj_list = []

        state = self.states[0]

        """get traj_list and last_done"""
        step = 0
        done = False
        last_done = 0
        while step < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [
                ten.cpu() for ten in self.act.get_action(ten_s.to(self.device))
            ]
            next_s, reward, done, _ = env.step(np.tanh(ten_a[0].numpy()))

            traj_list.append((ten_s, reward, done, ten_a, ten_n))

            step += 1
            if done:
                state = env.reset()
                last_done = step  # behind `step += 1`
            else:
                state = next_s

        last_done = (last_done,)
        self.states[0] = state
        # assert len(traj_list) == step
        # assert len(traj_list[0]) == 5
        # assert len(traj_list[0][0]) == self.env_num

        """convert traj_list -> buf_srdan"""
        buf_srdan = list(
            map(list, zip(*traj_list))
        )  # srdan: state, reward, done, action, noise
        del traj_list
        # assert len(buf_srdan) == 5
        # assert len(buf_srdan[0]) == step
        # assert len(buf_srdan[0][0]) == self.env_num
        buf_srdan = [
            torch.stack(buf_srdan[0]),
            (torch.tensor(buf_srdan[1], dtype=torch.float32) * self.reward_scale)
            .unsqueeze(0)
            .unsqueeze(1),
            ((1 - torch.tensor(buf_srdan[2], dtype=torch.float32)) * self.gamma)
            .unsqueeze(0)
            .unsqueeze(1),
            torch.stack(buf_srdan[3]),
            torch.stack(buf_srdan[4]),
        ]
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_srdan])
        return self.splice_trajectory(buf_srdan, last_done)

    def explore_vec_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []

        ten_s = self.states

        step = 0
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step < target_step:
            ten_a, ten_n = self.act.get_action(ten_s)
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a.tanh())

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n)
            )

            ten_s = ten_s_next

            step += 1
            last_done[torch.where(ten_dones)[0]] = step  # behind `step+=1`
            # if step % 64 == 0:
            #     print(';;', last_done.detach().cpu().numpy())

        self.states = ten_s
        # assert len(traj_list) == step
        # assert len(traj_list[0]) == 5
        # assert len(traj_list[0][0]) == self.env_num

        buf_srdan = list(map(list, zip(*traj_list)))
        del traj_list
        # assert len(buf_srdan) == 5
        # assert len(buf_srdan[0]) == step
        # assert len(buf_srdan[0][0]) == self.env_num
        buf_srdan[0] = torch.stack(buf_srdan[0])
        buf_srdan[1] = (torch.stack(buf_srdan[1]) * self.reward_scale).unsqueeze(2)
        buf_srdan[2] = ((1 - torch.stack(buf_srdan[2])) * self.gamma).unsqueeze(2)
        buf_srdan[3] = torch.stack(buf_srdan[3])
        buf_srdan[4] = torch.stack(buf_srdan[4])
        # assert all([buf_item.shape[:2] == (step, self.env_num)
        #             for buf_item in buf_srdan])

        return self.splice_trajectory(buf_srdan, last_done)

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
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
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                self.lambda_a_value / (buf_adv_v.std() + 1e-5)
            )
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        obj_critic = None
        obj_actor = None

        assert buf_len >= batch_size
        update_times = int(buf_len / batch_size * repeat_times)
        for _ in range(1, update_times + 1):
            indices = torch.randint(
                buf_len, size=(batch_size,), requires_grad=False, device=self.device
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
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optim_update(self.cri_optim, obj_critic / (r_sum.std() + 1e-6))
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(
        self, buf_len, buf_reward, buf_mask, buf_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimiated by the ``Critic`` network.
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

    def splice_trajectory(self, buf_srdan, last_done):
        out_srdan = []
        for j in range(5):
            cur_items = []
            buf_items = buf_srdan.pop(0)  # buf_srdan[j]

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_items.append(pre_item)

                cur_items.append(buf_items[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_items[last_step:, env_i]

            out_srdan.append(torch.vstack(cur_items))

        # print(';;;3', last_done.sum().item() / self.env_num, out_srdan[0].shape[0] / self.env_num)
        # print(';;;4', out_srdan[1][-4:, -3:])
        return [
            out_srdan,
        ]  # = [states, rewards, dones, actions, noises], len(states) == traj_len


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

    def __init__(self):
        AgentPPO.__init__(self)
        self.ClassAct = ActorDiscretePPO

    def explore_one_env(self, env, target_step):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        state = self.states[0]

        last_done = 0
        traj = []

        step = 0
        done = False
        while step < target_step or not done:
            ten_states = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            ten_actions, ten_noises = self.act.get_action(
                ten_states
            )  # ten_a_ints, ten_probs
            action = ten_actions.cpu().numpy()[0]
            # next_s, reward, done, _ = env.step(np.tanh(action))
            next_s, reward, done, _ = env.step(action)  # only different

            traj.append((ten_states, reward, done, ten_actions, ten_noises))

            if done:
                state = env.reset()
                last_done = step
            else:
                state = next_s

            step += 1

        self.states[0] = state

        traj_list = self.splice_trajectory(
            [
                traj,
            ],
            [
                last_done,
            ],
        )
        return self.convert_trajectory(traj_list)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step):
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        ten_states = self.states
        assert env.device.index == self.device.index

        env_num = len(self.traj_list)
        traj_list = [[] for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        step = 0
        ten_dones = [
            False,
        ] * self.env_num
        while step < target_step and not any(ten_dones):
            ten_actions, ten_noises = self.act.get_action(
                ten_states
            )  # ten_a_ints, ten_probs
            # tem_next_states, ten_rewards, ten_dones, _ = env.step(ten_actions.tanh())
            tem_next_states, ten_rewards, ten_dones, _ = env.step(
                ten_actions
            )  # only different

            for env_i in range(env_num):
                traj_list[env_i].append(
                    (
                        ten_states[env_i],
                        ten_rewards[env_i],
                        ten_dones[env_i],
                        ten_actions[env_i],
                        ten_noises[env_i],
                    )
                )
                if ten_dones[env_i]:
                    last_done_list[env_i] = step
            ten_states = tem_next_states

            step += 1

        self.states = ten_states
        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]


class AgentSharePPO(AgentPPO):  # [plan to]
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
