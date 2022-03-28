import numpy as np
import numpy.random as rd
import torch
from elegantrl.agents.net import ActorSAC, CriticTwin
from elegantrl.agents.AgentBase import AgentBase


class AgentSAC_H(AgentBase):  # [ElegantRL.2021.11.11]
    """
    Bases: ``AgentBase``

    Soft Actor-Critic algorithm. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. Tuomas Haarnoja et al.. 2018.

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
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

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

        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select actions given an array of states.

        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            """objective of alpha (temperature parameter automatic adjustment)"""
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            """objective of actor"""
            obj_h_term = self.get_obj_h_term()

            obj_actor = (
                obj_h_term - (self.cri(state, action_pg) + logprob * alpha).mean()
            )
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    def get_buf_h_term(self, buf_state, buf_action, buf_r_sum):
        buf_r_norm = buf_r_sum - buf_r_sum.mean()
        buf_r_diff = (
            torch.where(buf_r_norm[:-1] * buf_r_norm[1:] <= 0)[0].detach().cpu().numpy()
            + 1
        )
        buf_r_diff = list(buf_r_diff) + [
            buf_r_norm.shape[0],
        ]

        step_i = 0
        min_len = 16
        positive_list = list()
        for step_j in buf_r_diff:
            if buf_r_norm[step_i] > 0 and step_i + min_len < step_j:
                positive_list.append((step_i, step_j))
            step_i = step_j

        for step_i, step_j in positive_list:
            index = np.arange(step_i, step_j)

            ten_state = buf_state[index]
            ten_action = buf_action[index]
            ten_r_sum = buf_r_sum[index]

            q_avg = ten_r_sum.mean().item()
            q_min = ten_r_sum.min().item()
            q_max = ten_r_sum.max().item()

            self.h_term_buffer.append(
                (ten_state, ten_action, ten_r_sum, q_avg, q_min, q_max)
            )

        q_arg_sort = np.argsort([item[3] for item in self.h_term_buffer])
        self.h_term_buffer = [
            self.h_term_buffer[i]
            for i in q_arg_sort[max(0, len(self.h_term_buffer) // 4 - 1) :]
        ]

        q_min = np.min(np.array([item[4] for item in self.h_term_buffer]))
        q_max = np.max(np.array([item[5] for item in self.h_term_buffer]))
        self.h_term_r_min_max = (q_min, q_max)

    def get_obj_h_term(self):
        list_len = len(self.h_term_buffer)
        rd_list = rd.choice(list_len, replace=False, size=max(2, list_len // 2))

        ten_state = list()
        ten_action = list()
        ten_r_sum = list()
        for i in rd_list:
            ten_state.append(self.h_term_buffer[i][0])
            ten_action.append(self.h_term_buffer[i][1])
            ten_r_sum.append(self.h_term_buffer[i][2])
        ten_state = torch.vstack(ten_state)  # ten_state.shape == (-1, state_dim)
        ten_action = torch.vstack(ten_action)  # ten_action.shape == (-1, action_dim)
        ten_r_sum = torch.hstack(ten_r_sum)  # ten_r_sum.shape == (-1, )

        """rd sample"""
        ten_size = ten_state.shape[0]
        indices = torch.randint(
            ten_size, size=(ten_size // 2,), requires_grad=False, device=self.device
        )
        ten_state = ten_state[indices]
        ten_action = ten_action[indices]
        ten_r_sum = ten_r_sum[indices]

        """hamilton"""
        ten_logprob = self.act.get_logprob(ten_state, ten_action)
        ten_hamilton = ten_logprob.exp().prod(dim=1)

        n_min, n_max = self.h_term_r_min_max
        ten_r_norm = (ten_r_sum - n_min) / (n_max - n_min)
        return -(ten_hamilton * ten_r_norm).mean() * self.lambda_h_term
