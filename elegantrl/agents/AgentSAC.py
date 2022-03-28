import numpy as np
import torch
from copy import deepcopy
from elegantrl.agents.net import ActorSAC, ActorFixSAC, CriticTwin, ShareSPG
from elegantrl.agents.AgentBase import AgentBase


class AgentSAC(AgentBase):
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

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", ActorSAC)
        self.cri_class = getattr(self, "cri_class", CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)

    def update_net(self, buffer):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = obj_actor = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            """objective of alpha (temperature parameter automatic adjustment)"""
            a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log * (log_prob - self.target_entropy).detach()
            ).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)

            """objective of actor"""
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

            q_value_pg = self.cri(state, a_noise_pg)
            obj_actor = -(q_value_pg + log_prob * alpha).mean()
            self.optimizer_update(self.act_optimizer, obj_actor)
            # self.soft_update(self.act_target, self.act, self.soft_update_tau) # SAC don't use act_target network

        return (
            obj_critic.item(),
            -obj_actor.item(),
            self.alpha_log.exp().detach().item(),
        )

    def get_obj_critic_raw(self, buffer, batch_size):
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
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
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
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentModSAC(
    AgentSAC
):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    """
    Bases: ``AgentSAC``

    Modified SAC with introducing of reliable_lambda, to realize “Delayed” Policy Updates.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", ActorFixSAC)
        self.cri_class = getattr(self, "cri_class", CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

        self.lambda_a_log_std = getattr(args, "lambda_a_log_std", 2**-4)

    def update_net(self, buffer):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        with torch.no_grad():  # H term
            # buf_state = buffer.sample_batch_r_m_a_s()[3]
            if buffer.prev_idx <= buffer.next_idx:
                buf_state = buffer.buf_state[buffer.prev_idx : buffer.next_idx]
            else:
                buf_state = torch.vstack(
                    (
                        buffer.buf_state[buffer.prev_idx :],
                        buffer.buf_state[: buffer.next_idx],
                    )
                )
            buffer.prev_idx = buffer.next_idx

            avg_a_log_std = self.act.get_a_log_std(buf_state).mean(dim=0, keepdim=True)
            avg_a_log_std = avg_a_log_std * torch.ones(
                (self.batch_size, 1), device=self.device
            )
            del buf_state

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        obj_actor = torch.zeros(1)
        for update_c in range(
            1, int(2 + buffer.now_len * self.repeat_times / self.batch_size)
        ):
            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = (
                0.995 * self.obj_c + 0.005 * obj_critic.item()
            )  # for reliable_lambda

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            """objective of alpha (temperature parameter automatic adjustment)"""
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            reliable_lambda = np.exp(-self.obj_c**2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                obj_a_std = (
                    self.criterion(self.act.get_a_log_std(state), avg_a_log_std)
                    * self.lambda_a_log_std
                )

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean() + obj_a_std

                self.optimizer_update(self.act_optimizer, obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return self.obj_c, -obj_actor.item(), alpha.item()


# FIXME: this class is incomplete
class AgentShareSAC(AgentSAC):  # Integrated Soft Actor-Critic
    def __init__(self):
        AgentSAC.__init__(self)
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.cri_optim = None

        self.target_entropy = None
        self.alpha_log = None

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
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.target_entropy = np.log(action_dim)
        self.act = self.cri = ShareSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.cri_optim = torch.optim.Adam(
            [
                {"params": self.act.enc_s.parameters(), "lr": learning_rate * 1.5},
                {
                    "params": self.act.enc_a.parameters(),
                },
                {"params": self.act.net.parameters(), "lr": learning_rate * 1.5},
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.dec_d.parameters(),
                },
                {
                    "params": self.act.dec_q1.parameters(),
                },
                {
                    "params": self.act.dec_q2.parameters(),
                },
                {"params": (self.alpha_log,)},
            ],
            lr=learning_rate,
        )

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> tuple:  # 1111
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            alpha = self.alpha_log.exp()

            """objective of critic"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.0025 * obj_critic.item()
            )  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda

            """objective of alpha (temperature parameter automatic adjustment)"""
            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log
                * (logprob - self.target_entropy).detach()
                * reliable_lambda
            ).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(
                    *self.act_target.get_q1_q2(state, a_noise_pg)
                ).mean()  # twin critics
                obj_actor = -(
                    q_value_pg + logprob * alpha.detach()
                ).mean()  # policy gradient

                obj_united = obj_critic + obj_alpha + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic + obj_alpha

            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()
