from elegantrl.agents.net import (
    ActorSAC,
    CriticTwin,
)
from elegantrl.agents.AgentBase import AgentBase


class AgentDDPG(AgentBase):
    """
    Bases: ``AgentBase``

    Deep Deterministic Policy Gradient algorithm. “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.

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
        self.act.explore_noise = getattr(
            args, "explore_noise", 0.1
        )  # set for `get_action()`

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
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optimizer_update(self.act_optimizer, obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), -obj_actor.item()
