import torch

from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.net import ActorDiscretePPO, CriticPPO
from elegantrl.train.replay_buffer import ReplayBufferList

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentA2C(AgentPPO):  # A2C.2015, PPO.2016
    """
    A2C algorithm. “Asynchronous Methods for Deep Reinforcement Learning”. Mnih V. et al.. 2016.

    :param net_dim: the dimension of networks (the width of neural networks)
    :param state_dim: the dimension of state (the number of state vector)
    :param action_dim: the dimension of action (the number of discrete action)
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param args: the arguments for agent training. `args = Arguments()`
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
        print(
            "| AgentA2C: A2C is worse than PPO. We provide AgentA2C code just for teaching."
            "| Without TrustRegion, A2C needs special hyper-parameters, such as smaller repeat_times."
        )

    def update_net(self, buffer: ReplayBufferList):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        '''update network'''
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for _ in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            # logprob = buf_logprob[indices]

            """A2C: Advantage function"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            obj_actor = (-(adv_v * new_logprob.exp()).mean() + obj_entropy * self.lambda_entropy)
            self.optimizer_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple


class AgentDiscreteA2C(AgentA2C):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorDiscretePPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
