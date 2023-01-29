import torch
from typing import Tuple

from elegantrl.train.config import Config
from elegantrl.agents.AgentPPO import AgentPPO, AgentDiscretePPO
from elegantrl.agents.net import ActorDiscretePPO


class AgentA2C(AgentPPO):
    """
    A2C algorithm. “Asynchronous Methods for Deep Reinforcement Learning”. Mnih V. et al.. 2016.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

    def update_net(self, buffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]
            buffer_num = states.shape[1]

            '''get advantages and reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' to avoiding out of GPU memory.
            values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for i in range(0, buffer_size, bs):
                for j in range(buffer_num):
                    values[i:i + bs, j] = self.cri(states[i:i + bs, j])

            advantages = self.get_advantages(rewards, undones, values)  # shape == (buffer_size, buffer_num)
            reward_sums = advantages + values  # shape == (buffer_size, buffer_num)
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        # assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, buffer_num)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        sample_len = buffer_size - 1

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            ids = torch.randint(sample_len * buffer_num, size=(self.batch_size,), requires_grad=False)
            ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
            ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

            state = states[ids0, ids1]
            action = actions[ids0, ids1]
            # logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            obj_actor = (advantage * new_logprob).mean()  # obj_actor without Trust Region
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()


class AgentDiscreteA2C(AgentDiscretePPO):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", ActorDiscretePPO)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
