import torch

from elegantrl.agents.AgentPPO import AgentPPO

from elegantrl.agents.net import ActorDiscretePPO

"""vectorized env"""
from elegantrl.train.config import Config
from elegantrl.agents.AgentPPO import AgentVecPPO


class AgentA2C(AgentPPO):  # A2C.2015, PPO.2016
    """
    Bases: ``AgentPPO``

    A2C algorithm. “Asynchronous Methods for Deep Reinforcement Learning”. Mnih V. et al.. 2016.

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
        print(
            "| AgentA2C: A2C or A3C is worse than PPO. We provide AgentA2C code just for teaching."
            "| Without TrustRegion, A2C needs special hyper-parameters, such as smaller repeat_times."
        )

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]

            """get buf_r_sum, buf_logprob"""
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i: i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            # buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                    self.lambda_a_value / (buf_adv_v.std() + 1e-5)
            )
            # buf_adv_v: advantage_value in ReplayBuffer
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for _ in range(1, update_times + 1):
            indices = torch.randint(
                buf_len, size=(batch_size,), requires_grad=False, device=self.device
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            # logprob = buf_logprob[indices]

            """A2C: Advantage function"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            obj_actor = (
                    -(adv_v * new_logprob.exp()).mean() + obj_entropy * self.lambda_entropy
            )
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple


class AgentDiscreteA2C(AgentA2C):
    """
    Bases: ``AgentA2C``

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        AgentA2C.__init__(self)
        self.ClassAct = ActorDiscretePPO

    def explore_one_env(self, env, target_step):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :param reward_scale: a reward scalar to clip the reward.
        :param gamma: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        state = self.states[0]

        last_done = 0
        traj = []
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            a_int = ten_a_ints[0].numpy()
            next_s, reward, done, _ = env.step(a_int)  # only different

            traj.append((ten_states, reward, done, ten_a_ints, ten_probs))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory(
            [
                traj,
            ],
            [
                last_done,
            ],
        )
        return self.convert_trajectory(traj_list)

    def explore_vec_env(self, env, target_step):
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :param reward_scale: a reward scalar to clip the reward.
        :param gamma: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [[] for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_a_ints.numpy())

            for env_i in range(env_num):
                traj_list[env_i].append(
                    (
                        ten_states[env_i],
                        ten_rewards[env_i],
                        ten_dones[env_i],
                        ten_a_ints[env_i],
                        ten_probs[env_i],
                    )
                )
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]


"""vectorized env"""


class AgentVecA2C(AgentVecPPO):  # A2C.2015, PPO.2016
    """
    A2C algorithm. “Asynchronous Methods for Deep Reinforcement Learning”. Mnih V. et al.. 2016.

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        print("| AgentA2C: A2C or A3C is worse than PPO in any case. We provide AgentA2C code just for teaching.\n"
              "| Without TrustRegion, A2C needs special hyper-parameters, such as smaller repeat_times.")

    def update_net(self, buffer) -> [float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]
            buffer_num = states.shape[1]

            '''get advantages and reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' to avoiding out of GPU memory.
            values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for i in range(0, buffer_size, bs):
                for j in range(buffer_num):
                    values[i:i + bs, j] = self.cri(states[i:i + bs, j]).squeeze(1)

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

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            obj_actor = (advantage * new_logprob).mean()  # obj_actor without Trust Region
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()


class AgentDiscreteVecA2C(AgentVecA2C):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", ActorDiscretePPO)
        self.ClassAct = ActorDiscretePPO
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
