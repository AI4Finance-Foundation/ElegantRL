from torch import nn

from rlsolver.methods.eco_s2v.jumanji.train.config import *
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN
from .AgentBase import AgentBase

TEN = th.Tensor


class AgentPPO(AgentBase):
    """PPO algorithm + GAE
    “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    “Generalized Advantage Estimation”. John Schulman. et al..
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorPPO().to(self.device)
        self.cri = CriticPPO().to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)
        ######
        self.n_nodes = getattr(args, "num_nodes", 0.25)
        ########
        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.001)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)

    def _explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, logprobs, rewards, undones, unmasks)` for on-policy
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `logprobs.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
        states = th.zeros((horizon_len, self.num_envs, self.n_nodes + 7, self.n_nodes), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs), dtype=th.long).to(self.device)
        logprobs = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        #################################
        state = env.get_observation()  # shape == (num_envs, state_dim) for a vectorized env.
        ##########################
        for t in range(horizon_len):
            action, logprob = self.explore_action(state)

            states[t] = state
            actions[t] = action
            logprobs[t] = logprob
            ###########
            state, reward, terminal = env.step(action)  # next_state
            if terminal[0]:
                # print(th.mean(env.best_score))
                state = env.reset()
            ##########
            rewards[t] = reward
            terminals[t] = terminal

        self.last_state = state
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        return states, actions, logprobs, rewards, undones, unmasks

    def inference(self, env, max_steps):
        state = env.reset()
        for i in range(max_steps):
            action, logprob = self.explore_action(state)
            state = env.step(action)[0]
        return env.best_score

    def explore_action(self, state: TEN) -> tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state)
        return actions, logprobs

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        sample_len = states.shape[0]
        num_seqs = states.shape[1]
        ids = th.randint(sample_len * num_seqs, size=(self.batch_size,), requires_grad=False, device=self.device)
        ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        state = states[ids0, ids1]
        action = actions[ids0, ids1]
        unmask = unmasks[ids0, ids1]
        logprob = logprobs[ids0, ids1]
        advantage = advantages[ids0, ids1]
        reward_sum = reward_sums[ids0, ids1]

        value = self.cri(state.clone()).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        new_logprob, entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()

        # surrogate1 = advantage * ratio
        # surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        # surrogate = th.min(surrogate1, surrogate2)  # save as below
        surrogate = advantage * ratio * th.where(advantage.gt(0), 1 - self.ratio_clip, 1 + self.ratio_clip)

        obj_surrogate = (surrogate * unmask).mean()  # major actor objective
        obj_entropy = (entropy * unmask).mean()  # minor actor objective
        obj_actor_full = obj_surrogate - obj_entropy * self.lambda_entropy
        self.optimizer_backward(self.act_optimizer, -obj_actor_full)
        return obj_critic.item(), obj_surrogate.item(), obj_entropy.item()

    def get_advantages(self, states: TEN, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value
        #########
        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = self.last_state
        next_value = self.cri(next_state.clone()).detach().squeeze(-1)

        advantage = th.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        if self.if_use_v_trace:  # get advantage value in reverse time series (V-trace)
            for t in range(horizon_len - 1, -1, -1):
                next_value = rewards[t] + masks[t] * next_value
                advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
                next_value = values[t]
        else:  # get advantage value using the estimated value of critic network
            for t in range(horizon_len - 1, -1, -1):
                advantages[t] = rewards[t] - values[t] + masks[t] * advantage
                advantage = values[t] + self.lambda_gae_adv * advantages[t]
        return advantages

    def update_avg_std_for_normalization(self, states: TEN):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = (self.act.state_std * (1 - tau) + state_std * tau).clamp_min(1e-4)
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        self.act_target.state_avg[:] = self.act.state_avg
        self.act_target.state_std[:] = self.act.state_std
        self.cri_target.state_avg[:] = self.cri.state_avg
        self.cri_target.state_std[:] = self.cri.state_std


class AgentA2C(AgentPPO):
    """A2C algorithm.
    “Asynchronous Methods for Deep Reinforcement Learning”. 2016.
    """

    def update_net(self, buffer) -> tuple[float, float, float]:
        buffer_size = buffer[0].shape[0]

        '''get advantages reward_sums'''
        with th.no_grad():
            states, actions, logprobs, rewards, undones, unmasks = buffer
            ##################
            values = [self.cri(states[i].clone()) for i in range(0, buffer_size)]
            values = th.stack(values, dim=0)  # values.shape == (buffer_size, )
            #############
            advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
            assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])
        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        '''update network'''
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        for update_t in range(buffer_size):
            #######
            obj_critic, obj_actor = self.update_objectives(buffer, update_t)
            ########
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg, 0

    def update_objectives(self, buffer: tuple[TEN, ...], update_t: int) -> tuple[float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer
        ########
        buffer_size = states.shape[0]
        state = states[update_t]
        action = actions[update_t]
        unmask = unmasks[update_t]
        # logprob = logprobs[indices]
        advantage = advantages[update_t]
        reward_sum = reward_sums[update_t]
        ########
        value = self.cri(state.clone())  # critic network predicts the reward_sum (Q value) of state
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
        obj_actor = (advantage * new_logprob).mean()  # obj_actor without policy gradient clip
        self.optimizer_backward(self.act_optimizer, -obj_actor)
        return obj_critic.item(), obj_actor.item()

    ########
    def save(self, path):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if os.path.splitext(path)[-1] == '':
            path += '.pth'
        th.save(self.act.state_dict(), path)


###########
class ActorPPO(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MPNN(n_obs_in=7,
                        n_layers=3,
                        n_features=64,
                        n_hid_readout=[],
                        tied_weights=False)

        self.ActionDist = th.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, state: TEN) -> TEN:
        a_prob = th.softmax(self.net(state.clone()), dim=-1)  # action_prob without softmax
        return a_prob

    def get_action(self, state: TEN) -> (TEN, TEN):
        a_prob = th.softmax(self.net(state), dim=-1)
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> (TEN, TEN):
        a_prob = self.soft_max(self.net(state.clone()))  # action.shape == (batch_size, 1), action.dtype = th.int
        dist = self.ActionDist(a_prob)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy


class CriticPPO(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MPNN(n_obs_in=7,
                        n_layers=3,
                        n_features=64,
                        n_hid_readout=[],
                        tied_weights=False)

    def forward(self, state: TEN) -> TEN:
        value = th.mean(self.net(state), dim=-1)
        return value

        ##########
