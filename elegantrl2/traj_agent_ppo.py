import os
import numpy as np
import torch as th
from typing import Optional
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .traj_config import Config
from .traj_buffer import TrajBuffer

TEN = th.Tensor


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_discrete: bool = args.if_discrete
        self.if_off_policy: bool = args.if_off_policy

        self.net_dims = net_dims  # the networks dimension of each layer
        self.state_dim = state_dim  # feature number of state
        self.action_dim = action_dim  # feature number of continuous action or number of discrete action

        self.gamma = args.gamma  # discount factor of future rewards
        self.max_step = args.max_step  # limits the maximum number of steps an agent can take in a trajectory.
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.sample_len = args.sample_len  # length of sequence sampled from replay buf.
        self.sample_num = args.sample_num  # num of sequences sampled from replay buf.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state
        self.buffer_init_size = args.buffer_init_size  # train after samples over buffer_init_size for off-policy

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        self.prev_observ: Optional[TEN] = None  # shape == (num_envs, state_dims)
        self.prev_hidden: Optional[TEN] = None  # shape == (num_envs, state_dims, num_layers)
        self.curr_observ: Optional[TEN] = None  # shape == (num_envs, state_dims)
        self.curr_hidden: Optional[TEN] = None  # shape == (num_envs, state_dims, num_layers)
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        self.act = None
        self.cri = None
        self.act_target = self.act
        self.cri_target = self.cri

        '''optimizer'''
        self.act_optimizer: Optional[th.optim] = None
        self.cri_optimizer: Optional[th.optim] = None
        self.amp_scaler = th.amp.GradScaler()
        self.amp_autocast = th.amp.autocast

        self.criterion = getattr(args, 'criterion', th.nn.MSELoss(reduction="none"))
        self.if_vec_env = self.num_envs > 1  # use vectorized environment (vectorized simulator)
        self.if_use_per = getattr(args, 'if_use_per', None)  # use PER (Prioritized Experience Replay)
        self.lambda_fit_cum_r = getattr(args, 'lambda_fit_cum_r', 0.0)  # critic fits cumulative returns

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    def optimizer_backward(self, optimizer: th.optim, objective: TEN):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = th.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        # optimizer.zero_grad()
        #
        # with self.amp_autocast:
        #     objective = network(input_tensor)
        self.amp_scaler.scale(objective).backward()  # loss.backward()
        self.amp_scaler.unscale_(optimizer)  # amp

        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        self.amp_scaler.step(optimizer)  # optimizer.step()
        self.amp_scaler.update()  # optimizer.step()

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"

            if getattr(self, attr_name) is None:
                continue

            if if_save:
                th.save(getattr(self, attr_name).state_dict(), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, th.load(file_path, map_location=self.device))


class AgentPPO(AgentBase):
    """PPO algorithm + GAE
    “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.
    “Generalized Advantage Estimation”. John Schulman. et al..
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.001)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.if_use_v_trace = getattr(args, 'if_use_v_trace', True)

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> tuple[TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(observs, rewards, undones, unmasks, actions)`
            `observs.shape == (horizon_len, num_envs, state_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
        """
        assert isinstance(if_random, bool)
        observs = th.zeros((horizon_len, self.num_envs, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, self.num_envs), dtype=th.int32).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        observ = self.prev_observ = self.curr_observ  # shape == (num_envs, state_dim)
        hidden = self.prev_hidden = self.curr_hidden  # shape == (num_envs, state_dim, num_layers)

        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            action, hidden = self.act.get_action(observ, hidden)

            observs[t] = observ
            actions[t] = action

            observ, reward, terminal, truncate, _ = env.step(convert(action))  # next_state

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.curr_observ = observ
        self.curr_hidden = hidden

        rewards = rewards.view(horizon_len, self.num_envs, 1) * self.reward_scale
        undones = th.logical_not(terminals.view(horizon_len, self.num_envs, 1))
        unmasks = th.logical_not(truncates.view(horizon_len, self.num_envs, 1))
        return observs, rewards, undones, unmasks, actions

    def explore_action(self, observ: TEN, hidden: TEN) -> tuple[TEN, TEN]:
        actions, hidden = self.act.get_action(observ, hidden)
        return actions, hidden

    def update_net(self, buf: TrajBuffer) -> tuple[float, float, float]:
        buffer_size = buf.cur_size

        '''get advantages reward_sums'''
        th.set_grad_enabled(False)
        sample_len = np.ceil(self.sample_num * self.sample_len / buf.num_seqs)
        num_seqs = buf.num_seqs

        values = th.empty((buffer_size, num_seqs), dtype=th.float32, device=self.device)
        for sample_i in range(0, buffer_size, sample_len):
            seq = buf.sample_seqs(seq_num=self.sample_num, seq_len=sample_len, seq_i=sample_i)
            state = seq[:, :, buf.observ_i:buf.unmask_j]
            value = self.cri(state, self.prev_hidden).mean(dim=2)
            values[sample_i:sample_i + sample_len] = value

        advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
        reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
        del rewards, undones, values

        advantages = (advantages - advantages.mean()) / (advantages[::4, ::4].std() + 1e-5)  # avoid CUDA OOM
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, states.shape[1])

        '''update network'''
        obj_entropies = []
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor, obj_entropy = self.update_objectives(buf, update_t)
            obj_entropies.append(obj_entropy)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_entropy_avg = np.array(obj_entropies).mean() if len(obj_entropies) else 0.0
        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg, obj_entropy_avg

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

        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
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

    def get_advantages(self, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value

        # update undones rewards when truncated
        truncated = th.logical_not(unmasks)
        if th.any(truncated):
            rewards[truncated] += values[truncated]
            undones[truncated] = False

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = self.last_state.clone()
        next_value = self.cri(next_state).detach().squeeze(-1)

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


'''network'''


class ActorPPO(th.nn.Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int):
        super().__init__()
        self.inp_mlp = build_mlp(dims=[state_dim + 3, net_dims[0]], if_raw_out=False)
        self.mid_rnn = nn.GRU(input_size=net_dims[0], hidden_size=net_dims[1], num_layers=2, batch_first=True)
        self.out_mlp = build_mlp(dims=[net_dims[1:], action_dim], if_raw_out=True)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, observ: TEN, hidden: TEN) -> tuple[TEN, TEN]:  # for evaluation
        observ_enc = self.inp_mlp(observ)
        observ_enc, hidden = self.mid_rnn(observ_enc, hidden)
        action_avg = self.out_mlp(observ_enc)
        return action_avg, hidden

    def get_action(self, observ: TEN, hidden: TEN) -> tuple[TEN, TEN]:  # for exploration
        observ_enc = self.inp_mlp(observ)
        observ_enc, hidden = self.mid_rnn(observ_enc, hidden)
        action_avg = self.out_mlp(observ_enc)

        action_std = self.action_std_log.exp()

        action = action_avg + th.randn_like(action_avg) * action_std
        return action, hidden

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.tanh()


class CriticPPO(th.nn.Module):
    def __init__(self, net_dims: list[int], state_dim: int, action_dim: int, num_ensemble: int = 32):
        super().__init__()
        assert isinstance(action_dim, int)
        self.inp_mlp = build_mlp(dims=[state_dim + 3, net_dims[0]], if_raw_out=False)
        self.mid_rnn = nn.GRU(input_size=net_dims[0], hidden_size=net_dims[1], num_layers=2, batch_first=True)
        self.out_mlp = build_mlp(dims=[net_dims[1:], num_ensemble], if_raw_out=True)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, observ: TEN, hidden: TEN) -> tuple[TEN, TEN]:  # for evaluation
        observ_enc = self.inp_mlp(observ)
        observ_enc, hidden = self.mid_rnn(observ_enc, hidden)
        values_avg = self.out_mlp(observ_enc)
        return values_avg, hidden

    def get_values(self, observ: TEN, hidden: TEN) -> tuple[TEN, TEN]:  # for evaluation
        observ_enc = self.inp_mlp(observ)
        observ_enc, hidden = self.mid_rnn(observ_enc, hidden)
        values_avg = self.out_mlp(observ_enc)
        return values_avg, hidden


"""utils"""


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    net_dims: the middle dimension, `net_dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ELU
    net_list = nn.ModuleList()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
