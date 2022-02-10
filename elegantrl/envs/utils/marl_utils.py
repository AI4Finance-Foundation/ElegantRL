import logging
import operator
from collections import defaultdict

import torch as th
import torch.nn as nn
from torch.distributions import Categorical


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length, decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (
                (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            )

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))


class LinearIncreaseSchedule:
    def __init__(self, start, finish, time_length):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start - self.delta * T)


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length, decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (
                (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1
            )

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))


class LinearIncreaseSchedule:
    def __init__(self, start, finish, time_length):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start - self.delta * T)


# Directly from OpenAI Baseline implementation (https://github.com/openai/baselines)
class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        if mid + 1 <= start:
            return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self._operation(
                self._reduce_helper(start, mid, 2 * node, node_start, mid),
                self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
            )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx], self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx *= 2
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=min, neutral_element=float("inf"))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super().reduce(start, end)


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info[f"{key}_T"].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info[f"{key}_T"] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = f"{th.mean(th.tensor([float(x[1]) for x in self.stats[k][-window:]])):.4f}"
            log_str += f"{k + ':':<25}{item:>8}"
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
        # Reset stats to avoid accumulating logs in memory
        self.stats = defaultdict(lambda: [])


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger


import time
import numpy as np


def print_time(start_time, T, t_max, episode, episode_rewards):
    time_elapsed = time.time() - start_time
    T = max(1, T)
    time_left = time_elapsed * (t_max - T) / T
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    last_reward = r"N\A"
    if len(episode_rewards) > 5:
        last_reward = f"{np.mean(episode_rewards[-50:]):.2f}"
    print(
        f"[F[F[KEp: {episode:,}, T: {T:,}/{t_max:,}, Reward: {last_reward}, \n[KElapsed: {time_str(time_elapsed)}, Left: {time_str(time_left)}\n",
        " " * 10,
        end="\r",
    )


def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += f"{int(days):d} days, "
    if hours > 0:
        string += f"{int(hours):d} hours, "
    if minutes > 0:
        string += f"{int(minutes):d} minutes, "
    string += f"{int(seconds):d} seconds"
    return string


class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "test_noise", 0.0)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        return (
            pick_random * random_actions
            + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        )


# This multi-agent controller shares parameters between agents


def build_td_lambda_targets(
    rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda
):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
            rewards[:, t]
            + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
        )
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def build_gae_targets(rewards, masks, values, gamma, lambd):
    B, T, A, _ = values.size()
    T -= 1
    advantages = th.zeros(B, T, A, 1).to(device=values.device)
    advantage_t = th.zeros(B, A, 1).to(device=values.device)

    for t in reversed(range(T)):
        delta = rewards[:, t] + values[:, t + 1] * gamma * masks[:, t] - values[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t]
        advantages[:, t] = advantage_t

    returns = values[:, :T] + advantages
    return advantages, returns


def build_q_lambda_targets(
    rewards, terminated, mask, exp_qvals, qvals, gamma, td_lambda
):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = exp_qvals.new_zeros(*exp_qvals.shape)
    ret[:, -1] = exp_qvals[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        reward = rewards[:, t] + exp_qvals[:, t] - qvals[:, t]  # off-policy correction
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
            reward
            + (1 - td_lambda) * gamma * exp_qvals[:, t + 1] * (1 - terminated[:, t])
        )
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def build_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])

    # Tree diagram
    mac = mac[:, :-1]
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = th.cat(((t1 * mac)[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
    return target_q + tree_q_vals


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + "K"


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)
