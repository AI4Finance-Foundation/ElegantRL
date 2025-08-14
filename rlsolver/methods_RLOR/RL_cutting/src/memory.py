import numpy as np

class Memory(object):
    """general memory class that can hold states, actions, rewards, and values
    """

    def __init__(self):
        self.states = []  # each element of [Ab, c0, cuts]
        self.actions = []
        self.rewards = []  # each element contains trajectory of raw rewards
        self.intrinsic_rewards = [] # intrinsic rewards if RND is used
        self.isdone = []
        self.values = []  # discounted reward
        self.intrinsic_values = [] # discounted intrinsic reward
        self.reward_sums = []  # each trajectory's sum of rewards, for plotting
        self.advantages = []

    def clear(self):
        self.__init__()


class TrajMemory(Memory):
    def add_frame(self, condensed_s, a, r):
        self.states.append(condensed_s)
        self.actions.append(a)
        self.rewards.append(r)

class MasterMemory(Memory):
    """inherits a general memory class,
       so we can easily add trajectories to the MasterMemory
    """

    def add_trajectory(self, trajectory_memory):
        self.states.extend(trajectory_memory.states)
        self.actions.extend(trajectory_memory.actions)

        self.rewards.extend(trajectory_memory.rewards)
        self.intrinsic_rewards.extend(trajectory_memory.intrinsic_rewards)

        self.values.extend(trajectory_memory.values)
        self.intrinsic_values.extend(trajectory_memory.intrinsic_values)

        self.reward_sums.extend(trajectory_memory.reward_sums)
