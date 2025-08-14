from abc import ABC, abstractmethod

import numpy as np
import torch

from rlsolver.methods.eco_s2v.config import *


class SpinSolver(ABC):
    """Abstract base class for agents solving SpinSystem Ising problems."""

    def __init__(self, env, record_cut=False, record_rewards=False, record_qs=False, verbose=False):
        """Base initialisation of a SpinSolver.

        Args:
            env (SpinSystem): The environment (an instance of SpinSystem) with
                which the agent interacts.
            verbose (bool, optional): The logging verbosity.

        Attributes:
            env (SpinSystem): The environment (an instance of SpinSystem) with
                which the agent interacts.
            verbose (bool): The logging verbosity.
            total_reward (float): The cumulative total reward received.
        """

        self.env = env
        self.verbose = verbose
        self.record_cut = record_cut
        self.record_rewards = record_rewards
        self.record_qs = record_qs

        self.total_reward = 0

    def reset(self):
        self.total_reward = 0
        self.env.reset()

    def solve(self, *args):
        """Solve the SpinSystem by flipping individual spins until termination.

        Args:
            *args: The arguments passed through to the 'step' method to take the
                next action.  The implementation of 'step' depedens on the
                solver instance used.

        Returns:
            (float): The cumulative total reward received.

        """

        done = False
        while not done:
            reward, done = self.step(*args)
            self.total_reward += reward
        return self.total_reward

    @abstractmethod
    def step(self, *args):
        """Take the next step (flip the next spin).

        The implementation of 'step' depedens on the
                solver instance used.

        Args:
            *args: The arguments passed through to the 'step' method to take the
                next action.  The implementation of 'step' depedens on the
                solver instance used.

        Raises:
            NotImplementedError: Every subclass of SpinSolver must implement the
                step method.
        """

        raise NotImplementedError()


class Greedy(SpinSolver):
    """A greedy solver for a SpinSystem."""

    def __init__(self, *args, **kwargs):
        """Initialise a greedy solver.

        Args:
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            trial_env (SpinSystemMCTS): The environment with in the agent tests
                actions (a clone of self.env where the final actions are taken).
            current_snap: The current state of the environment.
        """

        super().__init__(*args, **kwargs)

    def step(self):
        """Take the action which maximises the immediate reward.

        Returns:
            reward (float): The reward recieved.
            done (bool): Whether the environment is in a terminal state after
                the action is taken.
        """
        rewards_avaialable = self.env.get_immeditate_rewards_avaialable()

        if self.env.reversible_spins:
            action = rewards_avaialable.argmax()
        else:
            masked_rewards_avaialable = rewards_avaialable.copy()
            np.putmask(masked_rewards_avaialable,
                       self.env.get_observation()[0, :] != self.env.get_allowed_action_states(),
                       -100)
            action = masked_rewards_avaialable.argmax()

        if rewards_avaialable[action] < 0:
            action = None
            reward = 0
            done = True
        else:
            observation, reward, done, _ = self.env.step(action)

        return reward, done


class Random(SpinSolver):
    """A random solver for a SpinSystem."""

    def step(self):
        """Take a random action.

        Returns:
            reward (float): The reward recieved.
            done (bool): Whether the environment is in a terminal state after
                the action is taken.
        """

        observation, reward, done, _ = self.env.step(self.env.action_space.sample())
        return reward, done


class Network(SpinSolver):
    """A network-only solver for a SpinSystem."""

    epsilon = 0.

    def __init__(self, network, *args, **kwargs):
        """Initialise a network-only solver.

        Args:
            network: The network.
            *args: Passed through to the SpinSolver constructor.

        Attributes:
            current_snap: The last observation of the environment, used to choose the next action.
        """

        super().__init__(*args, **kwargs)
        self.device = TRAIN_DEVICE
        self.network = network.to(self.device)
        self.network.eval()
        self.current_observation = self.env.get_observation()
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)

        self.history = []

    def reset(self, spins=None, clear_history=True):
        if spins is None:
            self.current_observation = self.env.reset()
        else:
            self.current_observation = self.env.reset(spins)
        self.current_observation = torch.FloatTensor(self.current_observation).to(self.device)
        self.total_reward = 0

        if clear_history:
            self.history = []

    @torch.no_grad()
    def step(self):

        # Q-values predicted by the network.
        qs = self.network(self.current_observation)

        if self.env.reversible_spins:
            if np.random.uniform(0, 1) >= self.epsilon:
                # Action that maximises Q function
                action = qs.argmax().item()
            else:
                # Random action
                action = np.random.randint(0, self.env.action_space.n)

        else:
            x = (self.current_observation[0, :] == self.env.get_allowed_action_states()).nonzero()
            if np.random.uniform(0, 1) >= self.epsilon:
                action = x[qs[x].argmax().item()].item()
                # Allowed action that maximises Q function
            else:
                # Random allowed action
                action = x[np.random.randint(0, len(x))].item()

        if action is not None:
            observation, reward, done, _ = self.env.step(action)
            self.current_observation = torch.FloatTensor(observation).to(self.device)

        else:
            reward = 0
            done = True

        if not self.record_cut and not self.record_rewards:
            record = [action]
        else:
            record = [action]
            if self.record_cut:
                record += [self.env.calculate_cut()]
            if self.record_rewards:
                record += [reward]
            if self.record_qs:
                record += [qs]

        record += [self.env.get_immeditate_rewards_avaialable()]

        self.history.append(record)

        return reward, done
