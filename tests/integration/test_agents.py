"""
This script tests whether or not each of the agents is able to train, depending
on whether or not it receives an environment with a discrete or continuous
action space.
"""

import gym
import unittest
from elegantrl.agent import *
from elegantrl.config import Arguments, get_gym_env_args
from elegantrl.run import train_and_evaluate


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.discrete_agents = [
            AgentDQN,
            AgentDuelingDQN,
            AgentDoubleDQN,
            AgentD3QN,
            AgentDiscretePPO,
        ]
        self.continuous_agents = [
            AgentDDPG,
            AgentTD3,
            AgentSAC,
            AgentModSAC,
            AgentREDqSAC,
            AgentPPO,
            AgentHtermPPO,
        ]
        self.discrete_env_args = get_gym_env_args(
            gym.make("LunarLander-v2"), if_print=False
        )
        self.continuous_env_args = get_gym_env_args(
            gym.make("BipedalWalker-v3"), if_print=False
        )

    def test_should_create_arguments_for_each_agent(self):
        for agent in self.discrete_agents:
            Arguments(agent, env_func=gym.make, env_args=self.discrete_env_args)
        for agent in self.continuous_agents:
            Arguments(agent, env_func=gym.make, env_args=self.continuous_env_args)

    def train_on(self, args: Arguments):
        args.eval_times = 2**4
        args.break_step = 1
        train_and_evaluate(args)

    def train_discrete(self, agent: AgentBase):
        args = Arguments(agent, env_func=gym.make, env_args=self.discrete_env_args)
        self.train_on(args)

    def train_continuous(self, agent: AgentBase):
        args = Arguments(agent, env_func=gym.make, env_args=self.continuous_env_args)
        self.train_on(args)

    # first, test discrete agents

    def test_should_train_DQN_on_discrete_action_space(self):
        self.train_discrete(AgentDQN)

    def test_should_not_train_DQN_on_continuous_action_space(self):
        self.assertRaises(Exception, self.train_continuous, AgentDQN)

    def test_should_train_DuelingDQN_on_discrete_action_space(self):
        self.train_discrete(AgentDuelingDQN)

    def test_should_not_train_DuelingDQN_on_continuous_action_space(self):
        self.assertRaises(Exception, self.train_continuous, AgentDuelingDQN)

    def test_should_train_DoubleDQN_on_discrete_action_space(self):
        self.train_discrete(AgentDoubleDQN)

    def test_should_not_train_DoubleDQN_on_continuous_action_space(self):
        self.assertRaises(Exception, self.train_continuous, AgentDoubleDQN)

    def test_should_train_D3QN_on_discrete_action_space(self):
        self.train_discrete(AgentD3QN)

    def test_should_not_train_D3QN_on_continuous_action_space(self):
        self.assertRaises(Exception, self.train_continuous, AgentD3QN)

    def test_should_train_DiscretePPO_on_discrete_action_space(self):
        self.train_discrete(AgentDiscretePPO)

    def test_should_not_train_DiscretePPO_on_continuous_action_space(self):
        self.assertRaises(Exception, self.train_continuous, AgentDiscretePPO)

    # next, test continuous agents

    def test_should_train_DDPG_on_continuous_action_space(self):
        self.train_continuous(AgentDDPG)

    def test_should_not_train_DDPG_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentDDPG)

    def test_should_train_TD3_on_continuous_action_space(self):
        self.train_continuous(AgentTD3)

    def test_should_not_train_TD3_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentTD3)

    def test_should_train_SAC_on_continuous_action_space(self):
        self.train_continuous(AgentSAC)

    def test_should_not_train_SAC_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentSAC)

    def test_should_train_ModSAC_on_continuous_action_space(self):
        self.train_continuous(AgentModSAC)

    def test_should_not_train_ModSAC_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentModSAC)

    def test_should_train_REDqSAC_on_continuous_action_space(self):
        self.train_continuous(AgentREDqSAC)

    def test_should_not_train_REDqSAC_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentREDqSAC)

    def test_should_train_PPO_on_continuous_action_space(self):
        self.train_continuous(AgentPPO)

    def test_should_not_train_PPO_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentPPO)

    def test_should_train_HtermPPO_on_continuous_action_space(self):
        self.train_continuous(AgentHtermPPO)

    def test_should_not_train_HtermPPO_on_discrete_action_space(self):
        self.assertRaises(Exception, self.train_discrete, AgentHtermPPO)


if __name__ == "__main__":
    unittest.main()
