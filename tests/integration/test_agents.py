import gym
import unittest
from elegantrl.agent import *
from elegantrl.config import Arguments, get_gym_env_args
from elegantrl.run import train_and_evaluate


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.agents = [
            AgentDQN,
            AgentDuelingDQN,
            AgentDoubleDQN,
            AgentD3QN,
            AgentDDPG,
            AgentTD3,
            AgentSAC,
            AgentModSAC,
            AgentHtermModSAC,
            AgentREDqSAC,
            AgentPPO,
            AgentHtermPPO,
            AgentDiscretePPO,
        ]
        self.discrete_env_args = get_gym_env_args(
            gym.make("LunarLander-v2"), if_print=False
        )
        self.continuous_env_args = get_gym_env_args(
            gym.make("BipedalWalker-v3"), if_print=False
        )

    def test_should_create_arguments_for_each_agent(self):
        for agent in self.agents:
            Arguments(agent, env_func=gym.make, env_args=self.discrete_env_args)
            Arguments(agent, env_func=gym.make, env_args=self.continuous_env_args)

    def train_on(self, args: Arguments):
        args.eval_times = 2**4
        args.break_step = 2**10
        train_and_evaluate(args)

    def test_should_train_discrete_DQN_agent(self):
        args = Arguments(AgentDQN, env_func=gym.make, env_args=self.discrete_env_args)
        self.train_on(args)

    def test_should_not_train_continuous_DQN_agent(self):
        args = Arguments(AgentDQN, env_func=gym.make, env_args=self.continuous_env_args)
        self.assertRaises(IndexError, self.train_on, args)


if __name__ == "__main__":
    unittest.main()
