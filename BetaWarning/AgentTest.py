from AgentRun import *
from AgentNet import *
from AgentZoo import *


def test__env_quickly():
    env_names = [
        # Classical Control
        "Pendulum-v0", "CartPole-v0",

        # Box2D
        "LunarLander-v2", "LunarLanderContinuous-v2",
        "BipedalWalker-v3", "BipedalWalkerHardcore-v3",
        'CarRacing-v0',  # Box2D pixel-level
        'MultiWalker',  # Box2D MultiAgent

        # py-bullet (MuJoCo is not free)
        "AntBulletEnv-v0", "MinitaurBulletEnv-v0",
    ]

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    for env_name in env_names:
        build_gym_env(env_name, if_print=True, if_norm=False)
        print()


if __name__ == '__main__':
    test__env_quickly()
