from AgentRun import *
from AgentNet import *
from AgentZoo import *


def test__env_quickly():
    env_names = ["Pendulum-v0", "LunarLanderContinuous-v2"]

    env_names = ["AntBulletEnv-v0", "MinitaurBulletEnv-v0"]

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    for env_name in env_names:
        build_gym_env(env_name, if_print=True, if_norm=False)


if __name__ == '__main__':
    test__env_quickly()
