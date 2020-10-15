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


def test__replay_buffer():
    max_memo = 2 ** 6
    max_step = 2 ** 4
    reward_scale = 1
    gamma = 0.99
    net_dim = 2 ** 7

    env_name = 'LunarLanderContinuous-v2'
    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    state = env.reset()

    '''offline'''
    buffer_offline = BufferArray(max_memo, state_dim, 1 if if_discrete else action_dim)
    _rewards, _steps = initial_exploration(env, buffer_offline, max_step, if_discrete, reward_scale, gamma, action_dim)
    print('Memory length of buffer_offline:', buffer_offline.now_len)

    '''online'''
    buffer_online = BufferTupleOnline(max_memo=max_step)
    agent = AgentPPO(state_dim, action_dim, net_dim)
    agent.state = env.reset()

    _rewards, _steps = agent.update_buffer(env, buffer_online, max_step, reward_scale, gamma)
    print('Memory length of buffer_online: ', len(buffer_online.storage_list))

    buffer_ary = buffer_online.convert_to_rmsas()
    buffer_offline.extend_memo(buffer_ary)
    buffer_offline.init_before_sample()
    print('Memory length of buffer_offline:', buffer_offline.now_len)


if __name__ == '__main__':
    # test__env_quickly()
    test__replay_buffer()
    print('; AgentTest Terminal.')
