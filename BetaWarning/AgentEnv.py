import numpy as np
import numpy.random as rd
import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)



# def build_env_20201111(env, if_print=True, if_norm=True):  # important function
#     assert env is not None
#
#     if not isinstance(env, gym.Env):
#         if hasattr(env, 'state_dim'):
#             state_dim = env.state_dim
#             action_dim = env.action_dim
#             target_reward = env.target_reward
#             if_discrete = env.if_discrete
#         else:
#             print('| build_env: Could tell me the value of following constant?')
#             raise RuntimeError(
#                 'in build_env(): \n'
#                 '\tCould tell me the value of these constants?\n'
#                 '\tstate_dim, action_dim, target_reward, if_discrete = (int, int, float, bool)'
#             )
#
#         return env, state_dim, action_dim, target_reward, if_discrete
#
#     assert isinstance(env, gym.Env)
#     # '''Don't show warning: WARN: Box bound precision lowered by casting to float32
#     # https://stackoverflow.com/questions/60149105/
#     # userwarning-warn-box-bound-precision-lowered-by-casting-to-float32
#     # '''
#     # gym.logger.set_level(40)  # not important
#
#     state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
#
#     print('exit')
#     exit()
#
#     if env_name == 'Pendulum-v0':
#         env = gym.make(env_name)
#         env.spec.reward_threshold = -200.0  # target_reward
#         state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
#     elif env_name == 'CarRacing-v0':
#         # | state_dim: (96, 96, 3), action_dim: 3, action_max: 1.0, target_reward: 900
#         frame_num = 3
#         env = gym.make(env_name)
#         env = fix_car_racing_env(env, frame_num=frame_num, action_num=frame_num)
#         state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
#         assert len(state_dim)
#         state_dim = (frame_num, state_dim[0], state_dim[1])  # two consecutive frame (96, 96)
#         # from AgentPixel import CarRacingEnv
#         # env = CarRacingEnv(img_stack=4, action_repeat=4)
#         # state_dim, action_dim, action_max = (4, 96, 96), 3, 1.0
#         # target_reward, if_discrete = 900, False
#     elif env_name == 'MultiWalker':
#         from multiwalker_base import MultiWalkerEnv, multi_to_single_walker_decorator
#         env = MultiWalkerEnv()
#         env = multi_to_single_walker_decorator(env)
#
#         state_dim = sum([box.shape[0] for box in env.observation_space])
#         action_dim = sum([box.shape[0] for box in env.action_space])
#         action_max = 1.0
#         target_reward = 50
#         if_discrete = False
#     elif env_name == 'FinRL':  # 2020-12-12
#         env = SingleStockFinEnv()
#
#         state_dim = 4
#         action_dim = 1
#         action_max = 1.0
#         target_reward = 800
#         if_discrete = False
#     else:
#         env = gym.make(env_name)
#         state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
#
#     # todo check action space from (-1, 1)
#     if isinstance(env, gym.Env):  # convert state to float32 and do state norm if_norm
#         avg = None
#         std = None
#         if if_norm:  # I use def print_norm() to get the following (avg, std)
#             # if env_name == 'Pendulum-v0':
#             #     state_mean = np.array([-0.00968592 -0.00118888 -0.00304381])
#             #     std = np.array([0.53825575 0.54198545 0.8671749 ])
#             if env_name == 'LunarLanderContinuous-v2':
#                 avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
#                                 -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
#                 std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
#                                 0.25886878, 0.277233, 0.27771219])
#             elif env_name == "BipedalWalker-v3":
#                 avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
#                                 -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
#                                 4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
#                                 -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
#                                 3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
#                                 5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
#                 std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
#                                 0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
#                                 0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
#                                 0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
#                                 0.11752805, 0.14116005, 0.13839757, 0.07760469])
#             elif env_name == 'AntBulletEnv-v0':
#                 avg = np.array([
#                     0.4838, -0.047, 0.3500, 1.3028, -0.249, 0.0000, -0.281, 0.0573,
#                     -0.261, 0.0000, 0.0424, 0.0000, 0.2278, 0.0000, -0.072, 0.0000,
#                     0.0000, 0.0000, -0.175, 0.0000, -0.319, 0.0000, 0.1387, 0.0000,
#                     0.1949, 0.0000, -0.136, -0.060])
#                 std = np.array([
#                     0.0601, 0.2267, 0.0838, 0.2680, 0.1161, 0.0757, 0.1495, 0.1235,
#                     0.6733, 0.4326, 0.6723, 0.3422, 0.7444, 0.5129, 0.6561, 0.2732,
#                     0.6805, 0.4793, 0.5637, 0.2586, 0.5928, 0.3876, 0.6005, 0.2369,
#                     0.4858, 0.4227, 0.4428, 0.4831])
#             # elif env_name == 'MinitaurBulletEnv-v0':
#             #     # avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
#             #     #                 1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
#             #     #                 0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
#             #     #                 -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
#             #     #                 0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
#             #     #                 -0.20753499, -0.47758384, 0.86756409])
#             #     # std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
#             #     #                 0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
#             #     #                 14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
#             #     #                 13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
#             #     #                 2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
#             #     #                 0.05903034, 0.1314812, 0.0221248])
#
#             # elif env_name == "BipedalWalkerHardcore-v3":
#             #     avg = np.array([-3.6378160e-02, -2.5788052e-03, 3.4413573e-01, -8.4189959e-03,
#             #                     -9.1864385e-02, 3.2804706e-04, -6.4693891e-02, -9.8939031e-02,
#             #                     3.5180664e-01, 6.8103075e-01, 2.2930240e-03, -4.5893672e-01,
#             #                     -7.6047562e-02, 4.6414185e-01, 3.9363885e-01, 3.9603019e-01,
#             #                     4.0758255e-01, 4.3053803e-01, 4.6186063e-01, 5.0293463e-01,
#             #                     5.7822973e-01, 6.9820738e-01, 8.9829963e-01, 9.8080903e-01])
#             #     std = np.array([0.5771428, 0.05302362, 0.18906464, 0.10137994, 0.41284004,
#             #                     0.68852615, 0.43710527, 0.87153363, 0.3210142, 0.36864948,
#             #                     0.6926624, 0.38297284, 0.76805115, 0.33138904, 0.09618598,
#             #                     0.09843876, 0.10035378, 0.11045089, 0.11910835, 0.13400233,
#             #                     0.15718603, 0.17106676, 0.14363566, 0.10100251])
#
#         env = decorate_env(env, action_max, avg, std, data_type=np.float32)
#     return env, state_dim, action_dim, target_reward, if_discrete


# def get_env_info(env, if_print=True) -> tuple:  # 2020-10-10
#     env_name = env.unwrapped.spec.id
#
#     state_shape = env.observation_space.shape
#     if len(state_shape) == 1:
#         state_dim = state_shape[0]
#     else:
#         state_dim = state_shape
#
#     try:
#         if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
#         if if_discrete:  # discrete
#             action_dim = env.action_space.n
#             action_max = int(1)
#         elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
#             action_dim = env.action_space.shape[0]
#             action_max = float(env.action_space.high[0])
#
#             action_high = np.array(env.action_space.high)
#             action_high[:] = action_max
#             action_low = np.array(env.action_space.low)
#             action_low[:] = -action_max
#             if any(action_high != env.action_space.high) and any(action_low != env.action_space.low):
#                 print(f'| Warning: '
#                       f'act_high {env.action_space.high}  '
#                       f'act_low  {env.action_space.low}')
#         else:
#             raise AttributeError
#     except AttributeError:
#         print("| Could you assign these value manually? \n"
#               "| I need: state_dim, action_dim, action_max, target_reward, if_discrete")
#         raise AttributeError
#
#     target_reward = env.spec.reward_threshold
#     if target_reward is None:
#         assert target_reward is not None
#
#     if if_print:
#         print("| env_name: {}, action space: {}".format(env_name, 'if_discrete' if if_discrete else 'Continuous'))
#         print("| state_dim: {}, action_dim: {}, action_max: {}, target_reward: {}".format(
#             state_dim, action_dim, action_max, target_reward))
#     return state_dim, action_dim, action_max, target_reward, if_discrete




"""check"""


def check__build_env():
    env_names = [
        # Classical Control
        "Pendulum-v0", "CartPole-v0", "Acrobot-v1",

        # Box2D
        "LunarLander-v2", "LunarLanderContinuous-v2",
        "BipedalWalker-v3", "BipedalWalkerHardcore-v3",
        'CarRacing-v0',  # Box2D pixel-level
        # 'MultiWalker',  # Box2D MultiAgent

        # py-bullet (MuJoCo is not free)
        "AntBulletEnv-v0", "Walker2DBulletEnv-v0", "HalfCheetahBulletEnv-v0",
        # "HumanoidBulletEnv-v0", "HumanoidFlagrunBulletEnv-v0", "HumanoidFlagrunHarderBulletEnv-v0",

        "ReacherBulletEnv-v0", "PusherBulletEnv-v0", "ThrowerBulletEnv-v0",
        # "StrikerBulletEnv-v0",

        "MinitaurBulletEnv-v0",
    ]

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    for env_name in env_names:
        print(f'| {env_name}')
        build_env(env_name, if_print=True, if_norm=False)
        print()


if __name__ == '__main__':
    # run()
    # test()
    # train()
    # train__baselines_rl()
    check__build_env()
