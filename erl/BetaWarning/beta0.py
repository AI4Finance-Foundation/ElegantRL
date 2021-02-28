from eRL.run import Arguments, train_and_evaluate, train_and_evaluate__multiprocessing
from eRL.env import decorate_env
import eRL.agent as agent
import gym

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

'''
beta3 GaePPO Reacher

ceta1 GaePPO Ant
ceta2 ModSAC Ant demo5 AgentModSAC
ceta3 ISAC   Ant demo5 AgentISAC
'''


def demo1__discrete_action_space():
    """DEMO 1: Discrete action env: CartPole-v0 of gym"""
    args = Arguments(agent_rl=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters

    args.agent_rl = agent.AgentD3QN  # choose an DRL algorithm
    args.env = decorate_env(env=gym.make('CartPole-v0'))
    args.net_dim = 2 ** 7  # change a default hyper-parameters
    # args.env = decorate_env(env=gym.make('LunarLander-v2'))
    # args.net_dim = 2 ** 8  # change a default hyper-parameters

    train_and_evaluate(args)


def demo2():
    """DEMO 2: Continuous action env, gym.Box2D"""
    if_on_policy = False
    args = Arguments(if_on_policy=if_on_policy)  # on-policy has different hyper-parameters from off-policy
    if if_on_policy:
        args.agent_rl = agent.AgentGaePPO  # on-policy: AgentPPO, AgentGaePPO
    else:
        args.agent_rl = agent.AgentModSAC  # off-policy: AgentSAC, AgentModPPO, AgentTD3, AgentDDPG

    env = gym.make('Pendulum-v0')
    env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
    args.env = decorate_env(env=env)
    args.net_dim = 2 ** 7  # change a default hyper-parameters
    # args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
    # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95

    train_and_evaluate(args)


def demo3():
    """DEMO 3: Custom Continuous action env: FinanceStock-v1"""
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO  # PPO+GAE (on-policy)

    from eRL.env import FinanceMultiStockEnv
    args.env = FinanceMultiStockEnv(if_train=True)  # a standard env for ElegantRL, not need decorate_env()
    args.env_eval = FinanceMultiStockEnv(if_train=False)
    args.break_step = int(5e6)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
    args.net_dim = 2 ** 8
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.eval_times1 = 2 ** 3

    # train_and_evaluate(args)
    args.rollout_num = 8
    args.if_break_early = False
    train_and_evaluate__multiprocessing(args)


def demo41():
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO  # agent.AgentPPO

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('ReacherBulletEnv-v0'))

    args.break_step = int(5e4 * 8)  # (5e4) 1e5, UsedTime: (400s) 800s
    args.repeat_times = 2 ** 3
    args.reward_scale = 2 ** 1  # (-15) 18 (30)
    args.eval_times1 = 2 ** 2
    args.eval_times1 = 2 ** 6

    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


def demo42():
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO  # agent.AgentPPO

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'))
    args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime: 25697s
    args.reward_scale = 2 ** -3  #
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 9
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 12
    args.show_gap = 2 ** 6
    args.eval_times1 = 2 ** 2

    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


def demo5():
    args = Arguments(if_on_policy=False)
    # args.agent_rl = agent.AgentModSAC
    args.agent_rl = agent.AgentInterSAC

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'))
    # args.env = decorate_env(gym.make('ReacherBulletEnv-v0'))

    args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
    args.max_memo = 2 ** 19
    args.net_dim = 2 ** 7  # todo
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder

    train_and_evaluate(args)
    # args.rollout_num = 4
    # train_and_evaluate__multiprocessing(args)


def render_pybullet():
    from eRL import agent
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO  # agent.AgentPPO

    import pybullet_envs  # for python-bullet-gym

    dir(pybullet_envs)
    args.env = decorate_env(gym.make('ReacherBulletEnv-v0'))
    args.gpu_id = 3

    args.init_before_training()

    net_dim = args.net_dim
    state_dim = args.env.state_dim
    action_dim = args.env.action_dim
    # cwd = args.cwd
    cwd = "ReacherBulletEnv-v0_2"
    env = args.env
    max_step = args.max_step

    agent = args.agent_rl(net_dim, state_dim, action_dim)
    agent.save_or_load_model(cwd, if_save=False)  # if_save=False means load history

    import cv2
    import os

    frame_save_dir = f'{cwd}/frame'
    os.makedirs(frame_save_dir, exist_ok=True)

    '''methods 1: Print Error in remote server: Invalid GLX version: major 1, minor 2'''
    # import pybullet as pb
    # phy = pb.connect(pb.GUI)
    '''methods 2: Print Error in remote server: Invalid GLX version: major 1, minor 2'''
    env.render()  # https://github.com/benelot/pybullet-gym/issues/25

    state = env.reset()
    for i in range(max_step):
        print(i)
        actions, _noises = agent.select_actions((state,))
        action = actions[0]
        next_state, reward, done, _ = env.step(action)
        frame = env.render()

        if isinstance(frame, np.ndarray):
            cv2.imshow(env.env_name, frame)
            cv2.waitKey(20)  # 1000ms = 1s
            cv2.imwrite(f'{frame_save_dir}/{i:06}.png', frame)

        if done:
            break
        state = next_state
    print('end')

render_pybullet()
# demo1()
# demo5()
# render_pybullet()
