import os
import time
from copy import deepcopy

import torch
import numpy as np
import numpy.random as rd

'''DEMO'''


class Arguments:
    def __init__(self, agent_rl=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent_rl = agent_rl  # Deep Reinforcement Learning algorithm
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating

        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8  # the network width
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.max_memo = 2 ** 17  # capacity of replay buffer
        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 8
            self.repeat_times = 2 ** 4
            self.max_memo = 2 ** 12
        self.max_step = 2 ** 10  # max steps in one training episode
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.eval_times = 2 ** 3  # evaluation times if 'eval_reward > target_reward'
        self.show_gap = 2 ** 8  # show the Reward and Loss value per show_gap seconds
        self.random_seed = 0  # initialize random seed in self.init_before_training(

    def init_before_training(self):
        assert self.agent_rl is not None
        assert self.env is not None
        if not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env = decorate_env(env)')

        import sys
        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.gpu_id = self.gpu_id if self.gpu_id.isdigit() else '0'
        self.cwd = f'./{self.agent_rl.__name__}/{self.env.env_name}_{self.gpu_id}' if self.cwd is None else self.cwd
        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

        import shutil  # remove history according to bool(if_remove)
        if self.if_remove is None:
            self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
        if self.if_remove:
            shutil.rmtree(self.cwd, ignore_errors=True)
            print("| Remove history")
        os.makedirs(self.cwd, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def run__demo():
    import elegantrl2.agent as agent
    from elegantrl2.env import decorate_env
    # from elegantrl2.main import Arguments, train_and_evaluate, train_and_evaluate__multiprocessing
    import gym

    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    """DEMO 1: Discrete action env: CartPole-v0 of gym"""
    args = Arguments(agent_rl=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters

    args.agent_rl = agent.AgentD3QN  # choose an DRL algorithm
    args.env = decorate_env(env=gym.make('CartPole-v0'))
    args.net_dim = 2 ** 7  # change a default hyper-parameters
    # args.env = decorate_env(env=gym.make('LunarLander-v2'))
    # args.net_dim = 2 ** 8  # change a default hyper-parameters

    train_and_evaluate(args)

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

    """DEMO 3: Custom Continuous action env: FinanceStock-v1"""
    args = Arguments(if_on_policy=True)
    args.agent_rl = agent.AgentGaePPO  # PPO+GAE (on-policy)

    from elegantrl2.env import FinanceMultiStockEnv
    args.env = FinanceMultiStockEnv(if_train=True)  # a standard env for ElegantRL, not need decorate_env()
    args.env_eval = FinanceMultiStockEnv(if_train=False)
    args.break_step = int(2e6)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
    args.net_dim = 2 ** 8
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.eval_times = 2 ** 3

    # train_and_evaluate(args)
    args.rollout_num = 8
    train_and_evaluate__multiprocessing(args)

    def draw_cumulative_return_curve(cwd='FinanceStock-v2_1.61_3'):
        from elegantrl2.env import FinanceMultiStockEnv
        env = FinanceMultiStockEnv(if_train=False)

        agent_rl = agent.AgentGaePPO  # PPO+GAE (on-policy)
        net_dim = 2 ** 8
        agent_ppo = agent_rl(net_dim, env.state_dim, env.action_dim)
        agent_ppo.save_or_load_model(cwd, if_save=False)

        import torch
        cumulative_returns = env.draw_cumulative_return(agent_ppo, torch)

        import matplotlib.pyplot as plt

        plt.plot(cumulative_returns)
        plt.show()

    '''DEMO 4: PyBullet(MuJoCo) Robot Env'''
    args = Arguments(if_on_policy=True)  # on-policy has different hyper-parameters from off-policy
    args.agent_rl = agent.AgentGaePPO  # on-policy: AgentPPO, AgentGaePPO

    env_name = 'AntBulletEnv-v0'
    assert env_name in {"AntBulletEnv-v0", "Walker2DBulletEnv-v0", "HalfCheetahBulletEnv-v0",
                        "HumanoidBulletEnv-v0", "HumanoidFlagrunBulletEnv-v0", "HumanoidFlagrunHarderBulletEnv-v0"}
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'))

    args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 2  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder

    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


'''DEMO wait for updating'''

# def run__demo_20200202():
#     import gym  # don't worry about 'WARN: Box bound precision lowered by casting to float32'
#     import AgentZoo
#     from AgentEnv import decorate_env
#     args = Arguments(agent_rl=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters
#
#     '''DEMO 1: Discrete action env: CartPole-v0 of gym'''
#     args.agent_rl = AgentZoo.AgentDoubleDQN  # choose an DRL algorithm
#     # args.env = decorate_env(env=gym.make('CartPole-v0'))
#     # args.net_dim = 2 ** 7  # change a default hyper-parameters
#     args.env = decorate_env(env=gym.make('LunarLander-v2'))
#     args.net_dim = 2 ** 8  # change a default hyper-parameters
#     train_and_evaluate(args)
#     exit()
#
#     '''DEMO 2: Continuous action env: gym.box2D'''
#     args.env = decorate_env(env=gym.make('Pendulum-v0'))
#     args.env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
#     # args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
#     # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
#     args.agent_rl = AgentZoo.AgentSAC  # off-policy
#     train_and_evaluate(args)
#     exit()
#
#     args = Arguments(if_on_policy=True)  # on-policy has different hyper-parameters from off-policy
#     # args.env = decorate_env(env=gym.make('Pendulum-v0'))
#     args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
#     # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
#     args.agent_rl = AgentZoo.AgentPPO  # on-policy
#     train_and_evaluate(args)
#     exit()
#
#     '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
#     from AgentEnv import FinanceMultiStockEnv
#     args = Arguments(if_on_policy=True)
#     args.env = FinanceMultiStockEnv()  # a standard env for ElegantRL, not need decorate_env()
#     args.agent_rl = AgentZoo.AgentPPO  # PPO+GAE (on-policy)
#
#     args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
#     args.net_dim = 2 ** 8
#     args.max_step = 1699
#     args.max_memo = (args.max_step - 1) * 16
#     args.batch_size = 2 ** 11
#     args.repeat_times = 2 ** 4
#     train_and_evaluate(args)
#     exit()


# def train__demo():
#     pass
#
#     '''DEMO 1: Standard gym env CartPole-v0 (discrete action) using D3QN (DuelingDoubleDQN, off-policy)'''
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     env = gym.make('CartPole-v0')
#     env = decorate_env(env, if_print=True)
#
#     from AgentZoo import AgentD3QN
#     args = Arguments(rl_agent=AgentD3QN, env=env, gpu_id=0)
#     args.break_step = int(1e5 * 8)  # UsedTime: 60s (reach target_reward 195)
#     args.net_dim = 2 ** 7
#     args.init_for_training()
#     train_agent(args)
#     exit()
#
#     '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     env = gym.make('LunarLanderContinuous-v2')
#     env = decorate_env(env, if_print=True)
#
#     from AgentZoo import AgentModSAC
#     args = Arguments(rl_agent=AgentModSAC, env=env, gpu_id=0)
#     args.break_step = int(6e4 * 8)  # UsedTime: 900s (reach target_reward 200)
#     args.net_dim = 2 ** 7
#     args.init_for_training()
#     # train_agent(args)  # Train agent using single process. Recommend run on PC.
#     train_agent_mp(args)  # Train using multi process. Recommend run on Server.
#     exit()
#
#     '''DEMO 3: Custom env FinanceStock (continuous action) using PPO (PPO2+GAE, on-policy)'''
#     env = FinanceMultiStockEnv()  # 2020-12-24
#
#     from AgentZoo import AgentPPO
#     args = Arguments(rl_agent=AgentPPO, env=env)
#     args.eval_times1 = 1
#     args.eval_times2 = 1
#     args.rollout_num = 4
#     args.if_break_early = True
#
#     args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
#     args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
#     args.net_dim = 2 ** 8
#     args.max_step = 1699
#     args.max_memo = 1699 * 16
#     args.batch_size = 2 ** 10
#     args.repeat_times = 2 ** 4
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     # from AgentZoo import AgentModSAC
#     # args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
#     # args.eval_times1 = 1
#     # args.eval_times2 = 2
#     #
#     # args.break_step = 2 ** 22  # UsedTime:
#     # args.net_dim = 2 ** 7
#     # args.max_memo = 2 ** 18
#     # args.batch_size = 2 ** 8
#     # args.init_for_training()
#     # train_agent_mp(args)  # train_agent(args)
#
#
# def train__discrete_action():
#     import AgentZoo as Zoo
#     args = Arguments(rl_agent=None, env=None, gpu_id=None)
#     args.rl_agent = [
#         Zoo.AgentDQN,  # 2014.
#         Zoo.AgentDoubleDQN,  # 2016. stable
#         Zoo.AgentDuelingDQN,  # 2016. stable and fast
#         Zoo.AgentD3QN,  # 2016+ Dueling + Double DQN (Not a creative work)
#     ][3]  # I suggest to use D3QN
#
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#
#     args.env = decorate_env(gym.make('CartPole-v0'), if_print=True)
#     args.break_step = int(1e4 * 8)  # (3e5) 1e4, used time 20s
#     args.reward_scale = 2 ** 0  # 0 ~ 200
#     args.net_dim = 2 ** 6
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('LunarLander-v2'), if_print=True)
#     args.break_step = int(1e5 * 8)  # (2e4) 1e5 (3e5), used time (200s) 1000s (2000s)
#     args.reward_scale = 2 ** -1  # (-1000) -150 ~ 200 (285)
#     args.net_dim = 2 ** 7
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#
# def train__continuous_action__off_policy():
#     import AgentZoo as Zoo
#     args = Arguments(rl_agent=None, env=None, gpu_id=None)
#     args.rl_agent = [
#         Zoo.AgentDDPG,  # 2016. simple, simple, slow, unstable
#         Zoo.AgentBaseAC,  # 2016+ modify DDPG, faster, more stable
#         Zoo.AgentTD3,  # 2018. twin critics, delay target update
#         Zoo.AgentSAC,  # 2018. twin critics, maximum entropy, auto alpha, fix log_prob
#         Zoo.AgentModSAC,  # 2018+ modify SAC, faster, more stable
#         Zoo.AgentInterAC,  # 2019. Integrated AC(DPG)
#         Zoo.AgentInterSAC,  # 2020. Integrated SAC(SPG)
#     ][4]  # I suggest to use ModSAC (Modify SAC)
#     # On-policy PPO is not here 'run__off_policy()'. See PPO in 'run__on_policy()'.
#
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     args.if_break_early = True  # break training if reach the target reward (total return of an episode)
#     args.if_remove_history = True  # delete the historical directory
#
#     env = gym.make('Pendulum-v0')  # It is easy to reach target score -200.0 (-100 is harder)
#     args.env = decorate_env(env, if_print=True)
#     args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
#     args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
#     args.init_for_training()
#     train_agent(args)  # Train agent using single process. Recommend run on PC.
#     # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
#     exit()
#
#     args.env = decorate_env(gym.make('LunarLanderContinuous-v2'), if_print=True)
#     args.break_step = int(5e5 * 8)  # (2e4) 5e5, used time 1500s
#     args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('BipedalWalker-v3'), if_print=True)
#     args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
#     args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('ReacherBulletEnv-v0'), if_print=True)
#     args.break_step = int(5e4 * 8)  # (4e4) 5e4
#     args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29)
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('AntBulletEnv-v0'), if_print=True)
#     args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
#     args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
#     args.batch_size = 2 ** 8
#     args.max_memo = 2 ** 20
#     args.eva_size = 2 ** 3  # for Recorder
#     args.show_gap = 2 ** 8  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('MinitaurBulletEnv-v0'), if_print=True)
#     args.break_step = int(4e6 * 4)  # (2e6) 4e6
#     args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
#     args.batch_size = (2 ** 8)
#     args.net_dim = int(2 ** 8)
#     args.max_step = 2 ** 11
#     args.max_memo = 2 ** 20
#     args.eval_times2 = 3  # for Recorder
#     args.eval_times2 = 9  # for Recorder
#     args.show_gap = 2 ** 9  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('BipedalWalkerHardcore-v3'), if_print=True)  # 2020-08-24 plan
#     args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (334)
#     args.break_step = int(4e6 * 8)  # (2e6) 4e6
#     args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
#     args.max_memo = int(2 ** 21)
#     args.batch_size = int(2 ** 8)
#     args.eval_times2 = 2 ** 5  # for Recorder
#     args.show_gap = 2 ** 8  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_offline_policy(args)
#     exit()
#
#
# def train__continuous_action__on_policy():
#     import AgentZoo as Zoo
#     args = Arguments(rl_agent=None, env=None, gpu_id=None)
#     args.rl_agent = [
#         Zoo.AgentPPO,  # 2018. PPO2 + GAE, slow but quite stable, especially in high-dim
#         Zoo.AgentInterPPO,  # 2019. Integrated Network, useful in pixel-level task (state2D)
#     ][0]
#
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     args.if_break_early = True  # break training if reach the target reward (total return of an episode)
#     args.if_remove_history = True  # delete the historical directory
#
#     args.net_dim = 2 ** 8
#     args.max_memo = 2 ** 12
#     args.batch_size = 2 ** 9
#     args.repeat_times = 2 ** 4
#     args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#
#     env = gym.make('Pendulum-v0')  # It is easy to reach target score -200.0 (-100 is harder)
#     args.env = decorate_env(env, if_print=True)
#     args.break_step = int(8e4 * 8)  # 5e5 means the average total training step of ModPPO to reach target_reward
#     args.reward_scale = 2 ** 0  # (-1800) -1000 ~ -200 (-50), UsedTime:  (100s) 200s
#     args.gamma = 0.9  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('LunarLanderContinuous-v2'), if_print=True)
#     args.break_step = int(3e5 * 8)  # (2e5) 3e5 , used time: (400s) 600s
#     args.reward_scale = 2 ** 0  # (-800) -200 ~ 200 (301)
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     # exit()
#
#     args.env = decorate_env(gym.make('BipedalWalker-v3'), if_print=True)
#     args.break_step = int(8e5 * 8)  # (4e5) 8e5 (4e6), UsedTimes: (600s) 1500s (8000s)
#     args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (325)
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('ReacherBulletEnv-v0'), if_print=True)
#     args.break_step = int(2e6 * 8)  # (1e6) 2e6 (4e6), UsedTimes: 2000s (6000s)
#     args.reward_scale = 2 ** 0  # (-15) 0 ~ 18 (25)
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('AntBulletEnv-v0'), if_print=True)
#     args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime: 25697s
#     args.reward_scale = 2 ** -3  #
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#     args.net_dim = 2 ** 9
#     args.init_for_training()
#     train_agent_mp(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('MinitaurBulletEnv-v0'), if_print=True)
#     args.break_step = int(1e6 * 8)  # (4e5) 1e6 (8e6)
#     args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.net_dim = 2 ** 8
#     args.max_memo = 2 ** 11
#     args.batch_size = 2 ** 9
#     args.repeat_times = 2 ** 4
#     args.init_for_training()
#     train_agent_mp(args)
#     exit()
#
#     # args.env = decorate_env(gym.make('BipedalWalkerHardcore-v3'), if_print=True)  # 2020-08-24 plan
#     # on-policy (like PPO) is BAD at learning on a environment with so many random factors (like BipedalWalkerHardcore).
#     # exit()
#
#     args.env = fix_car_racing_env(gym.make('CarRacing-v0'))
#     # on-policy (like PPO) is GOOD at learning on a environment with less random factors (like 'CarRacing-v0').
#     # see 'train__car_racing__pixel_level_state2d()'
#
#
# def run__fin_rl():
#     env = FinanceMultiStockEnv()  # 2020-12-24
#
#     from AgentZoo import AgentPPO
#
#     args = Arguments(rl_agent=AgentPPO, env=env)
#     args.eval_times1 = 1
#     args.eval_times2 = 1
#     args.rollout_num = 4
#     args.if_break_early = False
#
#     args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
#     args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
#     args.net_dim = 2 ** 8
#     args.max_step = 1699
#     args.max_memo = 1699 * 16
#     args.batch_size = 2 ** 10
#     args.repeat_times = 2 ** 4
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     # from AgentZoo import AgentModSAC
#     #
#     # args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
#     # args.eval_times1 = 1
#     # args.eval_times2 = 2
#     #
#     # args.break_step = 2 ** 22  # UsedTime:
#     # args.net_dim = 2 ** 7
#     # args.max_memo = 2 ** 18
#     # args.batch_size = 2 ** 8
#     # args.init_for_training()
#     # train_agent_mp(args)  # train_agent(args)
#
#
# def train__car_racing__pixel_level_state2d():
#     from AgentZoo import AgentPPO
#
#     '''DEMO 4: Fix gym Box2D env CarRacing-v0 (pixel-level 2D-state, continuous action) using PPO'''
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     env = gym.make('CarRacing-v0')
#     env = fix_car_racing_env(env)
#
#     args = Arguments(rl_agent=AgentPPO, env=env, gpu_id=None)
#     args.if_break_early = True
#     args.eval_times2 = 1
#     args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.
#     args.rollout_num = 4  # (num, step, time) (8, 1e5, 1360) (4, 1e4, 1860)
#     args.random_seed += 1943
#
#     args.break_step = int(5e5 * 4)  # (1e5) 2e5 4e5 (8e5) used time (7,000s) 10ks 30ks (60ks)
#     # Sometimes bad luck (5%), it reach 300 score in 5e5 steps and don't increase.
#     # You just need to change the random seed and retrain.
#     args.reward_scale = 2 ** -2  # (-1) 50 ~ 700 ~ 900 (1001)
#     args.max_memo = 2 ** 11
#     args.batch_size = 2 ** 7
#     args.repeat_times = 2 ** 4
#     args.net_dim = 2 ** 7
#     args.max_step = 2 ** 10
#     args.show_gap = 2 ** 8  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    cwd = args.cwd
    env = args.env
    env_eval = args.env_eval
    agent_id = args.gpu_id
    agent_rl = args.agent_rl  # basic arguments

    gamma = args.gamma  # training arguments
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    show_gap = args.show_gap  # evaluate arguments
    eval_times = args.eval_times
    break_step = args.break_step
    if_break_early = args.if_break_early
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    del deepcopy
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, Evaluator, ReplayBuffer'''
    agent = agent_rl(net_dim, state_dim, action_dim)  # build AgentRL
    agent.state = env.reset()
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=agent.device, env=env_eval,
                          eval_times=eval_times, show_gap=show_gap)  # build Evaluator

    if_on_policy = agent_rl.__name__ in {'AgentPPO', 'AgentGaePPO'}
    buffer = ReplayBuffer(max_memo, state_dim, if_on_policy=if_on_policy,
                          action_dim=1 if if_discrete else action_dim)  # build experience replay buffer
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # update replay buffer
            steps = _explore_before_train(env, buffer, max_step, reward_scale, gamma)
        agent.update_policy(buffer, max_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step = steps

    if_solve = False
    while not ((if_break_early and if_solve)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        obj_a, obj_c = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        with torch.no_grad():  # speed up running
            if_solve = evaluator.evaluate_act__save_checkpoint(agent.act, steps, obj_a, obj_c)


'''multiprocessing training'''


def train_and_evaluate__multiprocessing(args):
    args.init_before_training()
    act_workers = args.rollout_num

    import multiprocessing as mp  # Python built-in multiprocessing library

    pipe1_eva, pipe2_eva = mp.Pipe()  # Pipe() for Process mp_evaluate_agent()
    pipe2_exp_list = list()  # Pipe() for Process mp_explore_in_env()

    process_train = mp.Process(target=mp__update_params, args=(args, pipe2_eva, pipe2_exp_list))
    process_evaluate = mp.Process(target=mp_evaluate_agent, args=(args, pipe1_eva))
    process = [process_train, process_evaluate]

    for worker_id in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        pipe2_exp_list.append(exp_pipe1)
        process.append(mp.Process(target=mp_explore_in_env, args=(args, exp_pipe2, worker_id)))

    [p.start() for p in process]
    process_train.join()
    process_evaluate.join()
    [p.terminate() for p in process]
    print('\n')


def mp__update_params(args, pipe1_eva, pipe1_exp_list):
    agent_rl = args.agent_rl  # basic arguments
    env = args.env
    cwd = args.cwd
    rollout_num = args.rollout_num

    gamma = args.gamma  # training arguments
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    break_step = args.break_step
    if_break_early = args.if_break_early
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''build agent'''
    agent = agent_rl(net_dim, state_dim, action_dim)  # build AgentRL
    pipe1_eva.send(agent.act)  # act = pipe2_eva.recv()
    if_on_policy = agent_rl.__name__ in {'AgentPPO', 'AgentGaePPO'}
    print(';0')

    agent.state = [pipe.recv() for pipe in pipe1_exp_list]
    if if_on_policy:
        agent.action, agent.noise = agent.select_actions(agent.state)
    else:
        agent.action = agent.select_actions(agent.state)
    for i in range(rollout_num):
        pipe1_exp_list[i].send(agent.action[i])

    buffer_mp = ReplayBufferMP(max_memo + max_step * rollout_num, state_dim,
                               if_on_policy=if_on_policy,
                               action_dim=1 if if_discrete else action_dim,
                               rollout_num=rollout_num)  # build experience replay buffer

    steps = 0
    if not if_on_policy:
        with torch.no_grad():  # update replay buffer
            for _buffer in buffer_mp.l_buffer:
                steps += _explore_before_train(env, _buffer, max_step // rollout_num, reward_scale, gamma)
        agent.update_policy(buffer_mp, max_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step = steps
    pipe1_eva.send((agent.act, steps, 0, 0.5))  # pipe1_eva (act, steps, obj_a, obj_c)

    if_solve = False
    while not ((if_break_early and if_solve)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer__pipe(pipe1_exp_list, buffer_mp, max_step)
        total_step += steps

        obj_a, obj_c = agent.update_policy(buffer_mp, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        pipe1_eva.send((agent.act, steps, obj_a, obj_c))  # pipe1_eva act_cpu
        if_solve = pipe1_eva.recv()

        if pipe1_eva.poll():
            if_solve = pipe1_eva.recv()  # pipe2_eva.send(if_solve)

    buffer_mp.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                               env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12
    pipe1_eva.send('stop')  # eva_pipe stop  # send to mp_evaluate_agent
    time.sleep(4)
    # print('; quit: params')


def mp_explore_in_env(args, pipe2_exp, worker_id):
    env = args.env
    reward_scale = args.reward_scale
    gamma = args.gamma
    del args

    random_seed = int(2 ** 16 * rd.rand() + worker_id)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    next_state = env.reset()
    pipe2_exp.send(next_state)
    while True:
        action = pipe2_exp.recv()  # pipe1_exp.send(action)
        next_state, reward, done, _ = env.step(action)

        reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
        if done:
            next_state = env.reset()
        pipe2_exp.send((reward_mask, next_state))


def mp_evaluate_agent(args, pipe2_eva):
    env = args.env
    env_eval = args.env_eval
    cwd = args.cwd
    agent_id = args.gpu_id
    show_gap = args.show_gap  # evaluate arguments
    eval_times = args.eval_times

    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    device = torch.device("cpu")
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=device, env=env_eval,
                          eval_times=eval_times, show_gap=show_gap)  # build Evaluator

    '''act_cpu without gradient for pipe1_eva'''
    act = pipe2_eva.recv()  # pipe1_eva.send(agent.act)
    act_cpu = deepcopy(act).to(torch.device("cpu"))  # for pipe1_eva
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    with torch.no_grad():  # speed up running
        act, steps, obj_a, obj_c = pipe2_eva.recv()  # pipe2_eva (act, steps, obj_a, obj_c)

        if_loop = True
        while if_loop:
            '''update actor'''
            while not pipe2_eva.poll():  # wait until pipe2_eva not empty
                time.sleep(1)
            steps_sum = 0
            while pipe2_eva.poll():  # receive the latest object from pipe
                q_i_eva_get = pipe2_eva.recv()  # pipe2_eva act
                if q_i_eva_get == 'stop':
                    if_loop = False
                    break
                act, steps, obj_a, obj_c = q_i_eva_get
                steps_sum += steps
            act_cpu.load_state_dict(act.state_dict())
            if_solve = evaluator.evaluate_act__save_checkpoint(act_cpu, steps_sum, obj_a, obj_c)
            pipe2_eva.send(if_solve)  # if_solve = pipe1_eva.recv()

            evaluator.save_npy__draw_plot()

    '''save the model, rename the directory'''
    new_cwd = cwd[:-2] + f'_{evaluator.r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    else:
        print(f'| SavedDir: {new_cwd}    WARNING: file exit')
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - evaluator.start_time:.0f}')

    while pipe2_eva.poll():  # empty the pipe
        pipe2_eva.recv()
    # print('; quit: evaluate')


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times, show_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0
        self.save_path = ''

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.show_gap = show_gap
        self.eva_times = eval_times
        self.env = env
        self.target_reward = env.target_reward

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

        import matplotlib as mpl  # draw figure in Terminal
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        # plt.style.use('ggplot')

        self.plt = plt
        fig, self.axs = plt.subplots(2)

    def evaluate_act__save_checkpoint(self, act, steps, obj_a, obj_c):
        reward_list = [_get_episode_return(self.env, act, self.device) for _ in range(self.eva_times)]
        r_avg = np.average(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg

            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")
        self.total_step += steps  # update total training steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

        if_solve = bool(self.r_max > self.target_reward)  # check if_solve
        if if_solve and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return if_solve

    def save_npy__draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy__draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''save recorder as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''plot subplots'''
        plt = self.plt
        axs = self.axs

        recorder = np.array(self.recorder)  # recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        steps = recorder[:, 0]  # x-axis is training steps
        r_avg = recorder[:, 1]
        r_std = recorder[:, 2]
        obj_a = recorder[:, 3]
        obj_c = recorder[:, 4]

        axs0 = axs[0]
        color0 = 'lightcoral'
        axs0.plot(steps, r_avg, label='Episode Return', color=color0)
        axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

        axs11 = axs[1]
        color11 = 'royalblue'
        label = 'objA'
        axs11.set_ylabel(label, color=color11)
        axs11.plot(steps, obj_a, label=label, color=color11)
        axs11.tick_params(axis='y', labelcolor=color11)

        ax12 = axs[1].twinx()
        color12 = 'darkcyan'
        ax12.set_ylabel('objC', color=color12)
        ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
        ax12.tick_params(axis='y', labelcolor=color12)

        '''plot title'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
        plt.title(save_title, y=2.3)

        '''plot save'''
        if self.save_path:  # remove old plot figure
            os.remove(self.save_path)
        self.save_path = f"{self.cwd}/{save_title}.jpg"
        plt.savefig(self.save_path)
        # plt.show()
        # plt.close()
        plt.clf()


def _get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step if hasattr(env, 'max_step') else 2 ** 10
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside

        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, 'episode_return') else episode_return


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_for_ppo(
        self.if_on_policy = if_on_policy

        if if_on_policy:
            other_dim = 1 + 1 + action_dim * 2
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
        else:
            other_dim = 1 + 1 + action_dim
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

    def append_memo(self, state, other):  # CPU array to CPU array
        if not self.if_on_policy:
            state = torch.as_tensor(state, device=self.device)
            other = torch.as_tensor(other, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_memo(self, state, other):  # CPU array to CPU array
        # assert isinstance(other, np.ndarray)
        if not self.if_on_policy:
            state = torch.as_tensor(state, device=self.device)
            other = torch.as_tensor(other, device=self.device)

        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def random_sample(self, batch_size):
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_for_ppo(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| print_norm: state_avg, state_fix_std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")


class ReplayBufferMP:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy, rollout_num):
        self.now_len = 0
        self.max_len = max_len
        self.rollout_num = rollout_num

        _max_len = max_len // rollout_num
        self.l_buffer = [ReplayBuffer(_max_len, state_dim, action_dim, if_on_policy)
                         for _ in range(rollout_num)]

    def extend_memo_mp(self, state, other, i):
        self.l_buffer[i].extend_memo(state, other)

    def random_sample(self, batch_size):
        _batch_size = batch_size // self.rollout_num

        l__r_m_a_s_ns = [self.l_buffer[i].random_sample(_batch_size)
                         for i in range(self.rollout_num)]
        return (torch.cat([item[0] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[1] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[2] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[3] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[4] for item in l__r_m_a_s_ns], dim=0))

    def sample_for_ppo(self):
        l__r_m_a_n_s = [self.l_buffer[i].sample_for_ppo()
                        for i in range(self.rollout_num)]
        return (torch.cat([item[0] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[1] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[2] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[3] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[4] for item in l__r_m_a_n_s], dim=0))

    def update__now_len__before_sample(self):
        self.now_len = 0
        for buffer in self.l_buffer:
            buffer.update__now_len__before_sample()
            self.now_len += buffer.now_len

    def empty_memories__before_explore(self):
        for buffer in self.l_buffer:
            buffer.empty_memories__before_explore()

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        for buffer in self.l_buffer:
            buffer.print_state_norm(neg_avg, div_std)


def _explore_before_train(env, buffer, target_step, reward_scale, gamma):
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        buffer.append_memo(state, other)

        state = env.reset() if done else next_state
    return steps


if __name__ == '__main__':
    timer = time.time()
    run__demo()
    print(time.time() - timer)
