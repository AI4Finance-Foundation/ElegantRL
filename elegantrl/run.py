import os
import time
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.tutorial.agent import ReplayBuffer


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically

        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8  # the network width
        self.batch_size = 2 ** 8  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.target_step = 2 ** 10  # collect target_step, then update network
        self.max_memo = 2 ** 17  # capacity of replay buffer
        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 8
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.eval_times = 2 ** 1  # evaluation times if 'eval_reward > target_reward'
        self.show_gap = 2 ** 8  # show the Reward and Loss value per show_gap seconds
        self.random_seed = 0  # initialize random seed in self.init_before_training(

    def init_before_training(self):
        self.gpu_id = '0' if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.env.env_name}_{self.gpu_id}' if self.cwd is None else self.cwd
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
    # from elegantrl.tutorial.run import Arguments, train_and_evaluate
    from elegantrl.tutorial.env import PreprocessEnv

    import gym
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    """DEMO 1: Discrete action env of gym"""
    args = Arguments(agent=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters

    '''choose an DRL algorithm'''
    from elegantrl.tutorial.agent import AgentDoubleDQN  # AgentDQN
    args.agent = AgentDoubleDQN()

    '''choose environment'''
    # args.env = PreprocessEnv(env=gym.make('CartPole-v0'))
    # args.net_dim = 2 ** 7  # change a default hyper-parameters
    # args.batch_size = 2 ** 7
    "TotalStep: 2e3, TargetReward: , UsedTime: 10s"
    args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
    args.net_dim = 2 ** 8
    args.batch_size = 2 ** 8
    "TotalStep: , TargetReward: , UsedTime: "

    '''train and evaluate'''
    train_and_evaluate(args)
    exit()

    '''DEMO 2: Continuous action env if gym'''
    '''DEMO 2.1: choose an off-policy DRL algorithm'''
    from elegantrl.tutorial.agent import AgentSAC  # AgentTD3, AgentDDPG
    args = Arguments(if_on_policy=False)
    args.agent = AgentSAC()
    '''DEMO 2.2: choose an on-policy DRL algorithm'''
    from elegantrl.tutorial.agent import AgentPPO  # AgentGaePPO
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()

    '''choose environment'''
    env = gym.make('Pendulum-v0')
    env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
    args.env = PreprocessEnv(env=env)
    args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
    args.net_dim = 2 ** 7
    args.batch_size = 2 ** 7
    "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
    # args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    # args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302
    "TotalStep: 9e4, TargetReward: 200, UsedTime: 2500s"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334
    # args.break_step = int(2e5)
    # args.if_allow_break = False
    "TotalStep: 2e5, TargetReward: 300, UsedTime: 5000s"

    '''train and evaluate'''
    train_and_evaluate(args)
    # train_and_evaluate__multiprocessing(args)  # try multiprocessing in complete version
    exit()

    '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
    args = Arguments(if_on_policy=True)
    '''choose an DRL algorithm'''
    from elegantrl.tutorial.agent import AgentPPO
    args.agent = AgentPPO()

    from elegantrl.env import FinanceMultiStockEnv  # a standard env for ElegantRL, not need PreprocessEnv()
    args.env = FinanceMultiStockEnv(if_train=True, train_beg=0, train_len=1024)
    args.env_eval = FinanceMultiStockEnv(if_train=False, train_beg=0, train_len=1024)  # eva_len = 1699 - train_len
    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 <
    args.break_step = int(5e6)
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.if_allow_break = False
    "TotalStep:  2e5, TargetReward: 1.25, UsedTime:  200s"
    "TotalStep:  4e5, TargetReward: 1.50, UsedTime:  400s"
    "TotalStep: 10e5, TargetReward: 1.62, UsedTime: 1000s"

    '''train and evaluate'''
    train_and_evaluate(args)
    # args.rollout_num = 8
    # train_and_evaluate__multiprocessing(args)  # try multiprocessing in complete version
    exit()


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    show_gap = args.show_gap
    eval_times = args.eval_times
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    buffer = ReplayBuffer(max_len=max_memo + max_step, if_on_policy=if_on_policy, if_gpu=True,
                          state_dim=state_dim, action_dim=1 if if_discrete else action_dim)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times=eval_times, show_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env.reset()
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)

        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if 'cri_target' in dir(agent) else None
    total_step = steps

    '''start training'''
    if_reach_goal = False
    while not ((if_break_early and if_reach_goal)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.store_transition(env, buffer, target_step, reward_scale, gamma)

        total_step += steps

        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)

        with torch.no_grad():  # speed up running
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, obj_a, obj_c)


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times, show_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

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

    def evaluate_save(self, act, steps, obj_a, obj_c):
        reward_list = [get_episode_return(self.env, act, self.device)
                       for _ in range(self.eva_times)]
        r_avg = np.average(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

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


def get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step
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


def explore_before_training(env, buffer, target_step, reward_scale, gamma):
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
        buffer.append_buffer(state, other)

        state = env.reset() if done else next_state
    return steps


if __name__ == '__main__':
    run__demo()
