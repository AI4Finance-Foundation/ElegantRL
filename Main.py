import os
import sys
import time
import torch
import numpy as np
import numpy.random as rd


class Arguments:
    def __init__(self, rl_agent=None, env=None, gpu_id=None):
        self.rl_agent = rl_agent  # Deep Reinforcement Learning algorithm
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.break_step = 2 ** 17  # break training after 'total_step > break_step'
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.show_gap = 2 ** 8  # show the Reward and Loss of actor and critic per show_gap seconds
        self.eval_times = 2 ** 3  # evaluation times if 'eval_reward > target_reward'
        self.random_seed = 0

    def init_before_training(self):
        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.rl_agent.__name__}/{self.env.env_name}_{self.gpu_id}' if self.cwd is None else self.cwd
        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

        import shutil  # weather remove history?
        if self.if_remove is None:
            self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
        if self.if_remove:
            shutil.rmtree(self.cwd, ignore_errors=True)
            print("| Remove history")
        os.makedirs(self.cwd, exist_ok=True)
        del shutil

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def main():
    args = Arguments(rl_agent=None, env=None, gpu_id=0)
    from Env import decorate_env

    '''DEMO 1: Discrete action env: CartPole-v0 of gym'''
    import gym
    args.env = decorate_env(env=gym.make('CartPole-v0'))
    from Agent import AgentD3QN
    args.rl_agent = AgentD3QN  # Dueling Double DQN
    args.net_dim = 2 ** 7
    train_and_evaluate(args)
    exit()

    '''DEMO 2: Continuous action env: LunarLanderContinuous-v2 of gym.box2D'''
    import gym
    args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
    from Agent import AgentModSAC
    args.rl_agent = AgentModSAC  # Modified SAC (off-policy)

    args.break_step = int(6e4 * 8)  # UsedTime 900s (reach target_reward 200)
    args.net_dim = 2 ** 7
    train_and_evaluate(args)
    exit()

    '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
    from Env import FinanceMultiStockEnv
    args.env = FinanceMultiStockEnv()  # a standard env for ElegantRL, not need decorate_env()
    from Agent import AgentPPO
    args.rl_agent = AgentPPO  # PPO+GAE (on-policy)

    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = (args.max_step - 1) * 16
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.init_before_training()
    train_and_evaluate(args)
    exit()


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    rl_agent = args.rl_agent
    gpu_id = args.gpu_id
    env = args.env
    cwd = args.cwd

    '''training arguments'''
    gamma = args.gamma
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    '''evaluate arguments'''
    break_step = args.break_step
    if_break_early = args.if_break_early
    show_gap = args.show_gap
    eval_times = args.eval_times
    del args  # In order to show these hyper-parameters clearly, I put them above.

    if_on_policy = rl_agent.__name__ in {'AgentPPO', 'AgentGaePPO'}

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    target_reward = env.target_reward
    from copy import deepcopy  # built-in library of Python
    env_eval = deepcopy(env)
    del deepcopy

    '''build rl_agent'''
    agent = rl_agent(net_dim, state_dim, action_dim)
    agent.state = env.reset()

    '''build ReplayBuffer'''
    buffer = ReplayBuffer(max_memo, state_dim, action_dim=1 if if_discrete else action_dim,
                          if_on_policy=if_on_policy)
    total_step = 0
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # update replay buffer
            steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step += steps

    '''build Recorder'''
    evaluator = Evaluator(eval_times)
    with torch.no_grad():
        evaluator.evaluate_and_save_checkpoint(env_eval, agent.act, agent.device, steps, agent.obj_a, agent.obj_c)

    '''loop'''
    if_solve = False
    while not ((if_break_early and if_solve) or total_step > break_step or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)

        with torch.no_grad():  # for saving the GPU buffer
            if_save = evaluator.evaluate_and_save_checkpoint(env_eval, agent.act, agent.device, steps, agent.obj_a,
                                                             agent.obj_c)
            evaluator.save_checkpoint(cwd, agent.act, gpu_id) if if_save else None
            if_solve = evaluator.evaluate(target_reward, gpu_id, show_gap, cwd)


def explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim):
    state = env.reset()
    steps = 0

    while steps < max_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        memo_tuple = (scaled_reward, mask, *state, action, *next_state) if if_discrete else \
            (scaled_reward, mask, *state, *action, *next_state)  # not elegant but ok
        buffer.append_memo(memo_tuple)

        state = env.reset() if done else next_state
    return steps


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.next_idx = 0
        self.is_full = False
        self.max_len = max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim=1, done_dim=1
        self.action_idx = self.state_idx + action_dim

        last_dim = action_dim if if_on_policy else state_dim
        self.memo_dim = 1 + 1 + state_dim + action_dim + last_dim
        self.memories = np.empty((max_len, self.memo_dim), dtype=np.float32)

    def append_memo(self, memo_tuple):
        self.memories[self.next_idx] = memo_tuple
        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def random_sample(self, batch_size):
        indices = rd.randint(self.now_len, size=batch_size)
        memory = torch.as_tensor(self.memories[indices], device=self.device)
        return (memory[:, 0:1],  # rewards
                memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
                memory[:, 2:self.state_idx],  # states
                memory[:, self.state_idx:self.action_idx],  # actions
                memory[:, self.action_idx:],)  # next_states

    def all_sample(self):
        tensors = (self.memories[:self.now_len, 0:1],  # rewards
                   self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
                   self.memories[:self.now_len, 2:self.state_idx],  # states
                   self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
                   self.memories[:self.now_len, self.action_idx:],)  # next_states or log_prob_sum
        return [torch.tensor(ary, device=self.device) for ary in tensors]

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.is_full = False


class Evaluator:
    def __init__(self, eval_size):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.is_solved = False
        self.total_step = 0
        self.eva_size = eval_size  # constant

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def evaluate_and_save_checkpoint(self, env, act, device, step_sum, obj_a, obj_c):
        is_saved = False
        reward_list = [get_episode_return(env, act, device) for _ in range(self.eva_size)]

        r_avg = np.average(reward_list)
        if r_avg > self.r_max:  # check final
            self.r_max = r_avg
            is_saved = True

        r_std = float(np.std(reward_list))
        self.total_step += step_sum
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        return is_saved

    def evaluate(self, target_reward, agent_id, show_gap, _cwd):
        total_step, r_avg, r_std, obj_a, obj_c = self.recorder[-1]

        self.is_solved = bool(self.r_max > target_reward)
        if self.is_solved and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{agent_id:<2}  {total_step:8.2e}  {target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()
            print(f"{agent_id:<2}  {total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return self.is_solved

    def save_checkpoint(self, cwd, act, agent_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")


def get_episode_return(env, act, device) -> float:
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


if __name__ == '__main__':
    main()
