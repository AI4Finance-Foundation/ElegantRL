import os
import gym
import time
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.replay import ReplayBuffer

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


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
            self.batch_size = 2 ** 9
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.if_per = False  # Prioritized Experience Replay for sparse reward

        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)
        self.eval_gap = 2 ** 5  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main=True):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| There should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.env.env_name}_{self.gpu_id}'

        if if_main:
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


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    if_per = args.if_per
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    eval_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    if args.env_eval is not None:
        env_eval = args.env_eval
    else:
        env_eval = deepcopy(env)

    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, if_per)
    if_on_policy = getattr(agent, 'if_on_policy', False)


    buffer = ReplayBuffer(max_len=max_memo + max_step, state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                          if_on_policy=if_on_policy, if_per=if_per, if_gpu=True)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2, )

    '''prepare for training'''
    agent.state = env.reset()
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)
        print('Begin updating network before training')
        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps

    '''start training'''
    if_reach_goal = False
    print('start training')
    while not ((if_break_early and if_reach_goal)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        print('explore environment')
        steps = agent.explore_env(env, buffer, target_step, reward_scale, gamma)
        total_step += steps
        print('update environment: total_steps={}'.format(total_step))
        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)

        if_reach_goal = evaluator.evaluate_save(agent, steps, obj_a, obj_c)
        evaluator.draw_plot()

    print(f'| SavedDir: {cwd}\n| UsedTime: {time.time() - evaluator.start_time:.0f}')


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # a early time
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8}   {'avgQ':>8}  {'avgL':>8} |"
              f"{'avgS':>6}  {'stdS':>4}")

    def evaluate_save(self, agent, steps, obj_a, obj_c) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()

            rewards_steps_list = [get_episode_return(self.env, agent, self.device) for _ in range(self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [get_episode_return(self.env, agent, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                '''save actor.pth'''
                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(agent.act.state_dict(), act_save_path)
                print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_return:8.2f} |"
                      f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.6f}  {r_std:8.6f}   {obj_a:8.6f}  {obj_c:8.6f} |"
                  f"{s_avg:6.0f}  {s_std:4.0f}")
        else:
            if_reach_goal = False
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''convert to array and save as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return(env, agent, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        action = agent.get_best_act(s_tensor)
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    #episode_return = getattr(env, 'episode_return', episode_return)
    episode_return = env.get_episode_return()
    return episode_return, episode_step + 1


def save_learning_curve(recorder, cwd='.', save_title='learning curve'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_a = recorder[:, 3]
    obj_c = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, figsize=(20, 16))

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/plot_learning_curve.jpg")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def explore_before_training(env, buffer, target_step, reward_scale, gamma) -> int:
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0

    print('start exploring before training')
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
