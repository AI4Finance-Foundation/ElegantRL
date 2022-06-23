import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# import wandb


class Evaluator:
    def __init__(self, cwd, agent_id, eval_env, args):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'

        self.cwd = cwd
        self.agent_id = agent_id
        self.env_num = args.env_num
        self.eval_env = eval_env
        self.eval_gap = args.eval_gap
        self.eval_times = args.eval_times
        self.target_return = args.target_return
        self.target_step = args.target_step
        self.if_Isaac = args.if_Isaac
        self.tensorboard = SummaryWriter(f"{cwd}/tensorboard")
        self.evaluate_save_and_plot = self.evaluate_save_and_plot_reevaluate if args.reevaluate else self.evaluate_save_and_plot_raw

        self.r_max = -np.inf
        self.eval_time = 0
        self.eval_step = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'Time':>8} |"
              f"{'maxR':>8}{'curR':>8}{'curS':>8} |"
              f"{'objC':>8}{'objA':>7}{'etc.':>7}")

    def evaluate_save_and_plot_raw(self, act, steps, r_exp, step_exp, log_tuple) -> (bool, bool):
        self.total_step = steps

        # check if evaluate
        if self.total_step - self.eval_step < self.eval_gap:
            if_reach_goal = False
            if_checkpoint = False
        else:
            self.eval_time = time.time()
            self.eval_step = self.total_step

            '''save the policy network'''
            if_checkpoint = r_exp > self.r_max
            if if_checkpoint:  # save checkpoint with highest episode return
                    self.r_max = r_exp  # update max reward (episode return)
                    act_path = f"{self.cwd}/actor_{self.total_step:08}_{self.r_max:09.3f}.pth"
                    torch.save(act.state_dict(), act_path)  # save policy network in *.pth
                    # print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

            '''record the training information'''
            self.used_time = int(time.time() - self.start_time)
            self.recorder.append((self.total_step, r_exp, *log_tuple))  # update recorder

            wandb.log({'reward': r_exp, 'critic_loss': log_tuple[0], 'actor_loss': log_tuple[1]})

            self.tensorboard.add_scalar("info/critic_loss_sample", log_tuple[0], self.total_step)
            self.tensorboard.add_scalar("info/actor_obj_sample", -1 * log_tuple[1], self.total_step)
            self.tensorboard.add_scalar("info/env_step_sample", step_exp, self.total_step)
            self.tensorboard.add_scalar("reward/reward_sample", r_exp, self.total_step)

            self.tensorboard.add_scalar("info/critic_loss_time", log_tuple[0], self.used_time)
            self.tensorboard.add_scalar("info/actor_obj_time", -1 * log_tuple[1], self.used_time)
            self.tensorboard.add_scalar("info/env_step_sample", step_exp, self.used_time)
            self.tensorboard.add_scalar("reward/reward_time", r_exp, self.used_time)
            
            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return or self.total_step > self.target_step)  # check if_reach_goal

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.used_time:>8} |"
                    f"{self.r_max:8.2f}{r_exp:8.2f}{step_exp:8.2f} |"
                    f"{''.join(f'{n:7.2f}' for n in log_tuple)}")

            if len(self.recorder) == 0:
                    print("| save_npy_draw_plot() WARNING: len(self.recorder)==0")
                    return None
            np.save(self.recorder_path, self.recorder)

        return if_reach_goal, if_checkpoint

    def evaluate_save_and_plot_reevaluate(self, act, steps, r_exp, log_tuple) -> (bool, bool):
        self.total_step = steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
        # if self.total_step - self.eval_step < self.eval_gap:
            if_reach_goal = False
            if_checkpoint = False
        else:
            self.eval_time = time.time()
            self.eval_step = self.total_step

            if self.if_Isaac:
                rewards_steps_list = get_rewards_step_list_from_vec_env(self.eval_env, act)
                rewards_steps_ary = torch.Tensor(rewards_steps_list)
                r_avg, s_avg = rewards_steps_ary.mean(axis=0).cpu()  # average of episode return and episode step
                r_std, s_std = rewards_steps_ary.std(axis=0).cpu()  # standard dev. of episode return and episode step
            else:
                rewards_steps_list = [get_cumulative_returns_and_step(self.eval_env, act) for _ in range(self.eval_times)]
                rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
                r_avg, s_avg = rewards_steps_ary.mean(axis=0)
                r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step

            '''save the policy network'''
            if_checkpoint = r_avg > self.r_max
            if if_checkpoint:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_path = f"{self.cwd}/actor_{self.total_step:08}_{self.r_max:09.3f}.pth"
                torch.save(act.state_dict(), act_path)  # save policy network in *.pth

                print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

            '''record the training information'''
            self.used_time = int(time.time() - self.start_time)
            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder
            self.tensorboard.add_scalar("info/critic_loss_sample", log_tuple[0], self.total_step)
            self.tensorboard.add_scalar("info/actor_obj_sample", -1 * log_tuple[1], self.total_step)
            self.tensorboard.add_scalar("reward/avg_reward_sample", r_avg, self.total_step)
            self.tensorboard.add_scalar("reward/std_reward_sample", r_std, self.total_step)
            self.tensorboard.add_scalar("reward/exp_reward_sample", r_exp, self.total_step)

            self.tensorboard.add_scalar("info/critic_loss_time", log_tuple[0], self.used_time)
            self.tensorboard.add_scalar("info/actor_obj_time", -1 * log_tuple[1], self.used_time)
            self.tensorboard.add_scalar("reward/avg_reward_time", r_avg, self.used_time)
            self.tensorboard.add_scalar("reward/std_reward_time", r_std, self.used_time)
            self.tensorboard.add_scalar("reward/exp_reward_time", r_exp, self.used_time)

            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal:
                print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                      f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                      f"{'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                      f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8}  ########")

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")

            if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
                self.eval_env.curriculum_learning_for_evaluator(r_avg)

            '''plot learning curve figure'''
            if len(self.recorder) == 0:
                print("| save_npy_draw_plot() WARNING: len(self.recorder)==0")
                return None

            np.save(self.recorder_path, self.recorder)

            '''draw plot and save as figure'''
            train_time = int(time.time() - self.start_time)
            total_step = int(self.recorder[-1][0])
            save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

            save_learning_curve(self.recorder, self.cwd, save_title)

        return if_reach_goal, if_checkpoint

    def save_or_load_recoder(self, if_checkpoint):
        if if_checkpoint:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]


"""util"""


def get_cumulative_returns_and_step(env, act) -> (float, int):  # [ElegantRL.2022.03.03]
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pth'

    env = build_env(env_func=env_func, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        returns += reward
        if done:
            break
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1
    return returns, steps


def get_rewards_step_list_from_vec_env(env, act) -> list:  # todo
    device = env.device

    rewards_ary = list()
    dones_ary = list()
    state = env.reset()
    for _ in range(env.max_step):
        action = act(state.to(device))
        # assert action.shape == (env.env_num, env.action_dim)
        states, rewards, dones, info_dict = env.step(action)

        rewards_ary.append(rewards)
        dones_ary.append(dones)

    rewards_ary = torch.stack(rewards_ary)  # rewards_ary.shape == (env.max_step, env.env_num)
    dones_ary = torch.stack(dones_ary)
    # assert rewards_ary.shape == (env.env_num, )
    # assert dones_ary.shape == (env.env_num, )

    reward_list = list()
    steps_list = list()
    for i in range(env.env_num):
        dones_where = torch.where(dones_ary[:, i] == 1)[0]
        episode_num = dones_where.shape[0]

        if episode_num == 0:
            continue

        j0 = 0
        rewards_env = np.array(rewards_ary[:, i].cpu())
        for j1 in dones_where + 1:
            reward_list.append(rewards_env[j0:j1].sum())
            steps_list.append(j1 - j0 + 1)
            j0 = j1

    # reward_list = torch.tensor(reward_list, dtype=torch.float32)
    # steps_list = torch.tensor(steps_list, dtype=torch.float32)
    #
    # print(f'\n reward_list avg {reward_list.mean(0):9.2f}'
    #       f'\n             std {reward_list.std(0):9.2f}'
    #       f'\n  steps_list avg {steps_list.mean(0):9.2f}'
    #       f'\n             std {steps_list.std(0):9.2f}'
    #       f'\n     episode_num {steps_list.shape[0]}')
    # return reward_list, steps_list
    reward_step_list = list(zip(reward_list, steps_list))
    return reward_step_list


def save_learning_curve(
        recorder: list = None, cwd: str = '.',
        save_title: str = 'learning curve', fig_name: str = 'plot_learning_curve.jpg'
):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return', color=color0)
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


"""learning curve"""


def demo_evaluator_actor_pth():
    import gym
    from elegantrl.agents.AgentPPO import AgentPPO
    from elegantrl.train.config import build_env
    from elegantrl.train.config import Arguments

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent_class = AgentPPO

    env_func = gym.make
    env_args = {'env_num': 1,
                'env_name': 'LunarLanderContinuous-v2',
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False,
                'target_return': 200,

                'id': 'LunarLanderContinuous-v2'}

    # actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pth'
    eval_times = 4
    net_dim = 2 ** 7

    '''init'''
    args = Arguments(agent_class=agent_class, env_func=env_func, env_args=env_args)
    env = build_env(env_func=args.env_func, env_args=args.env_args)
    act = agent_class(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    # act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate'''
    r_s_ary = [get_cumulative_returns_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

    print('r_avg, s_avg', r_avg, s_avg)
    return r_avg, s_avg


def demo_evaluate_actors(dir_path: str, gpu_id: int, agent, env_args: dict, eval_times=2, net_dim=128):
    import gym
    from elegantrl.train.config import build_env
    # dir_path = './LunarLanderContinuous-v2_PPO_1'
    # gpu_id = 0
    # agent_class = AgentPPO
    # net_dim = 2 ** 7

    env_func = gym.make
    # env_args = {'env_num': 1,
    #             'env_name': 'LunarLanderContinuous-v2',
    #             'max_step': 1000,
    #             'state_dim': 8,
    #             'action_dim': 2,
    #             'if_discrete': False,
    #             'target_return': 200,
    #             'eval_times': 2 ** 4,
    #
    #             'id': 'LunarLanderContinuous-v2'}
    # eval_times = 2 ** 1

    '''init'''
    env = build_env(env_func=env_func, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act

    '''evaluate'''
    step_epi_r_s_ary = list()

    act_names = [name for name in os.listdir(dir_path) if len(name) == 19]
    from tqdm import tqdm
    for act_name in tqdm(act_names):
        act_path = f"{dir_path}/{act_name}"

        act.load_state_dict(torch.load(act_path, map_location=lambda storage, loc: storage))
        r_s_ary = [get_cumulative_returns_and_step(env, act) for _ in range(eval_times)]
        r_s_ary = np.array(r_s_ary, dtype=np.float32)
        r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

        step = int(act_name[6:15])

        step_epi_r_s_ary.append((step, r_avg, s_avg))

    step_epi_r_s_ary = np.array(step_epi_r_s_ary, dtype=np.float32)

    '''sort by step'''
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    return step_epi_r_s_ary


def demo_load_pendulum_and_render():
    import gym
    import torch
    from elegantrl.agents.AgentPPO import AgentPPO
    from elegantrl.train.config import build_env
    from elegantrl.train.config import Arguments

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent_class = AgentPPO

    env_func = gym.make
    env_args = {'env_num': 1,
                'env_name': 'Pendulum-v1',
                'max_step': 200,
                'state_dim': 3,
                'action_dim': 1,
                'if_discrete': False,

                'id': 'Pendulum-v1'}

    actor_path = './Pendulum-v1_PPO_0/actor.pth'
    net_dim = 2 ** 7

    '''init'''
    env = build_env(env_func=env_func, env_args=env_args)
    args = Arguments(agent_class, env=env)
    act = agent_class(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate'''
    # eval_times = 2 ** 7
    # from elegantrl.envs.CustomGymEnv import PendulumEnv
    # eval_env = PendulumEnv()
    # from elegantrl.train.evaluator import get_cumulative_returns_and_step
    # r_s_ary = [get_cumulative_returns_and_step(eval_env, act) for _ in range(eval_times)]
    # r_s_ary = np.array(r_s_ary, dtype=np.float32)
    # r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    #
    # print('r_avg, s_avg', r_avg, s_avg)
    # exit()

    '''render'''
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action * 2)  # todo
        returns += reward
        env.render()

        if done:
            break
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1

    print(f"\n| cumulative_returns {returns}"
          f"\n|      episode steps {steps}")


def run():
    from elegantrl.agents.AgentPPO import AgentPPO
    flag_id = 1  # int(sys.argv[1])

    gpu_id = [2, 3][flag_id]
    agent = AgentPPO
    env_args = [
        {'env_num': 1,
         'env_name': 'LunarLanderContinuous-v2',
         'max_step': 1000,
         'state_dim': 8,
         'action_dim': 2,
         'if_discrete': False,
         'target_return': 200,
         'eval_times': 2 ** 4,
         'id': 'LunarLanderContinuous-v2'},

        {'env_num': 1,
         'env_name': 'BipedalWalker-v3',
         'max_step': 1600,
         'state_dim': 24,
         'action_dim': 4,
         'if_discrete': False,
         'target_return': 300,
         'eval_times': 2 ** 3,
         'id': 'BipedalWalker-v3', },
    ][flag_id]
    env_name = env_args['env_name']

    print('gpu_id', gpu_id)
    print('env_name', env_name)

    '''save step_epi_r_s_ary'''
    # cwd_path = '.'
    # dir_names = [name for name in os.listdir(cwd_path)
    #              if name.find(env_name) >= 0 and os.path.isdir(name)]
    # for dir_name in dir_names:
    #     dir_path = f"{cwd_path}/{dir_name}"
    #     step_epi_r_s_ary = demo_evaluate_actors(dir_path, gpu_id, agent, env_args)
    #     np.savetxt(f"{dir_path}-step_epi_r_s_ary.txt", step_epi_r_s_ary)

    '''load step_epi_r_s_ary'''
    step_epi_r_s_ary = list()

    cwd_path = '.'
    ary_names = [name for name in os.listdir('.')
                 if name.find(env_name) >= 0 and name[-4:] == '.txt']
    for ary_name in ary_names:
        ary_path = f"{cwd_path}/{ary_name}"
        ary = np.loadtxt(ary_path)
        step_epi_r_s_ary.append(ary)
    step_epi_r_s_ary = np.vstack(step_epi_r_s_ary)
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    print('step_epi_r_s_ary.shape', step_epi_r_s_ary.shape)

    '''plot'''
    import matplotlib.pyplot as plt
    # plt.plot(step_epi_r_s_ary[:, 0], step_epi_r_s_ary[:, 1])

    plot_x_y_up_dw_step = list()
    n = 8
    for i in range(0, len(step_epi_r_s_ary), n):
        y_ary = step_epi_r_s_ary[i:i + n, 1]
        if y_ary.shape[0] <= 1:
            continue

        y_avg = y_ary.mean()
        y_up = y_ary[y_ary > y_avg].mean()
        y_dw = y_ary[y_ary <= y_avg].mean()

        y_step = step_epi_r_s_ary[i:i + n, 2].mean()
        x_avg = step_epi_r_s_ary[i:i + n, 0].mean()
        plot_x_y_up_dw_step.append((x_avg, y_avg, y_up, y_dw, y_step))

    if_show_episode_step = True
    color0 = 'royalblue'
    color1 = 'lightcoral'
    # color2 = 'darkcyan'
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    title = f"{env_name}_{agent.__name__}_ElegantRL"

    fig, ax = plt.subplots(1)

    plot_x = [item[0] for item in plot_x_y_up_dw_step]
    plot_y = [item[1] for item in plot_x_y_up_dw_step]
    plot_y_up = [item[2] for item in plot_x_y_up_dw_step]
    plot_y_dw = [item[3] for item in plot_x_y_up_dw_step]
    ax.plot(plot_x, plot_y, label='Episode Return', color=color0)
    ax.fill_between(plot_x, plot_y_up, plot_y_dw, facecolor=color0, alpha=0.3)
    ax.set_ylabel('Episode Return', color=color0)
    ax.tick_params(axis='y', labelcolor=color0)
    ax.grid(True)

    if if_show_episode_step:
        ax_twin = ax.twinx()
        plot_y_step = [item[4] for item in plot_x_y_up_dw_step]
        ax_twin.fill_between(plot_x, 0, plot_y_step, facecolor=color1, alpha=0.3)
        ax_twin.set_ylabel('Episode Step', color=color1)
        ax_twin.tick_params(axis='y', labelcolor=color1)
        ax_twin.set_ylim(0, np.max(plot_y_step) * 2)

    print('title', title)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # demo_evaluate_actors()
    run()
