from AgentRun import *
from AgentNet import *
from AgentZoo import *

""" ModSAC
print loss_a_avg los_a_std (adjust reward_scale to keep loss_a_std near 1)
"""


def draw_plot_with_2npy(cwd, train_time, max_reward):  # 2020-07-07
    record_explore = np.load('%s/record_explore.npy' % cwd)  # , allow_pickle=True)
    # record_explore.append((total_step, exp_r_avg, pg_avg, pg_std, loss_c_avg))
    record_evaluate = np.load('%s/record_evaluate.npy' % cwd)  # , allow_pickle=True)
    # record_evaluate.append((total_step, eva_r_avg, eva_r_std))

    if len(record_evaluate.shape) == 1:
        record_evaluate = np.array([[0, 0, 0]], dtype=np.float32)
    if len(record_explore.shape) == 1:  # todo fix bug
        record_explore = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

    train_time = int(train_time)
    total_step = int(record_evaluate[-1][0])
    save_title = f"plot_Step_Time_maxR_{total_step:8.3e}_{train_time:8.3e}_{max_reward:8.3e}"
    save_path = "{}/{}.png".format(cwd, save_title)

    """plot"""
    import matplotlib as mpl  # draw figure in Terminal
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # plt.style.use('ggplot')

    fig, axs = plt.subplots(2)
    plt.title(save_title, y=2.3)

    ax11 = axs[0]
    ax11_color = 'royalblue'
    ax11_label = 'explore R'
    exp_step = record_explore[:, 0]
    exp_reward = record_explore[:, 1]
    ax11.plot(exp_step, exp_reward, label=ax11_label, color=ax11_color)

    ax12 = axs[0]
    ax12_color = 'lightcoral'
    ax12_label = 'Epoch R'
    eva_step = record_evaluate[:, 0]
    r_avg = record_evaluate[:, 1]
    r_std = record_evaluate[:, 2]
    ax12.plot(eva_step, r_avg, label=ax12_label, color=ax12_color)
    ax12.fill_between(eva_step, r_avg - r_std, r_avg + r_std, facecolor=ax12_color, alpha=0.3, )

    ax21 = axs[1]
    ax21_color = 'royalblue'
    ax21_label = 'Q value'  # call it Q value estimate, policy gradient, or negative actor loss
    q_avg = -record_explore[:, 2]
    q_std = -record_explore[:, 3]
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(exp_step, q_avg, label=ax21_label, color=ax21_color)  # negative loss A
    ax21.fill_between(exp_step, q_avg - q_std, q_avg + q_std, facecolor=ax21_color, alpha=0.3, )
    ax21.tick_params(axis='y', labelcolor=ax21_color)

    ax22 = axs[1].twinx()
    ax22_color = 'darkcyan'
    ax22_label = 'lossC'
    exp_loss_c = record_explore[:, 4]
    ax22.set_ylabel(ax22_label, color=ax22_color)
    ax22.fill_between(exp_step, exp_loss_c, facecolor=ax22_color, alpha=0.2, )
    ax22.tick_params(axis='y', labelcolor=ax22_color)

    # todo remove prev figure
    prev_save_names = [name for name in os.listdir(cwd) if name[:9] == save_title[:9]]
    os.remove(f'{cwd}/{prev_save_names[0]}') if len(prev_save_names) > 0 else None

    plt.savefig(save_path)
    # plt.pause(4)
    # plt.show()
    plt.close()


class Recorder:
    def __init__(self, eval_size1=3, eval_size2=9):
        self.eva_r_max = -np.inf
        self.total_step = 0
        self.record_exp = list()  # total_step, exp_r_avg, pg_avg, pg_std, loss_c_avg
        self.record_eva = list()  # total_step, eva_r_avg, eva_r_std
        self.is_solved = False

        '''constant'''
        self.eva_size1 = eval_size1
        self.eva_size2 = eval_size2

        '''print_reward'''
        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()

        print(f"{'GPU':>3}{'Step':>8} {'MaxR':>8}|"
              f"{'avgR':>8} {'stdR':>8} {'ExpR':>8}|"
              f"{'avgQ':>8} {'stdQ':>8} {'LossC':>8}")

    def update__record_evaluate(self, env, act, max_step, device, is_discrete):  # todo self.eva_size2
        is_saved = False
        reward_list = [get_episode_reward(env, act, max_step, device, is_discrete)
                       for _ in range(self.eva_size1)]

        eva_r_avg = np.average(reward_list)
        if eva_r_avg > self.eva_r_max:  # check 1
            reward_list.extend([get_episode_reward(env, act, max_step, device, is_discrete)
                                for _ in range(self.eva_size2 - self.eva_size1)])
            eva_r_avg = np.average(reward_list)
            if eva_r_avg > self.eva_r_max:  # check final
                self.eva_r_max = eva_r_avg
                is_saved = True

        eva_r_std = np.std(reward_list)
        self.record_eva.append((self.total_step, eva_r_avg, eva_r_std))
        return is_saved

    def update__record_explore(self, exp_s_sum, exp_r_avg, pg_avg, pg_std, loss_c):
        """pg: policy gradient
        or call it Q value estimate, negative actor loss
        it is better to adjust reward_scale to keep it's std near 1.0, avg (10~100)
        """
        if isinstance(exp_s_sum, int):
            exp_s_sum = (exp_s_sum,)
            exp_r_avg = (exp_r_avg,)
        for s, r in zip(exp_s_sum, exp_r_avg):
            self.total_step += s
            self.record_exp.append((self.total_step, r, pg_avg, pg_std, loss_c))

    def save_act(self, cwd, act, gpu_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{gpu_id:<3}{self.total_step:8.2e} {self.eva_r_max:8.2f}|")

    def check_is_solved(self, target_reward, gpu_id, show_gap):
        if self.eva_r_max > target_reward:
            self.is_solved = True
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'GPU':>3}{'Step':>8} {'TargetR':>8}|"
                      f"{'avgR':>8} {'stdR':>8} {'ExpR':>8}|"
                      f"{'UsedTime':>8} ########")

                total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
                total_step, exp_r_avg, pg_avg, pg_std, loss_c_avg = self.record_exp[-1]
                print(f"{gpu_id:<3}{total_step:8.2e} {target_reward:8.2f}|"
                      f"{eva_r_avg:8.2f} {eva_r_std:8.2f} {exp_r_avg:8.2f}|"
                      f"{self.used_time:>8} ########")

        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()

            total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
            total_step, exp_r_avg, pg_avg, pg_std, loss_c_avg = self.record_exp[-1]
            print(f"{gpu_id:<3}{total_step:8.2e} {self.eva_r_max:8.2f}|"
                  f"{eva_r_avg:8.2f} {eva_r_std:8.2f} {exp_r_avg:8.2f}|"
                  f"{pg_avg:8.2f} {pg_std:8.2f} {loss_c_avg:8.2f}")
        return self.is_solved

    def save_npy__plot_png(self, cwd):
        np.save('%s/record_explore.npy' % cwd, self.record_exp)
        np.save('%s/record_evaluate.npy' % cwd, self.record_eva)
        draw_plot_with_2npy(cwd, train_time=time.time() - self.start_time, max_reward=self.eva_r_max)

    def demo(self):
        pass


def train_agent(
        rl_agent, env_name, gpu_id, cwd,
        net_dim, max_memo, max_step, batch_size, repeat_times, reward_scale, gamma,
        break_step, if_break_early, show_gap, eval_times1, eval_times2, **_kwargs):  # 2020-09-18
    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)

    '''init: agent, buffer, recorder'''
    recorder = Recorder(eval_size1=eval_times1, eval_size2=eval_times2)  # todo eva_size1
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    if_online_policy = bool(rl_agent.__name__ in {'AgentPPO', 'AgentGAE', 'AgentInterGAE', 'AgentDiscreteGAE'})
    if if_online_policy:
        buffer = BufferTupleOnline(max_memo)
    else:
        buffer = BufferArray(max_memo, state_dim, 1 if if_discrete else action_dim)
        with torch.no_grad():  # update replay buffer
            rewards, steps = initial_exploration(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        recorder.update__record_explore(steps, rewards, pg_avg=0, pg_std=0, loss_c=0)

    '''loop'''
    if_train = True
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.init_before_sample()
        pg_avg, pg_std, loss_c = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, pg_avg, pg_std, loss_c)

            if_save = recorder.update__record_evaluate(env, agent.act, max_step, agent.device, if_discrete)
            recorder.save_act(cwd, agent.act, gpu_id) if if_save else None
            recorder.save_npy__plot_png(cwd)

            if_solve = recorder.check_is_solved(target_reward, gpu_id, show_gap)

        '''break loop rules'''
        if_train = not ((if_break_early and if_solve)
                        or recorder.total_step > break_step
                        or os.path.exists(f'{cwd}/stop.mark'))
    recorder.save_npy__plot_png(cwd)
    buffer.print_state_norm()


def mp__update_params(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    class_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    if_stop = args.if_break_early
    del args

    state_dim, action_dim = q_o_buf.get()  # q_o_buf 1.
    agent = class_agent(state_dim, action_dim, net_dim)

    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_buf.put(act_cpu)  # q_i_buf 1.
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    buffer = BufferArrayGPU(max_memo, state_dim, action_dim)  # experiment replay buffer

    '''initial_exploration'''
    buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf 2.
    reward_avg = np.average(reward_list)
    step_sum = sum(step_list)
    buffer.extend_memo(buffer_array)
    q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0, 0))  # q_i_eva 1.

    total_step = step_sum
    if_train = True
    if_solve = False
    while if_train:
        buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf n.
        reward_avg = np.average(reward_list)
        step_sum = sum(step_list)
        total_step += step_sum
        buffer.extend_memo(buffer_array)

        buffer.init_before_sample()
        pg_avg, pg_std, loss_c_avg = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_buf.put(act_cpu)  # q_i_buf n.
        q_i_eva.put((act_cpu, reward_avg, step_sum, pg_avg, pg_std, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            if_solve = q_o_eva.get()  # q_o_eva n.
        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop.mark'))

    buffer.print_state_norm()

    q_i_buf.put('stop')
    q_i_eva.put('stop')
    while q_i_buf.qsize() > 0 or q_i_eva.qsize() > 0:
        time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def mp_evaluate_agent(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
    env_name = args.env_name
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_step
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    env, state_dim, action_dim, target_reward, is_discrete = build_gym_env(env_name, if_print=True)

    '''build evaluated only actor'''
    q_i_eva_get = q_i_eva.get()  # q_i_eva 1.
    act = q_i_eva_get  # q_i_eva_get == act.to(device_cpu), requires_grad=False

    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update__record_evaluate(env, act, max_step, device, is_discrete)

    is_training = True
    with torch.no_grad():  # for saving the GPU buffer
        while is_training:
            is_saved = recorder.update__record_evaluate(env, act, max_step, device, is_discrete)
            recorder.save_act(cwd, act, gpu_id) if is_saved else None
            recorder.save_npy__plot_png(cwd)

            is_solved = recorder.check_is_solved(target_reward, gpu_id, show_gap)
            q_o_eva.put(is_solved)  # q_o_eva n.

            '''update actor'''
            while q_i_eva.qsize() == 0:  # wait until q_i_eva has item
                time.sleep(1)
            while q_i_eva.qsize():  # get the latest actor
                q_i_eva_get = q_i_eva.get()  # q_i_eva n.
                if q_i_eva_get == 'stop':
                    is_training = False
                    break
                act, exp_r_avg, exp_s_sum, pg_avg, pg_std, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, pg_avg, pg_std, loss_c_avg)

    recorder.save_npy__plot_png(cwd)

    while q_o_eva.qsize() > 0:
        q_o_eva.get()
    while q_i_eva.qsize() > 0:
        q_i_eva.get()
    # print('; quit: evaluate')


def train_agent_mp(args):
    import multiprocessing as mp
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O
    process = [mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
               mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),
               mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


class AgentModSAC(AgentBasicAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        use_dn = True  # and use hard target update
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = ActorSAC(state_dim, action_dim, actor_dim, use_dn).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = ActorSAC(state_dim, action_dim, net_dim, use_dn).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        critic_dim = int(net_dim * 1.25)
        self.cri = CriticTwinShared(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwinShared(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim + 1) * 0.5
        self.log_alpha = torch.tensor((-self.target_entropy * np.e,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        q_eval_pg = None
        critic_loss = None

        alpha = self.log_alpha.exp().detach()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size = int(batch_size * k)  # increase batch_size
        train_step = int(max_step * k)  # increase training_step
        update_a = 0
        for update_c in range(1, train_step):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.cri_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # policy entropy
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = (self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)).mean()  # todo
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin # todo

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * loss_c_tmp  # soft update
            lamb = np.exp(-self.avg_loss_c ** 2)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''stochastic policy'''
            actions_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient

            '''auto temperature parameter: alpha'''
            alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
            alpha = self.log_alpha.exp().detach()

            if update_a / update_c < 0.5 + lamb:  # auto TTUR
                update_a += 1  # auto TTUR
                '''actor_loss'''
                q_eval_pg = torch.min(*self.cri.get__q1_q2(state, actions_noise))  # policy gradient, stable but slower

                actor_loss = -(q_eval_pg + log_prob * alpha).mean()  # policy gradient
                self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * lamb
                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

                """target update"""
                soft_target_update(self.act_target, self.act)  # soft target update
            """target update"""
            soft_target_update(self.cri_target, self.cri)  # soft target update

        return q_eval_pg.mean().item(), q_eval_pg.std().item(), critic_loss.item() * 0.5


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    # """offline policy"""
    # rl_agent = Zoo.AgentModSAC
    # assert rl_agent in {Zoo.AgentDDPG,  # 2014. simple, old, slow, unstable
    #                     Zoo.AgentBasicAC,  # 2014+ stable DDPG
    #                     Zoo.AgentTD3,  # 2018. twin critics, delay target update
    #                     Zoo.AgentSAC,  # 2018. twin critics, policy entropy, auto alpha
    #                     Zoo.AgentModSAC,  # 2018+ stable SAC
    #                     Zoo.AgentInterAC,  # 2019. Integrated AC(DPG)
    #                     Zoo.AgentInterSAC,  # 2020. Integrated SAC(SPG)
    #                     }  # PPO, GAE is online policy. See below.
    rl_agent = AgentModSAC

    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = True
    args.if_remove_history = True

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # # max_reward I get:
    # args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    # args.reward_scale = 2 ** -2
    # args.init_for_training()
    # train_agent(**vars(args))  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    # exit()

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(5e4 * 16)
    # args.reward_scale = 2 ** 0
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()
    #
    print(';;;;0')
    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)
    args.reward_scale = 2 ** 0
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    print(';;;;1')
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(5e6 * 4)
    args.reward_scale = 2 ** -3  # todo
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_offline_policy(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(1e6 * 4)
    args.reward_scale = 2 ** 4  # todo
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.net_dim = 2 ** 8
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training(cpu_threads=4)
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.break_step = int(4e6 * 8)
    args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    args.max_memo = int(2 ** 21)
    args.batch_size = int(2 ** 8)
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_offline_policy(**vars(args))
    exit()


run_continuous_action()
