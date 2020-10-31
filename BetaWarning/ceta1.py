from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""FixISAC
beta2 a_std_log.clamp_max(0), LL
beta3 a_std_log.clamp_max(0), BW

ceta2 -a_std_log.abs(), LL
ceta3 -a_std_log.abs(), BW
beta0 -a_std_log.abs(), Minitaur

ceta1 -a_std_log.abs(), Minitaur (tanh*1.01).clip(-1, 1)
ceta0 -a_std_log.abs(), Minitaur (tanh*1.1).clip(-1, 1)
ceta2 -a_std_log.abs(), Minitaur (tanh*1.1).clip(-1, 1), net 2 ** 7
ceta3 -a_std_log.abs(), Minitaur (tanh*1.1).clip(-1, 1), net 2 ** 7, reward_scale = 2 ** 4

beta2 -a_std_log.abs(), Minitaur (tanh*1.1).clip(-1, 1), net 2 ** 7, fix_term / 8
beta0 -a_std_log.abs(), net 2 ** 7, fix_term / 8
"""


def build_gym_env(env_name, if_print=True, if_norm=True):
    assert env_name is not None

    '''UserWarning: WARN: Box bound precision lowered by casting to float32
    https://stackoverflow.com/questions/60149105/
    userwarning-warn-box-bound-precision-lowered-by-casting-to-float32
    '''
    gym.logger.set_level(40)  # non-essential

    # env = gym.make(env_name)
    # state_dim, action_dim, action_max, target_reward, is_discrete = get_env_info(env, if_print)
    '''env compatibility'''  # some env need to adjust.
    if env_name == 'Pendulum-v0':
        env = gym.make(env_name)
        env.spec.reward_threshold = -200.0  # target_reward
        state_dim, action_dim, action_max, target_reward, is_discrete = get_env_info(env, if_print)
    elif env_name == 'CarRacing-v0':
        from AgentPixel import fix_car_racing_v0
        env = gym.make(env_name)
        env = fix_car_racing_v0(env)
        state_dim, action_dim, action_max, target_reward, is_discrete = get_env_info(env, if_print)
        assert len(state_dim)
        # state_dim = (2, state_dim[0], state_dim[1])  # two consecutive frame (96, 96)
        state_dim = (1, state_dim[0], state_dim[1])  # one frame (96, 96)
    elif env_name == 'MultiWalker':
        from multiwalker_base import MultiWalkerEnv, multi_to_single_walker_decorator
        env = MultiWalkerEnv()
        env = multi_to_single_walker_decorator(env)

        state_dim = sum([box.shape[0] for box in env.observation_space])
        action_dim = sum([box.shape[0] for box in env.action_space])
        action_max = 1.0
        target_reward = 50
        is_discrete = False
    elif env_name == "MinitaurBulletEnv-v0":
        env = gym.make(env_name)
        state_dim, action_dim, action_max, target_reward, is_discrete = get_env_info(env, if_print)

        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step((action * 1.1).clip(-1, 1))
                return state, reward, done, info

            return new_env_step

        env.step = decorator_step(env.step)

    else:
        env = gym.make(env_name)
        state_dim, action_dim, action_max, target_reward, is_discrete = get_env_info(env, if_print)

    '''env normalization'''  # adjust action into [-1, +1] using action_max is necessary.
    avg = None
    std = None
    if if_norm:
        '''norm no need'''
        # if env_name == 'Pendulum-v0':
        #     state_mean = np.array([-0.00968592 -0.00118888 -0.00304381])
        #     std = np.array([0.53825575 0.54198545 0.8671749 ])

        '''norm could be'''
        if env_name == 'LunarLanderContinuous-v2':
            avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
                            -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
            std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
                            0.25886878, 0.277233, 0.27771219])
        elif env_name == "BipedalWalker-v3":
            avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
                            -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
                            4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
                            -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
                            3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
                            5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
            std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
                            0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
                            0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
                            0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
                            0.11752805, 0.14116005, 0.13839757, 0.07760469])
        elif env_name == 'AntBulletEnv-v0':
            avg = np.array([
                0.4838, -0.047, 0.3500, 1.3028, -0.249, 0.0000, -0.281, 0.0573,
                -0.261, 0.0000, 0.0424, 0.0000, 0.2278, 0.0000, -0.072, 0.0000,
                0.0000, 0.0000, -0.175, 0.0000, -0.319, 0.0000, 0.1387, 0.0000,
                0.1949, 0.0000, -0.136, -0.060])
            std = np.array([
                0.0601, 0.2267, 0.0838, 0.2680, 0.1161, 0.0757, 0.1495, 0.1235,
                0.6733, 0.4326, 0.6723, 0.3422, 0.7444, 0.5129, 0.6561, 0.2732,
                0.6805, 0.4793, 0.5637, 0.2586, 0.5928, 0.3876, 0.6005, 0.2369,
                0.4858, 0.4227, 0.4428, 0.4831])
        elif env_name == 'MinitaurBulletEnv-v0':
            avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
                            1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
                            0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
                            -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
                            0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
                            -0.20753499, -0.47758384, 0.86756409])
            std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
                            0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
                            14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
                            13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
                            2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
                            0.05903034, 0.1314812, 0.0221248])
        # elif env_name == "BipedalWalkerHardcore-v3":
        #     pass
        '''norm necessary'''

    env = decorator__normalization(env, action_max, avg, std)
    return env, state_dim, action_dim, target_reward, is_discrete


def mp__update_params(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    class_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env_name = args.env_name
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
    buffer_ary, reward_list, step_list = q_o_buf.get()  # q_o_buf 2.
    reward_avg = np.average(reward_list)
    step_sum = sum(step_list)
    buffer.extend_memo(buffer_ary)

    '''pre training and hard update before training loop'''
    buffer.init_before_sample()
    agent.update_policy(buffer, max_step, batch_size, repeat_times)
    agent.act_target.load_state_dict(agent.act.state_dict())

    q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva 1.

    total_step = step_sum
    if_train = True
    if_solve = False
    while if_train:
        buffer_ary, reward_list, step_list = q_o_buf.get()  # q_o_buf n.
        reward_avg = np.average(reward_list)
        step_sum = sum(step_list)
        total_step += step_sum
        buffer.extend_memo(buffer_ary)

        buffer.init_before_sample()
        loss_a_avg, loss_c_avg = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_buf.put(act_cpu)  # q_i_buf n.
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            if_solve = q_o_eva.get()  # q_o_eva n.
        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)

    q_i_buf.put('stop')
    q_i_eva.put('stop')
    while q_i_buf.qsize() > 0 or q_i_eva.qsize() > 0:
        time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def mp__update_buffer(args, q_i_buf, q_o_buf):  # update replay buffer by interacting with env
    env_name = args.env_name
    max_step = args.max_step
    reward_scale = args.reward_scale
    gamma = args.gamma

    del args

    torch.set_num_threads(8)  # todo

    env, state_dim, action_dim, _, is_discrete = build_gym_env(env_name, if_print=False)  # _ is target_reward

    q_o_buf.put((state_dim, action_dim))  # q_o_buf 1.

    '''build evaluated only actor'''
    q_i_buf_get = q_i_buf.get()  # q_i_buf 1.
    act = q_i_buf_get  # act == act.to(device_cpu), requires_grad=False

    buffer_part, reward_list, step_list = get__buffer_reward_step(
        env, max_step, reward_scale, gamma, action_dim, is_discrete)

    q_o_buf.put((buffer_part, reward_list, step_list))  # q_o_buf 2.

    explore_noise = True
    state = env.reset()
    is_training = True
    while is_training:
        buffer_list = list()
        reward_list = list()
        reward_item = 0.0
        step_list = list()
        step_item = 0

        global_step = 0
        while global_step < max_step:
            '''select action'''
            s_tensor = torch.tensor((state,), dtype=torch.float32, requires_grad=False)
            a_tensor = act(s_tensor, explore_noise)
            action = a_tensor.detach_().numpy()[0]

            next_state, reward, done, _ = env.step(action)
            reward_item += reward
            step_item += 1

            adjust_reward = reward * reward_scale
            mask = 0.0 if done else gamma
            buffer_list.append((adjust_reward, mask, state, action, next_state))

            if done:
                global_step += step_item

                reward_list.append(reward_item)
                reward_item = 0.0
                step_list.append(step_item)
                step_item = 0

                state = env.reset()
            else:
                state = next_state

        buffer_part = np.stack([np.hstack(buf_tuple) for buf_tuple in buffer_list])
        q_o_buf.put((buffer_part, reward_list, step_list))  # q_o_buf n.

        q_i_buf_get = q_i_buf.get()  # q_i_buf n.
        if q_i_buf_get == 'stop':
            is_training = False
        else:
            act = q_i_buf_get  # act == act.to(device_cpu), requires_grad=False

    while q_o_buf.qsize() > 0:
        q_o_buf.get()
    while q_i_buf.qsize() > 0:
        q_i_buf.get()
    # print('; quit: buffer')


def mp_evaluate_agent(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
    env_name = args.env_name
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_memo
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
                act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

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


def modify_log_prob(a_noise, a_mean, a_std, a_std_log):
    # a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
    # log_prob_noise = a_delta + a_std_log  # + 0.919  # self.constant_log_sqrt_2pi
    #
    # a_noise_tanh = a_noise.tanh()
    # fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
    # log_prob = log_prob_noise + fix_term

    a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
    log_prob_noise = a_delta + -a_std_log.abs()  # + 0.919  # self.constant_log_sqrt_2pi

    a_noise_tanh = a_noise.tanh()
    fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
    log_prob = log_prob_noise + fix_term / 8  # todo
    return a_noise_tanh, log_prob.sum(1, keepdim=True)


class InterSPG(nn.Module):  # class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dim, action_dim, mid_dim):
        self.c = 0

        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # state
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # action without nn.Tanh()

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        # decoder
        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic

        layer_norm(self.dec_a[-1], std=0.01)  # net[-1] is output layer for action, it is no necessary.
        layer_norm(self.dec_q1[-1], std=0.1)
        layer_norm(self.dec_q2[-1], std=0.1)

        '''Not need to use both SpectralNorm and TwinCritic
        I choose TwinCritc instead of SpectralNorm, 
        because SpectralNorm is conflict with soft target update,

        if is_spectral_norm:
            self.dec_q1[1] = nn.utils.spectral_norm(self.dec_q1[1])
            self.dec_q2[1] = nn.utils.spectral_norm(self.dec_q2[1])
        '''

    def forward(self, s, noise_std=0.0):  # actor, in fact, noise_std is a boolean
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without tensor.tanh()

        if noise_std != 0.0:
            a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it is a_mean without .tanh()
        return a_mean.tanh()

    def get__a__log_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        # the above is not same as below, because it needs gradient
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        # '''compute log_prob according to mean and std of action (stochastic policy)'''
        # a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        # log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi
        #
        # a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.00001).log()
        # return a_noise_tanh, log_prob.sum(1, keepdim=True)
        return modify_log_prob(a_noise, a_mean, a_std, a_std_log)

    def get__a__std(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        return a_mean.tanh(), a_std_log

    def get__a__avg_std_noise_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        # the above is not same as below, because it needs gradient
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        # '''compute log_prob according to mean and std of action (stochastic policy)'''
        # a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        # log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi
        #
        # a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.00001).log()
        # return a_mean.tanh(), a_std_log, a_noise_tanh, log_prob.sum(1, keepdim=True)
        res1, res2 = modify_log_prob(a_noise, a_mean, a_std, a_std_log)
        return a_mean.tanh(), a_std_log, res1, res2

    def get__q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2

    def get__q1(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        return q1


class AgentFixInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.act_anchor = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_anchor.eval()
        self.act_anchor.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim) - action_dim * 0.919
        self.log_alpha = torch.tensor((-self.target_entropy,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        lamb = 0
        actor_loss = critic_loss = None
        a1_log_std = None

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size = int(batch_size * k)  # increase batch_size
        train_step = int(max_step * k)  # increase training_step

        alpha = self.log_alpha.exp().detach()  # auto temperature parameter

        update_a = 0
        for update_c in range(1, train_step):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # twin critic
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # # auto temperature parameter

            '''critic_loss'''
            q1_value, q2_value = self.act.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_loss.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            '''stochastic policy'''
            a1_mean, a1_log_std, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

            log_prob = log_prob.mean()

            '''auto temperature parameter: alpha'''
            alpha_loss = lamb * self.log_alpha * (log_prob - self.target_entropy).detach()  # todo sable
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
                alpha = self.log_alpha.exp()  # .detach()

            '''action correction term'''
            with torch.no_grad():
                a2_mean, a2_log_std = self.act_anchor.get__a__std(state)
            actor_term = self.criterion(a1_mean, a2_mean) + self.criterion(a1_log_std, a2_log_std)

            if update_a / update_c > 1 / (2 - lamb):
                united_loss = critic_loss + actor_term * (1 - lamb)
            else:
                update_a += 1  # auto TTUR
                '''actor_loss'''
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise)).mean()  # twin critics
                actor_loss = -(q_eval_pg + log_prob * alpha)  # policy gradient

                united_loss = critic_loss + actor_term * (1 - lamb) + actor_loss * lamb

            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)
        soft_target_update(self.act_anchor, self.act, tau=lamb if lamb > 0.1 else 0.0)
        return a1_log_std.mean().item(), critic_loss.item() / 2


def run_continuous_action(gpu_id=None):
    rl_agent = AgentFixInterSAC
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(5e4 * 16)  # (2e4) 5e4
    # args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    # args.env_name = "BipedalWalker-v3"
    # args.break_step = int(2e5 * 8)  # (1e5) 2e5
    # args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(1e6 * 8)  # (8e5) 10e5
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(4e6 * 4)  # (2e6) 4e6
    args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


run_continuous_action()
