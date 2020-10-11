import os
import sys
import time  # for reward recorder

import gym
import torch
import numpy as np
import numpy.random as rd

from AgentZoo import initial_exploration
from AgentZoo import BufferArray, BufferArrayGPU, BufferTupleOnline

"""Zen4Jia1Hao2, GitHub: YonV1943 ElegantRL (Pytorch model-free DRL)
I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
In my opinion, before 2020, the policy gradient algorithm agent didn't learn s policy.
Actually, they "learn game feel" or "get a soft touch". In Chinese "shou3 gan3 手感". 
Learn more about policy gradient algorithms in:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
"""


class Arguments:  # default working setting and hyper-parameter
    def __init__(self, rl_agent=None, env_name=None, gpu_id=None):
        self.rl_agent = rl_agent
        self.env_name = env_name
        self.gpu_id = sys.argv[-1][-4] if gpu_id is None else gpu_id
        self.cwd = None

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards

        '''Arguments for evaluate'''
        self.break_step = 2 ** 17  # break training after 'total_step > break_step'
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.if_remove_history = True  # remove the cwd folder? (True, False, None:ask me)
        self.show_gap = 2 ** 8  # show the Reward and Loss of actor and critic per show_gap seconds
        self.eval_times1 = 2 ** 3  # evaluation times if 'eval_reward > old_max_reward'
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > target_reward'
        self.random_seed = 1943 + 1  # Github: YonV 1943, ZenJiaHao

    def init_for_training(self, cpu_threads=4):
        assert self.rl_agent is not None
        assert self.env_name is not None
        if self.gpu_id is None:
            self.gpu_id = sys.argv[-1][-4]
        if not self.gpu_id.isnumeric():
            self.gpu_id = '0'
        self.cwd = f'./{self.rl_agent.__name__}/{self.env_name}_{self.gpu_id}'

        print('\n| GPU: {} | CWD: {}'.format(self.gpu_id, self.cwd))
        whether_remove_history(self.cwd, self.if_remove_history)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(cpu_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # env.seed(random_seed)  # sometimes env has random seed.

    def update_args(self, new_dict):  # useless
        for key, value in new_dict.items():
            setattr(self, key, value)


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
        recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

        # todo pre training and hard update before loop
        buffer.init_before_sample()
        agent.update_parameters(buffer, max_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict())

    '''loop'''
    if_train = True
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.init_before_sample()
        loss_a, loss_c = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, loss_a, loss_c)

            if_save = recorder.update__record_evaluate(env, agent.act, max_step, agent.device, if_discrete)
            recorder.save_act(cwd, agent.act, gpu_id) if if_save else None
            recorder.save_npy__plot_png(cwd)

            if_solve = recorder.check_is_solved(target_reward, gpu_id, show_gap)

        '''break loop rules'''
        if_train = not ((if_break_early and if_solve)
                        or recorder.total_step > break_step
                        or os.path.exists(f'{cwd}/stop.mark'))
    recorder.save_npy__plot_png(cwd)
    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)  # todo norm para


"""multi processing"""


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
    buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf 2.
    reward_avg = np.average(reward_list)
    step_sum = sum(step_list)
    buffer.extend_memo(buffer_array)

    # todo pre training and hard update before loop
    buffer.init_before_sample()
    agent.update_parameters(buffer, max_step, batch_size, repeat_times)
    agent.act_target.load_state_dict(agent.act.state_dict())

    q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva 1.

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
        loss_a_avg, loss_c_avg = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_buf.put(act_cpu)  # q_i_buf n.
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            if_solve = q_o_eva.get()  # q_o_eva n.
        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop.mark'))

    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)  # todo norm para

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

    torch.set_num_threads(4)

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


"""utils"""


def get_env_info(env, is_print=True):  # 2020-06-06
    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    if len(state_shape) == 1:
        state_dim = state_shape[0]
    else:
        state_dim = state_shape

    try:
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if is_discrete:  # discrete
            action_dim = env.action_space.n
            action_max = int(1)
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            action_max = float(env.action_space.high[0])

            action_high = np.array(env.action_space.high)
            action_high[:] = action_max
            action_low = np.array(env.action_space.low)
            action_low[:] = -action_max
            if any(action_high != env.action_space.high) and any(action_low != env.action_space.low):
                print(f'| Warning: '
                      f'act_high {env.action_space.high}  '
                      f'act_low  {env.action_space.low}')
        else:
            raise AttributeError
    except AttributeError:
        print("| Could you assign these value manually? \n"
              "| I need: state_dim, action_dim, action_max, target_reward, is_discrete")
        raise AttributeError

    target_reward = env.spec.reward_threshold
    if target_reward is None:
        assert target_reward is not None

    if is_print:
        print("| env_name: {}, action space: {}".format(env_name, 'is_discrete' if is_discrete else 'Continuous'))
        print("| state_dim: {}, action_dim: {}, action_max: {}, target_reward: {}".format(
            state_dim, action_dim, action_max, target_reward))
    return state_dim, action_dim, action_max, target_reward, is_discrete


def decorator__normalization(env, action_max=1, state_avg=None, state_std=None):
    if state_avg is None:
        neg_state_avg = 0
        div_state_std = 1

        env.neg_state_avg = neg_state_avg
        env.div_state_std = div_state_std
    else:
        neg_state_avg = -state_avg
        div_state_std = 1 / (state_std + 1e-5)

        env.neg_state_avg = neg_state_avg
        env.div_state_std = div_state_std

    '''decorator_step'''
    if state_avg is not None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return (state + neg_state_avg) * div_state_std, reward, done, info

            return new_env_step

        env.step = decorator_step(env.step)

    elif action_max is not None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return state, reward, done, info

            return new_env_step

        env.step = decorator_step(env.step)

    '''decorator_reset'''
    if state_avg is not None:
        def decorator_reset(env_reset):
            def new_env_reset():
                state = env_reset()
                return (state + neg_state_avg) * div_state_std

            return new_env_reset

        env.reset = decorator_reset(env.reset)

    return env


def build_gym_env(env_name, if_print=True, if_norm=True):
    assert env_name is not None

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
            avg = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00])
            std = np.array([0.10, 0.36, 0.15, 0.30, 0.08, 0.08, 0.40, 0.40])
        elif env_name == "BipedalWalker-v3":
            avg = np.array([
                0.15421079, -0.0019480261, 0.20461461, -0.010021029, -0.054185472,
                -0.0066469274, 0.043834914, -0.0623244, 0.47021484, 0.55891204,
                -0.0014871443, -0.18538311, -0.032906517, 0.4628296, 0.34264696,
                0.3465399, 0.3586852, 0.3805626, 0.41525024, 0.46849185,
                0.55162823, 0.68896055, 0.88635695, 0.997974])
            std = np.array([
                0.33242697, 0.04527563, 0.19229797, 0.0729273, 0.7084785,
                0.6366427, 0.54090905, 0.6944477, 0.49912727, 0.6371604,
                0.5867769, 0.56915027, 0.6196849, 0.49863166, 0.07042835,
                0.07128556, 0.073920645, 0.078663535, 0.08622651, 0.09801551,
                0.116571024, 0.14705327, 0.14093699, 0.019490194])
        elif env_name == 'AntBulletEnv-v0':
            avg = np.array([
                -0.21634328, 0.08877027, 0.92127347, 0.19477099, 0.01834413,
                -0.00399973, 0.05166896, -0.06077103, 0.30839303, 0.00338527,
                0.0065377, 0.00814168, -0.08944025, -0.00331316, 0.29353178,
                -0.00634391, 0.1048052, -0.00327279, 0.52993906, -0.00569263,
                -0.14778288, -0.00101847, -0.2781167, 0.00479939, 0.64501953,
                0.8638916, 0.8486328, 0.7150879])
            std = np.array([
                0.07903007, 0.35201055, 0.13954371, 0.21050458, 0.06752874,
                0.06185101, 0.15283841, 0.16655168, 0.6452229, 0.26257575,
                0.6666661, 0.14235465, 0.69359726, 0.32817268, 0.6647092,
                0.16925392, 0.6878494, 0.3009345, 0.6294114, 0.15175952,
                0.6949041, 0.27704775, 0.6775213, 0.18721068, 0.478522,
                0.3429141, 0.35841736, 0.45138636])
        elif env_name == 'MinitaurBulletEnv-v0':
            avg = np.array([
                1.25116920e+00, 2.35373068e+00, 1.77717030e+00, 2.72379971e+00,
                2.27262020e+00, 1.12126017e+00, 2.80015516e+00, 1.72379172e+00,
                -4.53610346e-02, 2.10091516e-01, -1.20424433e-03, 2.07291126e-01,
                1.69130951e-01, -1.16945259e-01, 1.06861845e-01, -5.53673357e-02,
                2.81922913e+00, 5.51327229e-01, 9.92989361e-01, -8.03717971e-01,
                7.90598467e-02, 2.99980807e+00, -1.27279997e+00, 1.76894355e+00,
                3.58282216e-02, 8.28480721e-02, 8.04320276e-02, 9.86465216e-01])
            std = np.array([
                2.0109391e-01, 3.5780826e-01, 2.2601920e-01, 5.0385582e-01,
                3.8282552e-01, 1.9690999e-01, 3.9662227e-01, 2.3809761e-01,
                8.9289074e+00, 1.4150095e+01, 1.0200104e+01, 1.1171419e+01,
                1.3293057e+01, 7.7480621e+00, 1.0750853e+01, 9.1990738e+00,
                2.7995987e+00, 2.9199743e+00, 2.3916528e+00, 2.6439502e+00,
                3.1360087e+00, 2.7837939e+00, 2.7758663e+00, 2.5578094e+00,
                4.1734818e-02, 3.2294787e-02, 7.8678936e-02, 1.0366816e-02])

        '''norm necessary'''

    env = decorator__normalization(env, action_max, avg, std)
    return env, state_dim, action_dim, target_reward, is_discrete


def draw_plot_with_2npy(cwd, train_time, max_reward):  # 2020-09-18
    record_explore = np.load('%s/record_explore.npy' % cwd)  # , allow_pickle=True)
    # record_explore.append((total_step, exp_r_avg, loss_a_avg, loss_c_avg))
    record_evaluate = np.load('%s/record_evaluate.npy' % cwd)  # , allow_pickle=True)
    # record_evaluate.append((total_step, eva_r_avg, eva_r_std))

    if len(record_evaluate.shape) == 1:
        record_evaluate = np.array([[0., 0., 0.]])
    if len(record_explore.shape) == 1:
        record_explore = np.array([[0., 0., 0., 0.]])

    train_time = int(train_time)
    total_step = int(record_evaluate[-1][0])
    save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{max_reward:.3f}"
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
    ax21_color = 'lightcoral'  # todo same color as ax11 (expR)
    ax21_label = 'lossA'
    exp_loss_a = record_explore[:, 2]
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(exp_step, exp_loss_a, label=ax21_label, color=ax21_color)  # negative loss A
    ax21.tick_params(axis='y', labelcolor=ax21_color)

    ax22 = axs[1].twinx()
    ax22_color = 'darkcyan'
    ax22_label = 'lossC'
    exp_loss_c = record_explore[:, 3]
    ax22.set_ylabel(ax22_label, color=ax22_color)
    ax22.fill_between(exp_step, exp_loss_c, facecolor=ax22_color, alpha=0.2, )
    ax22.tick_params(axis='y', labelcolor=ax22_color)

    # remove prev figure
    prev_save_names = [name for name in os.listdir(cwd) if name[:9] == save_title[:9]]
    os.remove(f'{cwd}/{prev_save_names[0]}') if len(prev_save_names) > 0 else None

    plt.savefig(save_path)
    # plt.pause(4)
    # plt.show()
    plt.close()


def whether_remove_history(cwd, is_remove=None):  # 2020-03-04
    import shutil

    if is_remove is None:
        is_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(cwd)) == 'y')
    if is_remove:
        shutil.rmtree(cwd, ignore_errors=True)
        print("| Remove")

    os.makedirs(cwd, exist_ok=True)

    # shutil.copy(sys.argv[-1], "{}/AgentRun-py-backup".format(cwd))  # copy *.py to cwd
    # shutil.copy('AgentZoo.py', "{}/AgentZoo-py-backup".format(cwd))  # copy *.py to cwd
    # shutil.copy('AgentNet.py', "{}/AgentNetwork-py-backup".format(cwd))  # copy *.py to cwd
    del shutil


class Recorder:
    def __init__(self, eval_size1=3, eval_size2=9):
        self.eva_r_max = -np.inf
        self.total_step = 0
        self.record_exp = [(0, 0, 0, 0), ]  # total_step, exp_r_avg, loss_a_avg, loss_c_avg
        self.record_eva = [(0, 0, 0), ]  # total_step, eva_r_avg, eva_r_std
        self.is_solved = False

        '''constant'''
        self.eva_size1 = eval_size1
        self.eva_size2 = eval_size2

        '''print_reward'''
        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()

        print(f"{'GPU':>3}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8} |"
              f"{'ExpR':>8}  {'LossA':>8}  {'LossC':>8}")

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

    def update__record_explore(self, exp_s_sum, exp_r_avg, loss_a, loss_c):
        if isinstance(exp_s_sum, int):
            exp_s_sum = (exp_s_sum,)
            exp_r_avg = (exp_r_avg,)
        for s, r in zip(exp_s_sum, exp_r_avg):
            self.total_step += s
            self.record_exp.append((self.total_step, r, loss_a, loss_c))

    def save_act(self, cwd, act, gpu_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{gpu_id:<3}  {self.total_step:8.2e}  {self.eva_r_max:8.2f} |")

    def check_is_solved(self, target_reward, gpu_id, show_gap):
        if self.eva_r_max > target_reward:
            self.is_solved = True
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'GPU':>3}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8} |"
                      f"{'ExpR':>8}  {'UsedTime':>8}  ########")

                total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
                total_step, exp_r_avg, loss_a_avg, loss_c_avg = self.record_exp[-1]
                print(f"{gpu_id:<3}  {total_step:8.2e}  {target_reward:8.2f} |"
                      f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                      f"{exp_r_avg:8.2f}  {self.used_time:>8}  ########")

        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()

            total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
            total_step, exp_r_avg, loss_a_avg, loss_c_avg = self.record_exp[-1]
            print(f"{gpu_id:<3}  {total_step:8.2e}  {self.eva_r_max:8.2f} |"
                  f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                  f"{exp_r_avg:8.2f}  {loss_a_avg:8.2f}  {loss_c_avg:8.2f}")
        return self.is_solved

    def save_npy__plot_png(self, cwd):
        np.save('%s/record_explore.npy' % cwd, self.record_exp)
        np.save('%s/record_evaluate.npy' % cwd, self.record_eva)
        draw_plot_with_2npy(cwd, train_time=time.time() - self.start_time, max_reward=self.eva_r_max)

    def demo(self):
        pass


def get_eva_reward(agent, env_list, max_step) -> list:  # class Recorder 2020-01-11
    """ Notice:
    this function is a bit complicated. I don't recommend you or me to change it.
    """
    agent.act.eval()

    env_list_copy = env_list.copy()
    eva_size = len(env_list_copy)

    sum_rewards = [0.0, ] * eva_size
    states = [env.reset() for env in env_list_copy]

    reward_sums = list()
    for iter_num in range(max_step):
        actions = agent.select_actions(states)

        next_states = list()
        done_list = list()
        for i in range(len(env_list_copy) - 1, -1, -1):
            next_state, reward, done, _ = env_list_copy[i].step(actions[i])

            next_states.insert(0, next_state)
            sum_rewards[i] += reward
            done_list.insert(0, done)
            if done:
                reward_sums.append(sum_rewards[i])
                del sum_rewards[i]
                del env_list_copy[i]
        states = next_states

        if len(env_list_copy) == 0:
            break
    else:
        reward_sums.extend(sum_rewards)
    agent.act.train()

    return reward_sums


def get_episode_reward(env, act, max_step, device, is_discrete) -> float:
    # better to 'with torch.no_grad()'
    reward_item = 0.0

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.tensor((state,), dtype=torch.float32, device=device)

        a_tensor = act(s_tensor).argmax(dim=1) if is_discrete else act(s_tensor)
        action = a_tensor.cpu().data.numpy()[0]

        next_state, reward, done, _ = env.step(action)
        reward_item += reward

        if done:
            break
        state = next_state
    return reward_item


def get__buffer_reward_step(env, max_step, reward_scale, gamma, action_dim, is_discrete,
                            **_kwargs) -> (np.ndarray, list, list):
    buffer_list = list()

    reward_list = list()
    reward_item = 0.0

    step_list = list()
    step_item = 0

    state = env.reset()

    global_step = 0
    while global_step < max_step:
        action = rd.randint(action_dim) if is_discrete else rd.uniform(-1, 1, size=action_dim)
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

    buffer_array = np.stack([np.hstack(buf_tuple) for buf_tuple in buffer_list])
    return buffer_array, reward_list, step_list


"""demo"""


def run__demo():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=Zoo.AgentModSAC, env_name="LunarLanderContinuous-v2", gpu_id=None)
    # args = Arguments(rl_agent=Zoo.AgentInterSAC, env_name="LunarLanderContinuous-v2", gpu_id=None)
    args.init_for_training()
    train_agent(**vars(args))


def run__discrete_action(gpu_id=None):
    import AgentZoo as Zoo

    """offline policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentDuelingDQN, gpu_id=gpu_id)
    assert args.rl_agent in {Zoo.AgentDQN, Zoo.AgentDoubleDQN, Zoo.AgentDuelingDQN}

    args.env_name = "CartPole-v0"
    args.break_step = int(1e4 * 8)
    args.net_dim = 2 ** 6
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    args.env_name = "LunarLander-v2"
    args.break_step = int(1e5 * 8)
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    """online policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentDiscreteGAE, gpu_id=gpu_id)
    assert args.rl_agent in {Zoo.AgentDiscreteGAE, }

    args.max_memo = 2 ** 10
    args.batch_size = 2 ** 8
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7

    args.env_name = "CartPole-v0"
    args.break_step = int(4e3 * 8)
    args.init_for_training()
    train_agent(**vars(args))

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8

    args.env_name = "LunarLander-v2"
    args.break_step = int(2e5 * 8)
    args.init_for_training()
    train_agent(**vars(args))


def run_continuous_action(gpu_id=None):
    import AgentZoo as Zoo
    """offline policy"""
    rl_agent = Zoo.AgentModSAC
    assert rl_agent in {Zoo.AgentDDPG,  # 2014. simple, old, slow, unstable
                        Zoo.AgentBasicAC,  # 2014+ stable DDPG
                        Zoo.AgentTD3,  # 2018. twin critics, delay target update
                        Zoo.AgentSAC,  # 2018. twin critics, policy entropy, auto alpha
                        Zoo.AgentModSAC,  # 2018+ stable SAC
                        Zoo.AgentInterAC,  # 2019. Integrated AC(DPG)
                        Zoo.AgentInterSAC,  # 2020. Integrated SAC(SPG)
                        }  # PPO, GAE is online policy. See below.
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = True
    args.if_remove_history = True

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # max_reward I get:
    args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    args.reward_scale = 2 ** -2
    args.init_for_training()
    train_agent(**vars(args))  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(5e4 * 16)
    args.reward_scale = 2 ** 0
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)
    args.reward_scale = 2 ** 0
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(5e6 * 4)
    args.reward_scale = 2 ** -3
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training(8)
    train_agent_mp(args)  # train_offline_policy(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(1e6 * 4)
    args.reward_scale = 2 ** 6
    args.batch_size = 2 ** 8
    args.repeat_times = 2 ** 0
    args.max_memo = 2 ** 20
    args.net_dim = 2 ** 8
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training()
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

    """online policy"""
    args = Arguments(rl_agent=Zoo.AgentGAE, gpu_id=gpu_id)
    assert args.rl_agent in {Zoo.AgentPPO, Zoo.AgentGAE, Zoo.AgentInterGAE}
    """PPO and GAE is online policy.
    The memory in replay buffer will only be saved for one episode.

    TRPO's author use a surrogate object to simplify the KL penalty, and get PPO.
    So I provide PPO instead of TRPO here.

    GAE is Generalization Advantage Estimate. (in high dimension)
    RL algorithm that use advantage function (such as A2C, PPO, SAC) can use this technique.
    AgentGAE is a PPO using GAE and output log_std of action by an actor network.
    """
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    args.break_step = int(1e5 * 4)
    args.reward_scale = 2 ** -2
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(1e5 * 16)
    args.reward_scale = 2 ** 0
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(3e6 * 4)
    args.reward_scale = 2 ** 0
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(1e6 * 8)
    args.reward_scale = 2 ** -3
    args.net_dim = 2 ** 9
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(2e7 * 8)
    args.reward_scale = 2 ** 4
    args.net_dim = 2 ** 9
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.break_step = int(2e7 * 8)
    args.reward_scale = 2 ** 0
    args.net_dim = 2 ** 8
    args.init_for_training()
    train_agent(**vars(args))
    exit()


if __name__ == '__main__':
    run__demo()
    print('Finish:', sys.argv[-1])
