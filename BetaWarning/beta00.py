from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
New version InterSAC 2020-09-09 
log_alpha, move auto-alpha outside 'if rho > 2 ** -8'
"""


def mp__update_params(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    class_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.max_total_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    del args

    state_dim, action_dim = q_o_buf.get()  # q_o_buf 1.
    agent = class_agent(state_dim, action_dim, net_dim)

    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_buf.put(act_cpu)  # q_i_buf 1.
    # q_i_buf.put(act_cpu)  # q_i_buf 2. # warning
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    buffer = BufferArrayGPU(max_memo, state_dim, action_dim)  # experiment replay buffer

    '''initial_exploration'''
    buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf 2.
    reward_avg = np.average(reward_list)
    step_sum = sum(step_list)
    buffer.extend_memo(buffer_array)

    q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva 1.

    total_step = step_sum
    is_training = True
    # is_solved = False
    while is_training:
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
            is_solved = q_o_eva.get()  # q_o_eva n.
        '''break loop rules'''
        # if is_solved: # todo
        #     is_training = False
        if total_step > max_total_step or os.path.exists(f'{cwd}/stop.mark'):
            is_training = False

    q_i_buf.put('stop')
    q_i_eva.put('stop')
    while q_i_buf.qsize() > 0 or q_i_eva.qsize() > 0:
        time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def build_for_mp(args):
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


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=AgentInterSAC, gpu_id=gpu_id)
    args.show_gap = 2 ** 9
    args.eval_times2 = 2 ** 5

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.max_total_step = int(1e4 * 4)
    # args.reward_scale = 2 ** -2
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))
    # exit()
    #
    # args.env_name = "LunarLanderContinuous-v2"
    # args.max_total_step = int(1e5 * 4)
    # args.init_for_training()
    # build_for_mp(args)  # train_agent(**vars(args))
    # exit()

    # args.env_name = "BipedalWalker-v3"
    # args.random_seed = 1945
    # args.max_total_step = int(2e5 * 4)
    # args.init_for_training()
    # # build_for_mp(args)
    # train_agent(**vars(args))
    # exit()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.max_total_step = int(5e5 * 4)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.max_step = 2 ** 10
    # args.batch_size = 2 ** 9
    # args.reward_scale = 2 ** -2
    # args.eva_size = 2 ** 3  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.max_total_step = int(1e6 * 2)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20  # todo
    # args.max_step = 2 ** 11  # todo 10
    # args.net_dim = 2 ** 8
    # args.batch_size = 2 ** 8
    # args.reward_scale = 2 ** 4
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 9  # for Recorder
    # args.init_for_training(cpu_threads=4)
    # build_for_mp(args)  # train_agent(**vars(args))

    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.max_total_step = int(4e6 * 8)
    args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    args.max_memo = int(2 ** 21)
    args.batch_size = int(2 ** 8)
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    # build_for_mp(args)
    train_agent(**vars(args))
    exit()


run_continuous_action()
