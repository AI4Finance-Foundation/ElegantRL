from AgentRun import *

"""
Change Queue to Pipe
"""


def mp_evaluate_agent(args, eva_pipe):  # 2020-12-12
    env = args.env
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_memo
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    '''init: env'''
    # state_dim = env.state_dim
    # action_dim = env.action_dim
    if_discrete = env.if_discrete
    target_reward = env.target_reward

    '''build evaluated only actor'''
    # act = q_i_eva.get()  # q_i_eva 1, act == act.to(device_cpu), requires_grad=False
    act = eva_pipe.recv()  # eva_pipe 1, act == act.to(device_cpu), requires_grad=False

    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update__record_evaluate(env, act, max_step, device, if_discrete)

    if_train = True
    with torch.no_grad():  # for saving the GPU buffer
        while if_train:
            is_saved = recorder.update__record_evaluate(env, act, max_step, device, if_discrete)
            recorder.save_act(cwd, act, gpu_id) if is_saved else None

            is_solved = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)
            # q_o_eva.put(is_solved)  # q_o_eva n.
            eva_pipe.send(is_solved)  # eva_pipe is_solved

            '''update actor'''
            while not eva_pipe.poll():  # wait until eva_pipe not empty
                time.sleep(1)
            while eva_pipe.poll():  # receive the latest object from pipe
                q_i_eva_get = eva_pipe.recv()  # eva_pipe act
                if q_i_eva_get == 'stop':
                    if_train = False
                    break  # it should break 'while q_i_eva.qsize():' and 'while if_train:'
                act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

            # while q_i_eva.qsize() == 0:  # wait until q_i_eva has item
            #     time.sleep(1)
            # while q_i_eva.qsize():  # get the latest actor
            #     q_i_eva_get = q_i_eva.get()  # q_i_eva n.
            #     if q_i_eva_get == 'stop':
            #         if_train = False
            #         break  # it should break 'while q_i_eva.qsize():' and 'while if_train:'
            #     act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
            #     recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.eva_r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - recorder.start_time:.0f}')

    while eva_pipe.poll():  # empty the pipe
        eva_pipe.recv()
    # print('; quit: evaluate')


def mp__update_params(args, eva_pipe, pipes):  # 2020-12-22
    rl_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env = args.env
    reward_scale = args.reward_scale
    if_stop = args.if_break_early
    gamma = args.gamma
    del args

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    # target_reward = env.target_reward

    '''build agent and act_cpu'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = [pipe.recv() for pipe in pipes]
    agent.action = agent.select_actions(agent.state)
    for i in range(len(pipes)):
        pipes[i].send(agent.action[i])

    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''build replay buffer, init: total_step, reward_avg'''
    reward_avg = None
    total_step = 0
    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentInterPPO'}):
        buffer = BufferArrayGPU(max_memo + max_step, state_dim, action_dim, if_ppo=True)
        # buffer = BufferArrayGPU(max_memo + max_step * workers_num, state_dim, action_dim, if_ppo=True)
        # exp_step = max_memo // workers_num
    else:
        buffer = BufferArrayGPU(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)
        # exp_step = max_step // workers_num

        '''initial exploration'''
        with torch.no_grad():  # update replay buffer
            rewards, steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        reward_avg = np.average(rewards)
        step_sum = sum(steps)

        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        if 'act_target' in dir(agent):
            agent.act_target.load_state_dict(agent.act.state_dict())

        total_step += step_sum
    if reward_avg is None:
        reward_avg = get_episode_return(env, agent.act, max_step, agent.device, if_discrete)

    # q_i_eva.put(act_cpu)  # q_i_eva 1.
    # for q_i_exp in qs_i_exp:
    #     q_i_exp.put((agent.act, exp_step))  # q_i_exp 1.

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer__pipe_list(pipes, buffer, max_step, reward_scale, gamma)

        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update__now_len__before_sample()
        loss_a_avg, loss_c_avg = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        act_cpu.load_state_dict(agent.act.state_dict())
        # q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.
        eva_pipe.send((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # eva_pipe act

        if eva_pipe.poll():
            if_solve = eva_pipe.recv()  # eva_pipe if_solve

        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12

    eva_pipe.send('stop')  # eva_pipe stop
    time.sleep(4)
    # print('; quit: params')


def train_agent_mp(args):  # 2021-01-01
    act_workers = args.rollout_num

    import multiprocessing as mp
    # q_i_eva = mp.Queue(maxsize=16)  # evaluate I
    # q_o_eva = mp.Queue(maxsize=16)  # evaluate O
    eva_pipe1, eva_pipe2 = mp.Pipe(duplex=True)
    process = list()

    exp_pipe2s = list()
    for i in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        exp_pipe2s.append(exp_pipe1)
        process.append(mp.Process(target=mp__explore_pipe_env, args=(args, exp_pipe2, i)))
    process.extend([
        mp.Process(target=mp_evaluate_agent, args=(args, eva_pipe1)),
        mp.Process(target=mp__update_params, args=(args, eva_pipe2, exp_pipe2s)),
    ])

    [p.start() for p in process]
    [p.join() for p in process[act_workers:]]
    [p.terminate() for p in process[:act_workers]]
    print('\n')


def train__demo():
    pass

    '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make("BipedalWalker-v3")
    env = decorate_env(env, if_print=True)

    from AgentZoo import AgentModSAC
    args = Arguments(rl_agent=AgentModSAC, env=env)
    args.rollout_num = 4
    args.if_break_early = False
    args.break_step = int(1e5)  # todo just for test
    # args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.init_for_training()
    # train_agent(args)  # Train agent using single process. Recommend run on PC.
    train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()


if __name__ == '__main__':
    train__demo()
