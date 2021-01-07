from AgentRun import *

"""
Change Queue to Pipe
"""


def mp__explore_pipe_env(args, pipe, _act_id):
    env = args.env
    reward_scale = args.reward_scale
    gamma = args.gamma

    next_state = env.reset()
    pipe.send(next_state)
    while True:
        action = pipe.recv()
        next_state, reward, done, _ = env.step(action)

        reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
        if done:
            next_state = env.reset()

        pipe.send((reward_mask, next_state))  # todo faster


def mp__update_params(args, q_i_eva, q_o_eva, pipes):  # 2020-12-22
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

    # workers_num = len(qs_i_exp)
    # workers_num = len(exp_pipes)
    workers_num = 4  # rollout number

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    # target_reward = env.target_reward

    # from copy import deepcopy  # built-in library of Python
    # env_list = [env, ] + [deepcopy(env) for _ in range(4-1)]  # todo deepcopy random seed
    # del env

    '''build agent and act_cpu'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    # agent.state = env.reset()
    # agent.state = [env.reset() for env in env_list]  # todo
    # agent.reward_sum = [0.0, ] * len(env_list)  # todo
    # agent.step = [0, ] * len(env_list)  # todo
    agent.state = [pipe.recv() for pipe in pipes]

    agent.action = agent.select_actions(agent.state)  # todo double
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
        # rewards = list()
        # steps = list()
        # for i, q_i_exp in enumerate(qs_i_exp):
        #     q_i_exp.put(agent.act)  # q_i_exp n.
        #     # q_i_exp.put(act_cpu)  # env_cpu--act_cpu
        # for i, q_o_exp in enumerate(qs_o_exp):
        #     _memo_array, _rewards, _steps = q_o_exp.get()  # q_o_exp n.
        #     buffer.extend_memo(_memo_array)
        #     rewards.extend(_rewards)
        #     steps.extend(_steps)

        # with torch.no_grad():  # speed up running
        #     rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
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
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            if_solve = q_o_eva.get()  # q_o_eva n.

        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12

    for queue in [q_i_eva, ]:  # + list(qs_i_exp):  # quit orderly and safely
        queue.put('stop')
        while queue.qsize() > 0:
            time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def train_agent_mp(args):  # 2021-01-01
    act_workers = args.rollout_num

    import multiprocessing as mp
    q_i_eva = mp.Queue(maxsize=16)  # evaluate I
    q_o_eva = mp.Queue(maxsize=16)  # evaluate O
    process = list()

    exp_pipes = list()
    for i in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        exp_pipes.append(exp_pipe1)
        process.append(mp.Process(target=mp__explore_pipe_env, args=(args, exp_pipe2, i)))
    process.extend([
        mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)),
        mp.Process(target=mp__update_params, args=(args, q_i_eva, q_o_eva, exp_pipes)),
    ])

    # qs_i_exp = [mp.Queue(maxsize=16) for _ in range(act_workers)]
    # qs_o_exp = [mp.Queue(maxsize=16) for _ in range(act_workers)]
    # for i in range(act_workers):
    #     process.append(mp.Process(target=mp__explore_a_env, args=(args, qs_i_exp[i], qs_o_exp[i], i)))
    #
    # process.extend([
    #     mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)),
    #     mp.Process(target=mp__update_params, args=(args, q_i_eva, q_o_eva, qs_i_exp, qs_o_exp)),
    # ])

    [p.start() for p in process]
    [p.join() for p in process[act_workers:]]
    [p.terminate() for p in process[:act_workers]]
    print('\n')


def train__demo():
    pass

    # '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
    # import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    # gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    # env = gym.make("BipedalWalker-v3")
    # env = decorate_env(env, if_print=True)
    #
    # from AgentZoo import AgentModSAC
    # args = Arguments(rl_agent=AgentModSAC, env=env)
    # args.rollout_num = 4  # todo beta2
    #
    # args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    # args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    # # args.max_step = 2 ** 11  # todo beta3
    # args.init_for_training()
    # # train_agent(args)  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    # exit()

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    env = gym.make("AntBulletEnv-v0")
    env = decorate_env(env, if_print=True)

    from AgentZoo import AgentModSAC
    args = Arguments(rl_agent=AgentModSAC, env=env)
    args.rollout_num = 4
    args.if_break_early = False
    args.break_step = int(3e5)

    # args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    # train_agent(args)  # Train agent using single process. Recommend run on PC.
    train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()


if __name__ == '__main__':
    train__demo()
