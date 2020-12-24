import os
import numpy as np
import numpy.random as rd
from AgentRun import *

"""
ceta0 
"""


def _mp__update_params(args, q_i_eva, q_o_eva, qs_i_exp, qs_o_exp):  # 2020-12-22
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

    workers_num = len(qs_i_exp)

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    # target_reward = env.target_reward

    '''build agent and act_cpu'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()
    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''build replay buffer, init: total_step, reward_avg'''
    reward_avg = None
    total_step = 0
    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentModPPO', 'AgentInterPPO'}):
        buffer = BufferArrayGPU(max_memo + max_step * workers_num, state_dim, action_dim, if_ppo=True)
        exp_step = max_memo // workers_num
        # todo workers_num
    else:
        buffer = BufferArrayGPU(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)
        exp_step = max_step // workers_num

        '''initial exploration'''
        with torch.no_grad():  # update replay buffer
            rewards, steps = _explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        reward_avg = np.average(rewards)
        step_sum = sum(steps)

        '''pre training and hard update before training loop'''
        buffer.update_pointer_before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        if 'act_target' in dir(agent):
            agent.act_target.load_state_dict(agent.act.state_dict())

        # q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva n.
        total_step += step_sum
    if reward_avg is None:
        # reward_avg = _get_episode_return(env, act_cpu, max_step, torch.device("cpu"), if_discrete)
        reward_avg = _get_episode_return(env, agent.act, max_step, agent.device, if_discrete)

    q_i_eva.put(act_cpu)  # q_i_eva 1.
    for q_i_exp in qs_i_exp:
        q_i_exp.put((agent.act, exp_step))  # q_i_exp 1.

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        rewards = list()
        steps = list()
        for i, q_i_exp in enumerate(qs_i_exp):
            q_i_exp.put(agent.act)  # q_i_exp n.
        for i, q_o_exp in enumerate(qs_o_exp):
            _memo_array, _rewards, _steps = q_o_exp.get()  # q_o_exp n.
            buffer.extend_memo(_memo_array)
            rewards.extend(_rewards)
            steps.extend(_steps)
        # with torch.no_grad():  # speed up running
        #     rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)

        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update_pointer_before_sample()
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

    for queue in [q_i_eva, ] + list(qs_i_exp):  # quit orderly and safely
        queue.put('stop')
        while queue.qsize() > 0:
            time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def _mp__explore_a_env(args, q_i_exp, q_o_exp, act_id):
    env = args.env
    rl_agent = args.rl_agent
    net_dim = args.net_dim
    reward_scale = args.reward_scale
    gamma = args.gamma

    torch.manual_seed(args.random_seed + act_id)
    np.random.seed(args.random_seed + act_id)
    del args

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    # target_reward = env.target_reward

    '''build agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    '''build replay buffer, init: total_step, reward_avg'''
    act, exp_step = q_i_exp.get()  # q_i_exp 1.  # todo plan to make it elegant: max_memo, max_step, exp_step

    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentModPPO', 'AgentInterPPO'}):
        buffer = BufferArray(exp_step * 2, state_dim, action_dim, if_ppo=True)
    else:
        buffer = BufferArray(exp_step, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)

    with torch.no_grad():  # speed up running
        while True:
            q_i_exp_get = q_i_exp.get()  # q_i_exp n.
            if q_i_exp_get == 'stop':
                break
            agent.act = q_i_exp_get  # q_i_exp n.
            rewards, steps = agent.update_buffer(env, buffer, exp_step, reward_scale, gamma)

            buffer.update_pointer_before_sample()
            q_o_exp.put((buffer.memories[:buffer.now_len], rewards, steps))  # q_o_exp n.
            buffer.empty_memories_before_explore()

    for queue in [q_o_exp, q_i_exp]:  # quit orderly and safely
        while queue.qsize() > 0:
            queue.get()
    # print('; quit: explore')


def _mp_evaluate_agent(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
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
    act = q_i_eva.get()  # q_i_eva 1, act == act.to(device_cpu), requires_grad=False

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
            q_o_eva.put(is_solved)  # q_o_eva n.

            '''update actor'''
            while q_i_eva.qsize() == 0:  # wait until q_i_eva has item
                time.sleep(1)
            while q_i_eva.qsize():  # get the latest actor
                q_i_eva_get = q_i_eva.get()  # q_i_eva n.
                if q_i_eva_get == 'stop':
                    if_train = False
                    break  # it should break 'while q_i_eva.qsize():' and 'while if_train:'
                act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.eva_r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    print(f'SavedDir: {cwd}\n'
          f'UsedTime: {time.time() - recorder.start_time:.0f}')

    for queue in [q_o_eva, q_i_eva]:  # quit orderly and safely
        while queue.qsize() > 0:
            queue.get()
    # print('; quit: evaluate')


def _explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim):
    state = env.reset()

    rewards = list()
    reward_sum = 0.0
    steps = list()
    step = 0

    if if_discrete:
        def random_action__discrete():
            return rd.randint(action_dim)

        get_random_action = random_action__discrete
    else:
        def random_action__continuous():
            return rd.uniform(-1, 1, size=action_dim)

        get_random_action = random_action__continuous

    global_step = 0
    while global_step < max_step or len(rewards) == 0:  # warning 2020-10-10?
        # action = np.tanh(rd.normal(0, 0.25, size=action_dim))  # zero-mean gauss exploration
        action = get_random_action()
        next_state, reward, done, _ = env.step(action * if_discrete)
        reward_sum += reward
        step += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        buffer.append_memo((adjust_reward, mask, state, action, next_state))

        state = next_state
        if done:
            rewards.append(reward_sum)
            steps.append(step)
            global_step += step

            state = env.reset()  # reset the environment
            reward_sum = 0.0
            step = 1

    buffer.update_pointer_before_sample()
    return rewards, steps


def _get_episode_return(env, act, max_step, device, if_discrete) -> float:  # todo 2020-12-21
    # faster to run 'with torch.no_grad()'
    episode_return = 0.0  # sum of rewards in an episode

    # Compatibility for ElegantRL 2020-12-21
    max_step = env.max_step if hasattr(env, 'max_step') else max_step

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.tensor((state,), dtype=torch.float32, device=device)

        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.cpu().data.numpy()[0]

        next_state, reward, done, _ = env.step(action)
        episode_return += reward

        if done:
            break
        state = next_state

    # Compatibility for ElegantRL 2020-12-21
    episode_return = env.episode_return if hasattr(env, 'episode_return') else episode_return
    return episode_return


def train_agent_mp_1223(args):  # 2020-12-12
    import multiprocessing as mp
    q_i_eva = mp.Queue(maxsize=16)  # evaluate I
    q_o_eva = mp.Queue(maxsize=16)  # evaluate O
    process = list()

    act_workers = args.act_workers  # todo act_workers
    qs_i_exp = [mp.Queue(maxsize=16) for _ in range(act_workers)]
    qs_o_exp = [mp.Queue(maxsize=16) for _ in range(act_workers)]
    for i in range(act_workers):
        process.append(mp.Process(target=_mp__explore_a_env, args=(args, qs_i_exp[i], qs_o_exp[i], i)))

    process.extend([
        mp.Process(target=_mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)),
        mp.Process(target=_mp__update_params, args=(args, q_i_eva, q_o_eva, qs_i_exp, qs_o_exp)),
    ])

    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


def run__fin_rl_1223():
    env = FinanceMultiStock1221()  # todo 2020-12-21 16:00

    from AgentZoo import AgentPPO

    args = Arguments(rl_agent=AgentPPO, env=env)
    args.eval_times1 = 1
    args.eval_times2 = 2
    args.act_workers = 4
    args.if_break_early = False

    args.reward_scale = 2 ** 0  # (0) 1.1 ~ 16 (19)
    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = 1699 * 16  # todo  larger is better?
    args.batch_size = 2 ** 10  # todo
    args.repeat_times = 2 ** 4  # larger is better?
    args.init_for_training()
    train_agent_mp_1223(args)  # train_agent(args)
    exit()

    from AgentZoo import AgentModSAC

    args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
    args.eval_times1 = 1
    args.eval_times2 = 2

    args.break_step = 2 ** 22  # UsedTime:
    args.net_dim = 2 ** 7
    args.max_memo = 2 ** 18
    args.batch_size = 2 ** 8
    args.init_for_training()
    train_agent_mp_1223(args)  # train_agent(args)


if __name__ == '__main__':
    run__fin_rl_1223()
