from AgentRun import *
from AgentZoo import BufferArray


def run__sn_ac(gpu_id, cwd='AC_SNAC'):
    from AgentZoo import AgentSNAC
    args = Arguments(AgentSNAC)
    args.gpu_id = gpu_id

    # args.env_name = "LunarLanderContinuous-v2"
    # args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    # args.init_for_training()
    # while not train_agent(**vars(args)):
    #     args.random_seed += 42

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42


def process__buffer(q_aggr, qs_dist, args,
                    **_kwargs):
    max_memo = args.max_memo
    env_name = args.env_name
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = 2

    reward_scale = args.reward_scale
    gamma = args.gamma

    '''init'''
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward = get_env_info(env, be_quiet=False)
    buffer = BufferArray(max_memo, state_dim, action_dim)  # experiment replay buffer

    workers_num = len(qs_dist)

    '''loop'''
    with torch.no_grad():  # update replay buffer
        # rewards, steps = agent.update_buffer(
        #     env, buffer, max_step, max_action, reward_scale, gamma)
        rewards, steps = initial_exploration(
            env, buffer, max_step, max_action, reward_scale, gamma, action_dim)

    while True:
        for _ in range(workers_num):
            memo_array = q_aggr.get()
            buffer.extend_memo(memo_array)

        buffer.init_before_sample()
        for _ in range(max_step * repeat_times):
            for q_dist in qs_dist:
                batch_arrays = buffer.random_sample(batch_size, device=None)
                q_dist.put(batch_arrays)


def process__workers(gpu_id, root_cwd, q_aggr, q_dist, args,
                     **_kwargs):
    class_agent = args.class_agent
    env_name = args.env_name
    cwd = args.cwd
    net_dim = args.net_dim
    max_step = args.max_step
    # max_memo = args.max_memo
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    gamma = args.gamma
    update_gap = args.update_gap
    reward_scale = args.reward_scale

    cwd = '{}/{}_{}'.format(root_cwd, cwd, gpu_id)
    os.makedirs(cwd, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    random_seed = 42 + gpu_id
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(8)

    env = gym.make(env_name)
    is_solved = False

    class BufferArrayMP(BufferArray):
        def init_before_sample(self):
            q_aggr.put(self.memories)
            # self.now_len = self.max_len if self.is_full else self.next_idx

        def random_sample(self, _batch_size, device=None):
            batch_arrays = q_dist.get()

            '''convert array into torch.tensor'''
            tensors = [torch.tensor(ary, device=device) for ary in batch_arrays]
            return tensors

    '''init'''
    state_dim, action_dim, max_action, target_reward = get_env_info(env, be_quiet=True)
    agent = class_agent(env, state_dim, action_dim, net_dim)  # training agent
    buffer = BufferArrayMP(max_step, state_dim, action_dim)  # experiment replay buffer
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name, **_kwargs)

    '''loop'''
    # with torch.no_grad():  # update replay buffer
    #     # rewards, steps = agent.update_buffer(
    #     #     env, buffer, max_step, max_action, reward_scale, gamma)
    #     rewards, steps = initial_exploration(
    #         env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    # recorder.show_reward(rewards, steps, 0, 0)
    try:
        for epoch in range(max_epoch):
            '''update replay buffer by interact with environment'''
            with torch.no_grad():  # for saving the GPU buffer
                rewards, steps = agent.update_buffer(env, buffer, max_step, max_action, reward_scale, gamma)

            '''update network parameters by random sampling buffer for stochastic gradient descent'''
            loss_a, loss_c = agent.update_parameters(buffer, max_step, batch_size, update_gap)

            '''show/check the reward, save the max reward actor'''
            with torch.no_grad():  # for saving the GPU buffer
                '''NOTICE! Recorder saves the agent with max reward automatically. '''
                recorder.show_reward(rewards, steps, loss_a, loss_c)

                is_solved = recorder.check_reward(cwd, loss_a, loss_c)
            if is_solved:
                break
    except KeyboardInterrupt:
        print("raise KeyboardInterrupt while training.")
    # except AssertionError:  # for BipedWalker BUG 2020-03-03
    #     print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")
    #     return False

    train_time = recorder.show_and_save(env_name, cwd)

    # agent.save_or_load_model(cwd, is_save=True)  # save max reward agent in Recorder
    # buffer.save_or_load_memo(cwd, is_save=True)

    draw_plot_with_npy(cwd, train_time)
    return True


def run__multi_workers(gpu_tuple=(0, 1), root_cwd='AC_Methods_MP'):
    print('GPU: {} | CWD: {}'.format(gpu_tuple, root_cwd))
    whether_remove_history(root_cwd, remove=True)

    from AgentZoo import AgentSAC
    args = Arguments(AgentSAC)
    # args.env_name = "BipedalWalker-v3"
    args.env_name = "LunarLanderContinuous-v2"

    '''run in multiprocessing'''
    import multiprocessing as mp
    workers_num = len(gpu_tuple)
    queue_aggr = mp.Queue(maxsize=workers_num)  # queue of aggregation
    queues_dist = [mp.Queue(maxsize=args.max_step) for _ in range(workers_num)]  # queue of distribution

    processes = [mp.Process(target=process__buffer, args=(queue_aggr, queues_dist, args))]
    processes.extend([mp.Process(target=process__workers, args=(gpu_id, root_cwd, queue_aggr, queue_dist, args))
                      for gpu_id, queue_dist in zip(gpu_tuple, queues_dist)])

    [process.start() for process in processes]
    # [process.join() for process in processes]
    [process.close() for process in processes]


if __name__ == '__main__':
    run__multi_workers(gpu_tuple=(0, 1), root_cwd='AC_SAC_MP0')
