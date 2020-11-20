from AgentRun import *
from AgentNet import *
from AgentZoo import *


def test__show_available_env():
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    env_names = list(gym.envs.registry.env_specs.keys())
    env_names.sort()
    for env_name in env_names:
        if env_name.find('Bullet') == -1:
            continue
        print(env_name)


def test__env_quickly():
    env_names = [
        # Classical Control
        "Pendulum-v0", "CartPole-v0", "Acrobot-v1",

        # Box2D
        "LunarLander-v2", "LunarLanderContinuous-v2",
        "BipedalWalker-v3", "BipedalWalkerHardcore-v3",
        'CarRacing-v0',  # Box2D pixel-level
        # 'MultiWalker',  # Box2D MultiAgent

        # py-bullet (MuJoCo is not free)
        "AntBulletEnv-v0", "Walker2DBulletEnv-v0", "HalfCheetahBulletEnv-v0",
        # "HumanoidBulletEnv-v0", "HumanoidFlagrunBulletEnv-v0", "HumanoidFlagrunHarderBulletEnv-v0",

        "ReacherBulletEnv-v0", "PusherBulletEnv-v0", "ThrowerBulletEnv-v0",
        # "StrikerBulletEnv-v0",

        "MinitaurBulletEnv-v0",
    ]

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    for env_name in env_names:
        print(f'| {env_name}')
        build_gym_env(env_name, if_print=True, if_norm=False)
        print()


def test__replay_buffer():
    max_memo = 2 ** 6
    max_step = 2 ** 4
    reward_scale = 1
    gamma = 0.99
    net_dim = 2 ** 7

    env_name = 'LunarLanderContinuous-v2'
    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    state = env.reset()

    '''offline'''
    buffer_offline = BufferArray(max_memo, state_dim, 1 if if_discrete else action_dim)
    _rewards, _steps = initial_exploration(env, buffer_offline, max_step, if_discrete, reward_scale, gamma, action_dim)
    print('Memory length of buffer_offline:', buffer_offline.now_len)

    '''online'''
    buffer_online = BufferTupleOnline(max_memo=max_step)
    agent = AgentPPO(state_dim, action_dim, net_dim)
    agent.state = env.reset()

    _rewards, _steps = agent.update_buffer(env, buffer_online, max_step, reward_scale, gamma)
    print('Memory length of buffer_online: ', len(buffer_online.storage_list))

    buffer_ary = buffer_online.convert_to_rmsas()
    buffer_offline.extend_memo(buffer_ary)
    buffer_offline.update_pointer_before_sample()
    print('Memory length of buffer_offline:', buffer_offline.now_len)


def test__evaluate_agent():
    # save_path = '/mnt/sdb1/yonv/code/ElegantRL_cwd/2020-10-10/PPO/MinitaurBulletEnv-v0_33.72'
    # env_name = "MinitaurBulletEnv-v0"
    # rl_agent = AgentPPO
    # net_dim = 2 ** 9
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)

    save_path = '/mnt/sdb1/yonv/code/ElegantRL_cwd/2020-10-10/InterSAC/BipedalWalkerHardcore-v3_313.8'
    env_name = "BipedalWalkerHardcore-v3"
    rl_agent = AgentInterSAC
    net_dim = 2 ** 8

    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)

    agent = rl_agent(state_dim, action_dim, net_dim)
    del agent.cri
    agent.save_or_load_model(save_path, if_save=False)
    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    reward_item = 0.0
    device = torch.device("cpu")

    print('start evaluation')
    # env.render()  # pybullet-gym put env.render() before env.reset()
    state = env.reset()
    done = False
    total_step = 0
    while not done:
        s_tensor = torch.tensor((state,), dtype=torch.float32, device=device)

        a_tensor = act_cpu(s_tensor).argmax(dim=1) if if_discrete else act_cpu(s_tensor)
        action = a_tensor.cpu().data.numpy()[0]

        next_state, reward, done, _ = env.step(action)
        total_step += 1
        env.render()
        if total_step % 100 == 0:
            print(total_step, reward_item)
        reward_item += reward

        state = next_state
    print(total_step, reward_item)


def test__network():
    net_dim = 2 ** 4
    env_name = "LunarLanderContinuous-v2"
    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)

    act = InterSPG(state_dim, action_dim, net_dim)
    act_optimizer = torch.optim.Adam([
        {'params': act.enc_s.parameters(), 'lr': 2e-4},
        # {'params': act.enc_a.parameters(), 'lr': 4e-4},
        {'params': act.net.parameters(), 'lr': 1e-4},
        {'params': act.dec_a.parameters(), },
        # {'params': act.dec_d.parameters(), 'lr': 4e-4},
        # {'params': act.dec_q1.parameters(), 'lr': 2e-4},
        # {'params': act.dec_q2.parameters(), 'lr': 2e-4},
    ], lr=4e-4)

    # print(act_optimizer)
    for param_grounp in act_optimizer.param_groups:
        print(param_grounp['lr'])


def test__log_prob():
    def show_tensor(t):
        print(t.cpu().data.numpy())

    env_name = "LunarLanderContinuous-v2"
    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1
    a_mean = torch.tensor(np.array((-0.9, 0.1, 0.8, 0.8)), dtype=torch.float32, device=device)
    a_std_log = torch.tensor(np.array((0.1, 0.1, 0.1, 0.2)), dtype=torch.float32, device=device)
    a_std = a_std_log.exp()

    noise = torch.randn_like(a_mean, requires_grad=True, device=device)
    a_noise = a_mean + a_std * noise

    '''verifier log_prob.sum(1, keepdim=True) + 0.919 * action_dim'''
    # a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
    # log_prob_noise = a_delta + a_std_log + 0.919  # self.constant_log_sqrt_2pi
    #
    # a_noise_tanh = a_noise.tanh()
    # fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
    # log_prob = log_prob_noise + fix_term
    #
    # show_tensor(a_noise_tanh)
    # show_tensor(log_prob.sum(1, keepdim=True))

    # a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
    # log_prob_noise = a_delta + a_std_log #+ 0.919  # self.constant_log_sqrt_2pi
    #
    # a_noise_tanh = a_noise.tanh()
    # fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
    # log_prob = log_prob_noise + fix_term
    #
    # show_tensor(a_noise_tanh)
    # show_tensor(log_prob.sum(1, keepdim=True) + 0.919 * action_dim)

    '''verifier '''
    a_noise_tanh = a_noise.tanh()

    fix_term = (-a_noise_tanh.pow(2) + 1.00001)  # .log()
    show_tensor(fix_term)
    fix_term = (((a_noise + 0.001).tanh() - a_noise_tanh) / 0.001 + 0.00001)  # .log()
    show_tensor(fix_term)


def test__run_train_agent():
    args = Arguments(AgentInterSAC, gpu_id=1)

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
    args.init_for_training()
    train_agent(**vars(args))  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()


if __name__ == '__main__':
    # test__network()
    # test__log_prob()
    # test__env_quickly()
    test__show_available_env()
    # test__replay_buffer()
    # test__evaluate_agent()
    # test__run_train_agent()
    print('; AgentTest Terminal.')
