from elegantrl.train.evaluator import *
from elegantrl.train.config import Arguments
from elegantrl.envs.CustomGymEnv import GymNormaEnv
from elegantrl.agents.AgentPPO import AgentPPO, AgentPPOgetObjHterm
from elegantrl.agents.AgentSAC import AgentSAC, AgentReSAC


def get_cumulative_returns_and_step(env, act, if_render=False) -> (float, int):
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
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = act(tensor_state).argmax(dim=1) if if_discrete else act(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        returns += reward

        if if_render:
            env.render()
        if done:
            break
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1
    return returns, steps


def demo_evaluator_actor_pth():
    from elegantrl.train.config import build_env

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU
    env_name = ['Hopper-v3',
                'Swimmer-v3',
                'HalfCheetah-v3',
                'Walker2d-v3',
                'Ant-v3',
                'Humanoid-v3',
                ][5]
    agent_class = [AgentPPO, ][0]  # using AgentPPO or AgentPPOHtermK is the same when evaluating

    if env_name == 'Hopper-v3':
        env_func = GymNormaEnv  # gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Hopper-v3',
            'max_step': 1000,
            'state_dim': 11,
            'action_dim': 3,
            'if_discrete': False,
            'target_return': 3500.,
        }
        actor_path = './actor_Hopper_PPO_hop.pth'
        # actor_path = './actor_Hopper_PPO_hop_fail.pth'
        # actor_path = './actor_Hopper_PPO_fail.pth'
        actor_path = './actor_Hopper_PPO_stand.pth'

        net_dim = 2 ** 8
        layer_num = 3
    elif env_name == 'HalfCheetah-v3':
        env_func = GymNormaEnv  # gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'HalfCheetah-v3',
            'max_step': 1000,
            'state_dim': 17,
            'action_dim': 6,
            'if_discrete': False,
            'target_return': 4800.0,
        }
        # actor_path = './actor_HalfCheetah_PPO_run.pth'
        # actor_path = './actor_HalfCheetah_PPO_kiss_ground.pth'
        actor_path = './actor_HalfCheetah_PPO_stand.pth'
        net_dim = 2 ** 7
        layer_num = 3
    elif env_name == 'Swimmer-v3':
        env_func = GymNormaEnv  # gym.make
        # import gym
        # env_func = gym.make
        env_args = {
            'action_dim': 2,
            'env_name': 'Swimmer-v3',
            'env_num': 1,
            'if_discrete': False,
            'max_step': 1000,
            'state_dim': 8,
            'target_return': 360.0
        }
        # agent_class = AgentPPO
        # actor_path = './actor_Swimmer_PPO_C_160.pth'
        # actor_path = './actor_Swimmer_PPO_C_134.pth'
        # actor_path = './actor_Swimmer_PPO_C_157.pth'
        # actor_path = './actor_Swimmer_PPO_C_152.pth'
        # actor_path = './actor_Swimmer_PPO_C_097.201.pth'
        actor_path = './actor_Swimmer_PPO_stay_031.pth'

        # agent_class = AgentReSAC
        # actor_path = './actor_Swimmer_ReSAC_S_211.pth'
        # actor_path = './actor_Swimmer_ReSAC_S_224.pth'
        # actor_path = './actor_Swimmer_ReSAC_S_286.pth'  # norm

        net_dim = 2 ** 8
        layer_num = 3
    elif env_name == 'Walker2d-v3':
        env_func = GymNormaEnv  # gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Walker2d-v3',
            'if_discrete': False,
            'max_step': 1000,
            'state_dim': 17,
            'action_dim': 6,
            'target_return': 7000,
        }
        # actor_path = './actor_Walker2d_run11_7870.pth'  # norm
        # actor_path = './actor_Walker2d_run11_7209.pth'  # norm
        # actor_path = './actor_Walker2d_run11_6812.pth'  # norm
        # actor_path = './actor_Walker2d_run11_6955.pth'  # norm
        # actor_path = './actor_Walker2d_run12_5461.pth'  # norm
        # actor_path = './actor_Walker2d_run12_3295.pth'  # norm
        # actor_path = './actor_Walker2d_jump_4008.pth'  # norm
        # actor_path = './actor_Walker2d_fail_4512.pth'  # norm
        # actor_path = './actor_Walker2d_fail_6792.pth'  # norm
        # actor_path = './actor_Walker2d_fail_4992.pth'  # norm
        actor_path = './actor_Walker2d_fail_0431.pth'  # norm

        net_dim = 2 ** 8
        layer_num = 3
    elif env_name == 'Ant-v3':
        env_func = GymNormaEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Ant-v3',
            'max_step': 1000,
            'state_dim': 111,
            'action_dim': 8,
            'if_discrete': False,
            'target_return': 6000.0,
        }
        # actor_path = './actor_Ant_PPO_run_4701.pth'
        # actor_path = './actor_Ant_PPO_run_2105.pth'
        # actor_path = './actor_Ant_PPO_fail_174.pth'
        # actor_path = './actor_Ant_PPO_stay_909.pth'
        actor_path = './actor_Ant_PPO_stay_986.pth'

        net_dim = 2 ** 8
        layer_num = 3
    elif env_name == 'Humanoid-v3':
        from elegantrl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 8000.,
        }
        # from elegantrl.agents.AgentSAC import AgentReSAC
        # agent_class = AgentReSAC

        agent_class = AgentPPO
        actor_path = './actor_Huamnoid_PPO_run_8021.pth'
        # actor_path = './actor_Huamnoid_PPO_run_7105.pth'
        # actor_path = './actor_Huamnoid_PPO_run_6437.pth'
        # actor_path = './actor_Huamnoid_PPO_run_5422.pth'
        # actor_path = './actor_Huamnoid_PPO_run_3491.pth'
        # actor_path = './actor_Huamnoid_PPO_lift_leg_7500.pth'
        # actor_path = './actor_Huamnoid_PPO_lift_leg_6076.pth'
        # actor_path = './actor_Huamnoid_PPO_lift_knee_5136.pth'
        # actor_path = './actor_Huamnoid_PPO_curl_leg_4244.pth'  # net_dim = 2 ** 7
        # actor_path = './actor_Huamnoid_PPO_curl_leg_6378.pth'
        # actor_path = './actor_Huamnoid_PPO_run_7194.pth'  # norm
        # actor_path = './actor_Huamnoid_PPO_lift_knee_6887.pth'
        # actor_path = './actor_Huamnoid_PPO_lift_knee_7585.pth'
        # actor_path = './actor_Huamnoid_PPO_lift_knee_5278.pth'
        # actor_path = './actor_Huamnoid_PPO_run_4759.pth'
        # actor_path = './actor__000108565781_07978.063.pth'  # (Humanoid-v3_PPOHtermK_6 from single to two legs)
        # actor_path = './actor_Huamnoid_PPO_run_9732.pth'  # norm, nice racing
        # actor_path = './actor_Huamnoid_PPO_run_10863.pth'  # norm, nice racing
        # actor_path = './actor__000027862483_10202.021.pth'  # norm, nice racing

        net_dim = 2 ** 9
        layer_num = 3
    else:
        raise ValueError('env_name:', env_name)

    eval_times = 2 ** 4

    '''init'''
    args = Arguments(agent_class=agent_class, env_func=env_func, env_args=env_args)
    args.net_dim = net_dim
    args.num_layer = layer_num
    env = build_env(env_func=args.env_func, env_args=args.env_args)
    act = agent_class(net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate file'''
    r_s_ary = [get_cumulative_returns_and_step(env, act, if_render=True) for _ in range(eval_times)]
    # r_s_ary = [get_cumulative_returns_and_step(env, act, if_render=False) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    print(f'{actor_path:64} | r_avg {r_avg:9.3f} | s_avg {s_avg:9.3f}')

    '''evaluate directory'''
    # dir_path = 'Humanoid-v3_PPO_4'
    # for name in os.listdir(dir_path):
    #     if name[-4:] != '.pth':
    #         continue
    #     actor_path = f"{dir_path}/{name}"
    #
    #     act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    #
    #     r_s_ary = [get_cumulative_returns_and_step(env, act, if_render=False) for _ in range(eval_times)]
    #     r_s_ary = np.array(r_s_ary, dtype=np.float32)
    #     r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    #     print(f'{actor_path:64} | r_avg {r_avg:9.3f} | s_avg {s_avg:9.3f}')


if __name__ == '__main__':
    demo_evaluator_actor_pth()
