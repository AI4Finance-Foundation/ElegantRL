import os
import time
import numpy as np
import torch as th

from elegantrl2.traj_config import Config, build_env
from elegantrl2.traj_evaluator import Evaluator
from elegantrl2.traj_buffer import TrajBuffer

TEN = th.Tensor
ARY = np.ndarray


def action_to_dist_ary(action: TEN, action_dim: int, show_max: int = 9) -> ARY:
    if action.shape[1] == 1:
        unique_values, counts = th.unique(action, return_counts=True)
        show_ary = np.zeros((action_dim,), dtype=int)
        show_ary[unique_values.numpy()] = counts * (999 / th.numel(action))
    else:
        q = th.tensor((0.1, 0.5, 0.9), dtype=th.float32, device=action.device)
        show_ary = th.quantile(action[:, :show_max // 3], q=q, dim=1).data.cpu().numpy()
        show_ary = (show_ary * 999).astype(int)
    return show_ary


def check__action_to_str():
    action_dim = 4
    sample_num = 1943
    remove_action_id = 2

    action = th.randint(0, action_dim, size=(sample_num,))
    action = action[action != remove_action_id]
    action_dist_ary = action_to_dist_ary(action=action, action_dim=action_dim)
    assert action_dist_ary[remove_action_id] == 0
    print(action_dist_ary)


def train_agent(args: Config):
    args.init_before_training()
    th.set_grad_enabled(False)

    '''init environment'''
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    state, info_dict = env.reset()
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, TEN)

    '''init agent'''
    agent:AgentPPO = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)
    agent.last_state = state.detach().to(agent.device)
    del state

    '''init buf'''
    buffer = TrajBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
        if_discrete=args.if_discrete,
        args=args,
    )

    '''init evaluator'''
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)

    '''train loop'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_off_policy = args.if_off_policy
    if_save_buffer = args.if_save_buffer

    if_discrete = env.if_discrete

    del args

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)
        """buffer_items
        buffer_items = (states, rewards, undones, unmasks, actions)

        item.shape == (horizon_len, num_workers * num_envs, ...)
        actions.shape == (horizon_len, num_workers * num_envs, action_dim)  # if_discrete=False
        actions.shape == (horizon_len, num_workers * num_envs)              # if_discrete=True
        """
        buffer.update_seqs(seqs=th.concat(buffer_items, dim=2))

        show_str = action_to_dist_ary(action=buffer_items[4].data.cpu())
        exp_r = buffer_items[2].mean().item()

        th.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        logging_tuple = (*logging_tuple, agent.explore_rate, show_str)
        th.set_grad_enabled(False)

        evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}', flush=True)

    env.close() if hasattr(env, 'close') else None
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
        buffer.save_or_load_history(cwd, if_save=True)


def train_ppo_for_lunar_lander_continuous(agent_class, gpu_id: int):
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': False,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    # get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step
    args.repeat_times = 64  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.04

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args)


"""utils"""


def get_logprob_by_dist_normal(action_avg, action_std, action):
    """
    使用 torch.distributions.normal.Normal 计算 logprob
    :param action_avg: 均值，形状为 (seq_num, action_dim)
    :param action_std: 标准差，形状为 (seq_num, action_dim)
    :param action: 采样值，形状为 (seq_num, action_dim)
    :return: logprob，形状为 (seq_num,)
    """
    import torch.distributions as dist  # 仅用于代码检查
    dist = dist.normal.Normal(action_avg, action_std)
    logprob = dist.log_prob(action).sum(1)  # 沿着 action_dim 维度求和
    return logprob


def get_logprob_by_raw(action_avg: TEN, action_std: TEN, action: TEN) -> TEN:
    """
    手动实现正态分布的对数概率密度函数
    :param action_avg: 均值，形状为 (seq_num, action_dim)
    :param action_std: 标准差，形状为 (seq_num, action_dim)
    :param action: 采样值，形状为 (seq_num, action_dim)
    :return: logprob，形状为 (seq_num,)
    """
    # log_2pi = th.log(2 * th.tensor(th.pi))  # 常数 log(2π)
    # logprob = (
    #         -0.5 * log_2pi
    #         - th.log(action_std)
    #         - 0.5 * ((action - action_avg) / action_std) ** 2
    # ).sum(1)  # 沿着 action_dim 维度求和
    '''以上计算可简化为'''
    log_2pi_action_std_sq = th.log(2 * th.pi * action_std ** 2)
    normalized_sq = ((action - action_avg) / action_std) ** 2
    return -0.5 * (log_2pi_action_std_sq + normalized_sq).sum(1)


def check__get_logprob():
    # 设置随机种子以确保结果可重复
    th.manual_seed(42)

    # 定义输入数据
    batch_size = 5
    action_dim = 3

    # 随机生成 action_avg, action_std, action
    action_avg = th.randn(batch_size, action_dim)  # 均值
    action_std = th.rand(batch_size, action_dim).abs() + 1e-6  # 标准差，确保为正数
    action = th.randn(batch_size, action_dim)  # 采样值

    # 使用方法 1 计算 logprob
    logprob_method1 = get_logprob_by_dist_normal(action_avg, action_std, action)

    # 使用方法 2 计算 logprob
    logprob_method2 = get_logprob_by_raw(action_avg, action_std, action)

    # 打印结果
    print("Logprob (Method 1 - torch.distributions):\n", logprob_method1)
    print("Logprob (Method 2 - Manual Calculation):\n", logprob_method2)

    # 检查两种方法的结果是否一致
    if th.allclose(logprob_method1, logprob_method2, atol=1e-6):
        print("\n两种方法的结果一致！")
    else:
        print("\n两种方法的结果不一致！")


if __name__ == '__main__':
    # test_logprob_functions()
    from elegantrl2.traj_agent_ppo import AgentPPO

    train_ppo_for_lunar_lander_continuous(agent_class=AgentPPO, gpu_id=0)
