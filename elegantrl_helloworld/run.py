import os
import time
import torch
import numpy as np

from elegantrl_helloworld.env import build_env
from elegantrl_helloworld.agent import ReplayBuffer, ReplayBufferList


def train_agent(args):
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env_func, args.env_args)

    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.states = [env.reset(), ]

    if args.if_off_policy:
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_capacity=args.max_capacity,
                              state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim, )
        trajectory = agent.explore_env(env, args.target_step * args.eval_gap)
        buffer.update_buffer((trajectory,))
    else:
        buffer = ReplayBufferList()

    '''start training'''
    cwd = args.cwd
    eval_gap = args.eval_gap
    break_step = args.break_step
    target_step = args.target_step
    del args

    start_time = time.time()
    total_step = 0
    eval_timer = start_time - eval_gap
    torch.set_grad_enabled(False)
    while True:
        trajectory = agent.explore_env(env, target_step)
        steps, r_exp = buffer.update_buffer((trajectory,))

        with torch.enable_grad():
            logging_tuple = agent.update_net(buffer)

        total_step += steps

        if eval_timer + eval_gap < time.time():
            eval_timer = time.time()
            print(f"| Step {total_step:8.2e}  ExpR {r_exp:8.2f}  "
                  f"| ObjC {logging_tuple[0]:8.2f}  ObjA {logging_tuple[1]:8.2f}")
            save_path = f"{cwd}/actor_{total_step:012.0f}_{time.time() - start_time:08.0f}_{r_exp:08.2f}.pth"
            torch.save(agent.act.state_dict(), save_path)

        if (total_step > break_step) or os.path.exists(f"{cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`

    print(f'| UsedTime: {time.time() - start_time:.0f} | SavedDir: {cwd}')


def evaluate_agent(args):
    args.if_remove = False
    args.init_before_training()
    gpu_id = args.learner_gpus

    env = build_env(env_func=args.env_func, env_args=args.env_args)
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    actor = agent.act

    cwd = args.cwd
    recorder_path = f"{cwd}/recorder.npy"

    recorder = list()

    torch.set_grad_enabled(False)
    model_names = [name for name in os.listdir(cwd) if name[:6] == 'actor_']
    model_names.sort()
    for model_name in model_names:
        model_path = f"{cwd}/{model_name}"
        actor.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        cumulative_returns_ary = [get_cumulative_rewards(env, actor) for _ in range(args.eval_times)]
        cumulative_returns_ary = np.array(cumulative_returns_ary)
        cumulative_returns_avg = cumulative_returns_ary.mean()
        cumulative_returns_std = cumulative_returns_ary.std()

        name_split = model_name.split('_')
        steps = int(name_split[1])
        used_time = int(name_split[2])
        recorder.append((steps, used_time, cumulative_returns_avg))
        print(f"| Steps {steps:8.2e}  | Returns avg {cumulative_returns_avg:9.3f}  std {cumulative_returns_std:9.3f}")
    recorder = np.array(recorder)
    np.save(recorder_path, recorder)

    import matplotlib.pyplot as plt
    x_axis = recorder[:, 0]
    y_axis = recorder[:, 2]
    plt.plot(x_axis, y_axis)
    plt.xlabel('#samples (Steps)')
    plt.ylabel('#Returns (Score)')
    plt.grid()
    plt.title(f'Learning curve {args.agent_class.__name__} in task {args.env_name}')
    plt.savefig(f"{cwd}/LearningCurve_{args.env_name}_{args.agent_class.__name__}.jpg")
    plt.show()


def get_cumulative_rewards(env, act) -> float:
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        cumulative_returns += reward
        if done:
            break

    cumulative_returns = getattr(env, 'cumulative_returns', cumulative_returns)
    return cumulative_returns
