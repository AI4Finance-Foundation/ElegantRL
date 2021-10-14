import torch

from elegantrl.run import Arguments, train_and_evaluate
from elegantrl.env import build_env
from elegantrl.agent import AgentD3QN

def train_save_evaluate_watch():
    args = Arguments(env=build_env('CartPole-v0'), agent=AgentD3QN())

    '''train and save'''
    args.cwd = 'demo_CartPole_D3QN'
    args.eval_gap = 2 ** 5
    args.target_return = 195

    train_and_evaluate(args)  # single process

    '''evaluate and watch'''
    env = build_env('CartPole-v0')

    agent = AgentD3QN()
    agent.init(args.net_dim, args.state_dim, args.action_dim, gpu_id=0)
    agent.save_or_load_agent(cwd=args.cwd, if_save=False)
    agent.explore_rate = 0.0

    state = env.reset()
    episode_return = 0
    for i in range(2 ** 10):
        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        a_tensor = agent.select_actions(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        next_state, reward, done, _ = env.step(int(action))

        episode_return += reward
        if done:
            print(f'Step {i:>6}, EpisodeReturn {episode_return:8.3f}')
            state = env.reset()
            episode_return = 0
            step = 0
        else:
            state = next_state
        env.render()

    print('done')

train_save_evaluate_watch()