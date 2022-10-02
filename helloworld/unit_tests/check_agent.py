import gym
import torch

from env import PendulumEnv
from agent import *


def check_agent_base(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device).detach()
    action = torch.rand(size=(batch_size, action_dim), dtype=torch.float32, device=device).detach()

    '''check AgentBase'''
    agent = AgentDDPG(net_dims, state_dim, action_dim, gpu_id=gpu_id, args=Config())
    AgentBase.__init__(agent, net_dims, state_dim, action_dim, gpu_id=gpu_id, args=Config())

    '''check for run.render_agent'''
    action_grad = agent.act(state)
    q_value = agent.cri(state, action_grad)
    obj_act = -q_value.mean()
    assert agent.optimizer_update(agent.act_optimizer, obj_act) is None
    q_value = agent.cri(state, action)
    obj_cri = agent.criterion(q_value, torch.zeros_like(q_value).detach()).mean()
    assert agent.optimizer_update(agent.cri_optimizer, obj_cri) is None

    current_net = agent.cri
    target_net = deepcopy(agent.cri)
    assert agent.soft_update(target_net=target_net, current_net=current_net, tau=3e-5) is None


def check_agent_dqn(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    from config import build_env
    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env = build_env(env_class=gym.make, env_args=env_args)
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']

    '''init agent'''
    from agent import ReplayBuffer
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=int(1e4), state_dim=state_dim, action_dim=1, )
    args = Config()
    args.batch_size = batch_size
    agent = AgentDQN(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = env.reset()

    '''check for agent.explore_env'''
    buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=True)
    buffer.update(buffer_items)
    states, actions, rewards, undones = buffer_items
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}
    assert actions.shape == (horizon_len, 1)
    assert actions.dtype in {torch.int, torch.long}
    assert rewards.shape == (horizon_len, 1)
    assert rewards.dtype == torch.float
    assert undones.shape == (horizon_len, 1)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}

    buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=False)
    buffer.update(buffer_items)
    states, actions, rewards, undones = buffer_items
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}
    assert actions.shape == (horizon_len, 1)
    assert actions.dtype in {torch.int, torch.long}
    assert rewards.shape == (horizon_len, 1)
    assert rewards.dtype == torch.float
    assert undones.shape == (horizon_len, 1)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}

    '''check for agent.update_net'''
    buffer.update(buffer_items)
    obj_critic, state = agent.get_obj_critic(buffer=buffer, batch_size=batch_size)
    assert obj_critic.shape == ()
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}

    logging_tuple = agent.update_net(buffer=buffer)
    assert isinstance(logging_tuple, tuple)
    assert any([isinstance(item, float) for item in logging_tuple])
    assert len(logging_tuple) >= 2


def check_agent_ddpg(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    from config import build_env
    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env = build_env(env_class=PendulumEnv, env_args=env_args)
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']

    '''init agent'''
    from agent import ReplayBuffer
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=int(1e4), state_dim=state_dim, action_dim=action_dim, )
    args = Config()
    args.batch_size = batch_size
    agent = AgentDDPG(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = env.reset()

    '''check for agent.explore_env'''
    buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=True)
    states, actions, rewards, undones = buffer_items
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}
    assert actions.shape == (horizon_len, action_dim)
    assert actions.dtype == torch.float
    assert rewards.shape == (horizon_len, 1)
    assert rewards.dtype == torch.float
    assert undones.shape == (horizon_len, 1)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}

    buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=False)
    states, actions, rewards, undones = buffer_items
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}
    assert actions.shape == (horizon_len, action_dim)
    assert actions.dtype == torch.float
    assert rewards.shape == (horizon_len, 1)
    assert rewards.dtype == torch.float
    assert undones.shape == (horizon_len, 1)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}

    '''check for agent.update_net'''
    buffer.update(buffer_items)
    obj_critic, state = agent.get_obj_critic(buffer=buffer, batch_size=batch_size)
    assert obj_critic.shape == ()
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}

    logging_tuple = agent.update_net(buffer=buffer)
    assert isinstance(logging_tuple, tuple)
    assert any([isinstance(item, float) for item in logging_tuple])
    assert len(logging_tuple) >= 2


def check_agent_ppo(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    from config import build_env
    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env = build_env(env_class=PendulumEnv, env_args=env_args)
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']

    '''init agent'''
    args = Config()
    args.batch_size = batch_size
    agent = AgentPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = env.reset()

    convert = agent.act.convert_action_for_env
    action = torch.rand(size=(batch_size, action_dim), dtype=torch.float32).detach() * 6 - 3
    assert torch.any((action < -1.0) | (+1.0 < action))
    action = convert(action)
    assert torch.any((-1.0 <= action) & (action <= +1.0))

    '''check for agent.explore_env'''
    buffer_items = agent.explore_env(env=env, horizon_len=horizon_len)
    states, actions, logprobs, rewards, undones = buffer_items
    assert states.shape == (horizon_len, state_dim)
    assert states.dtype in {torch.float, torch.int}
    assert actions.shape == (horizon_len, action_dim)
    assert actions.dtype == torch.float
    assert logprobs.shape == (horizon_len,)
    assert logprobs.dtype == torch.float
    assert rewards.shape == (horizon_len, 1)
    assert rewards.dtype == torch.float
    assert undones.shape == (horizon_len, 1)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}

    '''check for agent.update_net'''
    values = agent.cri(states).squeeze(1)
    assert values.shape == (horizon_len,)
    advantages = agent.get_advantages(rewards=rewards, undones=undones, values=values)
    assert advantages.shape == (horizon_len,)
    assert advantages.dtype in {torch.float, torch.int}

    logging_tuple = agent.update_net(buffer=buffer_items)
    assert isinstance(logging_tuple, tuple)
    assert any([isinstance(item, float) for item in logging_tuple])
    assert len(logging_tuple) >= 2


if __name__ == '__main__':
    check_agent_base()
    check_agent_dqn()
    check_agent_ddpg()
    check_agent_ppo()
    print('| Finish checking.')
