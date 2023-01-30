import gym
import torch
from copy import deepcopy
from typing import Tuple
from torch import Tensor

from elegantrl.train.config import Config, build_env
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.envs.CustomGymEnv import PendulumEnv


def _check_buffer_items_for_off_policy(
        buffer_items: Tuple[Tensor, ...], if_discrete: bool,
        horizon_len: int, num_envs: int, state_dim: int, action_dim: int
):
    states, actions, rewards, undones = buffer_items

    assert states.shape == (horizon_len, num_envs, state_dim)
    assert states.dtype in {torch.float, torch.int}

    if if_discrete:
        actions_shape = (horizon_len, num_envs, 1)
        actions_dtypes = {torch.int, torch.long}
    else:
        actions_shape = (horizon_len, num_envs, action_dim)
        actions_dtypes = {torch.float, }
    assert actions.shape == actions_shape
    assert actions.dtype in actions_dtypes

    assert rewards.shape == (horizon_len, num_envs)
    assert rewards.dtype == torch.float

    assert undones.shape == (horizon_len, num_envs)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}


def _check_buffer_items_for_ppo_style(
        buffer_items: Tuple[Tensor, ...], if_discrete: bool,
        horizon_len: int, num_envs: int, state_dim: int, action_dim: int
):
    states, actions, logprobs, rewards, undones = buffer_items

    assert states.shape == (horizon_len, num_envs, state_dim)
    assert states.dtype in {torch.float, torch.int}

    if if_discrete:
        actions_shape = (horizon_len, num_envs, 1)
        actions_dtypes = {torch.int, torch.long}
    else:
        actions_shape = (horizon_len, num_envs, action_dim)
        actions_dtypes = {torch.float, }

    assert actions.shape == actions_shape
    assert actions.dtype in actions_dtypes

    assert logprobs.shape == (horizon_len, num_envs)
    assert logprobs.dtype == torch.float

    assert rewards.shape == (horizon_len, num_envs)
    assert rewards.dtype == torch.float

    assert undones.shape == (horizon_len, num_envs)
    assert undones.dtype == torch.float  # undones is float, instead of int
    assert set(undones.squeeze(1).cpu().data.tolist()).issubset({0.0, 1.0})  # undones in {0.0, 1.0}


def check_agent_base(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    print("\n| check_agent_base()")

    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device).detach()
    action = torch.rand(size=(batch_size, action_dim), dtype=torch.float32, device=device).detach()

    '''check AgentBase.__init__'''
    from elegantrl.agents.AgentBase import AgentBase
    from elegantrl.agents.AgentDDPG import AgentDDPG
    agent = AgentDDPG(net_dims, state_dim, action_dim, gpu_id=gpu_id, args=Config())
    AgentBase.__init__(agent, net_dims, state_dim, action_dim, gpu_id=gpu_id, args=Config())

    '''check AgentBase attribution'''
    assert hasattr(agent, 'explore_env')
    assert hasattr(agent, 'explore_one_env')
    assert hasattr(agent, 'explore_vec_env')

    assert hasattr(agent, 'update_net')
    assert hasattr(agent, 'get_obj_critic')
    assert hasattr(agent, 'get_obj_critic_raw')
    assert hasattr(agent, 'get_obj_critic_per')

    assert hasattr(agent, 'update_avg_std_for_normalization')
    assert hasattr(agent, 'get_returns')
    assert hasattr(agent.act, 'state_avg')
    assert hasattr(agent.act, 'state_std')
    assert hasattr(agent.cri, 'state_avg')
    assert hasattr(agent.cri, 'state_std')
    assert hasattr(agent.cri, 'value_avg')
    assert hasattr(agent.cri, 'value_std')

    '''check agent.optimizer'''
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


def check_agent_dqn_style(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    print("\n| check_agent_dqn()")

    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env = build_env(env_class=gym.make, env_args=env_args)
    num_envs = env_args['num_envs']
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    if_discrete = env_args['if_discrete']

    '''init agent'''
    from elegantrl.agents.AgentDQN import AgentDQN, AgentDuelingDQN, AgentDoubleDQN, AgentD3QN
    for agent_class in (AgentDQN, AgentDuelingDQN, AgentDoubleDQN, AgentD3QN):
        print(f"  agent_class = {agent_class.__name__}")

        buffer = ReplayBuffer(gpu_id=gpu_id, max_size=int(1e4), state_dim=state_dim, action_dim=1, )
        args = Config()
        args.batch_size = batch_size
        agent = agent_class(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        assert isinstance(state, Tensor)
        assert state.shape == (num_envs, state_dim)
        agent.last_state = state

        '''check for agent.explore_env'''
        for if_random in (True, False):
            print(f"  if_random = {if_random}")

            buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=if_random)
            _check_buffer_items_for_off_policy(
                buffer_items=buffer_items, if_discrete=if_discrete,
                horizon_len=horizon_len, num_envs=num_envs,
                state_dim=state_dim, action_dim=action_dim
            )
            buffer.update(buffer_items)

        '''check for agent.update_net'''
        buffer.update(buffer_items)
        obj_critic, q_value = agent.get_obj_critic(buffer=buffer, batch_size=batch_size)
        assert obj_critic.shape == ()
        assert q_value.shape == (batch_size,)
        assert q_value.dtype == torch.float32

        logging_tuple = agent.update_net(buffer=buffer)
        assert isinstance(logging_tuple, tuple)
        assert any([isinstance(item, float) for item in logging_tuple])
        assert len(logging_tuple) >= 2


def check_agent_ddpg_style(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    print("\n| check_agent_ddpg_style()")

    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env = build_env(env_class=PendulumEnv, env_args=env_args)
    num_envs = env_args['num_envs']
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    if_discrete = env_args['if_discrete']

    '''init agent'''
    from elegantrl.agents.AgentDDPG import AgentDDPG
    from elegantrl.agents.AgentTD3 import AgentTD3
    from elegantrl.agents.AgentSAC import AgentSAC, AgentModSAC
    for agent_class in (AgentDDPG, AgentTD3, AgentSAC, AgentModSAC):
        print(f"  agent_class = {agent_class.__name__}")

        buffer = ReplayBuffer(gpu_id=gpu_id, max_size=int(1e4), state_dim=state_dim, action_dim=action_dim, )
        args = Config()
        args.batch_size = batch_size
        agent = agent_class(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        assert isinstance(state, Tensor)
        assert state.shape == (num_envs, state_dim)
        agent.last_state = state

        '''check for agent.explore_env if_random=True'''
        if_random = True
        buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=if_random)
        _check_buffer_items_for_off_policy(
            buffer_items=buffer_items, if_discrete=if_discrete,
            horizon_len=horizon_len, num_envs=num_envs,
            state_dim=state_dim, action_dim=action_dim
        )
        buffer.update(buffer_items)

        if_random = False
        buffer_items = agent.explore_env(env=env, horizon_len=horizon_len, if_random=if_random)
        _check_buffer_items_for_off_policy(
            buffer_items=buffer_items, if_discrete=if_discrete,
            horizon_len=horizon_len, num_envs=num_envs,
            state_dim=state_dim, action_dim=action_dim
        )
        buffer.update(buffer_items)

        '''check for agent.update_net'''
        buffer.update(buffer_items)
        obj_critic, state = agent.get_obj_critic(buffer=buffer, batch_size=batch_size)
        assert obj_critic.shape == ()
        assert state.shape == (batch_size, state_dim)
        assert state.dtype in {torch.float, torch.int}

        logging_tuple = agent.update_net(buffer=buffer)
        assert isinstance(logging_tuple, tuple)
        assert any([isinstance(item, float) for item in logging_tuple])
        assert len(logging_tuple) >= 2


def check_agent_ppo_style(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    print("\n| check_agent_ddpg_style()")

    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env = build_env(env_class=PendulumEnv, env_args=env_args)
    num_envs = env_args['num_envs']
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    if_discrete = env_args['if_discrete']

    '''init agent'''
    from elegantrl.agents.AgentPPO import AgentPPO  # , AgentDiscretePPO
    from elegantrl.agents.AgentA2C import AgentA2C  # , AgentDiscreteA2C
    for agent_class in (AgentPPO, AgentA2C):
        print(f"  agent_class = {agent_class.__name__}")

        args = Config()
        args.batch_size = batch_size
        agent = agent_class(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        assert isinstance(state, Tensor)
        assert state.shape == (num_envs, state_dim)
        agent.last_state = state

        '''check for agent.explore_env'''
        buffer_items = agent.explore_env(env=env, horizon_len=horizon_len)
        _check_buffer_items_for_ppo_style(
            buffer_items=buffer_items, if_discrete=if_discrete,
            horizon_len=horizon_len, num_envs=num_envs,
            state_dim=state_dim, action_dim=action_dim,
        )

        '''check for agent.update_net'''
        states, actions, logprobs, rewards, undones = buffer_items

        values = agent.cri(states)
        assert values.shape == (horizon_len, num_envs)

        advantages = agent.get_advantages(rewards, undones, values)
        assert advantages.shape == (horizon_len, num_envs)

        logging_tuple = agent.update_net(buffer=buffer_items)
        assert isinstance(logging_tuple, tuple)
        assert any([isinstance(item, float) for item in logging_tuple])
        assert len(logging_tuple) >= 2


def check_agent_ppo_discrete_style(batch_size=3, horizon_len=16, net_dims=(64, 32), gpu_id=0):
    print("\n| check_agent_ppo_discrete_style()")

    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env = build_env(env_class=gym.make, env_args=env_args)
    num_envs = env_args['num_envs']
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    if_discrete = env_args['if_discrete']

    '''init agent'''
    from elegantrl.agents.AgentPPO import AgentDiscretePPO
    from elegantrl.agents.AgentA2C import AgentDiscreteA2C
    for agent_class in (AgentDiscretePPO, AgentDiscreteA2C):
        print(f"  agent_class = {agent_class.__name__}")

        args = Config()
        args.batch_size = batch_size
        agent = agent_class(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        assert isinstance(state, Tensor)
        assert state.shape == (num_envs, state_dim)
        agent.last_state = state

        '''check for agent.explore_env'''
        buffer_items = agent.explore_env(env=env, horizon_len=horizon_len)
        _check_buffer_items_for_ppo_style(
            buffer_items=buffer_items, if_discrete=if_discrete,
            horizon_len=horizon_len, num_envs=num_envs,
            state_dim=state_dim, action_dim=action_dim,
        )

        '''check for agent.update_net'''
        states, actions, logprobs, rewards, undones = buffer_items

        values = agent.cri(states)
        assert values.shape == (horizon_len, num_envs)

        advantages = agent.get_advantages(rewards, undones, values)
        assert advantages.shape == (horizon_len, num_envs)

        logging_tuple = agent.update_net(buffer=buffer_items)
        assert isinstance(logging_tuple, tuple)
        assert any([isinstance(item, float) for item in logging_tuple])
        assert len(logging_tuple) >= 2


if __name__ == '__main__':
    print('\n| check_agents.py.')
    check_agent_base()

    check_agent_dqn_style()
    check_agent_ddpg_style()
    check_agent_ppo_style()
    check_agent_ppo_discrete_style()
