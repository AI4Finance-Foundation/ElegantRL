import torch.nn

from net import *


def check_class_q_net(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    '''check'''
    act = QNet(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)
    act.explore_rate = 0.1

    action = act(state=state)
    assert isinstance(action, Tensor)
    assert action.dtype in {torch.float}
    assert action.shape == (batch_size, action_dim)

    action = act.get_action(state=state)
    assert isinstance(action, Tensor)
    assert action.dtype in {torch.int, torch.long}
    assert action.shape == (batch_size, 1)


def check_class_actor(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    '''check'''
    act = Actor(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)
    act.explore_noise_std = 0.1  # standard deviation of exploration action noise

    action = act(state=state)
    assert isinstance(action, Tensor)
    assert action.dtype in {torch.float}
    assert action.shape == (batch_size, action_dim)
    assert torch.any((-1.0 < action) & (action < +1.0))

    action = act.get_action(state=state)
    assert isinstance(action, Tensor)
    assert action.dtype in {torch.float}
    assert action.shape == (batch_size, action_dim)
    assert torch.any((-1.0 < action) & (action < +1.0))


def check_class_critic(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)
    action = torch.rand(size=(batch_size, action_dim), dtype=torch.float32, device=device)

    '''check'''
    cri = Critic(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

    q = cri(state=state, action=action)
    assert isinstance(q, Tensor)
    assert q.dtype in {torch.float}
    assert q.shape == (batch_size, 1)


def check_class_actor_ppo(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    '''check'''
    act = ActorPPO(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)
    assert isinstance(act.action_std_log, nn.Parameter)
    assert act.action_std_log.requires_grad

    action = act(state=state)
    assert isinstance(action, Tensor)
    assert action.dtype in {torch.float}
    assert action.shape == (batch_size, action_dim)
    action = act.convert_action_for_env(action)
    assert torch.any((-1.0 < action) & (action < +1.0))

    action, logprob = act.get_action(state=state)
    assert isinstance(action, Tensor)
    assert action.dtype in {torch.float}
    assert action.shape == (batch_size, action_dim)
    assert torch.any((-1.0 < action) & (action < +1.0))
    assert isinstance(logprob, Tensor)
    assert logprob.shape == (batch_size,)

    action = torch.rand(size=(batch_size, action_dim), dtype=torch.float32, device=device)
    logprob, entropy = act.get_logprob_entropy(state=state, action=action)
    assert isinstance(logprob, Tensor)
    assert logprob.shape == (batch_size,)
    assert isinstance(entropy, Tensor)
    assert entropy.shape == (batch_size,)


def check_class_critic_ppo(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    '''check'''
    cri = CriticPPO(dims=net_dims, state_dim=state_dim, _action_dim=action_dim).to(device)

    q = cri(state=state)
    assert isinstance(q, Tensor)
    assert q.dtype in {torch.float}
    assert q.shape == (batch_size, 1)


def check_def_build_mlp():
    net_dims = (64, 32)
    net = build_mlp(dims=net_dims)
    assert isinstance(net, nn.Sequential)
    assert len(net) == 1 == (len(net_dims) - 1) * 2 - 1

    net_dims = (64, 32, 16)
    net = build_mlp(dims=net_dims)
    assert isinstance(net, nn.Sequential)
    assert len(net) == 3 == (len(net_dims) - 1) * 2 - 1

    net_dims = (64, 32, 16, 8)
    net = build_mlp(dims=net_dims)
    assert isinstance(net, nn.Sequential)
    assert len(net) == 5 == (len(net_dims) - 1) * 2 - 1


if __name__ == '__main__':
    check_class_q_net()
    check_class_actor()
    check_class_critic()
    check_class_actor_ppo()
    check_class_critic_ppo()
    check_def_build_mlp()
    print('| Finish checking.')
