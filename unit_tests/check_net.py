import torch
import torch.nn as nn
from torch import Tensor


def check_net_base(state_dim=4, action_dim=2, batch_size=3, gpu_id=0):
    print("\n| check_net_base()")

    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    '''check for agent.AgentBase.update_avg_std_for_normalization()'''
    from elegantrl.agents.net import QNetBase, ActorBase, CriticBase
    for net_base in (QNetBase, ActorBase, CriticBase):
        print(f"  net_base = {net_base.__name__}")
        net = net_base(state_dim=state_dim, action_dim=action_dim).to(device)

        state_avg = net.state_avg
        assert isinstance(state_avg, Tensor)
        assert not state_avg.requires_grad
        state_std = net.state_std
        assert isinstance(state_std, Tensor)
        assert not state_std.requires_grad

        _state = net.state_norm(state)
        assert isinstance(_state, Tensor)
        assert _state.shape == (batch_size, state_dim)

    for net_base in (QNetBase, CriticBase):
        print(f"  net_base = {net_base.__name__}")
        net = net_base(state_dim=state_dim, action_dim=action_dim).to(device)

        value_avg = net.value_avg
        assert isinstance(value_avg, Tensor)
        assert not value_avg.requires_grad
        value_std = net.value_std
        assert isinstance(value_std, Tensor)
        assert not value_std.requires_grad

        value = torch.rand((batch_size, 2), dtype=torch.float32, device=device)
        _value = net.value_re_norm(value)
        assert isinstance(_value, Tensor)
        assert _value.shape == value.shape


def check_q_net(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    print("\n| check_q_net()")

    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    '''check for agent.AgentDQN, ...'''
    from elegantrl.agents.net import QNet, QNetDuel
    from elegantrl.agents.net import QNetTwin, QNetTwinDuel

    for net_class in (QNet, QNetDuel, QNetTwin, QNetTwinDuel):
        print(f"  net_class = {net_class.__name__}")
        net = net_class(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)
        net.explore_rate = 0.1

        '''check for run.get_rewards_and_steps'''
        action = net(state=state)
        assert isinstance(action, Tensor)
        assert action.dtype in {torch.float}
        assert action.shape == (batch_size, action_dim)

        '''check for agent.AgentDQN.explore_env'''
        action = net.get_action(state=state)
        assert isinstance(action, Tensor)
        assert action.dtype in {torch.int, torch.long}
        assert action.shape == (batch_size, 1)

    '''check for agent.AgentDoubleDQN, agent.AgentD3DQN'''
    from elegantrl.agents.net import QNetTwin, QNetTwinDuel
    for net_class in (QNetTwin, QNetTwinDuel):
        print(f"  net_class = {net_class.__name__}")
        net = net_class(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

        '''check for run.get_rewards_and_steps'''
        action = net(state=state)
        assert isinstance(action, Tensor)
        assert action.dtype in {torch.float}
        assert action.shape == (batch_size, action_dim)

        '''check for agent.AgentDQN.explore_env'''
        q1, q2 = net.get_q1_q2(state=state)
        assert isinstance(q1, Tensor)
        assert isinstance(q2, Tensor)
        assert q1.dtype is torch.float
        assert q2.dtype is torch.float
        assert q1.shape == (batch_size, action_dim)
        assert q2.shape == (batch_size, action_dim)


def check_actor(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    print("\n| check_actor()")
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)

    from elegantrl.agents.net import Actor, ActorSAC, ActorFixSAC, ActorPPO, ActorDiscretePPO
    '''check for agent.explore_env()'''
    for actor_class in (Actor, ActorSAC, ActorFixSAC):
        print(f"  actor_class = {actor_class.__name__}")
        act = actor_class(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)
        act.explore_noise_std = 0.1  # standard deviation of exploration action noise

        action = act(state=state)
        assert isinstance(action, Tensor)
        assert action.dtype in {torch.float}
        assert action.shape == (batch_size, action_dim)
        assert torch.any((-1.0 <= action) & (action <= +1.0))

        if actor_class in {ActorPPO, ActorDiscretePPO}:  # on-policy
            action, logprob = act.get_action(state=state)
            assert isinstance(logprob, Tensor)
            assert logprob.dtype in {torch.float}
            assert logprob.shape == (batch_size, action_dim)
        else:  # if actor_class in {Actor, ActorSAC, ActorFixSAC}:  # off-policy
            action = act.get_action(state=state)
        assert isinstance(action, Tensor)
        assert action.dtype in {torch.float}
        assert action.shape == (batch_size, action_dim)
        assert torch.any((-1.0 <= action) & (action <= +1.0))

    '''check for agent.update_net()'''
    for actor_class in (ActorSAC, ActorFixSAC):
        print(f"  actor_class = {actor_class.__name__}")
        act = actor_class(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

        logprob, entropy = act.get_action_logprob(state)

        assert isinstance(logprob, Tensor)
        assert logprob.dtype in {torch.float}
        assert logprob.shape == (batch_size, action_dim)

        assert isinstance(entropy, Tensor)
        assert entropy.dtype in {torch.float}
        assert entropy.shape == (batch_size, 1)

    for actor_class in (ActorPPO, ActorDiscretePPO):
        print(f"  actor_class = {actor_class.__name__}")
        act = actor_class(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

        action = act(state)
        if actor_class in {ActorDiscretePPO}:
            action = action.unsqueeze(1)
        logprob, entropy = act.get_logprob_entropy(state, action)

        assert isinstance(logprob, Tensor)
        assert logprob.dtype in {torch.float}
        assert logprob.shape == (batch_size,)

        assert isinstance(entropy, Tensor)
        assert entropy.dtype in {torch.float}
        assert entropy.shape == (batch_size,)


def check_critic(state_dim=4, action_dim=2, batch_size=3, net_dims=(64, 32), gpu_id=0):
    print("\n| check_critic()")
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state = torch.rand(size=(batch_size, state_dim), dtype=torch.float32, device=device)
    action = torch.rand(size=(batch_size, action_dim), dtype=torch.float32, device=device)

    '''check Critic'''
    from elegantrl.agents.net import Critic
    cri = Critic(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

    value = cri(state=state, action=action)
    assert isinstance(value, Tensor)
    assert value.dtype in {torch.float}
    assert value.shape == (batch_size,)

    '''check CriticTwin'''
    from elegantrl.agents.net import CriticTwin
    cri = CriticTwin(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

    value = cri(state=state, action=action)
    assert isinstance(value, Tensor)
    assert value.dtype in {torch.float}
    assert value.shape == (batch_size,)

    value = cri.get_q_min(state=state, action=action)
    assert isinstance(value, Tensor)
    assert value.dtype in {torch.float}
    assert value.shape == (batch_size,)

    q1, q2 = cri.get_q1_q2(state=state, action=action)
    assert isinstance(q1, Tensor)
    assert isinstance(q2, Tensor)
    assert q1.dtype in {torch.float}
    assert q2.dtype in {torch.float}
    assert q1.shape == (batch_size,)
    assert q2.shape == (batch_size,)

    '''check CriticPPO'''
    from elegantrl.agents.net import CriticPPO
    cri = CriticPPO(dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(device)

    value = cri(state=state)
    assert isinstance(value, Tensor)
    assert value.dtype in {torch.float}
    assert value.shape == (batch_size,)


def check_build_mlp(net_dims: [int] = (64, 32)):
    print("\n| check_build_mlp()")
    from elegantrl.agents.net import build_mlp

    net = build_mlp(dims=net_dims)
    assert isinstance(net, nn.Sequential)
    assert len(net) == 1 == len(net_dims) * 2 - 3

    net_dims = (64, 32, 16)
    net = build_mlp(dims=net_dims)
    assert isinstance(net, nn.Sequential)
    assert len(net) == 3 == len(net_dims) * 2 - 3

    net_dims = (64, 32, 16, 8)
    net = build_mlp(dims=net_dims)
    assert isinstance(net, nn.Sequential)
    assert len(net) == 5 == len(net_dims) * 2 - 3


def check_cnn():
    print("\n| check_cnn()")
    from elegantrl.agents.net import ConvNet

    inp_dim = 3
    out_dim = 32
    batch_size = 5

    for image_size in (112, 224):
        print(f"  image_size={image_size}")
        conv_net = ConvNet(inp_dim=inp_dim, out_dim=out_dim, image_size=image_size)

        image = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
        output = conv_net(image)
        assert output.dtype in {torch.float}
        assert output.shape == (batch_size, out_dim)


if __name__ == '__main__':
    print('\n| check_net.py')
    check_net_base()

    check_q_net()
    check_actor()
    check_critic()

    check_build_mlp()
    check_cnn()
