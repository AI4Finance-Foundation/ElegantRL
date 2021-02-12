import torch
import torch.nn as nn


class QNetBase(nn.Module):  # There is not need to use ActorBase()
    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, state):
        return self.net(state)  # q value


class QNet(QNetBase):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(QNetBase, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )


class QNetTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # q2 value

    def forward(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp)  # q1 value

    def get__q1_q2(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # q1 q2 value


class QNetDuelTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )

        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1), )
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1), )
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim), )
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # single dueling q value

    def get__q1_q2(self, state):
        tmp = self.net__state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2


class ActorBase(nn.Module):  # There is not need to use ActorBase()
    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, state):
        return self.net(state).tanh()  # action


class Actor(ActorBase):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(ActorBase, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorPPO(ActorBase):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(ActorBase, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = 0.9189385332046727  # =np.log(np.sqrt(2 * np.pi))

    def get__action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def compute__log_prob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return log_prob.sum(1)


class ActorSAC(ActorBase):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActorBase, self).__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_action = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )
        self.net__a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )
        self.sqrt_2pi_log = 0.9189385332046727  # =np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        tmp = self.net__state(state)
        return self.net_action(tmp).tanh()  # action

    def get__action(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net_action(t_tmp)
        a_std = self.net__a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get__action__log_prob(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net_action(t_tmp)
        a_std_log = self.net__a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)
        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
        return a_tan, log_prob.sum(1, keepdim=True)


class CriticBase(nn.Module):  # There is not need to use ActorBase()
    def __init__(self):
        super().__init__()
        self.net = None


class Critic(CriticBase):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(CriticBase, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticAdv(CriticBase):
    def __init__(self, state_dim, mid_dim):
        super(CriticBase, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

    def forward(self, state):
        return self.net(state)  # q value


class CriticTwin(CriticBase):  # shared parameter
    def __init__(self, state_dim, action_dim, mid_dim):
        super(CriticBase, self).__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)

    def get__q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # q1, q2 value
