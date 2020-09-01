from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
share params GAE
1. soft target network
2. integrated network 
"""


class InterGAE0(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
        )  # state

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)

        # layer_norm(self.net[0], std=1.0)
        layer_norm(self.dec_a, std=0.01)  # output layer for action
        layer_norm(self.dec_d, std=0.01)  # output layer for std_log

        '''temp'''
        self.enc_q = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
        )  # state
        self.net_q = DenseNet(mid_dim)
        net_out_dim = self.net_q.out_dim
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic

    def forward(self, s):
        x = self.net(s)
        a_mean = self.dec_a(x)
        return a_mean

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return a_noise, log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return log_prob

    def get__q1_q2(self, s):
        x = self.enc_q(s)
        x = self.net_q(x)
        q1 = self.dec_q1(x)
        q2 = self.dec_q2(x)
        return q1, q2


class InterGAE(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
        )  # state

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)

        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic

        # layer_norm(self.net[0], std=1.0)
        layer_norm(self.dec_a, std=0.01)  # output layer for action
        layer_norm(self.dec_d, std=0.01)  # output layer for std_log
        layer_norm(self.dec_q1, std=0.1)  # output layer for q value
        layer_norm(self.dec_q1, std=0.1)  # output layer for q value

    def forward(self, s):
        x = self.net(s)
        a_mean = self.dec_a(x)
        return a_mean

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return a_noise, log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return log_prob

    def get__q1_q2(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        q1 = self.dec_q1(x)
        q2 = self.dec_q2(x)
        return q1, q2


class AgentGAE(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 2e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterGAE(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.cri = self.act.get__q1_q2

        self.criterion = nn.SmoothL1Loss()

    def update_parameters(self, buffer, _max_step, batch_size, repeat_times):
        self.act.train()
        # self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot seem to use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**2 ~ 2**4

        loss_a_sum = 0.0  # just for print
        loss_c_sum = 0.0  # just for print

        '''the batch for training'''
        max_memo = len(buffer)
        all_batch = buffer.sample()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]
        # with torch.no_grad():
        # all__new_v = self.cri(all_state).detach_()  # all new value
        # all__new_v = torch.min(*self.cri(all_state)).detach_()  # TwinCritic
        # all__new_v = torch.add(*self.cri(all_state)).detach_() * 0.5  # TwinCritic # todo
        with torch.no_grad():  # todo avoid OOM
            b_size = 128
            b__len = all_state.size()[0]
            all__new_v = [torch.add(*self.cri(all_state[i:i + b_size])) * 0.5
                          for i in range(0, b__len - 1, b_size)]
            all__new_v = torch.cat(all__new_v, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: Generalization Advantage Estimate. ICLR 2016. 
        https://arxiv.org/pdf/1506.02438.pdf
        '''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # delta of q value
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-6)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            # indices = rd.choice(max_memo, batch_size, replace=True)  # False)
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_loss) + (actor_loss)
            """

            '''critic_loss'''
            new_log_prob = self.act.compute__log_prob(state, action)
            new_value1, new_value2 = self.cri(state)  # TwinCritic
            # new_log_prob, new_value1, new_value2 = self.act_target.compute__log_prob(state, action)

            critic_loss = (self.criterion(new_value1, old_value) +
                           self.criterion(new_value2, old_value)) / (old_value.std() * 2 + 1e-6)
            loss_c_sum += critic_loss.item() * 0.5  # just for print
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = (new_log_prob - old_log_prob).exp()
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (new_log_prob.exp() * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            loss_a_sum += actor_loss.item()  # just for print
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        loss_a_avg = loss_a_sum / sample_times
        loss_c_avg = loss_c_sum / sample_times
        return loss_a_avg, loss_c_avg


def run_continuous_action(gpu_id=None):
    import AgentZoo as Zoo

    """online policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentGAE, gpu_id=gpu_id)
    assert args.rl_agent in {Zoo.AgentPPO, Zoo.AgentGAE}
    """PPO and GAE is online policy.
    The memory in replay buffer will only be saved for one episode.

    TRPO's author use a surrogate object to simplify the KL penalty and get PPO.
    So I provide PPO instead of TRPO here.

    GAE is Generalization Advantage Estimate.
    RL algorithm that use advantage function (such as A2C, PPO, SAC) can use this technique.
    AgentGAE is a PPO using GAE and output log_std of action by an actor network.
    """

    args.max_memo = 2 ** 12
    args.repeat_times = 2 ** 4
    args.batch_size = 2 ** 9
    args.net_dim = 2 ** 8

    args.env_name = "LunarLanderContinuous-v2"
    args.max_total_step = int(4e5 * 4)
    args.init_for_training()
    train_agent(**vars(args))

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    args.max_total_step = int(1e5 * 4)
    args.max_memo = 2 ** 11
    args.repeat_times = 2 ** 3
    args.batch_size = 2 ** 8
    args.net_dim = 2 ** 7
    args.reward_scale = 2 ** -1
    args.init_for_training()
    train_agent(**vars(args))
    exit()


if __name__ == '__main__':
    run_continuous_action()
