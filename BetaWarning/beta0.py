from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""     BW HardCore
beta0   grad_clip 4.0 
beta1   grad_clip 4.0 lamb > 0.1, self.learning_rate = 2e-4

beta11  lamb anchor smoothL1
beta12  lamb anchor MSE
"""


class AgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = self.act

        self.act_anchor = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_anchor.eval()
        self.act_anchor.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim + 1) * 0.5
        self.log_alpha = torch.tensor((-self.target_entropy * np.e,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        lamb = None
        actor_loss = critic_loss = None

        alpha = self.log_alpha.exp().detach()  # auto temperature parameter

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size = int(batch_size * k)  # increase batch_size
        train_step = int(max_step * k)  # increase training_step
        update_a = 0
        for update_c in range(1, train_step):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # twin critic
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # # auto temperature parameter

            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = (self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)).mean()
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * loss_c_tmp  # soft update
            lamb = np.exp(-self.avg_loss_c ** 2)

            '''stochastic policy'''
            a1_mean, a1_log_std, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

            '''auto temperature parameter: alpha'''
            alpha_loss = (lamb * self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
            alpha = self.log_alpha.exp().detach()

            '''action correction term'''
            a2_mean, a2_log_std = self.act_anchor.get__a__std(state)
            actor_term = (self.criterion(a1_mean, a2_mean) + self.criterion(a1_log_std, a2_log_std)).mean()

            if update_a / update_c > 1 / (2 - lamb):
                united_loss = critic_loss + actor_term * (1 - lamb)
            else:
                update_a += 1  # auto TTUR
                '''actor_loss'''
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise))  # twin critics
                actor_loss = -(q_eval_pg + log_prob * alpha).mean()  # policy gradient

                united_loss = critic_loss + actor_term * (1 - lamb) + actor_loss * lamb

            self.act_optimizer.zero_grad()
            united_loss.backward()
            nn.utils.clip_grad_norm_(self.act.parameters(), 4.0)  # todo
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)
        if lamb > 0.1:  # todo
            self.act_anchor.load_state_dict(self.act.state_dict())
        return actor_loss.item(), critic_loss.item() / 2


def run_continuous_action(gpu_id=None):
    rl_agent = AgentInterSAC
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 4

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(5e4 * 16)
    args.reward_scale = 2 ** 0
    args.init_for_training(8)
    train_agent_mp(args)  # train_agent(**vars(args))

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)
    args.reward_scale = 2 ** 0
    args.init_for_training(8)
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.break_step = int(4e6 * 8)
    args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    args.max_memo = int(2 ** 21)
    args.batch_size = int(2 ** 8)
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training(8)
    train_agent_mp(args)  # train_offline_policy(**vars(args))
    exit()


run_continuous_action()
