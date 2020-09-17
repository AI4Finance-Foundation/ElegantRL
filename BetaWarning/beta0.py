from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
auto TTUR
"""


class AgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = InterSPG(state_dim, action_dim, actor_dim).to(self.device)
        self.act.train()

        self.cri = self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim + 1) * 0.5  # todo
        self.log_alpha = torch.tensor((-self.target_entropy * np.e,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate, betas=(0.5, 0.999))
        print('log_alpha:', self.log_alpha.item())  # todo init log_alpha

        '''extension: auto learning rate of actor'''
        self.trust_rho = TrustRho()

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        loss_a_list = list()
        loss_c_list = list()

        alpha = self.log_alpha.exp().detach()
        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        # log_prob = None  # todo print
        # rho = None
        a_update_n = 0

        for i in range(1, update_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # twin critic
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # policy entropy
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_list.append(loss_c_tmp)
            rho = self.trust_rho.update_rho(loss_c_tmp)

            '''stochastic policy'''
            a1_mean, a1_log_std, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

            '''action correction term'''
            a2_mean, a2_log_std = self.act_target.get__a__std(state)
            actor_term = self.criterion(a1_mean, a2_mean) + self.criterion(a1_log_std, a2_log_std)

            '''auto alpha'''
            alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)  # todo fix bug
            alpha = self.log_alpha.exp().detach()

            '''actor_loss'''
            if a_update_n / i < rho + 0.5:
                a_update_n += 1

                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise))  # policy gradient
                actor_loss = -(q_eval_pg + log_prob * alpha).mean()  # policy gradient
                loss_a_list.append(actor_loss.item())
            else:
                actor_loss = 0

            united_loss = critic_loss + actor_term * (1 - rho) + actor_loss * rho
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)

        # print(np.array(
        #     (alpha.item(), self.log_alpha.item(),
        #      log_prob.mean().item(), self.target_entropy,
        #      rho)
        # ).round(3))  # todo show alpha
        loss_a_avg = (sum(loss_a_list) / len(loss_a_list)) if len(loss_a_list) > 0 else 0.0
        loss_c_avg = sum(loss_c_list) / len(loss_c_list)
        return loss_a_avg, loss_c_avg


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=AgentInterSAC, gpu_id=gpu_id)
    args.show_gap = 2 ** 9
    args.eval_times2 = 2 ** 5
    args.if_stop = False  # todo

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.max_total_step = int(1e4 * 4)
    # args.reward_scale = 2 ** -2
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))
    # exit()
    #
    # args.env_name = "LunarLanderContinuous-v2"
    # args.max_total_step = int(1e5 * 4)
    # args.init_for_training()
    # build_for_mp(args)  # train_agent(**vars(args))
    # exit()

    args.env_name = "BipedalWalker-v3"
    args.random_seed = 1945
    args.max_total_step = int(2e5 * 4)
    args.init_for_training()
    build_for_mp(args)  # train_agent(**vars(args))
    # exit()
    #
    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.max_total_step = int(4e6 * 8)
    args.net_dim = int(2 ** 8)
    args.max_memo = int(2 ** 21)  # todo
    args.batch_size = int(2 ** 8)
    args.init_for_training()
    build_for_mp(args)  # train_agent(**vars(args))

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.max_total_step = int(5e5 * 4)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.max_step = 2 ** 10
    # args.batch_size = 2 ** 9
    # args.reward_scale = 2 ** -2
    # args.eva_size = 2 ** 3  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.max_total_step = int(1e6 * 2)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20  # todo
    # args.max_step = 2 ** 11  # todo 10
    # args.net_dim = 2 ** 8
    # args.batch_size = 2 ** 8
    # args.reward_scale = 2 ** 4
    # args.eval_times2 = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 9  # for Recorder
    # args.init_for_training(cpu_threads=4)
    # build_for_mp(args)  # train_agent(**vars(args))


run_continuous_action()
