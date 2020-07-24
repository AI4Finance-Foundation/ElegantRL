from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
beta2 ArgumentsBeta
beta2     tau=5e-3 * (0.5 + rho)
beta2     args.max_memo = 2 ** 19  # todo    args.batch_size = 2 ** 8  # todo
"""


class AgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = InterSPG(state_dim, action_dim, actor_dim).to(self.device)
        self.act.train()

        # critic_dim = int(net_dim * 1.25)
        # self.cri = ActorCriticSPG(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        # self.cri.train()
        self.cri = self.act

        # self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        # self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        para_list = list(self.act.parameters())  # + list(self.cri.parameters())
        self.act_optimizer = torch.optim.Adam(para_list, lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        # self.cri_target = ActorCriticSPG(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        # self.cri_target.eval()
        # self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = np.log(1.0 / action_dim) * 0.98
        '''extension: auto learning rate of actor'''
        self.trust_rho = TrustRho()

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        update_freq = self.update_freq * repeat_times  # delay update frequency, for soft target update
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0
        rho = self.trust_rho()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_ + 1, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                next_q_target = next_q_target - next_log_prob * self.alpha  # SAC, alpha
                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_sum += loss_c_tmp
            self.trust_rho.append_loss_c(loss_c_tmp)

            '''actor correction term'''
            a_mean2, a_std2 = self.act_target.get__a__std(state)

            '''actor_loss'''
            if i % repeat_times == 0 and rho > 0.001:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                '''stochastic policy'''
                a_mean1, a_std1, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

                '''auto alpha'''
                alpha_loss = -(self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                '''policy gradient'''
                self.alpha = self.log_alpha.exp()
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise))

                actor_loss = (-q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                actor_term = self.criterion(a_mean1, a_mean2) + self.criterion(a_std1, a_std2)
                united_loss = critic_loss + actor_term * (1 - rho) + actor_loss * (rho * 0.5)
            else:
                a_mean1, a_std1 = self.act.get__a__std(state)

                actor_term = self.criterion(a_mean1, a_mean2) + self.criterion(a_std1, a_std2)
                united_loss = critic_loss + actor_term * (1 - rho)

            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            """target update"""
            soft_target_update(self.act_target, self.act, tau=5e-3 * (0.5 + rho))  # todo  # soft target update

            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0
                # self.act_target.load_state_dict(self.act.state_dict())  # hard target update
                rho = self.trust_rho.update_rho()

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg


def run__mp(gpu_id=None, cwd='MP__InterSAC'):
    import multiprocessing as mp
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O

    def build_for_mp():
        process = [mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
                   mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),
                   mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
        [p.start() for p in process]
        [p.join() for p in process]
        # [p.close() for p in process]
        [p.terminate() for p in process]  # use p.terminate() instead of p.close()
        time.sleep(8)

    import AgentZoo as Zoo
    class_agent = Zoo.AgentDeepSAC

    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="LunarLanderContinuous-v2")
    # build_for_mp()
    #
    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="BipedalWalker-v3")
    # build_for_mp()

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="AntBulletEnv-v0")
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.max_step = 2 ** 10
    # args.net_dim = 2 ** 8
    # args.batch_size = 2 ** 9
    # args.reward_scale = 2 ** -2
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # build_for_mp()
    #

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="MinitaurBulletEnv-v0")
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 19  # todo
    args.batch_size = 2 ** 8  # todo
    args.max_step = 2 ** 10
    args.reward_scale = 2 ** 4
    args.is_remove = True
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    build_for_mp()


if __name__ == '__main__':
    run__mp()
