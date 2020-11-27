from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
BW
beta1 SAC
ceta4 SAC BW
ceta1 SAC Reach

beta0 beta3 ModSAC a_lr cancel
beta2 beta2 ModSAC


ceta0 ModSAC 
"""


class AgentModSAC(AgentBaseAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        use_dn = True  # and use hard target update
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = ActorSAC(state_dim, action_dim, actor_dim, use_dn).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = ActorSAC(state_dim, action_dim, net_dim, use_dn).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        critic_dim = int(net_dim * 1.25)
        self.cri = CriticTwinShared(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwinShared(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

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

    def select_action(self, states):
        action = self.act.get__noise_action(states)
        return action

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        log_prob = None  # just for print

        alpha = self.log_alpha.exp().detach()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(max_step * k * repeat_times)

        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.cri_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # policy entropy
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_loss.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            actions_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient

            '''auto temperature parameter (alpha)'''
            alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
            alpha = self.log_alpha.exp().detach()

            '''actor_loss'''
            if update_a / update_c < 1 / (2 - lamb):  # auto TTUR
                update_a += 1

                q_eval_pg = torch.min(*self.cri.get__q1_q2(state, actions_noise))  # policy gradient, stable but slower

                actor_loss = -(q_eval_pg + log_prob * alpha).mean()  # policy gradient

                # self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * lamb
                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

                soft_target_update(self.act_target, self.act)
            soft_target_update(self.cri_target, self.cri)

        return log_prob.mean().item(), self.avg_loss_c


def test__train_agent():
    args = Arguments(gpu_id=None)
    args.rl_agent = AgentModSAC

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(5e5 * 8)  # (2e4) 5e5, used time 1500s
    # args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    args.env_name = "ReacherBulletEnv-v0"
    args.break_step = int(5e4 * 8)  # (4e4) 5e4
    args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29) # todo wait update
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


test__train_agent()
