from AgentRun import *
from AgentZoo import *
from AgentNet import *

import torch.nn.functional as nn_f
from torch.distributions import Categorical


class QNetTwinShared(nn.Module):  # 2020-06-18
    def __init__(self, state_dim, action_dim, mid_dim, use_dn, use_sn):  # todo
        super(QNetTwinShared, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 DenseNet(mid_dim), )
        self.net1 = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim))
        self.net2 = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim))

    def forward(self, state):
        # x = torch.cat((state, action), dim=1)
        x = self.net(state)
        q_value = self.net1(x)
        return q_value

    def get__q1_q2(self, state):
        # x = torch.cat((state, action), dim=1)
        x = self.net(state)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class AgentDiscreteSAC(AgentBasicAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        use_dn = True  # and use hard target update
        use_sn = True  # and use hard target update
        self.learning_rate = 2e-4
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
        self.cri = QNetTwinShared(state_dim, action_dim, critic_dim, use_dn, use_sn).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = QNetTwinShared(state_dim, action_dim, critic_dim, use_dn, use_sn).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

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
        self.target_entropy = -1  # why a negative number? Due to Gauss Dist. -> Tanh()?
        '''extension: auto learning rate of actor'''
        self.loss_c_sum = 0.0
        self.rho = 0.5

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update

    def select_actions(self, states, explore_noise=0.0):  # 2020-07-07
        states = torch.tensor(states, dtype=torch.float32, device=self.device)  # state.size == (1, state_dim)
        actions = self.act(states, explore_noise)  # discrete action space

        if explore_noise == 0.0:  # should be more elegant
            a_ints = actions.argmax(dim=1)
        else:
            a_prob = nn_f.softmax(actions, dim=1)
            a_dist = Categorical(a_prob)
            a_ints = a_dist.sample()
        return a_ints.cpu().data.numpy()

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        update_freq = self.update_freq * repeat_times  # delay update frequency, for soft target update
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, state, a_int, next_s = buffer.random_sample(batch_size_, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_a_prob = nn_f.softmax(next_a_noise, dim=1) # SAC discrete
                next_q_target = torch.min(*self.cri_target.get__q1_q2(next_s))  # CriticTwin
                next_q_target = (next_q_target * next_a_prob).sum(dim=1, keepdim=True)  # SAC discrete
                next_q_target = next_q_target - next_log_prob * self.alpha  # auto-alpha

                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state)  # CriticTwin
            q1_value = q1_value.gather(1, a_int.long())  # SAC discrete
            q2_value = q2_value.gather(1, a_int.long())  # SAC discrete

            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_sum += critic_loss.item() * 0.5  # CriticTwin

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            if i % repeat_times == 0 and self.rho > 0.001:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                # stochastic policy
                a_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient
                a_prob = nn_f.softmax(a_noise, dim=1) # SAC discrete

                # auto alpha
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # policy gradient
                self.alpha = self.log_alpha.exp()
                # q_eval_pg = self.cri(state, actions_noise)  # policy gradient
                q_eval_pg = torch.min(*self.cri.get__q1_q2(state))  # policy gradient, stable but slower
                q_eval_pg = (q_eval_pg * a_prob).sum(dim=1, keepdim=True)  # SAC discrete

                actor_loss = (-q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

            """target update"""
            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0
                # self.soft_target_update(self.act_target, self.act)  # soft target update
                # self.soft_target_update(self.cri_target, self.cri)  # soft target update
                self.act_target.load_state_dict(self.act.state_dict())  # hard target update
                self.cri_target.load_state_dict(self.cri.state_dict())  # hard target update

                rho = np.exp(-(self.loss_c_sum / update_freq) ** 2)
                self.rho = (self.rho + rho) * 0.5
                self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * self.rho
                self.loss_c_sum = 0.0

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg


def run__zoo(gpu_id, cwd='AC_Zoo'):
    args = Arguments(AgentDiscreteSAC)
    args.gpu_id = gpu_id

    args.env_name = "LunarLander-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent_discrete(**vars(args))


if __name__ == '__main__':
    # run__zoo(gpu_id=0, cwd='DiscreteSAC')
    run__multi_process(run__zoo, gpu_tuple=((0, 1), (2, 3))[0], cwd='DiscreteSAC')
