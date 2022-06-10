from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import util
import copy



class ActorVMPO(nn.Module):
    def __init__(self, action_dim, mid_dim, device, shared_net):
        super(ActorVMPO, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.shared_net = shared_net
        self.nn_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, self.action_dim))
        self.action_log_std = nn.Parameter(torch.zeros(action_dim, device=device))  # scale_tril std will bring to much extra bug

    def forward(self, states):
        states = states.squeeze()  
        tmp = self.shared_net(states)
        mean = self.nn_avg(tmp)
        return mean

    def get_mean(self, states):
        states = states.squeeze()  
        tmp = self.shared_net(states)
        mean = self.nn_avg(tmp)
        return mean

    def get_cov(self):
        action_std = self.action_log_std.clamp(min=-20., max=2.).exp()
        cov = torch.diag_embed(action_std)
        return cov  # shape = (action_dim, action_dim)
    
    # Action in VMPO should obey multivariate gaussian distribution.
    
    def get_action_4_explorer(self, states):  
        return torch.distributions.MultivariateNormal(self.get_mean(states), self.get_cov()).sample() # shape: (batch_size, action_dim)

    def entropy(self, action_mean, action_cov):  # get entropy of  certain dist
        return torch.distributions.MultivariateNormal(action_mean, action_cov).entropy()  # shape: (batch_size,)

    def log_prob(self, action_mean, action_cov, actions):
        return torch.distributions.MultivariateNormal(action_mean, action_cov).log_prob(actions)  # shape: (batch_size, )



class Critic(nn.Module):  
    def __init__(self,  mid_dim, shared_net):
        super(Critic, self).__init__()
        self.shared_net = shared_net
        self.net = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1)
        )

    def forward(self, states):
        states = states.squeeze()  
        return self.net(self.shared_net(states))


class AgentVMPO():
    def __init__(self, state_dim, action_dim, mid_dim, device,
                 epsilon_of_eta,
                 epsilon_of_alpha_mean_floor=0.005,
                 epsilon_of_alpha_mean_ceil=1,
                 epsilon_of_alpha_cov_floor=5e-6,
                 epsilon_of_alpha_cov_ceil=5e-5,
                 entropy_coef=5e-3, lr=1e-4, seq_len=1, gamma=0.99, lambda_gae=.98, use_topk=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.entropy_coef = entropy_coef
        self.shared_net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim), nn.ReLU()).to(device)
        self.actor = ActorVMPO(action_dim, mid_dim, device, self.shared_net).to(device)  # pi
        self.critic = Critic(mid_dim, self.shared_net).to(device)  # phi

        self.old_actor = None
        self.criterion = nn.SmoothL1Loss()
        self.epsilon_of_eta = epsilon_of_eta
        self.eta = nn.Parameter(torch.ones((1, ), device=device))  # eta temperature
        self.alpha_mean = nn.Parameter(torch.ones((1, ), device=device))
        self.alpha_cov = nn.Parameter(torch.ones((1, ), device=device))

        self.epsilon_of_alpha_mean_log_floor = np.log(epsilon_of_alpha_mean_floor)
        self.epsilon_of_alpha_mean_log_ceil = np.log(epsilon_of_alpha_mean_ceil)
        self.epsilon_of_alpha_cov_log_floor = np.log(epsilon_of_alpha_cov_floor)
        self.epsilon_of_alpha_cov_log_ceil = np.log(epsilon_of_alpha_cov_ceil)
        self.seq_len = seq_len
        self.lr = lr
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.use_topk = use_topk
        self.optim = torch.optim.Adam([
            {'params': self.eta, 'lr': self.lr},
            {'params': self.alpha_mean, 'lr': self.lr},
            {'params': self.alpha_cov, 'lr': self.lr},
            {'params': self.shared_net.parameters(), 'lr': self.lr},
            {'params': self.actor.nn_avg.parameters(), 'lr': self.lr},
            {'params': self.actor.action_log_std, 'lr': self.lr},
            {'params': self.critic.net.parameters(), 'lr': self.lr},
        ])
        self.calc_pi_loss = self.calc_pi_loss_use_topk if self.use_topk else self.calc_pi_loss_not_use_topk
        self.calc_eta_loss = self.calc_eta_loss_use_topk if self.use_topk else self.calc_eta_loss_not_use_topk
        self.re_calc_times1 = 1
        self.re_calc_times2 = 1
        self.re_calc_times_4_loss = 1

    def select_action(self, states):
        states = states.squeeze()  
        return self.actor.get_action_4_explorer(states)

    @staticmethod
    def gen_random_from_log_uniform(log_floor, log_ceil):
        return torch.distributions.Uniform(log_floor, log_ceil).sample().exp()

    def calc_pi_loss_not_use_topk(self, action_mean, action_cov, actions, advs_detached):
        
        # Clamp eta into the range [1e-8, +infty). (Lagrangian multipliers are always positive)
        eta_detached = self.eta.detach().clamp_min(1e-8)
        
        psi_detached = nn.functional.softmax(advs_detached / eta_detached, dim=0).squeeze_() 
        log_prob = self.actor.log_prob(action_mean, action_cov, actions) 
        loss = -(psi_detached * log_prob).sum()
        return loss

    def calc_pi_loss_use_topk(self, action_mean, action_cov, actions, advs_detached):
        
        # Clamp eta into the range [1e-8, +infty). (Lagrangian multipliers are always positive)
        eta_detached = self.eta.detach().clamp_min(1e-8)
        
        # Select top-k advantages.
        advs_topk_detached, idx_topk = torch.topk(advs_detached, advs_detached.numel() // 2, dim=0, sorted=False)
        
        psi_detached = nn.functional.softmax(advs_topk_detached / eta_detached, dim=0).squeeze_()  # bs//2
        log_prob = self.actor.log_prob(action_mean, action_cov, actions)[idx_topk]  # bs//2
        loss = -(psi_detached * log_prob).sum()
        return loss  # shape:[]

    def calc_eta_loss_not_use_topk(self, advs_detached):  # calc η temperature
        
        # Clamp eta into the range [1e-8, +infty). (Lagrangian multipliers eta are always positive)
        eta_clamp = self.eta.clamp_min(1e-8)
        
        D = advs_detached.numel()
        loss = eta_clamp * (self.epsilon_of_eta + (np.log(1 / D) + torch.logsumexp(advs_detached.squeeze() / eta_clamp, dim=0)))
        return loss

    def calc_eta_loss_use_topk(self, advs_detached):  # calc η temperature
        
        # Clamp eta into the range [1e-8, +infty). (Lagrangian multipliers are always positive)
        eta_clamp = self.eta.clamp_min(1e-8)
        
        # Select top-k advantages.
        advs_topk = torch.topk(advs_detached, advs_detached.shape[0] // 2, dim=0, sorted=False).values
        
        D_tilde = advs_detached.numel() // 2
        loss = eta_clamp * (self.epsilon_of_eta + (np.log(1 / D_tilde) + torch.logsumexp(advs_topk.squeeze() / eta_clamp, dim=0)))
        return loss 

    def calc_alpha_loss(self, old_mean_detached, old_cov_detached, new_mean, new_cov):
        # action_mean_of_old_pi_detached: old_mean_detached       bs,action_dim
        # action_std_of_old_pi_detached:  old_cov_detached        action_dim,action_dim
        # action_mean_of_new_pi:          new_mean                bs,action_dim
        # action_std_of_new_pi:           new_cov                 action_dim,action_dim

        # kl_mean
        inverse_cov_of_old_pi = old_cov_detached.inverse().unsqueeze_(0)  # 1, action_dim, action_dim
        tmp = (new_mean - old_mean_detached).unsqueeze_(-1)  # bs, action_dim, 1
        kl_mean = (0.5 * tmp.transpose(1, 2) @ inverse_cov_of_old_pi @ tmp).squeeze_()  # bs

        # kl_cov
        kl_cov = 0.5 * ((new_cov.inverse() @ old_cov_detached).trace() - self.action_dim + (torch.det(new_cov) / (torch.det(old_cov_detached) + 1e-6)).log())

        # # Clamps alpha into the range [ 1e-8, + infty ). (Lagrangian multipliers are always positive)
        alpha_m_clamp = self.alpha_mean.clamp_min(1e-8)
        alpha_c_clamp = self.alpha_cov.clamp_min(1e-8)

        # loss_of_kl_alpha_mean
        epsilon_of_alpha_mean = self.gen_random_from_log_uniform(self.epsilon_of_alpha_mean_log_floor, self.epsilon_of_alpha_mean_log_ceil)
        loss_of_kl_alpha_mean = alpha_m_clamp*(epsilon_of_alpha_mean-kl_mean.detach())+alpha_m_clamp.detach() * kl_mean

        # loss_of_kl_alpha_cov
        epsilon_of_alpha_cov = self.gen_random_from_log_uniform(self.epsilon_of_alpha_cov_log_floor, self.epsilon_of_alpha_cov_log_ceil)
        loss_of_kl_alpha_cov = alpha_c_clamp * (epsilon_of_alpha_cov - kl_cov.detach()) + alpha_c_clamp.detach() * kl_cov
        loss = loss_of_kl_alpha_mean.mean()+loss_of_kl_alpha_cov.mean()
        return loss

    def calc_critic_loss(self, v_predict, v_label_detached):  # mean v(phi) of old pi
        loss = self.criterion(v_predict, v_label_detached)
        return loss

    def calc_entropy_loss(self, action_mean, action_cov):  # author not mention it,add it 2 prevent premature
        loss = -self.entropy_coef * self.actor.entropy(action_mean, action_cov).mean()
        return loss

    @util.timeit()
    def update(self, buffer, repeat_times):
        self.old_actor = copy.deepcopy(self.actor)  # alias targ_actor need to fixed when updating
        buffer_size = buffer.buffer_size
        bs = buffer.bs
        with torch.no_grad():
            states, actions = buffer.get_whole_memo()[:2]
            indices = torch.arange(0, buffer_size, 1, device=self.device, dtype=torch.long)
            states = buffer.reform_to_seq_state_base_on_indice(indices)  # to seq_state

            # calc action_mean  for old actor
            while True:  # set a smaller 'bs: batch size' when out of GPU memory.
                try:
                    bs_ = buffer_size // self.re_calc_times1
                    action_mean_from_old_pi = torch.cat([self.old_actor.get_mean(states[i:i + bs_]) for i in range(0, buffer_size, bs_)], dim=0).detach()
                    break
                except:
                    self.re_calc_times1 *= 2
                    print(f're_calc_times1 = {self.re_calc_times1}')

            # calc vals & advs
            while True:  # set a smaller 'bs: batch size' when out of GPU memory.
                try:
                    bs_ = buffer_size // self.re_calc_times2
                    vals = torch.cat([self.critic(states[i:i + bs_]) for i in range(0, buffer_size, bs_)], dim=0).squeeze()  # bs
                    break
                except:
                    self.re_calc_times2 *= 2
                    print(f're_calc_times2={self.re_calc_times2}')
            advs = buffer.calc_gae(vals, self.gamma, self.lambda_gae, calc_rSum=False)  # !todo bs
            # advs = (advs - advs.mean()) / (advs.std() + 1e-7)#todo bs
            Gt = advs+vals
            advs = (advs - advs.mean()) / (advs.std() + 1e-7)  # todo bs
            action_cov_of_old_pi = self.old_actor.get_cov().detach()

        for _ in range(int(repeat_times * buffer_size / bs)):
            indices = torch.randint(buffer_size, size=(bs,), device=self.device)
            while True:
                try:
                    bs4gpuTrick = bs // self.re_calc_times_4_loss
                    for j in range(self.re_calc_times_4_loss):
                        idx = indices[bs4gpuTrick * j:bs4gpuTrick * (j + 1)]
                        state_minibatch = states[idx]  # detached
                        action_minibatch = actions[idx]  # detached
                        adv_minibatch = advs[idx]  # detached, shape: bs
                        old_pi_mean_minibatch = action_mean_from_old_pi[idx]  # detached
                        new_pi_mean_minibatch = self.actor.get_mean(state_minibatch)
                        new_pi_cov_minibatch = self.actor.get_cov()
                        v_predict_minibatch = self.critic(state_minibatch)
                        v_label_minibatch = Gt[idx].unsqueeze(-1)  # bs,1
                        pi_loss = self.calc_pi_loss(new_pi_mean_minibatch, new_pi_cov_minibatch, action_minibatch, adv_minibatch)
                        eta_loss = self.calc_eta_loss(adv_minibatch)
                        alpha_loss = self.calc_alpha_loss(old_pi_mean_minibatch, action_cov_of_old_pi, new_pi_mean_minibatch, new_pi_cov_minibatch)
                        critic_loss = self.calc_critic_loss(v_predict_minibatch, v_label_minibatch)
                        entropy_loss = self.calc_entropy_loss(new_pi_mean_minibatch, new_pi_cov_minibatch)  # prevent premature. (not mentioned in VMPO paper)
                        total_loss = pi_loss + eta_loss+alpha_loss + critic_loss + entropy_loss
                        if j == 0:
                            self.optim.zero_grad()
                        total_loss.backward()
                        if j + 1 == self.re_calc_times_4_loss:
                            self.optim.step()
                    break
                except Exception as e:
                    self.re_calc_times_4_loss *= 2
                    print(e)
                    exit()
                    print(f'self.re_calc_times_4_loss = {self.re_calc_times_4_loss}')

        return pi_loss.item(), critic_loss.item(), entropy_loss.item()
