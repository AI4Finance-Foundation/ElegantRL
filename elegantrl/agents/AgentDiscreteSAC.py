
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.net import DiscreteActSAC, DiscreteCriTwin

class DiscreteSAC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = DiscreteCriSAC
        self.ClassAct = DiscreteActSAC
        self.train_reward = []
        self.if_use_cri_target = True
        self.if_use_act_target = False
        self.trajectory_list = []
        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.train_iteraion = 0

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )

        #self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
        #                              requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_log = torch.zeros(1,dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)
        self.alpha = alpha = self.alpha_log.cpu().exp().item()
        self.trajectory_list = list()
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer
            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs], buf_action[i:i+bs,None]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        alpha = None
        for _ in range(int(buf_len * repeat_times / batch_size)):
            indices = torch.randint(buf_len - 1, size=(batch_size,), requires_grad=False, device=self.device)
            state = buf_state[indices]
            next_s = buf_state[indices+1]
            action = buf_action[indices]
            reward = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            mask = buf_mask[indices]
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                _,next_a, next_log_prob = self.act_target.get_action(next_s)  # stochastic policy
                next_q = (next_a * (torch.min(*self.cri_target.get_q1_q2(next_s)) - alpha * next_log_prob
        )).sum(dim=1)  # twin critics
            q_label = reward + mask * next_q
            q1, q2 = self.cri.get_q1_q2(state)
            q1 = q1.gather(1, action.unsqueeze(-1)).squeeze(-1)
            q2 = q2.gather(1, action.unsqueeze(-1)).squeeze(-1)
            obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            _,action_pg, log_prob = self.act.get_action(state)  # policy gradient
            obj_alpha = (alpha * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            obj_actor = (action_pg * (-(torch.min(*self.cri.get_q1_q2(state)) - log_prob * alpha.detach()))).sum(dim=1).mean()
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

            if self.train_iteraion%main_args.update_epoch == 0:
                self.explore_env_traj(self.env, 1000, 1000)
                H = 0.
                for k in range(K):
                    if len(self.traj_s[k])>main_args.num_sample:
                        index = np.random.choice(len(self.traj_s[k]),replace=False,size=main_args.num_sample)
                    else:
                        index = np.arange(len(self.traj_s[k]))
                    s = np.array(self.traj_s[k])[index]
                    a = np.array(self.traj_a[k])[index]
                    r = np.array(self.traj_r[k])[index]
                    x = np.arange(len(index))
                    s_vec = torch.from_numpy(np.array(s).reshape(-1,self.state_dim)).to(self.device)
                    a_vec = torch.from_numpy(np.array(a).reshape(-1)).to(self.device).long()
                    r = torch.from_numpy(r).to(self.device)
                    p = self.act.soft_max(self.act.net(s_vec))[np.arange(len(a_vec)), a_vec].reshape(-1, k+1)
                    traj_p = torch.prod(p, dim=1)
                    traj_r = torch.sum(r, dim=1)
                    # print('here',r.shape,traj_r.shape)
                    H -= torch.sum(traj_p * traj_r)
                self.optim_update(self.act_optim_H, H)

            if self.train_iteraion%5==0:
                ep = 0.
                for eval_i in range(num_eval):
                    episode_reward, _, = get_episode_return_and_step(self.env, self.act, self.device)
                    ep += episode_reward 
                self.train_reward.append([ep/num_eval,self.train_iteraion])
                print(self.train_iteraion,ep/num_eval)
            self.train_iteraion += 1
            
        return obj_critic, obj_actor.item(), alpha.item()
    
    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
