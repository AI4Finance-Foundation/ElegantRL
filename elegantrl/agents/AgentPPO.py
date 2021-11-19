import torch
import numpy as np
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorPPO, ActorDiscretePPO, CriticPPO, SharePPO


class AgentPPO(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorPPO
        self.ClassCri = CriticPPO

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )
        self.traj_list = [list() for _ in range(env_num)]
        self.env_num = env_num

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, state: torch.Tensor) -> tuple:
        """
        `tensor state` state.shape = (batch_size, state_dim)
        return `tensor action` action.shape = (batch_size, action_dim)
        return `tensor noise` noise.shape = (batch_size, action_dim)
        """
        state = state.to(self.device)
        action, noise = self.act.get_action(state)
        return action.detach().cpu(), noise.detach().cpu()

    def explore_one_env(self, env, target_step):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_actions, ten_noises = self.select_actions(ten_states)
            action = ten_actions[0].numpy()
            next_s, reward, done, _ = env.step(np.tanh(action))

            traj.append((ten_states, reward, done, ten_actions, ten_noises))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory([traj, ], [last_done, ])
        return self.convert_trajectory(traj_list)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_actions, ten_noises = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_actions.tanh())

            for env_i in range(env_num):
                traj_list[env_i].append((ten_states[env_i], ten_rewards[env_i], ten_dones[env_i],
                                         ten_actions[env_i], ten_noises[env_i]))
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            '''PPO: Surrogate objective of Trust Region'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        ten_bool = torch.not_equal(ten_mask, 0).float()
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_bool[i] * (pre_adv_v - ten_value[i])
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, traj_list, last_done_list):
        for env_i in range(self.env_num):
            last_done = last_done_list[env_i]
            traj_temp = traj_list[env_i]

            traj_list[env_i] = self.traj_list[env_i] + traj_temp[:last_done + 1]
            self.traj_list[env_i] = traj_temp[last_done:]
        return traj_list

    def convert_trajectory(self, traj_list):
        for traj in traj_list:
            temp = list(map(list, zip(*traj)))  # 2D-list transpose

            ten_state = torch.stack(temp[0])
            ten_reward = torch.as_tensor(temp[1], dtype=torch.float32) * self.reward_scale
            ten_mask = (1.0 - torch.as_tensor(temp[2], dtype=torch.float32)) * self.gamma
            ten_action = torch.stack(temp[3])
            ten_noise = torch.stack(temp[4])

            traj[:] = (ten_state, ten_reward, ten_mask, ten_action, ten_noise)
        return traj_list


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        AgentPPO.__init__(self)
        self.ClassAct = ActorDiscretePPO

    def explore_one_env(self, env, target_step):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            a_int = ten_a_ints[0].numpy()
            next_s, reward, done, _ = env.step(a_int)  # only different

            traj.append((ten_states, reward, done, ten_a_ints, ten_probs))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory([traj, ], [last_done, ])
        return self.convert_trajectory(traj_list)

    def explore_vec_env(self, env, target_step):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_a_ints.numpy())

            for env_i in range(env_num):
                traj_list[env_i].append((ten_states[env_i], ten_rewards[env_i], ten_dones[env_i],
                                         ten_a_ints[env_i], ten_probs[env_i]))
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]


class AgentSharePPO(AgentPPO):
    def __init__(self):
        AgentPPO.__init__(self)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if if_per_or_gae:
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.act = self.cri = SharePPO(state_dim, action_dim, net_dim).to(self.device)

        self.cri_optim = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': learning_rate * 0.9},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / torch.std(buf_adv_v) + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]  # advantage value
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            '''PPO: Surrogate objective of Trust Region'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            obj_united = obj_critic + obj_actor
            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple
