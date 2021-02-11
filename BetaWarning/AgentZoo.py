import os
import numpy as np
import torch
import torch.nn as nn
from AgentNet import ActorSAC, CriticTwin
from AgentNet import ActorPPO, CriticAdv

"""plan to 
- move old comment to here
- move DQN and etc, DDPG, TD3 etc from ElegantRLFull.py to here
- add Model-based RL: MBPO MuZero
"""

class AgentBase:  # DDPG-style
    def __init__(self, ):
        self.state = self.action = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = None
        self.cri = None
        self.criterion = None
        self.optimizer = None

        self.obj_a = 0.0
        self.obj_c = (-np.log(0.5)) ** 0.5

    def select_actions(self, states):  # states = (state, )
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        return actions.detach().cpu().numpy()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_state, reward, done, _ = env.step(action)
            buffer.append_memo((reward * reward_scale, 0.0 if done else gamma, *self.state, *action, *next_state))
            self.state = env.reset() if done else next_state
        return max_step

    def save_or_load_model(self, cwd, if_save):
        for net, name in ((self.act, 'act'), (self.cri, 'cri')):
            save_path = f'{cwd}/{name}.pth'
            if if_save:
                torch.save(net.state_dict(), save_path)
            elif os.path.exists(save_path):
                net = torch.load(save_path, map_location=lambda storage, loc: storage)
                net.load_state_dict(net)


class AgentModSAC(AgentBase):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBase.__init__(self)
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), requires_grad=True,
                                      dtype=torch.float32, device=self.device)

        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_target = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.parameters(), 'lr': learning_rate},
            {'params': self.cri.parameters(), 'lr': learning_rate},
            {'params': (self.alpha_log,), 'lr': learning_rate},
        ], lr=learning_rate)

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get__a_noisy(states)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(max_step * k * repeat_times)

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                q_label = reward + mask * (torch.min(*self.cri_target(next_s, next_a_noise)) + next_log_prob * alpha)

            q1, q2 = self.cri(state, action)
            cri_obj = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.obj_c = 0.995 * self.obj_c + 0.0025 * cri_obj.item()

            a_noise_pg, log_prob = self.act.get__a__log_prob(state)  # policy gradient
            alpha_obj = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

            lamb = np.exp(-self.obj_c ** 2)
            if_update_a = update_a / update_c < 1 / (2 - lamb)
            if if_update_a:  # auto TTUR
                update_a += 1

                alpha = self.alpha_log.exp().detach()
                act_obj = -(torch.min(*self.cri_target(state, a_noise_pg)) + log_prob * alpha).mean()
                self.obj_a = 0.995 * self.obj_a + 0.005 * q_label.mean().item()

                united_obj = cri_obj + alpha_obj + act_obj
            else:
                united_obj = cri_obj + alpha_obj

            self.optimizer.zero_grad()
            united_obj.backward()
            self.optimizer.step()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act) if if_update_a else None


class AgentGaePPO(AgentBase):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBase.__init__(self)

        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.cri = CriticAdv(state_dim, net_dim).to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.parameters(), 'lr': learning_rate},
            {'params': self.cri.parameters(), 'lr': learning_rate},
        ], lr=learning_rate)

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)  # plan to detach() here
        a_noise, noise = self.act.get__a_noisy__noise(states)
        return a_noise.detach().cpu().numpy(), noise.detach().cpu().numpy()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        buffer.empty_memories__before_explore()

        step_counter = 0
        target_step = buffer.max_len - max_step
        while step_counter < target_step:
            state = env.reset()
            for _ in range(max_step):
                action, noise = self.select_actions((state,))
                action = action[0]
                noise = noise[0]

                next_state, reward, done, _ = env.step(np.tanh(action))
                step_counter += 1

                buffer.append_memo((reward * reward_scale, 0.0 if done else gamma, *state, *action, *noise))
                if done:
                    break
                state = next_state
        return step_counter

    def update_policy(self, buffer, _max_step, batch_size, repeat_times=8):
        buffer.update__now_len__before_sample()

        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02

        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample()

        all__new_v = list()
        all_log_prob = list()
        with torch.no_grad():
            b_size = 2 ** 10
            a_std_log__sqrt_2pi_log = self.act.a_std_log + self.act.sqrt_2pi_log
            for i in range(0, all_state.size()[0], b_size):
                new_v = self.cri(all_state[i:i + b_size])
                all__new_v.append(new_v)

                log_prob = -(all_noise[i:i + b_size].pow(2) / 2 + a_std_log__sqrt_2pi_log).sum(1)
                all_log_prob.append(log_prob)

            all__new_v = torch.cat(all__new_v, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):  # could be more elegant
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]
        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        actor_obj = critic_obj = None
        for _ in range(int(repeat_times * max_memo / batch_size)):
            indices = torch.randint(max_memo, size=(batch_size,), device=self.device)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_obj
            new_value = self.cri(state)
            critic_obj = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)

            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio  # surrogate objective of TRPO
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy
            actor_obj = surrogate_obj + loss_entropy * lambda_entropy

            united_obj = actor_obj + critic_obj
            self.optimizer.zero_grad()
            united_obj.backward()
            self.optimizer.step()

        self.obj_a = actor_obj.item()
        self.obj_c = critic_obj.item()


def soft_target_update(target, current, tau=5e-3):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
