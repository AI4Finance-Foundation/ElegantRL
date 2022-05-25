import torch
from copy import deepcopy
from elegantrl_helloworld.net import QNet, Actor, Critic, ActorPPO, CriticPPO


class AgentBase:
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.gamma = getattr(args, 'gamma', 0.99)
        self.num_layer = getattr(args, 'num_layer', 1)
        self.batch_size = getattr(args, 'batch_size', 128)
        self.repeat_times = getattr(args, 'repeat_times', 1.)
        self.reward_scale = getattr(args, 'reward_scale', 1.)
        self.learning_rate = getattr(args, 'learning_rate', 2 ** -14)
        self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)

        self.if_off_policy = args.if_off_policy
        self.if_act_target = args.if_act_target
        self.if_cri_target = args.if_cri_target

        self.states = None  # assert self.states == (1, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = act_class(net_dim, self.num_layer, state_dim, action_dim).to(self.device)
        self.cri = cri_class(net_dim, self.num_layer, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_target = deepcopy(self.act) if self.if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if self.if_cri_target else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        """attribute"""
        self.criterion = torch.nn.SmoothL1Loss()

    def convert_traj_to_transition(self, traj_list):
        traj_list = list(map(list, zip(*traj_list)))  # state, reward, done, action, noise

        '''stack items'''
        traj_list[0] = torch.stack(traj_list[0]).squeeze(1)
        traj_list[1] = (torch.tensor(traj_list[1], dtype=torch.float32) * self.reward_scale).unsqueeze(1)
        traj_list[2] = ((1 - torch.tensor(traj_list[2], dtype=torch.float32)) * self.gamma).unsqueeze(1)
        traj_list[3:] = [torch.stack(item).squeeze(1) for item in traj_list[3:]]
        return traj_list

    @staticmethod
    def optimizer_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentDQN(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        args.if_act_target = getattr(args, 'if_act_target', True)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.act.explore_rate = getattr(args, "explore_rate", 0.25)
        # the probability of choosing action randomly in epsilon-greedy

    def explore_env(self, env, target_step) -> list:
        traj_list = []
        state = self.states[0]

        i = 0
        done = False
        get_action = self.act.get_action
        while i < target_step or not done:
            tensor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            tensor_action = get_action(tensor_state.to(self.device)).detach().cpu()
            next_state, reward, done, _ = env.step(tensor_action[0, 0].numpy())

            traj_list.append((tensor_state, reward, done, tensor_action))

            i += 1
            state = env.reset() if done else next_state

        self.states[0] = state
        return self.convert_traj_to_transition(traj_list)

    def update_net(self, buffer) -> tuple:
        obj_critic = q_value = torch.zeros(1)

        update_times = int(1 + buffer.cur_capacity * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q
        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value


class AgentDDPG(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', Critic)
        args.if_act_target = getattr(args, 'if_act_target', True)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.act.explore_noise_std = getattr(args, 'explore_noise', 0.1)  # set for `self.act.get_action()`

    def explore_env(self, env, target_step: int) -> list:
        traj_list = []
        state = self.states[0]

        i = 0
        done = False
        get_action = self.act.get_action
        while i < target_step or not done:
            tenor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            tensor_action = get_action(tenor_state.to(self.device)).detach().cpu()
            next_state, reward, done, _ = env.step(tensor_action[0].numpy())

            traj_list.append((tenor_state, reward, done, tensor_action))

            i += 1
            state = env.reset() if done else next_state

        self.states[0] = state
        return self.convert_traj_to_transition(traj_list)

    def update_net(self, buffer) -> tuple:
        obj_critic = obj_actor = torch.zeros(1)

        update_times = int(1 + buffer.cur_capacity * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            action = self.act(state)
            obj_actor = self.cri_target(state, action).mean()
            self.optimizer_update(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    def get_obj_critic(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_state = buffer.sample_batch(batch_size)
            next_action = self.act_target(next_state)
            next_q = self.cri_target(next_state, next_action)
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        obj_critic = self.criterion(q, q_label)
        return obj_critic, state


class AgentPPO(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, "if_cri_target", False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, target_step: int) -> list:
        traj_list = []
        state = self.states[0]

        i = 0
        done = False
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        while i < target_step or not done:
            tensor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            tensor_action, tensor_noise = [ten.cpu() for ten in get_action(tensor_state.to(self.device))]
            next_state, reward, done, _ = env.step(convert(tensor_action)[0].numpy())

            traj_list.append((tensor_state, reward, done, tensor_action, tensor_noise))

            i += 1
            state = env.reset() if done else next_state

        self.states[0] = state
        return self.convert_traj_to_transition(traj_list)

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)  # buffer data of advantage value
            del buf_noise

        '''update network'''
        obj_critic = obj_actor = torch.zeros(1)
        assert buf_len >= self.batch_size
        update_times = int(1 + buf_len * self.repeat_times / self.batch_size)
        for _ in range(update_times):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate - obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum(self, buf_len, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v


'''replay buffer'''


class ReplayBuffer:  # for off-policy
    def __init__(self, max_capacity, state_dim, action_dim, gpu_id=0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_capacity = 0
        self.max_capacity = max_capacity

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.buf_action = torch.empty((max_capacity, action_dim), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)
        self.buf_mask = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)

        buf_state_size = (max_capacity, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

    def update_buffer(self, traj_list):
        traj_items = list(map(list, zip(*traj_list)))

        states, rewards, masks, actions = [torch.cat(item, dim=0) for item in traj_items]
        p = self.p + rewards.shape[0]  # pointer

        if p > self.max_capacity:
            self.buf_state[self.p:self.max_capacity] = states[:self.max_capacity - self.p]
            self.buf_reward[self.p:self.max_capacity] = rewards[:self.max_capacity - self.p]
            self.buf_mask[self.p:self.max_capacity] = masks[:self.max_capacity - self.p]
            self.buf_action[self.p:self.max_capacity] = actions[:self.max_capacity - self.p]
            self.if_full = True

            p = p - self.max_capacity
            self.buf_state[0:p] = states[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_mask[0:p] = masks[-p:]
            self.buf_action[0:p] = actions[-p:]
        else:
            self.buf_state[self.p:p] = states
            self.buf_reward[self.p:p] = rewards
            self.buf_mask[self.p:p] = masks
            self.buf_action[self.p:p] = actions
        self.p = p

        self.cur_capacity = self.max_capacity if self.if_full else self.p

        steps = rewards.shape[0]
        r_exp = rewards.mean().item()
        return steps, r_exp

    def sample_batch(self, batch_size) -> tuple:
        indices = torch.randint(self.cur_capacity - 1, size=(batch_size,), device=self.device)
        return (
            self.buf_reward[indices],
            self.buf_mask[indices],
            self.buf_action[indices],
            self.buf_state[indices],
            self.buf_state[indices + 1]  # next state
        )


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        traj_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in traj_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp
