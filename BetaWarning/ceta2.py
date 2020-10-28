from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""OffPPO
beta0 FixOffPPO LL
beta1 FixOffPPO BW
beta2 FixOffPPO BW
beta3 OffPPO Ministaur

ceta4 FixOffPPO Minitaur
"""


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__mean = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, action_dim), )
        self.net__std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net__mean[0], std=1.0)
        layer_norm(self.net__mean[2], std=1.0)
        layer_norm(self.net__mean[4], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net__mean(s)
        # return a_mean.tanh()
        return (a_mean * 1.01).tanh().clamp(-1, 1)  # todo fix tanh 1.01

    def get__a__log_prob(self, state):  # todo plan to cancel
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):  # todo may plan to cancel
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)


class AgentOffPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.act_optimizer = torch.optim.SGD(self.act.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.cri_optimizer = torch.optim.SGD(self.cri.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.criterion = nn.SmoothL1Loss()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        noise_std = np.exp(self.act.net__std_log.cpu().data.numpy()[0])
        # assert noise_std.shape = (action_dim, )

        step_counter = 0
        max_memo = buffer.max_len - max_step
        while step_counter < max_memo:
            reward_sum = 0
            step_sum = 0

            state = env.reset()
            for step_sum in range(max_step):
                states = torch.tensor((state,), dtype=torch.float32, device=self.device)
                a_mean = self.act.net__mean(states).cpu().data.numpy()[0]  # todo fix bug
                noise = rd.normal(scale=noise_std)
                action = a_mean + noise

                # next_state, reward, done, _ = env.step(np.tanh(action))
                next_state, reward, done, _ = env.step(np.tanh(action * 1.01).clip(-1, 1))  # todo fix tanh 1.01
                reward_sum += reward

                mask = 0.0 if done else gamma

                reward_ = reward * reward_scale
                buffer.append_memo((reward_, mask, state, action, noise))

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        self.act.train()
        self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_loss = critic_loss = None  # just for print

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = [
            torch.tensor(ary, device=self.device)
            for ary in (buffer.memories[:, 0:1],
                        buffer.memories[:, 1:2],
                        buffer.memories[:, 2:buffer.state_idx],
                        buffer.memories[:, buffer.state_idx:buffer.action_idx],
                        buffer.memories[:, buffer.action_idx:],)
        ]

        b_size = 2 ** 10
        with torch.no_grad():
            a_log_std = self.act.net__std_log

            all__new_v = torch.cat([
                self.cri(all_state[i:i + b_size])
                for i in range(0, all_state.size()[0], b_size)
            ], dim=0)
            # all_log_prob = torch.cat([  # todo fix action.tanh()
            #     -(all_noise[i:i + b_size].pow(2) + a_log_std +
            #       (-all_action[i:i + b_size].tanh().pow(2) + 1.000001).log()).sum(1)
            #     for i in range(0, all_state.size()[0], b_size)
            # ], dim=0)
            all_log_prob = torch.cat([  # todo fix action.tanh()   # todo fix tanh 1.01
                -(all_noise[i:i + b_size].pow(2) + a_log_std +
                  (-(all_action[i:i + b_size]*1.01).tanh().clamp(-1, 1).pow(2) + 1.000001).log()).sum(1)
                for i in range(0, all_state.size()[0], b_size)
            ], dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value
        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv
            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]
        # all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # todo cancel value_norm

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            action = all_action[indices]
            old_log_prob = all_log_prob[indices]

            '''critic_loss'''
            new_value = self.cri(state)

            # critic_loss = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)  # value_norm
            critic_loss = self.criterion(new_value, old_value)  # todo cancel value_norm
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            a_mean = self.act.net__mean(state)  # todo fix bug
            a_log_std = self.act.net__std_log.expand_as(a_mean)

            # todo fix action.tanh()
            # new_log_prob = -((a_mean - action).pow(2) + a_log_std
            #                  + (-action.tanh().pow(2) + 1.000001).log()).sum(1)
            new_log_prob = -((a_mean - action).pow(2) + a_log_std
                             + (-(action*1.01).tanh().clamp(-1, 1).pow(2) + 1.000001).log()).sum(1)
            # todo fix tanh 1.01

            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        self.cri.eval()
        # return actor_loss.item(), critic_loss.item()# todo
        return self.act.net__std_log.mean().item(), critic_loss.item()  # todo

    def save_or_load_model(self, cwd, if_save):  # 2020-05-20
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path)
            torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            load_torch_file(self.cri, cri_save_path) if has_cri else None
        else:
            print("FileNotFound when load_model: {}".format(cwd))


def run_continuous_action_off_ppo(gpu_id=None):
    rl_agent = AgentOffPPO
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 2
    args.eval_times1 = 2
    args.eval_times2 = 4

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3  # 4

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(1e5 * 8)
    # args.reward_scale = 2 ** -3
    # args.init_for_training()
    # train_agent_1020(**vars(args))
    # exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(3e6 * 4)
    args.reward_scale = 2 ** -1
    args.max_step = 2 ** 11
    args.max_memo = 2 ** 11
    args.init_for_training()
    train_agent_1020(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # todo
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_1020(**vars(args))
    exit()


run_continuous_action_off_ppo()
