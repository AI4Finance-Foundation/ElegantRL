from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
Conv2D dim 8, leaklyReLU
"""


def _layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            # return int(16 * (i + 1))  # todo
            return int(16 + i * 8)  # todo

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        ) if isinstance(state_dim, int) else nn.Sequential(
            NnnReshape(*state_dim),  # -> [batch_size, frame_num, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
            nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),
            nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        # layer_norm(self.net__a[0], std=1.0)
        # layer_norm(self.net__a[2], std=1.0)
        _layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, s):
        a_avg = self.net(s)
        return a_avg.tanh()

    def get__a_noise__noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        # a_noise = torch.normal(a_avg, a_std) # same as below
        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def compute__log_prob(self, state, a_noise):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        a_delta = ((a_avg - a_noise) / a_std).pow(2) / 2
        log_prob = -(a_delta + (self.a_std_log + self.sqrt_2pi_log))
        return log_prob.sum(1)


class CriticAdv(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            # return int(16 * (i + 1))  # todo
            return int(16 + i * 8)  # todo

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        ) if isinstance(state_dim, int) else nn.Sequential(
            NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
            nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),
            nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        # layer_norm(self.net[0], std=1.0)
        # layer_norm(self.net[2], std=1.0)
        _layer_norm(self.net[-1], std=1.0)  # output layer for action

    def forward(self, s):
        q = self.net(s)
        return q


class AgentPPO:
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

    def select_action(self, states):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        a_noise, noise = self.act.get__a_noise__noise(states)
        a_noise = a_noise.cpu().data.numpy()[0]
        noise = noise.cpu().data.numpy()[0]
        return a_noise, noise  # not tanh()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        step_counter = 0
        target_step = buffer.max_len - max_step
        while step_counter < target_step:
            state = env.reset()

            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                a_noise, noise = self.select_action((state,))

                next_state, reward, done, _ = env.step(np.tanh(a_noise))
                reward_sum += reward

                reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
                buffer.append_memo((reward_mask, state, a_noise, noise))

                if done:
                    break

                state = next_state

            # Compatibility for ElegantRL 2020-12-21
            episode_return = env.episode_return if hasattr(env, 'episode_return') else reward_sum
            rewards.append(episode_return)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        """Contribution of PPO (Proximal Policy Optimization
        1. the surrogate objective of TRPO, PPO simplified calculation of TRPO
        2. use the advantage function of A3C (Asynchronous Advantage Actor-Critic)
        3. add GAE. ICLR 2016. Generalization Advantage Estimate and use trajectory to calculate Q value
        """
        buffer.update__now_len__before_sample()

        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_obj = critic_obj = None  # just for print return

        '''the batch for training'''
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

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_obj) + (actor_obj)
            """

            """critic_obj"""
            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_obj
            new_value = self.cri(state)

            critic_obj = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)

            self.cri_optimizer.zero_grad()
            critic_obj.backward()
            self.cri_optimizer.step()

            """actor_obj"""
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_obj = surrogate_obj + loss_entropy * lambda_entropy

            self.act_optimizer.zero_grad()
            actor_obj.backward()
            self.act_optimizer.step()

        return actor_obj.item(), critic_obj.item()

    def save_or_load_model(self, cwd, if_save):  # 2020-07-07
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_act = 'act' in dir(self)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path) if has_act else None
            torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", cwd)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path) if has_act else None
            load_torch_file(self.cri, cri_save_path) if has_cri else None
            print("Loaded act and cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


def train__pixel_level_state2d__car_racing():
    # from AgentZoo import AgentPPO
    '''
    700, 20e5, 31550, 1577(per 1e5)
    '''

    '''DEMO 4: Fix gym Box2D env CarRacing-v0 (pixel-level 2D-state, continuous action) using PPO'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)
    env.target_reward = 900  # todo

    args = Arguments(rl_agent=AgentPPO, env=env, gpu_id=None)
    args.if_break_early = True
    args.eval_times2 = 1
    args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.
    args.rollout_workers_num = 8  # (num, step, time) (8, 1e5, 1360) (4, 1e4, 1860)

    args.break_step = int(2e6 * 8)  # (1e5) 2e5 4e5 (8e5) used time (7,000s) 10ks 30ks (60ks)
    # Sometimes bad luck (5%), it reach 300 score in 5e5 steps and don't increase.
    # You just need to change the random seed and retrain.
    args.reward_scale = 2 ** -2  # (-1) 50 ~ 700 ~ 900 (1001)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


if __name__ == '__main__':
    train__pixel_level_state2d__car_racing()
