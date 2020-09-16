from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
soft target update for GAE
"""


def train_agent(
        rl_agent, net_dim, batch_size, repeat_times, gamma, reward_scale, cwd,
        env_name, max_step, max_memo, max_total_step,
        eva_size, gpu_id, show_gap, **_kwargs):  # 2020-06-01
    env, state_dim, action_dim, max_action, target_reward, is_discrete = build_gym_env(env_name, is_print=False)

    '''init: agent, buffer, recorder'''
    recorder = Recorder()
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    is_online_policy = bool(rl_agent.__name__ in {'AgentPPO', 'AgentGAE', 'AgentInterGAE', 'AgentDiscreteGAE'})
    if is_online_policy:
        buffer = BufferTupleOnline(max_memo)
    else:
        buffer = BufferArray(max_memo, state_dim, 1 if is_discrete else action_dim)
        with torch.no_grad():  # update replay buffer
            rewards, steps = initial_exploration(env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
        recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

    '''loop'''
    is_training = True
    while is_training:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # for saving the GPU buffer
            rewards, steps = agent.update_buffer(
                env, buffer, max_step, max_action, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.init_before_sample()
        loss_a, loss_c = agent.update_parameters(
            buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, loss_a, loss_c)

            is_saved = recorder.update__record_evaluate(
                env, agent.act, max_step, max_action, eva_size, agent.device, is_discrete)
            recorder.save_act(cwd, agent.act, gpu_id) if is_saved else None

            is_solved = recorder.check_is_solved(target_reward, gpu_id, show_gap)
        '''break loop rules'''
        # if is_solved or recorder.total_step > max_total_step or os.path.exists(f'{cwd}/stop.mark'):
        if recorder.total_step > max_total_step or os.path.exists(f'{cwd}/stop.mark'):
            is_training = False

    recorder.save_npy__plot_png(cwd)


class AgentInterGAE(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 1e-4  # learning rate of actor todo better than 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterGAE(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.cri = self.act.get__q1_q2
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.act_target = InterGAE(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())
        self.cri_target = self.act_target.get__q1_q2

        self.criterion = nn.SmoothL1Loss()

    def update_parameters(self, buffer, _max_step, batch_size, repeat_times):
        self.act.train()
        # self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot seem to use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**2 ~ 2**4

        loss_a_sum = 0.0  # just for print
        loss_c_sum = 0.0  # just for print

        '''the batch for training'''
        max_memo = len(buffer)
        all_batch = buffer.sample()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]
        # with torch.no_grad():
        # all__new_v = self.cri(all_state).detach_()  # all new value
        # all__new_v = torch.min(*self.cri(all_state)).detach_()  # TwinCritic
        with torch.no_grad():
            b_size = 128
            all__new_v = torch.cat([torch.min(*self.cri_target(all_state[i:i + b_size]))
                                    for i in range(0, all_state.size()[0], b_size)], dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: Generalization Advantage Estimate. ICLR 2016. 
        https://arxiv.org/pdf/1506.02438.pdf
        '''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # delta of q value
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

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-6)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            # indices = rd.choice(max_memo, batch_size, replace=True)  # False)
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_loss) + (actor_loss)
            """

            '''critic_loss'''
            new_log_prob = self.act.compute__log_prob(state, action)
            new_value1, new_value2 = self.cri(state)  # TwinCritic
            # new_log_prob, new_value1, new_value2 = self.act_target.compute__log_prob(state, action)

            critic_loss = (self.criterion(new_value1, old_value) +
                           self.criterion(new_value2, old_value)) / (old_value.std() * 2 + 1e-6)
            loss_c_sum += critic_loss.item()  # just for print
            # self.cri_optimizer.zero_grad()
            # critic_loss.backward()
            # self.cri_optimizer.step()

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = (new_log_prob - old_log_prob).exp()
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (new_log_prob.exp() * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            loss_a_sum += actor_loss.item()  # just for print

            united_loss = actor_loss + critic_loss
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)

        loss_a_avg = loss_a_sum / sample_times
        loss_c_avg = loss_c_sum / sample_times
        return loss_a_avg, loss_c_avg


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=AgentInterGAE, gpu_id=gpu_id)
    # assert args.rl_agent in {Zoo.AgentPPO, Zoo.AgentGAE, Zoo.AgentInterGAE}
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4

    # args.env_name = "BipedalWalker-v3"
    # args.max_total_step = int(3e6 * 4)
    # args.init_for_training()
    # train_agent(**vars(args))

    args.env_name = "MultiWalker"
    args.max_total_step = int(3e6 * 4)
    args.init_for_training()
    train_agent(**vars(args))
    exit()


run_continuous_action()
