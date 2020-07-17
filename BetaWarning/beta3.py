import multiprocessing as mp

from AgentZoo import *
from AgentRun import *
from AgentNet import *


class AgentDeepSAC(AgentBasicAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        use_dn = True  # and use hard target update
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
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        '''extension: auto learning rate of actor'''
        self.loss_c_sum = 0.0
        self.rho = 0.5

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update

    def update_params(self, buffer, max_step, batch_size, repeat_times):
        update_freq = self.update_freq * repeat_times  # delay update frequency, for soft target update
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.cri_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                next_q_target = next_q_target - next_log_prob * self.alpha  # SAC, alpha
                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_sum += critic_loss.item() * 0.5  # CriticTwin

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            if i % repeat_times == 0 and self.rho > 0.001:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                # stochastic policy
                actions_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient
                # auto alpha
                alpha_loss = -(self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # policy gradient
                self.alpha = self.log_alpha.exp()
                # q_eval_pg = self.cri(state, actions_noise)  # policy gradient
                q_eval_pg = torch.min(*self.cri.get__q1_q2(state, actions_noise))  # policy gradient, stable but slower

                actor_loss = (-q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

            """target update"""
            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0
                # soft_target_update(self.act_target, self.act)  # soft target update
                # soft_target_update(self.cri_target, self.cri)  # soft target update
                self.act_target.load_state_dict(self.act.state_dict())  # hard target update
                self.cri_target.load_state_dict(self.cri.state_dict())  # hard target update

                rho = np.exp(-(self.loss_c_sum / update_freq) ** 2)
                self.rho = (self.rho + rho) * 0.5
                self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * self.rho
                self.loss_c_sum = 0.0

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg


def get__buffer_reward_step(env, max_step, max_action, reward_scale, gamma, action_dim, is_discrete,
                            **_kwargs) -> (np.ndarray, list, list):
    buffer_list = list()

    reward_list = list()
    reward_item = 0.0

    step_list = list()
    step_item = 0

    state = env.reset()

    global_step = 0
    while global_step < max_step:
        action = rd.randint(action_dim) if is_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action * max_action)
        reward_item += reward
        step_item += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        buffer_list.append((adjust_reward, mask, state, action, next_state))

        if done:
            global_step += step_item

            reward_list.append(reward_item)
            reward_item = 0.0
            step_list.append(step_item)
            step_item = 0

            state = env.reset()
        else:
            state = next_state

    buffer_array = np.stack([np.hstack(buf_tuple) for buf_tuple in buffer_list])
    return buffer_array, reward_list, step_list


'''multi-process'''


def mp__update_params(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_epoch = args.max_epoch
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    args.init_for_training()
    del args

    state_dim, action_dim = q_o_buf.get()  # q__buf 1.
    agent = AgentDeepSAC(state_dim, action_dim, net_dim)

    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_buf.put(act_cpu)  # q_i_buf 1.
    # q_i_buf.put(act_cpu)  # q_i_buf 2. # todo warning
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    buffer = BufferArray(max_memo, state_dim, action_dim)  # experiment replay buffer

    '''initial_exploration'''
    buffer_array, reward_list, step_list = q_o_buf.get()  # q__buf 2.
    buffer.extend_memo(buffer_array)

    for epoch in range(max_epoch):  # epoch is episode
        buffer_array, reward_list, step_list = q_o_buf.get()  # q__buf n.
        reward_avg = np.average(reward_list)
        step_sum = sum(step_list)

        buffer.extend_memo(buffer_array)
        buffer.init_before_sample()

        loss_a_avg, loss_c_avg = agent.update_params(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_buf.put(act_cpu)  # q_i_buf n.
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            is_solved = q_o_eva.get()  # q_o_eva n.
            if is_solved:
                break

    q_i_buf.put(None)  # q_i_buf -1.
    q_i_eva.put(None)  # q_i_eva -1.
    pass


def mp__update_buffer(args, q_i_buf, q__buf):  # update replay buffer by interacting with env
    env_name = args.env_name
    max_step = args.max_step
    reward_scale = args.reward_scale
    gamma = args.gamma
    del args

    torch.set_num_threads(8)

    env = gym.make(env_name)
    state_dim, action_dim, max_action, _, is_discrete = get_env_info(env, is_print=False)
    q__buf.put((state_dim, action_dim))  # q__buf 1.

    '''build evaluated only actor'''
    q_i_buf_get = q_i_buf.get()  # q_i_buf 1.
    act = q_i_buf_get  # q_i_buf_get == act.to(device_cpu), requires_grad=False

    buffer_array, reward_list, step_list = get__buffer_reward_step(
        env, max_step, max_action, reward_scale, gamma, action_dim, is_discrete)

    q__buf.put((buffer_array, reward_list, step_list))  # q__buf 2.

    explore_noise = True
    state = env.reset()
    while q_i_buf_get is not None:
        buffer_list = list()

        reward_list = list()
        reward_item = 0.0

        step_list = list()
        step_item = 0

        global_step = 0
        while global_step < max_step:
            '''select action'''
            s_tensor = torch.tensor((state,), dtype=torch.float32, requires_grad=False)
            a_tensor = act(s_tensor, explore_noise)
            action = a_tensor.detach_().numpy()[0]

            next_state, reward, done, _ = env.step(action * max_action)
            reward_item += reward
            step_item += 1

            adjust_reward = reward * reward_scale
            mask = 0.0 if done else gamma
            buffer_list.append((adjust_reward, mask, state, action, next_state))

            if done:
                global_step += step_item

                reward_list.append(reward_item)
                reward_item = 0.0
                step_list.append(step_item)
                step_item = 0

                state = env.reset()
            else:
                state = next_state

        buffer_array = np.stack([np.hstack(buf_tuple) for buf_tuple in buffer_list])
        q__buf.put((buffer_array, reward_list, step_list))  # q__buf n.

        q_i_buf_get = q_i_buf.get()  # q_i_buf n.
        act = q_i_buf_get
    pass


def mp__evaluated_act(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
    max_step = args.max_step
    cwd = args.cwd

    print_gap = 2 ** 6
    env_name = args.env_name
    gpu_id = args.gpu_id
    del args

    torch.set_num_threads(8)

    '''recorder'''
    eva_r_max = -np.inf
    exp_r_avg = -np.inf
    total_step = 0
    loss_a_avg = 0
    loss_c_avg = 0
    recorder_exp = list()  # exp_r_avg, total_step, loss_a_avg, loss_c_avg
    recorder_eva = list()  # eva_r_avg, eva_r_std

    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)

    '''build evaluated only actor'''
    q_i_eva_get = q_i_eva.get()  # q_i_eva 1.
    act = q_i_eva_get  # q_i_eva_get == act.to(device_cpu), requires_grad=False

    print(f"{'GPU':3}  {'MaxR':>8}  {'R avg':>8}  {'R std':>8} |"
          f"{'ExpR':>8}  {'ExpS':>8}  {'LossA':>8}  {'LossC':>8}")

    is_solved = False
    start_time = timer()
    print_time = timer()

    def get_episode_reward():  # env, eval_num, act, max_step, max_action, ):
        reward_item = 0.0

        state = env.reset()
        for _ in range(max_step):
            s_tensor = torch.tensor((state,), dtype=torch.float32, requires_grad=False)
            a_tensor = act(s_tensor)
            action = a_tensor.detach_().numpy()[0]

            next_state, reward, done, _ = env.step(action * max_action)
            reward_item += reward

            if done:
                break
            state = next_state
        return reward_item

    while q_i_eva_get is not None:
        '''update actor'''
        while q_i_eva.qsize():  # get the latest
            q_i_eva_get = q_i_eva.get()  # q_i_eva n.
            act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
            total_step += exp_s_sum
            recorder_exp.append((exp_r_avg, total_step, loss_a_avg, loss_c_avg))

        '''evaluate actor'''
        reward_list = [get_episode_reward() for _ in range(16)]
        eva_r_avg = np.average(reward_list)
        if eva_r_avg > eva_r_max:  # check 1
            reward_list.extend([get_episode_reward() for _ in range(100 - len(reward_list))])
            eva_r_avg = np.average(reward_list)
            if eva_r_avg > eva_r_max:  # check 2
                eva_r_max = eva_r_avg

                act_save_path = f'{cwd}/actor.pth'
                print(f'{gpu_id:<3}  {eva_r_max:8.2f}  saved actor in {act_save_path}')
                torch.save(act.state_dict(), act_save_path)

        eva_r_std = np.std(reward_list)
        recorder_eva.append((eva_r_avg, eva_r_std))

        if eva_r_max > target_reward:
            is_solved = True
            used_time = int(timer() - start_time)
            print(f'######### solve: {used_time:8}  {total_step:8.2e}  \n'
                  f'######### solve: {eva_r_avg:8.2f}  {eva_r_std:8.2f} ')

        q_o_eva.put(is_solved)  # q_o_eva n.

        if timer() - print_time > print_gap:
            print_time = timer()
            print(f'{gpu_id:<3}  {eva_r_max:8.2f}  {eva_r_avg:8.2f}  {eva_r_std:8.2f} |'
                  f'{exp_r_avg:8.2f}  {total_step:8.2e}  {loss_a_avg:8.2f}  {loss_c_avg:8.2f}')

    pass


def run__mp():
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O

    cwd = 'MP_SAC'
    gpu_id = sys.argv[-1][-4]

    args = Arguments()
    args.class_agent = True  # todo
    # args.env_name = "BipedalWalker-v3"
    # args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LunarLander_{}'.format(cwd, gpu_id)
    args.gpu_id = gpu_id

    process = [
        mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
        # main process, train agent
        mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),  # assist, collect buffer
        mp.Process(target=mp__evaluated_act, args=(args, q_i_eva, q_o_eva)),  # assist, evaluate agent
    ]

    [p.start() for p in process]
    process[2].join()  # waiting the stop signal from process[2]
    [p.close() for p in process]


if __name__ == '__main__':
    run__mp()
