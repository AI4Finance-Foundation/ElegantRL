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


def mp__update_params(args, q__net, q__buf, q__eva, ):  # update network parameters using replay buffer
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_epoch = args.max_epoch
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    del args

    state_dim, action_dim = q__buf.get()  # q__buf 1.
    agent = AgentDeepSAC(state_dim, action_dim, net_dim)
    q__net.put(agent.act)  # q__net 1.
    q__eva.put(agent.act)  # q__eva 1.

    buffer = BufferArray(max_memo, state_dim, action_dim)  # experiment replay buffer

    '''initial_exploration'''
    buffer_array, reward_list, step_list = q__buf.get()  # q__buf 2.
    buffer.extend_memo(buffer_array)

    start_time = timer()
    for epoch in range(int(2 ** 12)):  # epoch is episode
        buffer_array, reward_list, step_list = q__buf.get()  # q__buf n.
        reward_avg = np.average(reward_list)
        step_sum = sum(step_list)

        buffer.extend_memo(buffer_array)
        buffer.init_before_sample()

        agent.update_params(buffer, max_step, batch_size, repeat_times)
        q__net.put(agent.act.state_dict())  # q__net n.
        q__eva.put((agent.act.state_dict(), reward_avg, step_sum))  # q__eva n.

    q__net.put(None)  # q__net -1.
    q__eva.put(None)  # q__eva -1.
    print("UsedTime:", int(timer() - start_time))
    pass


def mp__update_buffer(args, q__net, q__buf):  # update replay buffer by interacting with env
    env_name = args.env_name
    max_step = args.max_step
    reward_scale = args.reward_scale
    gamma = args.gamma
    del args

    env = gym.make(env_name)
    state_dim, action_dim, max_action, _, is_discrete = get_env_info(env, is_print=False)
    q__buf.put((state_dim, action_dim))  # q__buf 1.

    '''build evaluated only actor'''
    q__net_get = q__net.get()  # q__net 1.
    act = q__net_get.to(torch.device("cpu"))  # q__net_get ==  act.to(device_gpu)
    act.eval()
    [setattr(param, 'requires_grad', False) for param in act.parameters()]

    buffer_array, reward_list, step_list = get__buffer_reward_step(
        env, max_step, max_action, reward_scale, gamma, action_dim, is_discrete)

    q__buf.put((buffer_array, reward_list, step_list))  # q__buf 2.

    explore_noise = True
    state = env.reset()
    while q__net_get is not None:
        buffer_list = list()

        reward_list = list()
        reward_item = 0.0

        step_list = list()
        step_item = 0

        global_step = 0
        while global_step < max_step:
            '''select action'''
            # action = rd.randint(action_dim) if is_discrete else rd.uniform(-1, 1, size=action_dim)
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

        q__net_get = q__net.get()  # q__net n.
        act.load_state_dict(q__net_get)  # q__net_get == act.state_dict()
    pass


def mp__evaluate_agent(args, q__eva, ):  # evaluate agent and get its total reward of an episode
    max_step = args.max_step
    eval_num = 4
    env_name = args.env_name
    del args

    '''recorder'''
    exp_r_avg_list = list()  # explored reward average
    exp_s_sum_list = list()  # explored step sum
    global_step = 0
    eva_r_avg_list = list()  # evaluated reward average
    eva_r_std_list = list()  # evaluated reward standard deviation

    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)

    '''build evaluated only actor'''
    q__eva_get = q__eva.get()  # q__eva 1.
    act = q__eva_get.to(torch.device("cpu"))  # q__eva_get == act.to(device_gpu)
    act.eval()
    [setattr(param, 'requires_grad', False) for param in act.parameters()]

    while q__eva_get is not None:
        '''update actor'''
        q__eva_get = q__eva.get()  # q__eva n.
        act_load_dict, exp_r_avg, exp_s_sum = q__eva_get

        act.load_state_dict(act_load_dict)
        exp_r_avg_list.append(exp_r_avg)
        exp_s_sum_list.append(exp_s_sum)
        global_step += exp_s_sum

        '''evaluate actor'''
        reward_list = list()
        while len(reward_list) < eval_num:
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
            reward_list.append(reward_item)

        eva_r_avg = np.average(reward_list)
        eva_r_avg_list.append(eva_r_avg)
        eva_r_std = np.std(reward_list)
        eva_r_std_list.append(eva_r_std)
        print(f'|  ExpR {exp_r_avg:8.2f}  Step {exp_s_sum:8.2e}  '
              f'|  AvgR {eva_r_avg:8.2f}  StdR {eva_r_std:8.2f}')

    pass


def run__mp():
    q__net = mp.Queue(maxsize=2)
    q__buf = mp.Queue(maxsize=2)
    q__eva = mp.Queue(maxsize=2)

    cwd = 'MP_SAC'
    gpu_id = '3'

    args = Arguments()
    args.class_agent = True  # todo
    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    # while not train_agent(**vars(args)):
    #     args.random_seed += 42

    process = [
        mp.Process(target=mp__update_params, args=(args, q__net, q__buf, q__eva,)),  # main process, train agent
        mp.Process(target=mp__update_buffer, args=(args, q__net, q__buf,)),  # assist, collect buffer
        mp.Process(target=mp__evaluate_agent, args=(args, q__eva,)),  # assist, evaluate agent
    ]

    [p.start() for p in process]
    [p.join() for p in process]
    [p.close() for p in process]


if __name__ == '__main__':
    run__mp()
    pass
