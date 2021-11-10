import sys
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from elegantrl.net import NnReshape, layer_norm, DenseNet
from elegantrl.agent import AgentBase
from elegantrl.env import build_env
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp

'''net'''


class BiConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim  # inp_for_cnn.shape == (N, C, H, W)

        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_h_dim * mid_dim)
            nn.Linear(i_h_dim * mid_dim, out_dim),
        )
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_w_dim * mid_dim)
            nn.Linear(i_w_dim * mid_dim, out_dim),
        )

    def forward(self, state):
        xh = self.cnn_h(state)
        xw = self.cnn_w(state)
        return xw + xh


class ActorBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 1), nn.ReLU(),
            DenseNet(mid_dim * 1), nn.ReLU(),
            nn.Linear(mid_dim * 4, action_dim),
        )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        action = self.net(state)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))

    def get_action(self, state, action_std):
        action = self.net(state)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))


class CriticBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))
        action_dim = (i_c_dim, i_w_dim, i_h_dim)  # (2, bs_n, ur_n)

        self.cnn_s = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 2, mid_dim * 1),
        )
        self.cnn_a = nn.Sequential(
            NnReshape(*action_dim),
            BiConvNet(mid_dim, action_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 2, mid_dim * 1),
        )

        self.out_net = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1), nn.Hardswish(),
            nn.Linear(mid_dim * 1, 1),
        )
        layer_norm(self.out_net[-1], std=0.1)  # output layer for action

    def forward(self, state, action):
        xs = self.cnn_s(state)
        xa = self.cnn_a(action)
        return self.out_net(xs + xa)  # Q value


class ShareBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))

        state_tuple = (i_c_dim, i_h_dim, i_w_dim)
        self.enc_s = nn.Sequential(
            # NnReshape(*state_tuple),
            BiConvNet(mid_dim, state_tuple, mid_dim * 4),
        )
        action_tuple = (i_c_dim, i_w_dim, i_h_dim)
        self.enc_a = nn.Sequential(
            NnReshape(*action_tuple),
            BiConvNet(mid_dim, action_tuple, mid_dim * 4),
        )

        self.mid_n = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1), nn.ReLU(),
            DenseNet(mid_dim),
        )

        self.dec_a = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, action_dim),
        )
        layer_norm(self.dec_a[-1], std=0.1)  # output layer for action
        self.dec_q = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, 1),
        )
        layer_norm(self.dec_q[-1], std=0.1)  # output layer for action

    def forward(self, state):  # actor
        xs = self.enc_s(state)
        xn = self.mid_n(xs)
        action = self.dec_a(xn)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))

    def critic(self, state, action):
        xs = self.enc_s(state)
        xa = self.enc_a(action)
        xn = self.mid_n(xs + xa)
        return self.dec_q(xn)  # Q value

    def get_action(self, state, action_std):  # actor, get noisy action
        xs = self.enc_s(state)
        xn = self.mid_n(xs)
        action = self.dec_a(xn)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))


class ActorSimplify:
    def __init__(self, gpu_id, actor_net):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else 'cpu')
        self.actor_net = actor_net.to(self.device)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        states = torch.as_tensor(state[np.newaxis], dtype=torch.float32, device=self.device)
        action = self.actor_net(states)[0]
        return action.detach().cpu().numpy()


def check_network():
    from envs.DownLink import DownLinkEnv
    env = DownLinkEnv()

    gpu_id = 1
    mid_dim = 128
    state_dim = env.state_dim
    action_dim = env.action_dim

    if_check_actor = 0
    if if_check_actor:
        net = ActorBiConv(mid_dim, state_dim, action_dim)
        act = ActorSimplify(gpu_id, net)

        state = env.reset()

        action = env.get_action_mmse(state)
        reward = env.get_reward(action)
        print(f"| mmse            : {reward:8.3f}")

        action = env.get_action_mmse(state)
        action = env.get_action_norm_power(action)
        reward = env.get_reward(action)
        print(f"| mmse (max Power): {reward:8.3f}")

        action = np.ones(env.action_dim)
        reward = env.get_reward(action)
        print(f"| ones (max Power): {reward:8.3f}")

        action = env.get_action_norm_power(action=None)  # random.normal action
        reward = env.get_reward(action)
        print(f"| rand (max Power): {reward:8.3f}")

        action = act.get_action(state)
        action = env.get_action_norm_power(action)
        reward = env.get_reward(action)
        print(f"| net  (max Power): {reward:8.3f}")

        # state, reward, done, _ = env.step(action)
        # print(reward)

    if_check_critic = 1
    if if_check_critic:
        batch_size = 3
        device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else 'cpu')
        critic_net = CriticBiConv(mid_dim, state_dim, action_dim).to(device)

        ten_state = torch.randn(size=(batch_size, *state_dim), dtype=torch.float32, device=device)
        ten_action = torch.randn(size=(batch_size, action_dim), dtype=torch.float32, device=device)

        q_value = critic_net(ten_state, ten_action)
        print(q_value)


'''agent'''


class AgentStep1AC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorBiConv
        self.ClassCri = CriticBiConv
        self.if_use_cri_target = False
        self.if_use_act_target = False
        self.explore_noise = 2 ** -8
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.MSELoss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw
        self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.obj_critic = 0.99 * self.obj_critic + 0.01 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

            obj_actor = -self.cri(state, self.act(state)).mean()  # policy gradient
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            q_label, action, state, is_weights = buffer.sample_batch_one_step(batch_size)

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


class AgentShareStep1AC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ShareBiConv
        self.ClassCri = self.ClassAct
        self.if_use_cri_target = True
        self.if_use_act_target = True
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )
        self.act = self.cri = self.ClassAct(net_dim, state_dim, action_dim).to(self.device)
        if self.if_use_act_target:
            self.act_target = self.cri_target = deepcopy(self.act)
        else:
            self.act_target = self.cri_target = self.act

        self.cri_optim = torch.optim.Adam(
            [{'params': self.act.enc_s.parameters(), 'lr': learning_rate * SHA_LR},
             {'params': self.act.enc_a.parameters(), },
             {'params': self.act.mid_n.parameters(), 'lr': learning_rate * SHA_LR},
             {'params': self.act.dec_a.parameters(), },
             {'params': self.act.dec_q.parameters(), },
             ], lr=learning_rate)
        self.act_optim = self.cri_optim

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.MSELoss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        update_a = 0
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            '''objective of critic'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.obj_critic = 0.995 * self.obj_critic + 0.005 * obj_critic.item()  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.act_target.critic(state, action_pg).mean()

                obj_united = obj_critic + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic

            self.optim_update(self.act_optim, obj_united, self.act.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.act.critic(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            q_label, action, state, is_weights = buffer.sample_batch_one_step(batch_size)

        q_value = self.act.critic(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


'''run'''


def demo_down_link_task():
    env_name = ['DownLinkEnv-v0', 'DownLinkEnv-v1'][ENV_ID]
    agent_class = [AgentStep1AC, AgentShareStep1AC][DRL_ID]
    args = Arguments(env=build_env(env_name), agent=agent_class())
    args.random_seed += GPU_ID

    args.net_dim = 2 ** 8
    args.batch_size = int(args.net_dim * 2 ** -1)

    args.max_memo = 2 ** 17
    args.target_step = int(args.max_memo * 2 ** -4)
    args.repeat_times = 0.75
    args.reward_scale = 2 ** 2
    args.agent.exploration_noise = 2 ** -5

    args.eval_gpu_id = GPU_ID
    args.eval_gap = 2 ** 9
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1

    args.learner_gpus = (GPU_ID,)

    if_use_single_process = 0
    if if_use_single_process:
        train_and_evaluate(args, )
    else:
        args.worker_num = 4
        train_and_evaluate_mp(args, )


"""
| Remove cwd: ./AgentOneStepPG_DownLinkEnv-v0_(1,)
################################################################################
ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  6.55e+04  708.26 |  708.26    5.9   1023     0 |    0.69   0.02  -0.68
0  1.34e+06  857.54 |  857.54    2.9   1023     0 |    0.84   0.01  -0.55
0  2.20e+06  890.26 |  890.26   10.5   1023     0 |    0.88   0.01  -0.65
0  2.62e+06 1071.47 | 1071.47    6.4   1023     0 |    1.02   0.02  -0.64
0  5.18e+06 1162.44 | 1162.44    5.1   1023     0 |    1.12   0.02  -0.68
0  7.73e+06 1303.46 | 1303.46    2.4   1023     0 |    1.32   0.02  -0.64
0  1.00e+07 1619.88 | 1619.88    6.2   1023     0 |    1.55   0.02  -0.73
0  1.51e+07 1687.30 | 1687.30   22.1   1023     0 |    1.65   0.02  -0.81
0  2.01e+07 1822.41 | 1714.13    0.0   1023     0 |    1.67   0.02  -0.90
0  2.53e+07 1822.41 | 1623.96    0.0   1023     0 |    1.56   0.02  -0.88
0  3.21e+07 2098.69 | 2098.69    4.4   1023     0 |    2.03   0.02  -0.97

| Remove cwd: ./AgentOneStepPG_DownLinkEnv-v0_(3,)
################################################################################
ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  6.55e+04  716.41 |  716.41    5.2   1023     0 |    0.69   0.02  -0.69
0  1.34e+06  809.24 |  809.24    1.6   1023     0 |    0.72   0.01  -0.62
0  5.18e+06 1002.50 | 1002.50    1.4   1023     0 |    0.97   0.01  -0.62
0  1.03e+07 1239.68 | 1140.18    0.0   1023     0 |    1.12   0.01  -0.37
0  1.50e+07 1489.36 | 1489.36    2.6   1023     0 |    1.41   0.02  -0.42
0  2.01e+07 1767.12 | 1561.77    0.0   1023     0 |    1.50   0.02  -0.70
0  3.03e+07 1767.12 | 1579.06    0.0   1023     0 |    1.54   0.02  -0.73
0  4.01e+07 1767.12 | 1559.15    0.0   1023     0 |    1.54   0.02  -0.82
0  4.35e+07 1820.31 | 1820.31   21.5   1023     0 |    1.70   0.02  -0.92
0  4.52e+07 2006.18 | 2006.18   13.6   1023     0 |    1.85   0.02  -0.95
| UsedTime:   28538 | SavedDir: ./AgentOneStepPG_DownLinkEnv-v0_(3,)

| Remove cwd: ./AgentOneStepPG_DownLinkEnv-v0_(4,)
################################################################################
0  6.55e+04  713.66 |  713.66    0.8   1023     0 |    0.69   0.02  -0.67
0  1.34e+06  843.31 |  843.31    3.1   1023     0 |    0.80   0.01  -0.59
0  5.18e+06  993.18 |  952.90    0.0   1023     0 |    0.92   0.01  -0.48
0  1.03e+07 1309.81 | 1221.37    0.0   1023     0 |    1.22   0.01  -0.56
0  2.01e+07 1441.55 | 1441.55   25.8   1023     0 |    1.38   0.02  -0.23
0  3.03e+07 1582.27 | 1480.24    0.0   1023     0 |    1.42   0.02  -0.50
0  4.01e+07 1722.72 | 1539.62    0.0   1023     0 |    1.51   0.02  -0.66
0  5.03e+07 1893.11 | 1893.11    4.3   1023     0 |    1.78   0.02  -0.88
0  5.59e+07 1968.71 | 1968.71   17.7   1023     0 |    1.92   0.02  -1.02
| UsedTime:   35421 | SavedDir: ./AgentOneStepPG_DownLinkEnv-v0_(4,)
"""

""" 2021-11-09
GPU 83 GPU 1   B_SIZE=net_dim*1.0 NOISE=-7  RP_TIM=1.0          1922, 22e6, 43e6
GPU 83 GPU 2   beta0 RP_TIM=0.75 T_STEP=-4  B_SIZE=net_dim*1    1927, 10e6, 21e6
GPU 111  GPU 0 beta0 RP_TIM=0.75 T_STEP=-4  B_SIZE=net_dim*0.5  1805, 7e6

GPU 111 GPU 1  beta0 RP_TIM=0.75 T_STEP=-4  B_SIZE=net_dim*1    1696, 
GPU 111 GPU 2  B_SIZE=net_dim*0.5 NOISE=-5                      1938, 19e6, 28e6 ### NOISE=-5
GPU 111 GPU 3  B_SIZE=net_dim*0.5 NOISE=-7                      1842, 09e6, 14e6
GPU 83  GPU 3  B_SIZE=net_dim*0.5 NOISE=-7  RP_TIM=1.0          1871, 10e6, 21e6

GPU 83  GPU 4  B_SIZE=net_dim*0.5 NOISE=-5 beta1                1946, 10e6, 19e6 ### NOISE=-5 beta0
GPU 83  GPU 0  B_SIZE=net_dim*0.5 NOISE=-7 beta1                1819, 09e6, 15e6
"""

"""2021-11-10
GPU 83  GPU 2  MLP      RP_TIM=0.75 T_STEP=-4  B_SIZE=net_dim*1    1927, 10e6, 21e6 
GPU 83  GPU 3  DenseNet Env-v1 Step1AC
GPU 83  GPU 4  DenseNet B_SIZE=net_dim*0.5 NOISE=-5                1946, 10e6, 19e6 
GPU 83  GPU 0  DenseNet Env-v1 ShareStep1AC lr=1.1
GPU 83  GPU 1  DenseNet Env-v1 ShareStep1AC lr=1.25

GPU 111 GPU 1  DenseNet Env-v1 ShareStep1AC lr=0.75
GPU 111 GPU 2  DenseNet Env-v1 ShareStep1AC lr=1.50                2010, 06e6, 08e6
GPU 111 GPU 3  DenseNet Env-v1 ShareStep1AC lr=2.00
"""

GPU_ID = int(eval(sys.argv[1]))
DRL_ID = int(eval(sys.argv[2]))
ENV_ID = 1  # int(eval(sys.argv[2]))

SHA_LR = float(eval(sys.argv[3]))

if __name__ == '__main__':
    dir(sys)
    # check_network()
    demo_down_link_task()
