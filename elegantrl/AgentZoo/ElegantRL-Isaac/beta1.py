# import os
import sys
import numpy as np
# import numpy.random as rd
import torch
import torch.nn as nn
# from copy import deepcopy
# from torch.nn.utils import clip_grad_norm_

from elegantrl.net import NnReshape, layer_norm
from elegantrl.agent import AgentBase
from elegantrl.env import build_env
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp

"""
GPU 1 beta2, ur_n = 8
GPU 3 beta3, ur_n = 4
"""

'''net'''


class ActorBiCNN(nn.Module):  # todo attention
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            BiCNN(mid_dim, state_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1), nn.Hardswish(),
            nn.Linear(mid_dim * 1, action_dim),
        )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        action = self.net(state)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))

    def get_action(self, state, action_std):
        action = self.net(state)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))


class CriticBiCNN(nn.Module):  # todo attention
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))
        action_dim = (i_c_dim, i_w_dim, i_h_dim)  # (2, bs_n, ur_n)

        self.cnn_s = nn.Sequential(
            # NnReshape(*state_dim),
            BiCNN(mid_dim, state_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2),

        )
        self.cnn_a = nn.Sequential(
            NnReshape(*action_dim),
            BiCNN(mid_dim, action_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2),
        )

        self.att_s = nn.Sequential(
            nn.Linear(mid_dim * 2, mid_dim), nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 1, mid_dim * 2), nn.Sigmoid(),
        )
        self.att_a = nn.Sequential(
            nn.Linear(mid_dim * 2, mid_dim), nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 1, mid_dim * 2), nn.Sigmoid(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1), nn.Hardswish(),
            nn.Linear(mid_dim * 1, 1),
        )
        layer_norm(self.out_net[-1], std=0.1)  # output layer for action

    def forward(self, state, action):
        xs = self.cnn_s(state)
        xa = self.cnn_a(action)
        xc = torch.cat((xs * self.att_a(xa),
                        xa * self.att_s(xs)), dim=1)
        return self.out_net(xc)  # Q value


class BiCNN(nn.Module):  # todo attention
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        # inp_for_cnn.shape == (N, C, H, W)
        i_c_dim, i_h_dim, i_w_dim = inp_dim

        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim, mid_dim * 2, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_h_dim * mid_dim)
        )
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim, mid_dim * 2, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_w_dim * mid_dim)
        )

        hd_dim = i_h_dim * mid_dim
        wd_dim = i_w_dim * mid_dim
        self.att_h = nn.Sequential(
            nn.Linear(hd_dim, hd_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(hd_dim // 2, wd_dim), nn.Sigmoid(),
        )
        self.att_w = nn.Sequential(
            nn.Linear(wd_dim, wd_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(wd_dim // 2, hd_dim), nn.Sigmoid(),
        )

        self.out_net = nn.Linear(hd_dim + wd_dim, out_dim)

    def forward(self, state):
        xh = self.cnn_h(state)
        xw = self.cnn_w(state)
        xc = torch.cat((xh * self.att_w(xw),
                        xw * self.att_h(xh)), dim=1)
        return self.out_net(xc)


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
        net = ActorBiCNN(mid_dim, state_dim, action_dim)
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
        critic_net = CriticBiCNN(mid_dim, state_dim, action_dim).to(device)

        ten_state = torch.randn(size=(batch_size, *state_dim), dtype=torch.float32, device=device)
        ten_action = torch.randn(size=(batch_size, action_dim), dtype=torch.float32, device=device)

        q_value = critic_net(ten_state, ten_action)
        print(q_value)


'''agent'''


class AgentOneStepPG(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorBiCNN
        self.ClassCri = CriticBiCNN
        self.if_use_cri_target = False
        self.if_use_act_target = False
        self.explore_noise = 2 ** -8

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        AgentBase.init(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                       reward_scale=reward_scale, gamma=gamma,
                       learning_rate=learning_rate, if_per_or_gae=if_per_or_gae,
                       env_num=env_num, gpu_id=gpu_id, )
        self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state


'''run'''


def demo_down_link_task():
    env_name = ['DownLinkEnv-v0'][ENV_ID]
    agent_class = [AgentOneStepPG, ][0]
    args = Arguments(env=build_env(env_name), agent=agent_class())

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 17
    args.target_step = 2 ** 13
    args.batch_size = args.net_dim * 2
    args.repeat_times = 1.0
    args.agent.exploration_noise = 1 / NOISE

    args.eval_gpu_id = GPU_ID
    args.eval_gap = 2 ** 8
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
GPU 83 GPU 3   beta0 NOISE=64
GPU 83 GPU 4   beta0 NOISE=128

GPU 83 GPU 1   beta1 NOISE=128  attention
GPU 83 GPU 2   beta1 NOISE=256  attention
"""

if __name__ == '__main__':
    # check_network()
    ENV_ID = 0
    GPU_ID = int(eval(sys.argv[1]))
    NOISE = int(eval(sys.argv[2]))
    demo_down_link_task()
