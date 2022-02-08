import os
from copy import deepcopy

import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils import clip_grad_norm_


class AgentBase:  # [ElegantRL.2021.11.11]
    def __init__(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        """initialize

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = None
        self.states = None
        self.device = None
        self.action_dim = None
        self.reward_scale = None
        self.if_off_policy = True

        self.env_num = env_num
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        # self.amp_scale = None  # automatic mixed precision

        """attribute"""
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = (
            self.cri_target
        ) = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = (
            self.act_target
        ) = self.if_use_act_target = self.act_optim = self.ClassAct = None

        assert isinstance(gpu_id, int)
        assert isinstance(env_num, int)
        assert isinstance(net_dim, int)
        assert isinstance(state_dim, int)
        assert isinstance(action_dim, int)
        assert isinstance(if_per_or_gae, bool)
        assert isinstance(gamma, float)
        assert isinstance(reward_scale, float)
        assert isinstance(learning_rate, float)

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )

        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(
            self.device
        )
        self.act = (
            self.ClassAct(net_dim, state_dim, action_dim).to(self.device)
            if self.ClassAct
            else self.cri
        )
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = (
            torch.optim.Adam(self.act.parameters(), learning_rate)
            if self.ClassAct
            else self.cri
        )

        def get_optim_param(
            optim,
        ):  # optim = torch.optim.Adam(network_param, learning_rate)
            params_list = []
            for params_dict in optim.state_dict()["state"].values():
                params_list.extend(
                    [t for t in params_dict.values() if isinstance(t, torch.Tensor)]
                )
            return params_list

        from types import MethodType

        self.act_optim.parameters = MethodType(get_optim_param, self.act_optim)
        self.cri_optim.parameters = MethodType(get_optim_param, self.cri_optim)

        assert isinstance(if_per_or_gae, bool)
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action via a given state.

        :param state: a state in a shape (state_dim, ).
        :return: action [array], action.shape == (action_dim, ) where each action is clipped into range(-1, 1).
        """
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        return a_tensor.detach().cpu().numpy()

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Select continuous actions for exploration

        :param state: states.shape==(batch_size, state_dim, )
        :return: actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """

        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(
                -1, 1
            )
        return action.detach().cpu()

    def explore_one_env(self, env, target_step: int) -> list:
        """actor explores in single Env, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step()
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        state = self.states[0]
        traj = []
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + self.action_dim)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2:] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s

        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state, traj_other),
        ]
        return self.convert_trajectory(traj_list)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step: int) -> list:
        """actor explores in VectorEnv, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        ten_states = self.states

        traj = []
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat(
                (ten_rewards.unsqueeze(0), ten_dones.unsqueeze(0), ten_actions)
            )
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        # traj = [(env_ten, ...), ...], env_ten = (env1_ten, env2_ten, ...)
        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state[:, env_i, :], traj_other[:, env_i, :])
            for env_i in range(len(self.states))
        ]
        # traj_list = [traj_env_0, ...], traj_env_0 = (ten_state, ten_other)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def update_net(
        self, buffer, batch_size: int, repeat_times: float, soft_update_tau: float
    ) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer

        :param buffer: Experience replay buffer
        :param batch_size: sample batch_size of data for Stochastic Gradient Descent
        :param repeat_times: `batch_sampling_times = int(target_step * repeat_times / batch_size)`
        :param soft_update_tau: soft target update: `target_net = target_net * (1-tau) + current_net * tau`,
        """

    def optim_update(self, optimizer, objective):  # [ElegantRL 2021.11.11]
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(
            parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm
        )
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     """minimize the optimization objective via update the network parameters
    #
    #     amp: Automatic Mixed Precision
    #
    #     :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
    #     :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
    #     :param params: `params = net.parameters()` the network parameters which need to be updated.
    #     """
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update target network via current network

        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [
            ("actor", self.act),
            ("act_target", self.act_target),
            ("act_optim", self.act_optim),
            ("critic", self.cri),
            ("cri_target", self.cri_target),
            ("cri_optim", self.cri_optim),
        ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def convert_trajectory(self, traj_list: list) -> list:  # off-policy
        """convert trajectory (env exploration type) to trajectory (replay buffer type)

        convert `other = concat((      reward, done, ...))`
        to      `other = concat((scale_reward, mask, ...))`

        :param traj_list: `traj_list = [(tensor_state, other_state), ...]`
        :return: `traj_list = [(tensor_state, other_state), ...]`
        """
        for ten_state, ten_other in traj_list:
            ten_other[:, 0] = ten_other[:, 0] * self.reward_scale  # ten_reward
            ten_other[:, 1] = (
                1.0 - ten_other[:, 1]
            ) * self.gamma  # ten_mask = (1.0 - ary_done) * gamma
        return traj_list
