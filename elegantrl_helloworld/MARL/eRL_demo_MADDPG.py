# from elegantrl.agents.AgentMADDPG import AgentMADDPG
import os
import shutil
import time
from copy import deepcopy

import gym
import numpy as np
import numpy.random as rd
import torch
from net import Actor
from net import Critic
from torch import nn
from tqdm import tqdm

"""[ElegantRL.2021.09.09](https://github.com/AI4Finance-Foundation/ElegantRL)"""


def save_learning_curve(
    recorder=None,
    cwd=".",
    save_title="learning curve",
    fig_name="plot_learning_curve.jpg",
):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    # obj_c = recorder[:, 4]
    # obj_a = recorder[:, 5]

    """plot subplots"""
    import matplotlib as mpl

    mpl.use("Agg")

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)

    """axs[0]"""
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = "darkcyan"
    ax01.set_ylabel("Explore AvgReward", color=color01)
    ax01.plot(
        steps,
        r_exp,
        color=color01,
        alpha=0.5,
    )
    ax01.tick_params(axis="y", labelcolor=color01)

    color0 = "lightcoral"
    ax00.set_ylabel("Episode Return")
    ax00.plot(steps, r_avg, label="Episode Return", color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    """axs[1]"""
    ax10 = axs[1]
    ax10.cla()
    """plot save"""
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close(
        "all"
    )  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`


def get_episode_return_and_step_marl(env, agent, device) -> (float, int):
    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode

    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(100):
        action = agent.select_actions(state)

        state, reward, done, _ = env.step(action)
        # for i in range(agent.n_agents):
        episode_return += reward[0]
        global_done = True

    episode_return = getattr(env, "episode_return", episode_return)
    return episode_return, episode_step


def build_env(env, if_print=False):
    env_name = getattr(env, "env_name", env)
    assert isinstance(env_name, str)

    if env_name in {
        "LunarLanderContinuous-v2",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
        "CartPole-v0",
        "LunarLander-v2",
    }:
        env = gym.make(env_name)
        env = PreprocessEnv(env, if_print=if_print)
    elif env_name in {
        "ReacherBulletEnv-v0",
        "AntBulletEnv-v0",
        "HumanoidBulletEnv-v0",
        "MinitaurBulletEnv-v0",
    }:
        import pybullet_envs

        dir(pybullet_envs)
        env = gym.make(env_name)
        env = PreprocessEnv(env, if_print=if_print)
    elif env_name == "Pendulum-v0":
        env = gym.make("Pendulum-v0")
        env.target_return = -200
        env = PreprocessEnv(env=env, if_print=if_print)
    elif env_name == "CarRacingFix":  # Box2D
        from elegantrl.envs.CarRacingFix import CarRacingFix

        env = CarRacingFix()
    else:
        assert not isinstance(env, str)
        env = deepcopy(env)
        # raise ValueError(f'| build_env_from_env_name: need register: {env_name}')
    return env


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


class AgentBase:
    def __init__(self):
        self.states = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = False
        self.explore_rate = 1.0
        self.explore_noise = None
        self.traj_list = None  # trajectory_list
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

    def init(
        self,
        net_dim,
        state_dim,
        action_dim,
        learning_rate=1e-4,
        marl=False,
        n_agents=1,
        if_per_or_gae=False,
        env_num=1,
        agent_id=0,
    ):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `float learning_rate` learning rate of optimizer
        `bool if_per_or_gae` PER (off-policy) or GAE (on-policy) for sparse reward
        `int env_num` the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        `int agent_id` if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
        """
        self.action_dim = action_dim
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [[] for _ in range(env_num)]
        self.device = torch.device(
            f"cuda:{agent_id}"
            if (torch.cuda.is_available() and (agent_id >= 0))
            else "cpu"
        )
        # assert 0
        if not marl:
            self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(
                self.device
            )
        else:
            self.cri = self.ClassCri(
                int(net_dim * 1.25), state_dim * n_agents, action_dim * n_agents
            ).to(self.device)
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
        del self.ClassCri, self.ClassAct

        if env_num > 1:  # VectorEnv
            self.explore_env = self.explore_vec_env
        else:
            self.explore_env = self.explore_one_env

    def select_actions(self, states) -> np.ndarray:
        """Select continuous actions for exploration

        `array states` states.shape==(batch_size, state_dim, )
        return `array actions` actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = (actions + torch.randn_like(actions) * self.explore_noise).clamp(
                -1, 1
            )
        return actions.detach().cpu().numpy()

    def explore_one_env(self, env, target_step):
        """actor explores in one env, then returns the traj (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        traj = []
        state = self.states[0]
        for _ in range(target_step):
            action = self.select_actions((state,))[0]
            next_s, reward, done, _ = env.step(action)
            traj.append((state, (reward, done, *action)))

            state = env.reset() if done else next_s
        self.states[0] = state

        return [
            traj,
        ]  # traj_list [traj_env_0, ]

    def explore_vec_env(self, env, target_step):
        """actor explores in VectorEnv, then returns the trajectory (env transition)

        `object env` RL training environment. env.reset() env.step()
        `int target_step` explored target_step number of step in env
        return `[traj, ...]` for off-policy ReplayBuffer, `traj = [(state, other), ...]`
        """
        env_num = len(self.traj_list)
        states = self.states

        traj_list = [[] for _ in range(env_num)]
        for _ in range(target_step):
            actions = self.select_actions(states)
            s_r_d_list = env.step(actions)

            next_states = []
            for env_i in range(env_num):
                next_state, reward, done = s_r_d_list[env_i]
                traj_list[env_i].append(
                    (states[env_i], (reward, done, *actions[env_i]))
                )
                next_states.append(next_state)
            states = next_states

        self.states = states
        return traj_list  # (traj_env_0, ..., traj_env_i)

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer.
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        `float repeat_times` the times of sample batch = int(target_step * repeat_times) in off-policy
        `float soft_update_tau` target_net = target_net * (1-tau) + current_net * tau
        `return tuple` training logging. tuple = (float, float, ...)
        """

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
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
        """soft update a target network via current network

        `nn.Module target_net` target network update via a current network, it is more stable
        `nn.Module current_net` current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.

        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
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


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(
        self,
        net_dim,
        state_dim,
        action_dim,
        learning_rate=1e-4,
        marl=False,
        n_agents=1,
        if_use_per=False,
        env_num=1,
        agent_id=0,
    ):
        super().init(
            net_dim,
            state_dim,
            action_dim,
            learning_rate,
            marl,
            n_agents,
            if_use_per,
            env_num,
            agent_id,
        )
        self.ou_noise = OrnsteinUhlenbeckNoise(
            size=action_dim, sigma=self.explore_noise
        )
        self.loss_td = torch.nn.MSELoss()
        if if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(
                reduction="none" if if_use_per else "mean"
            )
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(
                reduction="none" if if_use_per else "mean"
            )
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, states) -> np.ndarray:
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states).detach().cpu().numpy()

        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = self.ou_noise()
            actions = (actions + ou_noise[np.newaxis]).clip(-1, 1)

        return actions[0]

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = (self.criterion(q_value, q_label) * is_weights).mean()

        td_error = (q_label - q_value.detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


class AgentMADDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

    def init(
        self,
        net_dim,
        state_dim,
        action_dim,
        learning_rate=1e-4,
        marl=True,
        n_agents=1,
        if_use_per=False,
        env_num=1,
        agent_id=0,
        gamma=0.95,
    ):
        self.agents = [AgentDDPG() for i in range(n_agents)]
        self.explore_env = self.explore_one_env
        self.if_on_policy = False
        self.n_agents = n_agents
        for i in range(self.n_agents):
            self.agents[i].init(
                net_dim,
                state_dim,
                action_dim,
                learning_rate=1e-4,
                marl=True,
                n_agents=self.n_agents,
                if_use_per=False,
                env_num=1,
                agent_id=0,
            )
        self.n_states = state_dim
        self.n_actions = action_dim

        self.batch_size = net_dim
        self.gamma = gamma
        self.update_tau = 0
        self.device = torch.device(
            f"cuda:{agent_id}"
            if (torch.cuda.is_available() and (agent_id >= 0))
            else "cpu"
        )

    def update_agent(self, rewards, dones, actions, observations, next_obs, index):
        # rewards, dones, actions, observations, next_obs = buffer.sample_batch(self.batch_size)
        curr_agent = self.agents[index]
        curr_agent.cri_optim.zero_grad()
        all_target_actions = []
        for i in range(self.n_agents):
            if i == index:
                all_target_actions.append(curr_agent.act_target(next_obs[:, index]))
            if i != index:
                action = self.agents[i].act_target(next_obs[:, i])
                all_target_actions.append(action)
        action_target_all = (
            torch.cat(all_target_actions, dim=1)
            .to(self.device)
            .reshape(actions.shape[0], actions.shape[1] * actions.shape[2])
        )

        target_value = rewards[:, index] + self.gamma * curr_agent.cri_target(
            next_obs.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]),
            action_target_all,
        ).detach().squeeze(dim=1)
        # vf_in = torch.cat((observations.reshape(next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]), actions.reshape(actions.shape[0], actions.shape[1],actions.shape[2])), dim = 2)
        actual_value = curr_agent.cri(
            observations.reshape(
                next_obs.shape[0], next_obs.shape[1] * next_obs.shape[2]
            ),
            actions.reshape(actions.shape[0], actions.shape[1] * actions.shape[2]),
        ).squeeze(dim=1)
        vf_loss = curr_agent.loss_td(actual_value, target_value.detach())

        # vf_loss.backward()
        # curr_agent.cri_optim.step()

        curr_agent.act_optim.zero_grad()
        curr_pol_out = curr_agent.act(observations[:, index])
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(self.n_agents):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(actions[:, i])
        # vf_in = torch.cat((observations, torch.cat(all_pol_acs, dim = 0).to(self.device).reshape(actions.size()[0], actions.size()[1], actions.size()[2])), dim = 2)

        pol_loss = -torch.mean(
            curr_agent.cri(
                observations.reshape(
                    observations.shape[0], observations.shape[1] * observations.shape[2]
                ),
                torch.cat(all_pol_acs, dim=1)
                .to(self.device)
                .reshape(actions.shape[0], actions.shape[1] * actions.shape[2]),
            )
        )

        curr_agent.act_optim.zero_grad()
        pol_loss.backward()
        curr_agent.act_optim.step()
        curr_agent.cri_optim.zero_grad()
        vf_loss.backward()
        curr_agent.cri_optim.step()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()
        self.batch_size = batch_size
        self.update_tau = soft_update_tau
        self.update(buffer)
        self.update_all_agents()
        return

    def update(self, buffer):
        rewards, dones, actions, observations, next_obs = buffer.sample_batch(
            self.batch_size
        )
        for index in range(self.n_agents):
            self.update_agent(rewards, dones, actions, observations, next_obs, index)

    def update_all_agents(self):
        for agent in self.agents:
            self.soft_update(agent.cri_target, agent.cri, self.update_tau)
            self.soft_update(agent.act_target, agent.act, self.update_tau)

    def explore_one_env(self, env, target_step) -> list:
        traj_temp = []
        k = 0
        for _ in range(target_step):
            k += 1
            actions = []
            for i in range(self.n_agents):
                action = self.agents[i].select_actions(self.states[i])
                actions.append(action)
            # print(actions)
            next_s, reward, done, _ = env.step(actions)
            traj_temp.append((self.states, reward, done, actions))
            global_done = all(global_done for _ in range(self.n_agents))
            if global_done or k > 100:
                state = env.reset()
                k = 0
            else:
                state = next_s
        self.states = state
        return traj_temp

    def select_actions(self, states):
        actions = []
        for i in range(self.n_agents):
            action = self.agents[i].select_actions(states[i])
            actions.append(action)
        return actions

    def save_or_load_agent(self, cwd, if_save):
        for i in range(self.n_agents):
            self.agents[i].save_or_load_agent(cwd + "/" + str(i), if_save)


class OrnsteinUhlenbeckNoise:  # NOT suggest to use it
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process
        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.
        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise
        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True, if_norm=False):
        """Preprocess a standard OpenAI gym environment for training.

        `object env` a standard OpenAI gym environment, it has env.reset() and env.step()
        `bool if_print` print the information of environment. Such as env_name, state_dim ...
        `object data_type` convert state (sometimes float64) to data_type (float32).
        """
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (
            self.env_name,
            self.state_dim,
            self.action_dim,
            self.action_max,
            self.max_step,
            self.if_discrete,
            self.target_return,
        ) = get_gym_env_info(self.env, if_print)
        self.env.env_num = getattr(self.env, "env_num", 1)
        self.env_num = 1

        if if_norm:
            state_avg, state_std = get_avg_std__for_state_norm(self.env_name)
            self.neg_state_avg = -state_avg
            self.div_state_std = 1 / (state_std + 1e-4)

            self.reset = self.reset_norm
            self.step = self.step_norm
        else:
            self.reset = self.reset_type
            self.step = self.step_type

    def reset_type(self):
        tmp = self.env.reset()
        return [
            tmp[0].astype(np.float32),
            tmp[1].astype(np.float32),
            tmp[2].astype(np.float32),
        ]

    def step_type(self, action):
        # print(self.action_max)
        # assert 0
        # print(action * self.action_max)
        state, reward, done, info = self.env.step(action * self.action_max)
        # return state.astype(np.float32), reward, done, info
        return state, reward, done, info

    def reset_norm(self) -> np.ndarray:
        """convert the data type of state from float64 to float32
        do normalization on state

        return `array state` state.shape==(state_dim, )
        """
        state = self.env.reset()
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(np.float32)

    def step_norm(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        """convert the data type of state from float64 to float32,
        adjust action range to (-action_max, +action_max)
        do normalization on state

        return `array state`  state.shape==(state_dim, )
        return `float reward` reward of one step
        return `bool done` the terminal of an training episode
        return `dict info` the information save in a dict. OpenAI gym standard. Send a `None` is OK
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(np.float32), reward, done, info


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.

    `object env` a standard OpenAI gym environment, it has env.reset() and env.step()
    `bool if_print` print the information of environment. Such as env_name, state_dim ...
    return `env_name` the environment name, such as XxxXxx-v0
    return `state_dim` the dimension of state
    return `action_dim` the dimension of continuous action; Or the number of discrete action
    return `action_max` the max action of continuous action; action_max == 1 when it is discrete action space
    return `max_step` the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step
    return `if_discrete` Is this env a discrete action space?
    return `target_return` the target episode return, if agent reach this score, then it pass this game (env).
    """
    assert isinstance(env, gym.Env)

    # env_name = getattr(env, 'env_name', None)
    env_name = "simple_spread"
    # env_name = env.unwrapped.spec.id if env_name is None else env_name

    state_shape = env.observation_space[0].shape
    state_dim = (
        state_shape[0] if len(state_shape) == 1 else state_shape
    )  # sometimes state_dim is a list

    target_return = getattr(env, "target_return", None)
    target_return_default = getattr(env.spec, "reward_threshold", None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2**16

    max_step = getattr(env, "max_step", None)
    max_step_default = getattr(env, "_max_episode_steps", None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2**10

    if_discrete = isinstance(env.action_space[0], gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space[0].n
        action_max = int(1)
    elif isinstance(
        env.action_space, gym.spaces.Box
    ):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError(
            "| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0"
        )

    if if_print:
        print(
            f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
            f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
            f"\n| max_step:  {max_step:4}, target_return: {target_return}"
        )
    return (
        env_name,
        state_dim,
        action_dim,
        action_max,
        max_step,
        if_discrete,
        target_return,
    )


class Evaluator:
    def __init__(
        self,
        cwd,
        agent_id,
        device,
        eval_env,
        eval_gap,
        eval_times1,
        eval_times2,
    ):
        self.recorder = []  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f"{cwd}/recorder.npy"
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = eval_env.target_return
        self.r_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(
            f"{'#' * 80}\n"
            f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
            f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
            f"{'expR':>8}{'objC':>7}{'etc.':>7}"
        )

    def evaluate_and_save(
        self, act, steps, r_exp, log_tuple
    ) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()

            """evaluate first time"""
            rewards_steps_list = [
                get_episode_return_and_step(self.eval_env, act, self.device)
                for _ in range(self.eval_times1)
            ]

            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(
                rewards_steps_list
            )

            """evaluate second time"""
            if (
                r_avg > self.r_max
            ):  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [
                    get_episode_return_and_step(self.eval_env, act, self.device)
                    for _ in range(self.eval_times2 - self.eval_times1)
                ]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(
                    rewards_steps_list
                )

            """save the policy network"""
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_save_path = f"{self.cwd}/actor.pth"
                torch.save(
                    act.state_dict(), act_save_path
                )  # save policy network in *.pth

                print(
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                )  # save policy and print

            self.recorder.append(
                (self.total_step, r_avg, r_std, r_exp, *log_tuple)
            )  # update recorder

            """print some information to Terminal"""
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(
                    f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                    f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                    f"{'UsedTime':>8}  ########\n"
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                    f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                    f"{self.used_time:>8}  ########"
                )

            print(
                f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}"
            )
            self.draw_plot()
        return if_reach_goal, if_save

    def evaluate_and_save_marl(self, agent, steps, r_exp) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()

            """evaluate first time"""
            rewards_steps_list = [
                get_episode_return_and_step_marl(self.eval_env, agent, self.device)
                for _ in range(self.eval_times1)
            ]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(
                rewards_steps_list
            )
            """evaluate second time"""
            if (
                r_avg > self.r_max
            ):  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [
                    get_episode_return_and_step_marl(self.eval_env, agent, self.device)
                    for _ in range(self.eval_times2 - self.eval_times1)
                ]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(
                    rewards_steps_list
                )

            """save the policy network"""
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                for i in range(agent.n_agents):
                    act_save_path = f"{self.cwd}/actor{i}.pth"
                    torch.save(
                        agent.agents[i].act.state_dict(), act_save_path
                    )  # save policy network in *.pth

                print(
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                )  # save policy and print
            self.recorder.append(
                (self.total_step, r_avg, r_std, r_exp)
            )  # update recorder
            # self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

            """print some information to Terminal"""
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(
                    f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                    f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                    f"{'UsedTime':>8}  ########\n"
                    f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                    f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                    f"{self.used_time:>8}  ########"
                )

            print(
                f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
            )
            # f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
            self.draw_plot()
        return if_reach_goal, if_save

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg = rewards_steps_ary.mean(
            axis=0
        )  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(
            axis=0
        )  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        np.save(self.recorder_path, self.recorder)

        """draw plot and save as png"""
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = (
            f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
        )

        save_learning_curve(self.recorder, self.cwd, save_title)


class ReplayBufferMARL:
    def __init__(self, max_len, state_dim, action_dim, n_agents, if_use_per, gpu_id=0):
        """Experience Replay Buffer
        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        `int max_len` the maximum capacity of ReplayBuffer. First In First Out
        `int state_dim` the dimension of state
        `int action_dim` the dimension of action (action_dim==1 for discrete action)
        `bool if_on_policy` on-policy or off-policy
        `bool if_gpu` create buffer space on CPU RAM or GPU
        `bool if_per` Prioritized Experience Replay for sparse reward
        """
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )
        self.per_tree = BinarySearchTree(max_len) if if_use_per else None
        self.buf_action = torch.empty(
            (max_len, n_agents, action_dim), dtype=torch.float32, device=self.device
        )
        self.buf_reward = torch.empty(
            (max_len, n_agents), dtype=torch.float32, device=self.device
        )
        self.buf_done = torch.empty(
            (max_len, n_agents), dtype=torch.float32, device=self.device
        )
        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty(
                (max_len, n_agents, state_dim), dtype=torch.float32, device=self.device
            )

        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty(
                (max_len, n_agents, *state_dim), dtype=torch.uint8, device=self.device
            )

        else:
            raise ValueError("state_dim")

    def append_buffer(self, state, reward, done, action):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_reward[self.next_idx] = reward
        self.buf_action[self.next_idx] = action
        self.buf_done[self.done] = done

        if self.per_tree:
            self.per_tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, reward, done, action):  # CPU array to CPU array
        size = len(reward)
        next_idx = self.next_idx + size

        if self.per_tree:
            self.per_tree.update_ids(
                data_ids=np.arange(self.next_idx, next_idx) % self.max_len
            )
        if next_idx > self.max_len:
            self.buf_state[self.next_idx : self.max_len] = state[
                : self.max_len - self.next_idx
            ]
            self.buf_reward[self.next_idx : self.max_len] = reward[
                : self.max_len - self.next_idx
            ]
            self.buf_done[self.next_idx : self.max_len] = done[
                : self.max_len - self.next_idx
            ]
            self.buf_action[self.next_idx : self.max_len] = action[
                : self.max_len - self.next_idx
            ]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_reward[0:next_idx] = reward[-next_idx:]
            self.buf_done[0:next_idx] = done[-next_idx:]
            self.buf_action[0:next_idx] = action[-next_idx:]
        else:

            self.buf_state[self.next_idx : next_idx] = state
            self.buf_action[self.next_idx : next_idx] = action
            self.buf_reward[self.next_idx : next_idx] = reward
            self.buf_done[self.next_idx : next_idx] = done
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        if self.per_tree:
            beg = -self.max_len
            end = (
                (self.now_len - self.max_len) if (self.now_len < self.max_len) else None
            )

            indices, is_weights = self.per_tree.get_indices_is_weights(
                batch_size, beg, end
            )
            return (
                self.buf_reward[indices].type(torch.float32),  # reward
                self.buf_done[indices].type(torch.float32),  # mask
                self.buf_action[indices].type(torch.float32),  # action
                self.buf_state[indices].type(torch.float32),  # state
                self.buf_state[indices + 1].type(torch.float32),  # next state
                torch.as_tensor(is_weights, dtype=torch.float32, device=self.device),
            )  # important sampling weights
        else:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            return (
                self.buf_reward[indices],  # reward
                self.buf_done[indices],  # mask
                self.buf_action[indices],  # action
                self.buf_state[indices],
                self.buf_state[indices + 1],
            )

    def update_now_len(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer"""
        self.now_len = self.max_len if self.if_full else self.next_idx

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        """print the state norm information: state_avg, state_std

        We don't suggest to use running stat state.
        We directly do normalization on state using the historical avg and std
        eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
        neg_avg = -states.mean()
        div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

        :array neg_avg: neg_avg.shape=(state_dim)
        :array div_std: div_std.shape=(state_dim)
        """
        max_sample_size = 2**14

        """check if pass"""
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(
                f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. "
            )
            return None

        """sample state"""
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[
            :max_sample_size
        ]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        """compute state norm"""
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(
                f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. "
            )
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = (
            (np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std
        ) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print("print_state_norm: state_avg, state_std (fixed)")
        print(f"avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"std = np.{repr(ary_std).replace('=float32', '=np.float32')}")

    def td_error_update(self, td_error):
        self.per_tree.td_error_update(td_error)

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"
        if_load = None

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state[0].shape
            reward_dim = self.n_agents
            done_dim = self.n_agents
            action_dim = self.buf_action[0].shape

            buf_state_data_type = (
                np.float16
                if self.buf_state.dtype in {np.float, np.float64, np.float32}
                else np.uint8
            )

            buf_state = np.empty((self.max_len, state_dim), dtype=buf_state_data_type)
            buf_reward = np.empty((self.max_len, reward_dim), dtype=np.float16)
            buf_done = np.empty((self.max_len, done_dim), dtype=np.float16)
            buf_action = np.empty((self.max_len, action_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = (
                self.buf_state[self.now_len : self.max_len].detach().cpu().numpy()
            )
            buf_reward[0:temp_len] = (
                self.buf_reward[self.now_len : self.max_len].detach().cpu().numpy()
            )
            buf_done[0:temp_len] = (
                self.buf_done[self.now_len : self.max_len].detach().cpu().numpy()
            )
            buf_action[0:temp_len] = (
                self.buf_action[self.now_len : self.max_len].detach().cpu().numpy()
            )

            buf_state[temp_len:] = self.buf_state[: self.now_len].detach().cpu().numpy()
            buf_reward[temp_len:] = (
                self.buf_reward[: self.now_len].detach().cpu().numpy()
            )
            buf_done[temp_len:] = self.buf_done[: self.now_len].detach().cpu().numpy()
            buf_action[temp_len:] = (
                self.buf_action[: self.now_len].detach().cpu().numpy()
            )

            np.savez_compressed(
                save_path,
                buf_state=buf_state,
                buf_reward=buf_reward,
                buf_done=buf_done,
                buf_action=buf_action,
            )
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict["buf_state"]
            buf_reward = buf_dict["buf_reward"]
            buf_done = buf_dict["buf_done"]
            buf_action = buf_dict["buf_action"]

            buf_state = torch.as_tensor(
                buf_state, dtype=torch.float32, device=self.device
            )
            buf_reward = torch.as_tensor(
                buf_reward, dtype=torch.float32, device=self.device
            )
            buf_done = torch.as_tensor(
                buf_done, dtype=torch.float32, device=self.device
            )
            buf_action = torch.as_tensor(
                buf_action, dtype=torch.float32, device=self.device
            )

            self.extend_buffer(buf_state, buf_reward, buf_done, buf_action)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")
            if_load = True
        else:
            # print(f"| ReplayBuffer FileNotFound: {save_path}")
            if_load = False
        return if_load


class Arguments:
    def __init__(self, if_on_policy=False):
        self.env = None  # the environment for training
        self.agent = None  # Deep Reinforcement Learning algorithm

        """Arguments for training"""
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = (
            2**0
        )  # an approximate target reward usually be closed to 256
        self.learning_rate = 2**-15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2**-8  # 2 ** -8 ~= 5e-3

        self.if_on_policy = if_on_policy
        if self.if_on_policy:  # (on-policy)
            self.net_dim = 2**9  # the network width
            self.batch_size = (
                self.net_dim * 2
            )  # num of transitions sampled from replay buffer.
            self.repeat_times = 2**3  # collect target_step, then update network
            self.target_step = (
                2**12
            )  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2**8  # the network width
            self.batch_size = (
                self.net_dim
            )  # num of transitions sampled from replay buffer.
            self.repeat_times = (
                2**0
            )  # repeatedly update network to keep critic's loss small
            self.target_step = 2**10  # collect target_step, then update network
            self.max_memo = 2**21  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        """Arguments for device"""
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        self.worker_num = (
            2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        )
        self.thread_num = (
            8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        )
        self.visible_gpu = (
            "0"  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        )
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        """Arguments for evaluate and save"""
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2**20  # break training after 'total_step > break_step'
        self.if_allow_break = (
            True  # allow break training when reach goal (early termination)
        )

        self.eval_env = (
            None  # the environment for evaluating. None means set automatically.
        )
        self.eval_gap = 2**7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2**3  # number of times that get episode return in first
        self.eval_times2 = 2**4  # number of times that get episode return in second
        self.eval_device_id = -1  # -1 means use cpu, >=0 means use GPU

    def init_before_training(self, if_main):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.visible_gpu)

        """env"""
        if self.env is None:
            raise RuntimeError(
                "\n| Why env=None? For example:\n| args.env = XxxEnv()\n| args.env = str(env_name)\n| args.env = build_env(env_name), from elegantrl.env import build_env"
            )

        if not (isinstance(self.env, str) or hasattr(self.env, "env_name")):
            raise RuntimeError("\n| What is env.env_name? use env=PreprocessEnv(env).")

        """agent"""
        if self.agent is None:
            raise RuntimeError(
                "\n| Why agent=None? Assignment `args.agent = AgentXXX` please."
            )

        if not hasattr(self.agent, "init"):
            raise RuntimeError(
                "\n| why hasattr(self.agent, 'init') == False\n| Should be `agent=AgentXXX()` instead of `agent=AgentXXX`."
            )

        if self.agent.if_on_policy != self.if_on_policy:
            raise RuntimeError(
                f"\n| Why bool `if_on_policy` is not consistent?"
                f"\n| self.if_on_policy: {self.if_on_policy}"
                f"\n| self.agent.if_on_policy: {self.agent.if_on_policy}"
            )

        """cwd"""
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, "env_name", self.env)
            self.cwd = f"./{agent_name}_{env_name}_{self.visible_gpu}"
        if if_main:
            # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(
                    input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == "y"
                )
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)


"""single processing training"""


def mpe_make_env(scenario_name, benchmark=False):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    return (
        MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
        if benchmark
        else MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    )


def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=True)

    env = build_env(args.env, if_print=False)
    """init: Agent"""
    agent = args.agent
    agent.init(
        args.net_dim,
        env.state_dim,
        env.action_dim,
        args.learning_rate,
        args.marl,
        args.n_agents,
        args.if_per_or_gae,
        args.env_num,
    )
    """init Evaluator"""
    eval_env = build_env(env) if args.eval_env is None else args.eval_env
    evaluator = Evaluator(
        args.cwd,
        agent_id,
        agent.device,
        eval_env,
        args.eval_gap,
        args.eval_times1,
        args.eval_times2,
    )
    evaluator.save_or_load_recoder(if_save=False)
    """init ReplayBuffer"""
    buffer = ReplayBufferMARL(
        max_len=args.max_memo,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=3,
        if_use_per=args.if_per_or_gae,
    )
    buffer.save_or_load_history(args.cwd, if_save=False)

    """start training"""
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    """choose update_buffer()"""

    def update_buffer(_trajectory_list):
        _steps = 0
        _r_exp = 0
        # print(_trajectory_list.shape)
        for _trajectory in _trajectory_list:
            ten_state = torch.as_tensor(
                [item[0] for item in _trajectory], dtype=torch.float32
            )

            ten_reward = torch.as_tensor([item[1] for item in _trajectory])

            ten_done = torch.as_tensor([item[2] for item in _trajectory])
            ten_action = torch.as_tensor([item[3] for item in _trajectory])
            ten_reward = ten_reward * reward_scale  # ten_reward
            ten_mask = (
                1.0 - ten_done * 1
            ) * gamma  # ten_mask = (1.0 - ary_done) * gamma
            buffer.extend_buffer(ten_state, ten_reward, ten_mask, ten_action)
            _steps += ten_state.shape[0]
            _r_exp += ten_reward.mean()  # other = (reward, mask, action)
        return _steps, _r_exp

    """init ReplayBuffer after training start"""
    agent.states = env.reset()
    agent.if_on_policy = True
    if not agent.if_on_policy:
        # if_load = buffer.save_or_load_history(cwd, if_save=False)
        if_load = 0
        if not if_load:
            trajectory = explore_before_training(env, target_step)
            trajectory = [
                trajectory,
            ]

            steps, r_exp = update_buffer(trajectory)

            evaluator.total_step += steps

    """start training loop"""
    if_train = True
    # cnt_train = 0
    state = env.reset()
    for cnt_train in tqdm(range(2000000)):
        #    while if_train or cnt_train < 2000000:
        if cnt_train % 100 == 0 and cnt_train > 0:
            state = env.reset()
        with torch.no_grad():
            traj_temp = []
            actions = []
            for i in range(agent.n_agents):
                action = agent.agents[i].select_actions(state[i])
                actions.append(action)
            next_s, reward, done, _ = env.step(actions)
            traj_temp.append((state, reward, done, actions))
            state = next_s
            steps, r_exp = update_buffer(
                [
                    traj_temp,
                ]
            )
        if cnt_train > agent.batch_size:
            agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        if cnt_train % 1000 == 0:
            with torch.no_grad():
                temp = evaluator.evaluate_and_save_marl(agent, steps, r_exp)
                if_reach_goal, if_save = temp
                if_train = not (
                    (if_allow_break and if_reach_goal)
                    or evaluator.total_step > break_step
                    or os.path.exists(f"{cwd}/stop")
                )

    print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}")

    env.close()
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not agent.if_on_policy else None
    evaluator.save_or_load_recoder(if_save=True)


if __name__ == "__main__":
    gym.logger.set_level(40)  # Block warning
    env = mpe_make_env("simple_spread")
    args = Arguments(if_on_policy=False)  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent = AgentMADDPG()
    args.env = PreprocessEnv(env)
    args.reward_scale = 2**-1  # RewardRange: -200 < -150 < 300 < 334
    args.gamma = 0.95
    args.marl = True
    args.max_step = 100
    args.n_agents = 3
    args.rollout_num = 2  # the number of rollout workers (larger is not always faster)
    train_and_evaluate(
        args
    )  # the training process will terminate once it reaches the target reward.
