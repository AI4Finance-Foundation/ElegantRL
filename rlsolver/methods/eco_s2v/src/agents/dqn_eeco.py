"""
Implements a DQN learning agent.
"""

import itertools
import math
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rlsolver.methods.eco_s2v.config import *
from rlsolver.methods.eco_s2v.src.agents.utils import Logger, TestMetric, set_global_seed
from rlsolver.methods.eco_s2v.src.agents.utils import eeco_ReplayBuffer as ReplayBuffer
from rlsolver.methods.eco_s2v.src.envs.util_envs import ExtraAction

fix_seed = False # if test stepVsObj, set it as True; and False otherwise.
if fix_seed:
    seed = 74
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DQN:
    """
    # Required parameters.
    envs : List of environments to use.
    network : Choice of neural network.

    # Initial network parameters.
    init_network_params : Pre-trained network to load upon initialisation.
    init_weight_std : Standard deviation of initial network weights.

    # DQN parameters
    double_dqn : Whether to use double DQN (DDQN).
    update_target_frequency : How often to update the DDQN target network.
    gamma : Discount factor.
    clip_Q_targets : Whether negative Q targets are clipped (generally True/False for irreversible/reversible agents).

    # Replay buffer.
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.

    # Learning rate
    update_learning_rate : Whether to dynamically update the learning rate (if False, initial_learning_rate is always used).
    initial_learning_rate : Initial learning rate.
    peak_learning_rate : The maximum learning rate.
    peak_learning_rate_step : The timestep (from the start, not from when training starts) at which the peak_learning_rate is found.
    final_learning_rate : The final learning rate.
    final_learning_rate_step : The timestep of the final learning rate.

    # Optional regularization.
    max_grad_norm : The norm grad to clip gradients to (None means no clipping).
    weight_decay : The weight decay term for regularisation.

    # Exploration
    update_exploration : Whether to update the exploration rate (False would tend to be used with NoisyNet layers).
    initial_exploration_rate : Inital exploration rate.
    final_exploration_rate : Final exploration rate.
    final_exploration_step : Timestep at which the final exploration rate is reached.

    # Loss function
    adam_epsilon : epsilon for ADAM optimisation.
    loss="mse" : Loss function to use.

    # Saving the agent
    save_network_frequency : Frequency with which the network parameters are saved.
    network_save_path : Folder into which the network parameters are saved.

    # Testing the agent
    evaluate : Whether to test the agent during training.
    test_envs : List of test environments.  None means the training environments (envs) are used.
    test_episodes : Number of episodes at each test point.
    test_obj_frequency : Frequency of tests.
    test_save_path : Folder into which the test scores are saved.
    test_metric : The metric used to quantify performance.

    # Other
    logging : Whether to log.
    seed : The global seed to set.  None means randomly selected.
    """

    def __init__(
            self,
            envs,
            network,

            # Initial network parameters.
            init_network_params=None,
            init_weight_std=None,

            # DQN parameters
            double_dqn=True,
            update_target_frequency=10000,
            gamma=0.99,
            clip_Q_targets=False,

            # Replay buffer.
            replay_start_size=50000,
            replay_buffer_size=1000000,
            minibatch_size=32,
            update_frequency=1,

            # Learning rate
            update_learning_rate=True,
            initial_learning_rate=0,
            peak_learning_rate=1e-3,
            peak_learning_rate_step=10000,
            final_learning_rate=5e-5,
            final_learning_rate_step=200000,

            # Optional regularization.
            max_grad_norm=None,
            weight_decay=0,

            # Exploration
            update_exploration=True,
            initial_exploration_rate=1,
            final_exploration_rate=0.1,
            final_exploration_step=1000000,

            # Loss function
            adam_epsilon=1e-8,
            loss="mse",

            # Saving the agent
            save_network_frequency=10000,
            network_save_path='network',

            # Testing the agent
            evaluate=True,
            test_envs=None,
            test_episodes=20,
            test_obj_frequency=10,
            test_save_path='test_scores',
            test_metric=TestMetric.ENERGY_ERROR,

            # Other
            logging=True,
            seed=None,
            test_sampling_speed=False,
            logger_save_path=None,
            sampling_patten=None,
            sample_device=None,
            buffer_device=None,
            sampling_speed_save_path=None,
            args=None,
    ):

        self.double_dqn = double_dqn
        self.train_device = TRAIN_DEVICE
        self.sample_device = sample_device
        self.buffer_device = buffer_device
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.clip_Q_targets = clip_Q_targets
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size

        self.update_learning_rate = update_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.peak_learning_rate_step = peak_learning_rate_step
        self.final_learning_rate = final_learning_rate
        self.final_learning_rate_step = final_learning_rate_step

        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.update_frequency = update_frequency
        self.update_exploration = update_exploration
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.test_sampling_speed = test_sampling_speed
        self.logger_save_path = logger_save_path
        self.sampling_patten = sampling_patten
        self.sampling_speed_save_path = sampling_speed_save_path
        self.args = args
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        self.env = envs
        matrix_index_cycle = itertools.cycle(range(math.ceil(self.replay_buffer_size / (self.env.max_steps * self.env.n_sims))))
        self.n_matrix = math.ceil(self.replay_buffer_size / (self.env.max_steps * self.env.n_sims)) * self.env.n_sims
        self.acting_in_reversible_spin_env = self.env.reversible_spins
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, sampling_patten=self.sampling_patten
                                          , device=self.buffer_device, n_matrix=self.n_matrix
                                          , matrix_index_cycle=matrix_index_cycle)
        self.seed = random.randint(0, 1000000) if seed is None else seed

        set_global_seed(self.seed, self.env)

        self.network = network().to(self.train_device)
        self.init_network_params = init_network_params
        self.init_weight_std = init_weight_std
        if self.init_network_params != None:
            print("Pre-loading network parameters from {}.\n".format(init_network_params))
            self.load(init_network_params)
        else:
            if self.init_weight_std != None:
                def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        print("Setting weights for", m)
                        m.weight.normal_(0, init_weight_std)

                with torch.no_grad():
                    self.network.apply(init_weights)

        self.target_network = network().to(self.train_device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay)

        self.evaluate = evaluate
        if test_envs in [None, [None]]:
            # By default, test on the same environment(s) as are trained on.
            self.test_envs = self.envs
        else:
            if type(test_envs) != list:
                test_envs = [test_envs]
            self.test_envs = test_envs
        self.test_episodes = int(test_episodes)
        self.test_obj_frequency = test_obj_frequency
        self.test_save_path = test_save_path
        self.test_metric = test_metric

        self.losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "losses.pkl")

        if not self.acting_in_reversible_spin_env:
            for env in self.envs:
                assert env.extra_action == ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."
            for env in self.test_envs:
                assert env.extra_action == ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."

        self.allowed_action_state = self.env.get_allowed_action_states()

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path

    def get_replay_buffer_for_env(self, env):
        return self.replay_buffers[env.action_space.n]

    def get_random_replay_buffer(self):
        return random.sample(self.replay_buffers.items(), k=1)[0][1]

    def learn(self, timesteps, start_time=None, verbose=False):
        start_time_of_learn = time.time()
        total_time = 0

        if self.logging:
            if not self.test_sampling_speed:
                logger = Logger(save_path=self.logger_save_path, args=self.args, n_sims=self.env.n_sims)
            else:
                logger = Logger(save_path=self.sampling_speed_save_path, args=self.args, n_sims=self.env.n_sims)
        path = self.network_save_path
        path_main, path_ext = os.path.splitext(path)
        if path_ext == '':
            path_ext += '.pth'
        self.save(path_main + "_0" + path_ext)
        last_record_obj_time = time.time()

        # Initialise the state
        state = self.env.reset()
        state = state.to(self.sample_device)
        self.replay_buffer.record_matrix(state[:, 7:, :])
        if USE_TWO_DEVICES_IN_ECO_S2V:
            score = torch.zeros((self.env.n_sims), device=self.sample_device, dtype=torch.float)
        else:
            score = torch.zeros((self.env.n_sims), device=self.train_device, dtype=torch.float)

        losses_eps = []
        t1 = time.time()
        test_scores = []
        losses = []
        is_training_ready = False
        if_buffer_full = False
        for timestep in range(timesteps):
            start_time_this_step = time.time()
            if timestep * self.env.n_sims >= self.replay_buffer_size:
                if_buffer_full = True
            if not is_training_ready:
                # if all([len(rb) >= self.replay_start_size for rb in self.replay_buffers.values()]):
                # if time
                if self.replay_start_size <= timestep * self.env.n_sims:
                    print('\nAll buffers have {} transitions stored - training is starting!\n'.format(
                        self.replay_start_size))
                    is_training_ready = True
                    training_ready_step = timestep

            # Choose action
            state_device = self.sample_device if USE_TWO_DEVICES_IN_ECO_S2V else self.train_device
            action = self.act(state.to(state_device).float(), is_training_ready=is_training_ready).to(state_device)

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            # Update learning rate
            if self.update_learning_rate:
                self.update_lr(timestep)

            # Perform action in environment
            state_next, reward, done = self.env.step(action)

            # print("score.device: ", score.device)
            # print("reward.device: ", reward.device)
            # if score.device != reward.device:
            #     aaa = 1
            score += reward
            # Store transition in replay buffer
            self.replay_buffer.add(state.half(), action, reward.half(), state_next.half(), done, score)

            if self.test_sampling_speed:  # save log
                num_samples_per_second = self.env.n_sims / (time.time() - start_time_this_step)
                logger.add_scalar('step_vs_num_samples_per_second', timestep, num_samples_per_second)

            if done[0]:
                # Reinitialise the state
                if verbose:
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, episode time: {}, score : {}, mean loss: {}, time : {} s".format(
                        timestep,
                        self.env.current_step,
                        torch.mean(score),
                        loss_str,
                        round(time.time() - t1, 3), ))
                t1 = time.time()
                state = self.env.reset()
                self.replay_buffer.record_matrix(state[:, 7:, :])
                score_device = SAMPLE_DEVICE_IN_ECO_S2V if USE_TWO_DEVICES_IN_ECO_S2V else TRAIN_DEVICE
                score = torch.zeros((self.env.n_sims), device=score_device, dtype=torch.float)
                losses_eps = []
            else:
                state = state_next
            if is_training_ready:
                # Update the main network
                if timestep % self.update_frequency == 0:
                    # Sample a batch of transitions
                    # transitions = self.get_random_replay_buffer().sample(self.minibatch_size, self.train_device)
                    if if_buffer_full:
                        transitions = self.replay_buffer.sample(self.minibatch_size)
                    else:
                        transitions = self.replay_buffer.sample(self.minibatch_size, (timestep + 1) * self.env.n_sims)
                    # Train on selected batch
                    loss = self.train_step(transitions)
                    losses.append([timestep, loss])
                    losses_eps.append(loss)
                # Periodically update target network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
            if timestep % self.test_obj_frequency == 0 and self.evaluate and is_training_ready:
                total_time += time.time() - start_time_of_learn
                test_score = self.evaluate_agent()
                start_time = time.time()
                print('\nTest score: {:.2f}\n'.format(test_score))
                if self.test_metric in [TestMetric.FINAL_CUT, TestMetric.MAX_CUT, TestMetric.CUMULATIVE_REWARD]:
                    best_network = all([test_score > score for t, score in test_scores])
                elif self.test_metric in [TestMetric.ENERGY_ERROR, TestMetric.BEST_ENERGY]:
                    best_network = all([test_score < score for t, score in test_scores])
                else:
                    raise NotImplementedError("{} is not a recognised TestMetric".format(self.test_metric))

                if self.logging:
                    if timestep == 0:
                        logger.add_scalar('step_vs_obj', 0, test_score)
                        logger.add_scalar('time_vs_obj', 0, test_score)
                    else:
                        logger.add_scalar('step_vs_obj', timestep - training_ready_step, test_score)
                        logger.add_scalar('time_vs_obj', total_time, test_score)

                if best_network:
                    self.save(path_main + "_0" + path_ext)

                test_scores.append([timestep, test_score])

            curr_time = time.time()
            if (curr_time - last_record_obj_time >= self.save_network_frequency) and is_training_ready:
                total_time += curr_time - start_time

                path_main_ = path_main + '_' + str(int(total_time))
                if self.logging and timestep % self.update_frequency == 0:
                    logger.add_scalar('step_vs_loss', timestep - training_ready_step, loss)
                    logger.add_scalar('time_vs_loss', total_time, loss)

                self.save(path_main_ + path_ext)
                start_time = curr_time
                last_record_obj_time = curr_time
        if self.logging:
            logger.save()

    @torch.no_grad()
    def __only_bad_actions_allowed(self, state, network):
        x = (state[0, :] == self.allowed_action_state).nonzero()
        q_next = network(state.to(self.train_device).float())[x].max()
        return True if q_next < 0 else False

    # def train_step(self, transitions,scaler):

    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions
        states = states.to(torch.float)
        rewards = rewards.to(torch.float)
        states_next = states_next.to(torch.float)
        # Calculate target Q
        with torch.no_grad():

            network_output = self.network(states_next.clone())
            target_network_output = self.target_network(states_next.clone())
            greedy_actions = network_output.argmax(-1, True)
            q_value_target = target_network_output.gather(-1, greedy_actions)
        if self.clip_Q_targets:
            q_value_target[q_value_target < 0] = 0

        # Calculate TD target
        # dones以bool存储的，
        td_target = rewards + dones.logical_not() * self.gamma * q_value_target.squeeze(-1)
        # Calculate Q value
        q_value = self.network(states.clone()).gather(-1, actions.unsqueeze(-1))

        # Calculate loss
        loss = self.loss(q_value, td_target.unsqueeze(-1), reduction='mean')
        # Update weights

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:  # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # 使用 PyTorch 进行 Q 函数预测
            with torch.no_grad():  # 关闭梯度计算
                action = self.predict(state.clone()).squeeze(-1)  # 假设 self.predict 返回的 action
        else:
            if self.acting_in_reversible_spin_env:
                # 在可逆环境中，随机选择动作
                # action = np.random.randint(0, self.env.action_space.n)
                action_device = self.sample_device if USE_TWO_DEVICES_IN_ECO_S2V else self.train_device
                action = torch.randint(0, self.env.action_space.n,
                                       (self.env.n_sims,), device=action_device,
                                       dtype=torch.long)
            else:
                # 从尚未翻转的 spin 中随机选择一个
                state_tensor = torch.tensor(state, dtype=torch.float32)  # 转换为 PyTorch Tensor
                allowed_actions = (state_tensor[0, :] == self.allowed_action_state).nonzero(as_tuple=True)[0]
                action = allowed_actions[torch.randint(0, len(allowed_actions), (1,))].item()
        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
                timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def update_lr(self, timestep):
        if timestep <= self.peak_learning_rate_step:
            lr = self.initial_learning_rate - (self.initial_learning_rate - self.peak_learning_rate) * (
                    timestep / self.peak_learning_rate_step
            )
        elif timestep <= self.final_learning_rate_step:
            lr = self.peak_learning_rate - (self.peak_learning_rate - self.final_learning_rate) * (
                    (timestep - self.peak_learning_rate_step) / (
                    self.final_learning_rate_step - self.peak_learning_rate_step)
            )
        else:
            lr = None

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    @torch.no_grad()
    def predict(self, states, acting_in_reversible_spin_env=None):

        if acting_in_reversible_spin_env is None:
            acting_in_reversible_spin_env = self.acting_in_reversible_spin_env

        states = states.to(self.train_device)
        qs = self.network(states)

        if acting_in_reversible_spin_env:
            if qs.dim() == 1:
                actions = qs.argmax().item()
            else:
                actions = qs.argmax(-1, True)
            return actions
        else:
            if qs.dim() == 1:
                x = (states[0, :] == self.allowed_action_state).nonzero()
                actions = x[qs[x].argmax().item()].item()
            else:
                disallowed_actions_mask = (states[:, :, 0] != self.allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -10000)
                actions = qs_allowed.argmax(1, True).squeeze(1)
            return actions

    @torch.no_grad()
    def evaluate_agent(self):
        """
        Evaluates agent's current performance.  Run multiple evaluations at once
        so the network predictions can be done in batches.
        """
        obs = self.test_envs[0].reset()
        test_env = deepcopy(self.test_envs[0])

        # self.predict(obs).squeeze(-1)

        done = torch.zeros((test_env.n_sims), dtype=torch.bool, device=test_env.device)
        actions_device = SAMPLE_DEVICE_IN_ECO_S2V if USE_TWO_DEVICES_IN_ECO_S2V else TRAIN_DEVICE
        actions = self.predict(obs).squeeze(-1)
        actions = actions.to(actions_device)
        while not done[0]:
            obs, rew, done = test_env.step(actions)
            actions = self.predict(obs).squeeze(-1)
            actions = actions.to(actions_device)

            if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                test_scores += rew
            if done[0]:
                if self.test_metric == TestMetric.BEST_ENERGY:
                    test_scores = test_env.best_energy
                elif self.test_metric == TestMetric.ENERGY_ERROR:
                    test_scores = abs(test_env.best_energy - test_env.calculate_best()[0])
                elif self.test_metric == TestMetric.MAX_CUT:
                    test_scores = test_env.get_best_cut()
                elif self.test_metric == TestMetric.FINAL_CUT:
                    test_scores = test_env.calculate_cut()

        if self.test_metric == TestMetric.ENERGY_ERROR:
            print("\n{}/{} graphs solved optimally".format(np.count_nonzero(np.array(test_scores) == 0),
                                                           self.test_episodes), end="")
        test_env.matrix_obs, test_env.state = None, None

        return torch.mean(test_scores)

    def save(self, path='network.pth'):
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if os.path.splitext(path)[-1] == '':
            path += '.pth'
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.train_device))
