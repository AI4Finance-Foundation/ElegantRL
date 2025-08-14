from abc import ABC, abstractmethod
from collections import namedtuple
from operator import matmul

import numpy as np
import torch
import torch.multiprocessing as mp

from rlsolver.methods.eco_s2v.src.envs.util_envs_eeco import HistoryBuffer
from rlsolver.methods.eco_s2v.src.envs.util_envs import (Observable,
                                                         RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis,
                                                         DEFAULT_OBSERVABLES)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple("action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class SpinSystemFactory(object):
    '''
    Factory class for returning new SpinSystem.
    '''

    @staticmethod
    def get(graph_generator=None,
            max_steps=20,
            observables=DEFAULT_OBSERVABLES,
            reward_signal=RewardSignal.DENSE,
            extra_action=ExtraAction.PASS,
            optimisation_target=OptimisationTarget.ENERGY,
            spin_basis=SpinBasis.SIGNED,
            norm_rewards=False,
            memory_length=None,  # None means an infinite memory.
            horizon_length=None,  # None means an infinite horizon.
            stag_punishment=None,  # None means no punishment for re-visiting states.
            basin_reward=None,  # None means no reward for reaching a local minima.
            reversible_spins=True,  # Whether the spins can be flipped more than once (i.e. True-->Georgian MDP).
            init_snap=None,
            seed=None,
            device=None,
            n_sims=None):

        if graph_generator.biased:
            return SpinSystemBiased(graph_generator, max_steps,
                                    observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                    norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                    reversible_spins,
                                    init_snap, seed)
        else:
            return SpinSystemUnbiased(graph_generator, max_steps,
                                      observables, reward_signal, extra_action, optimisation_target, spin_basis,
                                      norm_rewards, memory_length, horizon_length, stag_punishment, basin_reward,
                                      reversible_spins,
                                      init_snap, seed, device, n_sims)


class SpinSystemBase(ABC):
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class action_space():
        def __init__(self, n_actions, device):
            self.n = n_actions
            self.device = device
            self.actions = torch.arange(self.n, device=self.device)

        def sample(self, n=1):
            return torch.tensor(self.actions, device=self.device)[
                torch.multinomial(torch.ones(len(self.actions)), n, replacement=True)].tolist()

    class observation_space():
        def __init__(self, n_spins, n_observables):
            self.shape = [n_spins, n_observables]

    def __init__(self,
                 graph_generator=None,
                 max_steps=20,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal=RewardSignal.DENSE,
                 extra_action=ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.ENERGY,
                 spin_basis=SpinBasis.SIGNED,
                 norm_rewards=False,
                 memory_length=None,  # None means an infinite memory.
                 horizon_length=None,  # None means an infinite horizon.
                 stag_punishment=None,
                 basin_reward=None,
                 reversible_spins=False,
                 init_snap=None,
                 seed=None,
                 device=None,
                 n_sims=None):
        '''
        Init method.

        Args:
            graph_generator: A GraphGenerator (or subclass thereof) object.
            max_steps: Maximum number of steps before termination.
            reward_signal: RewardSignal enum determining how and when rewards are returned.
            extra_action: ExtraAction enum determining if and what additional action is allowed,
                          beyond simply flipping spins.
            init_snap: Optional snapshot to load spin system into pre-configured state for MCTS.
            seed: Optional random seed.
        '''

        if seed != None:
            np.random.seed(seed)

        # Ensure first observable is the spin state.
        # This allows us to access the spins as self.state[0,:self.n_spins.]
        assert observables[0] == Observable.SPIN_STATE, "First observable must be Observation.SPIN_STATE."
        self.n_sims = n_sims
        self.observables = list(enumerate(observables))
        self.device = device
        self.extra_action = extra_action

        self.gg = graph_generator
        self.n_spins = self.gg.n_spins  # Total number of spins in episode
        self.max_steps = max_steps  # Number of actions before reset

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards

        self.n_actions = self.n_spins
        if extra_action != ExtraAction.NONE:
            self.n_actions += 1

        self.action_space = self.action_space(self.n_actions, self.device)
        self.observation_space = self.observation_space(self.n_spins, len(self.observables))

        self.current_step = 0

        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
            self.bias = None

        self.optimisation_target = optimisation_target
        self.spin_basis = spin_basis

        self.memory_length = memory_length
        self.horizon_length = horizon_length if horizon_length is not None else self.max_steps
        self.stag_punishment = stag_punishment
        self.basin_reward = basin_reward
        self.reversible_spins = reversible_spins

        self.reset()

        # self.score = self.calculate_score()
        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score.clone()
        self.best_spins = self.state[:, 0, :].clone()

        if init_snap != None:
            self.load_snapshot(init_snap)

    def reset(self, spins=None):
        """
        Explanation here
        """
        self.current_step = 0
        if self.gg.biased:
            # self.matrix, self.bias = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix, self.bias = self.gg.get()
            self.bias = self.bias.to(self.device)
        else:
            # self.matrix = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix = self.gg.get()
        self.matrix = self.matrix.to(self.device)
        self._reset_graph_observables()
        spinsOne = torch.ones(self.n_sims, self.n_spins, device=self.device)
        local_rewards_available = self._get_immeditate_cuts_avaialable(spinsOne, self.matrix)
        if torch.any(torch.eq(torch.sum(torch.abs(local_rewards_available), dim=-1), 0)):
            # We've generated an empty graph, this is pointless, try again.
            self.reset()
        else:  # max_local_reward_available存的是得分最大点的值
            self.max_local_reward_available_ = torch.max(local_rewards_available, dim=-1).values
            if (self.max_local_reward_available_ == 0).any():
                self.reset()
            self.max_local_reward_available = self.max_local_reward_available_.unsqueeze(1).expand(-1, self.n_spins)
        self.state = self._reset_state(spins)
        self.score = self.calculate_score()

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score.clone()

        self.best_score = self.score.clone()
        self.best_obs_score = self.score.clone()
        self.best_spins = self.state[:, 0, :self.n_spins].clone()
        self.best_obs_spins = self.state[:, 0, :self.n_spins].clone()

        if self.memory_length is not None:
            self.score_memory = torch.full((self.n_sims, self.memory_length,), self.best_score,
                                           device=self.device)
            self.spins_memory = torch.full((self.n_sims, self.memory_length, self.best_spins.shape[0]),
                                           self.best_spins, device=self.device)
            self.idx_memory = 1

        self._reset_graph_observables()

        if self.stag_punishment is not None or self.basin_reward is not None:
            self.history_buffer = HistoryBuffer(n_sims=self.n_sims, device=self.device)

        return self.get_observation()

    def _reset_graph_observables(self):
        # Reset observed adjacency matrix
        if self.extra_action != self.extra_action.NONE:
            # Pad adjacency matrix for disconnected extra-action spins of value 0.
            self.matrix_obs = torch.zeros((self.matrix.shape[0] + 1, self.matrix.shape[0] + 1), device=self.device)
            self.matrix_obs[:-1, :-1] = self.matrix
        else:
            self.matrix_obs = self.matrix

        # Reset observed bias vector,
        if self.gg.biased:
            if self.extra_action != self.extra_action.NONE:
                # Pad bias for disconnected extra-action spins of value 0.
                self.bias_obs = torch.cat((self.bias, torch.tensor([0], device=self.device)))
            else:
                self.bias_obs = self.bias

    def _reset_state(self, spins=None):
        state = torch.zeros(self.n_sims, self.observation_space.shape[1], self.n_actions,
                            device=self.device)

        if spins is None:
            if self.reversible_spins:
                # For reversible spins, initialise randomly to {+1,-1}.
                state[:, 0, :self.n_spins] = 2 * torch.randint(0, 2, (self.n_sims, self.n_spins,),
                                                               device=self.device, dtype=torch.float) - 1
            else:
                # For irreversible spins, initialise all to +1 (i.e. allowed to be flipped).
                state[0, :self.n_spins] = 1
        else:
            state[0, :] = self._format_spins_to_signed(spins)

        # If any observables other than "immediate energy available" require setting to values other than
        # 0 at this stage, we should use a 'for k,v in enumerate(self.observables)' loop.
        immeditate_rewards_avaialable = self._get_immeditate_cuts_avaialable(state[:, 0, :self.n_spins], self.matrix)
        for idx, obs in self.observables:
            if obs == Observable.IMMEDIATE_REWARD_AVAILABLE:
                state[:, idx, :self.n_spins] = immeditate_rewards_avaialable / self.max_local_reward_available
            elif obs == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                state[:, idx, :self.n_spins] = (
                        1 - torch.sum(immeditate_rewards_avaialable <= 0, dim=-1).float() / self.n_spins).unsqueeze(
                    -1)

        return state

    def _get_spins(self, basis=SpinBasis.SIGNED):
        spins = self.state[:, 0, :self.n_spins]

        if basis == SpinBasis.SIGNED:
            pass
        elif basis == SpinSystemBiased:
            # convert {1,-1} --> {0,1}
            spins[0, :] = (1 - spins[0, :]) / 2
        else:
            raise NotImplementedError("Unrecognised SpinBasis")

        return spins

    def calculate_best_energy(self):
        if self.n_spins <= 10:
            # Generally, for small systems the time taken to start multiple processes is not worth it.
            res = self.calculate_best_brute()

        else:
            # Start up processing pool
            n_cpu = int(mp.cpu_count()) / 2

            pool = mp.Pool(mp.cpu_count())

            # Split up state trials across the number of cpus
            iMax = 2 ** (self.n_spins)
            args = np.round(np.linspace(0, np.ceil(iMax / n_cpu) * n_cpu, n_cpu + 1))
            arg_pairs = [list(args) for args in zip(args, args[1:])]

            # Try all the states.
            #             res = pool.starmap(self._calc_over_range, arg_pairs)
            try:
                res = pool.starmap(self._calc_over_range, arg_pairs)
                # Return the best solution,
                idx_best = np.argmin([e for e, s in res])
                res = res[idx_best]
            except Exception as e:
                # Falling back to single-thread implementation.
                # res = self.calculate_best_brute()
                res = self._calc_over_range(0, 2 ** (self.n_spins))
            finally:
                # No matter what happens, make sure we tidy up after outselves.
                pool.close()

            if self.spin_basis == SpinBasis.BINARY:
                # convert {1,-1} --> {0,1}
                best_score, best_spins = res
                best_spins = (1 - best_spins) / 2
                res = best_score, best_spins

            if self.optimisation_target == OptimisationTarget.CUT:
                best_energy, best_spins = res
                best_cut = self.calculate_cut(best_spins)
                res = best_cut, best_spins
            elif self.optimisation_target == OptimisationTarget.ENERGY:
                pass
            else:
                raise NotImplementedError()

        return res

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def step(self, action):
        done = torch.zeros((self.n_sims), device=self.device, dtype=torch.bool)
        rew = torch.zeros((self.n_sims), device=self.device, dtype=torch.float)  # Default reward to zero.
        randomised_spins = False
        self.current_step += 1

        if self.current_step > self.max_steps:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError

        new_state = self.state.clone()

        ############################################################
        # 1. Performs the action and calculates the score change. #
        ############################################################

        if torch.any(action == self.n_spins):
            if self.extra_action == ExtraAction.PASS:
                delta_score = 0
            elif self.extra_action == ExtraAction.RANDOMISE:
                # Randomise the spin configuration.
                randomised_spins = True
                random_actions = torch.randint(0, 2, (self.n_spins,)).to(
                    torch.float32) * 2 - 1  # Random choice between 1 and -1
                new_state[0, :] = self.state[0, :] * random_actions
                new_score = self.calculate_score(new_state[0, :])
                delta_score = new_score - self.score
                self.score = new_score.clone()
        else:
            # Perform the action and calculate the score change.
            sim_indices = torch.arange(self.state.size(0))
            # print("sim_indices.device: ", sim_indices.device)
            # print("action.device: ", action.device)
            # print("new_state.device: ", new_state.device)
            # print("self.state.device: ", self.state.device)
            new_state[sim_indices, 0, action] = -self.state[sim_indices, 0, action]
            if self.gg.biased:
                delta_score = self._calculate_score_change(new_state[:, 0, :self.n_spins], self.matrix, self.bias,
                                                           action)
            else:
                spins_score = self._get_immeditate_cuts_avaialable(new_state[:, 0, :self.n_spins], self.matrix)
                # delta_score = spins_score[action.unsqueeze(-1)]
                delta_score = -spins_score[sim_indices, action]
                # delta_score = -torch.gather(spins_score, dim=2, index=action.unsqueeze(-1)).squeeze(-1)
                # delta_score = self._calculate_score_change(new_state[:,:,0, :self.n_spins], self.matrix, action)
            # score_new = self.calculate_score(new_state[:,0, :self.n_spins])
            self.score += delta_score

        #############################################################################################
        # 2. Calculate reward for action and update anymemory buffers.                              #
        #   a) Calculate reward (always w.r.t best observable score).                              #
        #   b) If new global best has been found: update best ever score and spin parameters.      #
        #   c) If the memory buffer is finite (i.e. self.memory_length is not None):                #
        #          - Add score/spins to their respective buffers.                                  #
        #          - Update best observable score and spins w.r.t. the new buffers.                #
        #      else (if the memory is infinite):                                                    #
        #          - If new best has been found: update best observable score and spin parameters. #                                                                        #
        #############################################################################################

        # 更新state
        self.state = new_state.clone()

        # 获取即时奖励
        immeditate_rewards_avaialable = spins_score.clone()

        # 更新奖励
        improvement = self.score - self.best_obs_score
        if self.reward_signal == RewardSignal.BLS:
            rew = torch.where(improvement > 0, improvement, torch.tensor(0.0, dtype=improvement.dtype))
        elif self.reward_signal == RewardSignal.CUSTOM_BLS:
            rew = torch.where(improvement > 0, improvement / (improvement + 0.1),
                              torch.tensor(0.0, dtype=improvement.dtype))

        if self.reward_signal == RewardSignal.DENSE:
            rew = delta_score
        elif self.reward_signal == RewardSignal.SINGLE and done:
            rew = self.score - self.init_score

        if self.norm_rewards:
            rew /= self.n_spins

        if self.stag_punishment is not None or self.basin_reward is not None:
            visiting_new_state = self.history_buffer.update(self.state[:, 0, :])
            # visiting_new_state = torch.zeros((self.n_sims), device=self.device, dtype=torch.bool)
        if self.stag_punishment is not None:
            # 原先单环境逻辑： if not visiting_new_state: rew -= self.stag_punishment
            # 现在对二维环境：把 visiting_new_state 当作布尔掩码
            # 对于那些 "not visiting_new_state" 的位置做惩罚
            rew[~visiting_new_state] -= self.stag_punishment

        if self.basin_reward is not None:
            all_negative_mask = torch.all(immeditate_rewards_avaialable <= 0, dim=-1)
            # 这里 all_negative_mask 的形状是 (N1, N2)，bool
            # 第二步：将 all_negative_mask 和 visiting_new_state 结合，得到真正需要奖励的环境
            mask = all_negative_mask & visiting_new_state
            # 第三步：对这些环境加上 basin_reward
            rew[mask] += self.basin_reward

        # 更新best_score和best_spins
        update_mask = self.score > self.best_score
        self.best_score = torch.where(update_mask, self.score, self.best_score)
        self.best_spins = torch.where(update_mask.unsqueeze(-1).expand(-1, self.n_spins),
                                      self.state[:, 0, :self.n_spins], self.best_spins)

        # 处理有限记忆长度的情况
        if self.memory_length is not None:
            # 对于有限的记忆长度
            self.score_memory[:, :, self.idx_memory] = self.score
            self.spins_memory[:, :, self.idx_memory] = self.state[0, :self.n_spins].clone()  # 使用clone避免共享引用
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = self.score_memory.max().clone()
            self.best_obs_spins = self.spins_memory[self.score_memory.argmax()].clone()  # 使用clone避免共享引用
        else:
            self.best_obs_score = self.best_score.clone()
            self.best_obs_spins = self.best_spins.clone()  # 使用clone避免共享引用

        #############################################################################################
        # 3. Updates the state of the system (except self.state[0,:] as this is always the spin     #
        #    configuration and has already been done.                                               #
        #   a) Update self.state local features to reflect the chosen action.                       #                                                                  #
        #   b) Update global features in self.state (always w.r.t. best observable score/spins)     #
        #############################################################################################

        for idx, observable in self.observables:

            ### Local observables ###
            if observable == Observable.IMMEDIATE_REWARD_AVAILABLE:
                self.state[:, idx, :self.n_spins] = immeditate_rewards_avaialable / self.max_local_reward_available

            elif observable == Observable.TIME_SINCE_FLIP:
                self.state[:, idx, :] += (1. / self.max_steps)
                if randomised_spins:
                    self.state[:, idx, :] = self.state[:, idx, :] * (random_actions > 0).to(
                        torch.float32)  # 使用 .to(torch.float32) 使其为浮点数
                else:
                    self.state[sim_indices, idx, action] = 0

            ### Global observables ###
            elif observable == Observable.EPISODE_TIME:
                self.state[:, idx, :] += (1. / self.max_steps)

            elif observable == Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                self.state[:, idx, :] = torch.max(torch.tensor(0., device=self.device),
                                                  torch.tensor(
                                                      (self.current_step - self.max_steps) / self.horizon_length,
                                                      device=self.device) + 1)

            elif observable == Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                # 使用 torch.sum() 来替代 np.sum()
                # 这里为什么要用1-？
                self.state[:, idx, :] = (1 - torch.sum(immeditate_rewards_avaialable <= 0, dim=-1).to(
                    torch.float32) / self.n_spins).unsqueeze(-1)

            elif observable == Observable.DISTANCE_FROM_BEST_SCORE:
                self.state[:, idx, :] = (
                        torch.abs(self.score - self.best_obs_score) / self.max_local_reward_available[:,
                                                                      0]).unsqueeze(-1)

            elif observable == Observable.DISTANCE_FROM_BEST_STATE:
                # 使用 torch.count_nonzero 来代替 np.count_nonzero
                self.state[:, idx, :self.n_spins] = torch.count_nonzero(
                    (self.best_obs_spins[:, :self.n_spins] - self.state[:, 0, :self.n_spins])
                    , dim=-1).unsqueeze(-1)

        #############################################################################################
        # 4. Check termination criteria.                                                            #
        #############################################################################################
        if self.current_step == self.max_steps:
            # Maximum number of steps taken --> done.
            # print("Done : maximum number of steps taken")
            done[:] = True

        if not self.reversible_spins:
            # 使用 PyTorch 的 `nonzero()` 来替代 NumPy 的 `nonzero()`
            if len((self.state[0, :self.n_spins] > 0).nonzero()) == 0:
                # If no more spins to flip --> done.
                # print("Done : no more spins to flip")
                done = True

        return (self.get_observation(), rew, done)

    def get_observation(self):
        state = self.state.clone()
        if self.spin_basis == SpinBasis.BINARY:
            # convert {1,-1} --> {0,1}
            state[:, 0, :] = (1 - state[:, 0, :]) / 2

        if self.gg.biased:
            return torch.cat((state, self.matrix_obs, self.bias_obs), dim=0)
        else:
            return torch.cat((state, self.matrix_obs), dim=-2)

    def get_immeditate_rewards_avaialable(self, spins=None):
        if spins is None:
            spins = self._get_spins()

        if self.optimisation_target == OptimisationTarget.ENERGY:
            immediate_reward_function = lambda *args: -1 * self._get_immeditate_energies_avaialable_jit(*args)
        elif self.optimisation_target == OptimisationTarget.CUT:
            immediate_reward_function = self._get_immeditate_cuts_avaialable_jit
        else:
            raise NotImplementedError("Optimisation target {} not recognised.".format(self.optimisation_ta))

        if self.gg.biased:
            bias = self.bias.astype('float64')
            return immediate_reward_function(spins, self.matrix, bias)
        else:
            return immediate_reward_function(spins, self.matrix)

    def get_allowed_action_states(self):
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0, 1)
            elif self.spin_basis == SpinBasis.SIGNED:
                return (1, -1)
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis == SpinBasis.BINARY:
                return 0
            if self.spin_basis == SpinBasis.SIGNED:
                return 1

    def calculate_score(self, spins=None):
        if self.optimisation_target == OptimisationTarget.CUT:
            score = self.calculate_cut(spins)
        elif self.optimisation_target == OptimisationTarget.ENERGY:
            score = -1. * self.calculate_energy(spins)
        else:
            raise NotImplementedError
        return score

    def _calculate_score_change(self, new_spins, matrix, action):
        if self.optimisation_target == OptimisationTarget.CUT:
            delta_score = self._calculate_cut_change(new_spins, matrix, action)
        elif self.optimisation_target == OptimisationTarget.ENERGY:
            delta_score = -1. * self._calculate_energy_change(new_spins, matrix, action)
        else:
            raise NotImplementedError
        return delta_score

    def _format_spins_to_signed(self, spins):
        if self.spin_basis == SpinBasis.BINARY:
            if not np.isin(spins, [0, 1]).all():
                raise Exception("SpinSystem is configured for binary spins ([0,1]).")
            # Convert to signed spins for calculation.
            spins = 2 * spins - 1
        elif self.spin_basis == SpinBasis.SIGNED:
            if not np.isin(spins, [-1, 1]).all():
                raise Exception("SpinSystem is configured for signed spins ([-1,1]).")
        return spins

    @abstractmethod
    def calculate_energy(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def calculate_cut(self, spins=None):
        raise NotImplementedError

    @abstractmethod
    def get_best_cut(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_over_range(self, i0, iMax):
        raise NotImplementedError

    @abstractmethod
    def _calculate_energy_change(self, new_spins, matrix, action):
        raise NotImplementedError

    @abstractmethod
    def _calculate_cut_change(self, new_spins, matrix, action):
        raise NotImplementedError


##########
# Classes for implementing the calculation methods with/without biases.
##########
class SpinSystemUnbiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        else:
            spins = self._format_spins_to_signed(spins)

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')

        return self._calculate_energy_jit(spins, matrix)

    # 不传入spins就用self.spins算，传的话就用传进来的算
    def calculate_cut(self, spins=None):
        if spins is None:
            spins = self._get_spins()
        # else:
        #     spins = self._format_spins_to_signed(spins)
        return ((1 / 4) * (torch.matmul(self.matrix, spins.unsqueeze(-1)).squeeze(-1) * -spins).sum(dim=-1) + (1 / 4)
                * torch.sum(self.matrix, dim=(-1, -2)))

    def get_best_cut(self):
        if self.optimisation_target == OptimisationTarget.CUT:
            return self.best_score
        else:
            raise NotImplementedError("Can't return best cut when optimisation target is set to energy.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix)

    @staticmethod
    # @jit(float64(float64[:],float64[:,:],int64), nopython=True)
    def _calculate_energy_change(new_spins, matrix, action):
        return -2 * new_spins[action] * matmul(new_spins.T, matrix[:, action])

    @staticmethod
    # @jit(float64(float64[:],float64[:,:],int64), nopython=True)
    def _calculate_cut_change(new_spins, matrix, action):
        action_expanded = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 20)
        # 使用 gather 对第 2 维（索引为 2）进行索引
        return (-1 * torch.gather(new_spins, dim=2, index=action.unsqueeze(-1))
                * torch.matmul(new_spins.unsqueeze(-2), torch.gather(matrix, dim=2, index=action_expanded).squeeze(2)))

    @staticmethod
    # @jit(float64(float64[:],float64[:,:]), nopython=True)
    def _calculate_energy_jit(spins, matrix):
        return - torch.matmul(spins.T, torch.matmul(matrix, spins)) / 2

    @staticmethod
    def __calc_over_range(list_spins, matrix):
        # 初始化高能值
        energy = float('inf')  # 代替 1e50
        best_spins = None
        for spins in list_spins:
            spins = torch.tensor(spins, dtype=torch.float64)
            current_energy = -torch.matmul(spins.T, torch.matmul(matrix, spins)) / 2
            if current_energy < energy:
                energy = current_energy
                best_spins = spins.clone()
        return energy, best_spins

    @staticmethod
    def _get_immediate_energies_available(spins, matrix):
        spins = torch.tensor(spins, dtype=torch.float)
        matrix = torch.tensor(matrix, dtype=torch.float)
        return 2 * spins * torch.matmul(matrix, spins)

    @staticmethod
    def _get_immeditate_cuts_avaialable(spins, matrix):
        return torch.matmul(matrix, spins.unsqueeze(-1)).squeeze(-1) * spins


class SpinSystemBiased(SpinSystemBase):

    def calculate_energy(self, spins=None):
        if type(spins) == type(None):
            spins = self._get_spins()

        spins = spins.astype('float64')
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')

        return self._calculate_energy_jit(spins, matrix, bias)

    def calculate_cut(self, spins=None):
        raise NotImplementedError("MaxCut not defined/implemented for biased SpinSystems.")

    def get_best_cut(self):
        raise NotImplementedError("MaxCut not defined/implemented for biased SpinSystems.")

    def _calc_over_range(self, i0, iMax):
        list_spins = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_spins))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        bias = self.bias.astype('float64')
        return self.__calc_over_range_jit(list_spins, matrix, bias)
