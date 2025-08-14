import numpy as np
import os
import sys
from contextlib import contextmanager
from torch.multiprocessing import Pool, set_start_method


from src.memory import TrajMemory, MasterMemory
from src.helper import discounted_rewards



class RolloutGenerator(object):
    """a parallel process trajectory generator, with preprocessing functionality
    """

    def __init__(self, num_processes, num_trajs_per_process, verbose = False):
        self.num_processes = num_processes
        self.num_trajs_per_process = num_trajs_per_process
        self.verbose = verbose # prints the sum of the rewards for each trajectory


    def _condense_state(self, s):
        """Takes A, b, c0, cuts_a, cuts_b and concatenates Ab and cuts
        """
        def append_col(A, b):
            expanded_b = np.expand_dims(b, 1)
            return np.append(A, expanded_b, 1)

        A, b, c0, cuts_a, cuts_b = s
        Ab = append_col(A, b)
        cuts = append_col(cuts_a, cuts_b)
        return (Ab, c0, cuts)


    def _preprocess_state(self, state):
        """
        Given a dataset of { [Ai bi], c0, [Ei di] } i = 0 to N,
        1. min1 = min { Ai, Ei }, max1 = max {Ai, Ei }
        2. min2 = min { bi, di }, max2 = max { bi, di }
        3. Normalize Ai and Ei with min1 and max1. Normalize bi and di with min2 and max2.
        """

        def normalize_matrix(matrix, min_val, max_val):
            """normalizes all elements in matrix to be between min val and max val
            """
            return (max_val - min_val) * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) + min_val

        A, b, c0, cuts_a, cuts_b = state

        min_lhs = np.min([np.min(A), np.min(cuts_a)])
        max_lhs = np.max([np.max(A), np.max(cuts_a)])
        min_rhs = np.min([np.min(b), np.min(cuts_b)])
        max_rhs = np.max([np.max(b), np.max(cuts_b)])

        A = normalize_matrix(A, min_lhs, max_lhs)
        cuts_a = normalize_matrix(cuts_a, min_lhs, max_lhs)

        b = normalize_matrix(b, min_rhs, max_rhs)
        cuts_b = normalize_matrix(cuts_b, min_rhs, max_rhs)

        processed_state = (A, b, c0, cuts_a, cuts_b)
        condensed_state = self._condense_state(processed_state)
        return condensed_state



    def _generate_traj_process(self, env, actor, gamma, process_num, rnd, intrinsic_gamma):
        @contextmanager # supress Gurobi message
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

        np.random.seed()
        trajs = []

        if self.verbose:
            print(f"[{process_num}] Starting trajectory Rollout.")
        for num in range(self.num_trajs_per_process):
            with suppress_stdout():  # remove the Gurobi std out
                s = env.reset()  # samples a random instance every time env.reset() is called
            processed_s = self._preprocess_state(s)
            d = False
            t = 0
            traj_memory = TrajMemory()
            rews = 0
            if rnd != None: # add intrinsic reward for the starts of instances
                intrinsic_r = rnd.compute_intrinsic_reward(processed_s)
                traj_memory.intrinsic_rewards.append(intrinsic_r)

            while not d:
                actsize = len(processed_s[-1])  # k
                prob = actor.compute_prob(processed_s)
                prob /= np.sum(prob)

                a = np.random.choice(actsize, p=prob.flatten())

                new_s, r, d, _ = env.step([a])
                rews += r # keep the original reward tracking unchanged
                t += 1
                traj_memory.add_frame(processed_s, a, r)

                if t > 20 and (np.array(traj_memory.rewards[-10:]) == 0).all() == True:
                    d = True


                if not d:
                    processed_s = self._condense_state(new_s)
                    if rnd != None:
                        intrinsic_r= rnd.compute_intrinsic_reward(processed_s)
                        traj_memory.intrinsic_rewards.append(intrinsic_r)

                # todo: try early stopping?
                delta = 0
                if r < delta and t > 20:
                    break

            if self.verbose:
                print(f"[{process_num}] rews: {rews} \t t: {t}")


            traj_memory.reward_sums.append(rews)

            traj_memory.values = discounted_rewards(traj_memory.rewards, gamma)
            if rnd != None:
                traj_memory.intrinsic_values = discounted_rewards(traj_memory.intrinsic_rewards, intrinsic_gamma)

            trajs.append(traj_memory)
        return trajs

    def generate_trajs(self, env, actor, rnd, gamma, intrinsic_gamma):
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        if self.num_processes == 1: # don't run in parallel
            DATA = []
            DATA.append(self._generate_traj_process(env, actor, gamma, 0, rnd, intrinsic_gamma))
        else:
            env_list = [env] * self.num_processes
            actor_list = [actor] * self.num_processes
            gamma_list = [gamma] * self.num_processes
            rnd_list = [rnd] * self.num_processes
            i_list = np.arange(self.num_processes)
            intrinsic_gamma_list = [intrinsic_gamma] * self.num_processes

            with Pool(processes=self.num_processes) as pool:
                DATA = pool.starmap(self._generate_traj_process,
                                    zip(env_list,
                                        actor_list,
                                        gamma_list,
                                        i_list,
                                        rnd_list,
                                        intrinsic_gamma_list
                                        )
                                    )
            # unpack data
        master_mem = MasterMemory()
        for trajs in DATA:
            for traj in trajs:
                master_mem.add_trajectory(traj)

        return master_mem
